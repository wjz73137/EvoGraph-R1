#!/usr/bin/env python3
"""Build a four-document real GraphR1 smoke KB, or serve its retrieval API.

Only the first passage of the first four distinct train articles is used.
Official KB text is extracted by local Qwen, indexed by real GME, and grounded
to existing GLDv2 images. No fabricated records, QA answers, API keys, training,
legacy expr copying, deletions, or dependency changes are involved.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evograph_mm.kb.gldv2_subset import atomic_json, image_info
from scripts.download_gldv2_thumbnails import now
from scripts.run_evqa_gpu_smoke import ROOT, SUBSET, MODEL, LOGS, idle_gpu, sanitized
from scripts.run_evqa_retrieval_smoke import GME, chunks, sha256

OUTPUT_ROOT = ROOT / 'expr_mm/evqa_native_graph_smoke'
OUTPUT = OUTPUT_ROOT / 'E-VQA'
PREFIX = 'evqa_native_graph_smoke'
MAX_NEW_TOKENS = 1024
PROMPT_VERSION = None
REVIEW_EXTRACTION = False
REVIEW_PROMPT = None


def normalize_record_delimiters(output):
    """Repair only a missing '<' before a final numeric score, not any facts.

    Some local outputs use '|>96)' rather than '<|>96)'. Preserve the raw
    generation and count this syntax-only adaptation for the native parser.
    """
    lines, repairs = [], 0
    for line in output.splitlines(keepends=True):
        if line.lstrip().startswith(('("entity"<|>', '("hyper-relation"<|>')):
            line, n = re.subn(r'(?<!<)\|>([0-9]+(?:\.[0-9]+)?)(?=\)(?:##)?\s*$)',
                              r'<|>\1', line)
            repairs += n
        lines.append(line)
    return ''.join(lines), repairs


def validate_extraction_output(output):
    """Reject empty/terminator-only generations before they can enter the graph."""
    if '("hyper-relation"' not in output or '("entity"' not in output:
        raise RuntimeError('local graph LLM returned no parseable hyper-relations/entities')
    return output


def graph_bundle(rows, pages, images, count=4):
    """Use URL/image associations, never questions/answers/evidence labels."""
    from evograph_mm.kb.store import RecordBundle, text_embedding_id, visual_embedding_id
    docs, visual, links, seen = [], [], [], set()
    for row in rows:
        url = row['wikipedia_url'].split('|')[0].strip()
        if url in seen:
            continue
        page = pages[url]
        image_id = row['dataset_image_ids'].strip()
        image_path = images[image_id]
        section_id = next((i for i, t in enumerate(page['section_texts']) if t.strip()), None)
        if section_id is None or len(page['section_texts']) != len(page['section_titles']):
            raise ValueError('selected official page has no aligned real source text')
        _, _, passage = next(chunks(page['section_texts'][section_id]))
        identity = hashlib.sha256(url.encode()).hexdigest()[:20]
        data_id, text_id = f'wiki::{identity}', f'text::wiki::{identity}'
        metadata = {'wikipedia_url': url, 'wikipedia_title': page['title'],
                    'section_id': section_id, 'section_title': page['section_titles'][section_id],
                    'source_row': {'wikipedia_url': url, 'wikipedia_title': page['title']}}
        docs.append({'text_doc_id': text_id, 'data_id': data_id, 'split': 'train',
                     'image_id': image_id, 'image_path': str(image_path),
                     'source_metadata': metadata,
                     'contents': f"\"{page['title']}\"\nSection: {page['section_titles'][section_id]}\n{passage}"})
        visual.append({'visual_record_id': f'visual::{image_id}', 'image_id': image_id,
                       'image_path': str(image_path), 'data_id': data_id, 'split': 'train',
                       'source_metadata': metadata, 'image_missing': False})
        for source, target, relation in [
            (data_id, text_id, 'has_text_document'),
            (text_id, text_embedding_id(text_id), 'has_text_embedding'),
            (f'visual::{image_id}', visual_embedding_id(image_id), 'has_visual_embedding'),
        ]:
            links.append({'source_id': source, 'target_id': target, 'relation': relation,
                          'data_id': data_id, 'split': 'train'})
        seen.add(url)
        if len(docs) == count:
            break
    if len(docs) != count or len({r['image_id'] for r in visual}) != count:
        raise ValueError('not enough distinct aligned train articles/images for smoke graph')
    return RecordBundle(docs, visual, links)


def prepare():
    from evograph_mm.kb.layout import build_layout
    subset = ROOT / 'datasets_mm/E-VQA/subsets' / SUBSET
    kb_file = ROOT / 'datasets_mm/E-VQA/raw/kb' / f'{SUBSET}_wiki_pages.json'
    kb = json.loads(kb_file.read_text())
    manifest = json.loads((subset / 'manifest.json').read_text())
    if not kb['complete'] or kb['missing_urls'] or manifest['summary']['status'] != 'complete':
        raise RuntimeError('source subset/pages are incomplete')
    images = {r['image_id']: ROOT / r['subset_path'] for r in manifest['copied_images']}
    with (subset / 'qa_train.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    bundle = graph_bundle(rows, kb['pages'], images)
    for record in bundle.visual_records:
        path = Path(record['image_path'])
        expected = next(r['sha256'] for r in manifest['copied_images']
                        if r['image_id'] == record['image_id'])
        if not image_info(path) or sha256(path) != expected:
            raise RuntimeError('selected GLDv2 image failed decoding/hash verification')
    plan = {'pipeline_version': 1, 'dataset': 'E-VQA', 'subset': SUBSET,
            'scope': 'first passage of first four distinct existing train articles',
            'source_pages_sha256': sha256(kb_file),
            'qa_hashes': {s: sha256(subset / f'qa_{s}.csv') for s in ['train', 'test']},
            'documents': [{k: d[k] for k in ['text_doc_id', 'image_id', 'contents', 'source_metadata']}
                          for d in bundle.text_documents],
            'graph_llm': f'local/{MODEL.name}', 'embedding': 'local real GME',
            'gleaning_passes': 0, 'llm_max_new_tokens': MAX_NEW_TOKENS,
            'review_extraction': REVIEW_EXTRACTION,
            'no_gold_qa_in_kb': True, 'no_remote_api_calls': True}
    if PROMPT_VERSION:
        plan['prompt_version'] = PROMPT_VERSION
    owner = OUTPUT / 'owner.json'
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        if not owner.is_file() or json.loads(owner.read_text()) != plan:
            raise RuntimeError('existing graph output belongs to another configuration; nothing overwritten')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_json(owner, plan)
    layout = build_layout(ROOT, 'E-VQA', SUBSET, OUTPUT_ROOT)
    return plan, layout, bundle


def graph_provenance(bundle):
    """Validate every extracted graph node's source, then write URL sidecars."""
    import networkx as nx
    from graphr1.utils import compute_mdhash_id
    graph = nx.read_graphml(OUTPUT / 'graph_chunk_entity_relation.graphml')
    chunks_by_id = json.loads((OUTPUT / 'kv_store_text_chunks.json').read_text())
    sources = {compute_mdhash_id(d['contents'].strip(), prefix='doc-'): d['source_metadata']
               for d in bundle.text_documents}
    full_docs = json.loads((OUTPUT / 'kv_store_full_docs.json').read_text())
    if set(full_docs) != set(sources) or not chunks_by_id:
        raise RuntimeError('native GraphR1 did not persist all four source documents/chunks')
    sidecar = {'entity': {}, 'hyperedge': {}}
    for node, attributes in graph.nodes(data=True):
        chunk_ids = [v for v in attributes.get('source_id', '').split('<SEP>') if v]
        if not chunk_ids or any(c not in chunks_by_id for c in chunk_ids):
            raise RuntimeError('extracted graph node does not map to a real source chunk')
        metadata = [sources[chunks_by_id[c]['full_doc_id']] for c in chunk_ids]
        kind = 'hyperedge' if attributes.get('role') == 'hyperedge' else 'entity'
        sidecar[kind][node] = {'wikipedia_urls': sorted({m['wikipedia_url'] for m in metadata}),
                               'wikipedia_titles': sorted({m['wikipedia_title'] for m in metadata}),
                               'source_chunk_ids': chunk_ids}
    counts = {'entities': len(sidecar['entity']), 'hyperedges': len(sidecar['hyperedge']),
              'nodes': graph.number_of_nodes(), 'edges': graph.number_of_edges(),
              'full_docs': len(full_docs), 'text_chunks': len(chunks_by_id)}
    if not counts['entities'] or not counts['hyperedges'] or not counts['edges']:
        raise RuntimeError('native graph has no real extracted entities/hyperedges/relations')
    atomic_json(OUTPUT / 'mm_store/graph/graphr1_hit_source_sidecar.json', sidecar)
    return counts


def serve(port):
    # Keep all GPUs available once construction is complete. Retrieval uses a
    # single CPU model, four CPU threads, and a localhost-only HTTP listener.
    os.environ.update(CUDA_VISIBLE_DEVICES='', MM_EMBED_DEVICE='cpu',
                      EVOGRAPH_MM_EMBED_RUNTIME_DEVICE='cpu', EVOGRAPH_MM_ENABLE_BGE_TEXT='0',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    import torch
    import faiss
    import uvicorn
    from evograph_mm.kb.indexing.encoders import GMEQwen2VLEncoder
    from evograph_mm.kb.api import create_app
    torch.set_num_threads(4)
    faiss.omp_set_num_threads(4)
    report = json.loads((LOGS / f'{PREFIX}_report.json').read_text())
    if report['status'] != 'complete':
        raise RuntimeError('cannot serve an incomplete native graph')
    with (LOGS / '.evqa_native_retrieval_api.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with socket.socket() as check:
            check.bind(('127.0.0.1', port))
        app = create_app(working_dir=OUTPUT, model_path=GME, dataset='E-VQA', subset=SUBSET,
                         encoder_factory=lambda *a, **kw: GMEQwen2VLEncoder(GME, batch_size=1),
                         reload_interval=0)
        status = app.state.mm_api.status()
        if status['status'] != 'ready':
            raise RuntimeError('native retrieval API load blocked: ' + str(status['blockers']))
        # The smoke service is retrieval-only: do not expose the underlying
        # graph-edit/reload mutations while validating the baseline.
        from starlette.responses import JSONResponse

        @app.middleware('http')
        async def retrieval_only(request, call_next):
            if request.method not in {'GET', 'HEAD', 'OPTIONS'} and not (
                    request.method == 'POST' and request.url.path == '/search'):
                return JSONResponse({'error': 'read-only smoke retrieval service'}, status_code=405)
            return await call_next(request)

        atomic_json(LOGS / 'evqa_native_retrieval_api_status.json',
                    {'pid': os.getpid(), 'started_at': now(), 'device': 'cpu', 'cpu_threads': 4,
                     'host': '127.0.0.1', 'port': port, 'read_only': True,
                     'working_dir': str(OUTPUT), 'status': status})
        uvicorn.run(app, host='127.0.0.1', port=port, workers=1, access_log=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, choices=range(4), default=0)
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--port', type=int, default=8003)
    args = parser.parse_args(argv)
    if args.serve:
        if not 1024 <= args.port <= 65535:
            parser.error('port must be between 1024 and 65535')
        return serve(args.port)
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / f'.{PREFIX}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = {'pid': os.getpid(), 'status': 'running', 'started_at': now(),
                  'output_dir': str(OUTPUT), 'physical_gpu': args.gpu,
                  'mock_llm': False, 'mock_encoder': False, 'training_started': False,
                  'remote_api_calls': False, 'full_64_16_graph': False}

        def checkpoint(phase, **values):
            report.update(values, phase=phase, updated_at=now())
            atomic_json(LOGS / f'{PREFIX}_progress.json', report)
            print(json.dumps({'phase': phase, **values}), flush=True)

        def interrupted(signum, frame):
            raise InterruptedError(f'received signal {signum}; existing progress preserved')

        signal.signal(signal.SIGTERM, interrupted)
        try:
            if shutil.disk_usage(ROOT).free < 100_000_000_000:
                raise RuntimeError('free disk space below 100 GB')
            plan, layout, bundle = prepare()
            report['plan'] = plan
            report['gpu_preflight'] = idle_gpu(args.gpu)
            os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), MM_EMBED_DEVICE='cuda',
                              EVOGRAPH_MM_EMBED_RUNTIME_DEVICE='cuda',
                              EVOGRAPH_MM_ENABLE_BGE_TEXT='0', HF_HUB_OFFLINE='1',
                              TRANSFORMERS_OFFLINE='1',
                              TIKTOKEN_CACHE_DIR=str(MODEL.parent / '.tiktoken_cache'))
            import torch
            import faiss
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            from graphr1 import GraphR1
            from graphr1.utils import EmbeddingFunc
            from evograph_mm.kb.store import write_store, visual_embedding_id, text_embedding_id
            from evograph_mm.kb.mm_graph import build_mm_graph_records, write_mm_graph
            from evograph_mm.kb.build import build_text_graphr1_indexes
            from evograph_mm.kb.indexing.encoders import GMEQwen2VLEncoder, GME_MODEL_REPO_ID
            from evograph_mm.kb.indexing.faiss_store import write_vector_index
            torch.set_num_threads(4)
            faiss.omp_set_num_threads(4)
            os.chdir(OUTPUT)
            atomic_json(OUTPUT / '.graphr1_seeded.json',
                        {'seeded': False, 'source': 'evograph_mm', 'copied': []})
            report['store_counts'] = write_store(layout, bundle)
            mm_graph = build_mm_graph_records(text_documents=bundle.text_documents,
                                             image_records=bundle.visual_records)
            write_mm_graph(OUTPUT / 'mm_store/graph', mm_graph)
            checkpoint('loading_local_qwen_for_native_graph_extraction')
            torch.cuda.reset_peak_memory_stats()
            processor = AutoProcessor.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False)
            policy = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(MODEL), local_files_only=True, trust_remote_code=False,
                dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': 'cuda:0'})
            policy.eval()
            response_file = OUTPUT / 'local_llm_responses.json'
            responses = json.loads(response_file.read_text()) if response_file.is_file() else {}

            async def local_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
                key = hashlib.sha256(json.dumps([prompt, system_prompt, history_messages],
                                                ensure_ascii=False).encode()).hexdigest()
                if key in responses:
                    result, repairs = normalize_record_delimiters(responses[key]['output'])
                    responses[key].update(native_parser_output=result, delimiter_repairs=repairs)
                    atomic_json(response_file, responses)
                    return validate_extraction_output(result)
                messages = []
                if system_prompt:
                    messages.append({'role': 'system', 'content': system_prompt})
                messages.extend(history_messages or [])
                messages.append({'role': 'user', 'content': prompt})
                limit = min(int(kwargs.get('max_tokens', MAX_NEW_TOKENS)), MAX_NEW_TOKENS)

                def generate(generation_messages):
                    text = processor.apply_chat_template(
                        generation_messages, tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=[text], return_tensors='pt').to('cuda:0')
                    with torch.inference_mode():
                        output = policy.generate(**inputs, max_new_tokens=limit, do_sample=False)
                    continuation = output[:, inputs['input_ids'].shape[-1]:]
                    decoded = processor.batch_decode(continuation, skip_special_tokens=True)[0]
                    return decoded, int(inputs['input_ids'].shape[-1]), int(continuation.shape[-1])

                start = time.perf_counter()
                draft, input_tokens, draft_tokens = generate(messages)
                result, output_tokens = draft, draft_tokens
                if REVIEW_EXTRACTION:
                    if not REVIEW_PROMPT:
                        raise RuntimeError('review extraction enabled without a review prompt')
                    validate_extraction_output(draft)
                    review_messages = messages + [
                        {'role': 'assistant', 'content': draft},
                        {'role': 'user', 'content': REVIEW_PROMPT},
                    ]
                    result, _, output_tokens = generate(review_messages)
                responses[key] = {'output': result, 'input_tokens': input_tokens,
                                  'output_tokens': output_tokens,
                                  'draft_output': draft if REVIEW_EXTRACTION else None,
                                  'draft_output_tokens': draft_tokens if REVIEW_EXTRACTION else None,
                                  'reviewed': REVIEW_EXTRACTION,
                                  'reached_token_limit': output_tokens == limit,
                                  'seconds': round(time.perf_counter()-start, 3)}
                parsed_result, repairs = normalize_record_delimiters(result)
                responses[key].update(native_parser_output=parsed_result, delimiter_repairs=repairs)
                atomic_json(response_file, responses)
                checkpoint('native_local_llm_extraction', completed_llm_calls=len(responses))
                return validate_extraction_output(parsed_result)

            async def no_implicit_embedding(texts):
                # This checkout uses JSON KV stores during extraction; reject
                # unexpected vector calls until the real indexing stage below.
                raise RuntimeError('unexpected embedding call during native KV-only extraction')

            rag = GraphR1(working_dir=str(OUTPUT), llm_model_func=local_llm,
                          llm_model_name=f'local/{MODEL.name}',
                          llm_model_max_async=1, embedding_func_max_async=1,
                          embedding_func=EmbeddingFunc(1536, 1800, no_implicit_embedding, concurrent_limit=1),
                          entity_extract_max_gleaning=0, chunk_token_size=768,
                          chunk_overlap_token_size=64, entity_summary_to_max_tokens=500,
                          enable_llm_cache=False, addon_params={'example_number': 1})
            for i, doc in enumerate(bundle.text_documents):
                checkpoint('inserting_native_source_document', document=i+1, total_documents=4)
                rag.insert(doc['contents'])
            report['graph_counts'] = graph_provenance(bundle)
            report['llm_calls'] = len(responses)
            report['llm_token_limit_count'] = sum(bool(v['reached_token_limit']) for v in responses.values())
            report['syntax_only_delimiter_repairs'] = sum(v.get('delimiter_repairs', 0) for v in responses.values())
            report['raw_llm_outputs_preserved'] = True
            report['factual_consistency_verified'] = False
            report['policy_peak_allocated_mib'] = round(torch.cuda.max_memory_allocated()/1024**2, 2)
            del policy, processor
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            checkpoint('indexing_real_native_graph_with_gme')
            encoder = GMEQwen2VLEncoder(GME, batch_size=1)
            report['graph_indexes'] = build_text_graphr1_indexes(
                output_dir=OUTPUT, encoder=encoder, model_path=GME,
                model_repo_id=GME_MODEL_REPO_ID, encoder_mode='real_gme')
            image_vectors = encoder.encode_images([r['image_path'] for r in bundle.visual_records])
            report['image_index'] = write_vector_index(
                layout.indexing_store_root, 'image',
                [visual_embedding_id(r['image_id']) for r in bundle.visual_records], image_vectors,
                GME, GME_MODEL_REPO_ID, 'GME default image prompt', 'real_gme')
            text_vectors = encoder.encode_texts([d['contents'] for d in bundle.text_documents])
            report['text_index'] = write_vector_index(
                layout.indexing_store_root, 'text',
                [text_embedding_id(d['text_doc_id']) for d in bundle.text_documents], text_vectors,
                GME, GME_MODEL_REPO_ID, 'GME default document prompt', 'real_gme')
            atomic_json(OUTPUT / 'metadata.json', {'dataset': 'E-VQA', 'subset': SUBSET,
                        'source_subset': SUBSET, 'embedding_dimension': 1536,
                        'scope': plan['scope'], 'model_path': str(GME), 'encoder_mode': 'real_gme'})
            atomic_json(OUTPUT / 'build_report.json', {'status': 'complete', 'scope': plan['scope'],
                        'graph_counts': report['graph_counts'], 'mock_llm': False, 'mock_encoder': False})
            checkpoint('testing_native_retrieval_api')
            from fastapi.testclient import TestClient
            from evograph_mm.kb.api import create_app
            app = create_app(working_dir=OUTPUT, model_path=GME, dataset='E-VQA', subset=SUBSET,
                             encoder_factory=lambda *a, **kw: encoder, rag_factory=lambda *a: rag,
                             reload_interval=0)
            status = app.state.mm_api.status()
            if status['status'] != 'ready':
                raise RuntimeError('native API blocked: ' + str(status['blockers']))
            api_results = []
            with TestClient(app) as client:
                for query in [
                    {'queries': [bundle.text_documents[0]['source_metadata']['wikipedia_title']],
                     'entity_top_k': 2, 'hyperedge_top_k': 2, 'rag_top_k': 0, 'image_top_k': 0},
                    {'queries': ['<img>'], 'image_ids': [bundle.visual_records[0]['image_id']],
                     'visual_entity_top_k': 2, 'visual_hyperedge_top_k': 2, 'rag_top_k': 0},
                ]:
                    response = client.post('/search', json=query)
                    if response.status_code != 200:
                        raise RuntimeError('native API HTTP test failed: ' + str(response.status_code))
                    payload = json.loads(response.json()[0])
                    if payload.get('error') or not payload.get('results'):
                        raise RuntimeError('native API returned no real hits: ' + str(payload))
                    api_results.append({'request': query, 'response': payload})
            atomic_json(OUTPUT / 'native_api_tests.json', api_results)
            report['api_test'] = {'http_requests': len(api_results), 'real_results': True,
                                  'self_image_query_is_wiring_test_not_quality_score': True, 'status': status}
            report['gme_peak_allocated_mib'] = round(torch.cuda.max_memory_allocated()/1024**2, 2)
            subset = ROOT / 'datasets_mm/E-VQA/subsets' / SUBSET
            report['raw_qa_unchanged'] = all(sha256(subset / f'qa_{s}.csv') == h
                                             for s, h in plan['qa_hashes'].items())
            if not report['raw_qa_unchanged']:
                raise RuntimeError('raw QA changed during graph smoke test')
            report.update(status='complete', finished_at=now(), native_hypergraph_built=True,
                          limitation='Four leading passages only; not full small-subset graph or training readiness')
            checkpoint('finished')
        except (Exception, KeyboardInterrupt) as error:
            report.update(status='interrupted' if isinstance(error, (InterruptedError, KeyboardInterrupt)) else 'failed',
                          error_type=type(error).__name__, error=sanitized(error), finished_at=now())
            checkpoint('stopped')
        atomic_json(LOGS / f'{PREFIX}_report.json', report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
