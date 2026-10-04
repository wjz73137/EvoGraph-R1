#!/usr/bin/env python3
"""Build the four-source GraphR1 baseline using configured API extraction only."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evograph_mm.kb.gldv2_subset import atomic_json, image_info
from scripts.run_evqa_gpu_smoke import ROOT, SUBSET, sanitized
from scripts import run_evqa_native_graph_smoke as smoke
from scripts.run_evqa_retrieval_smoke import GME, sha256

OUTPUT_ROOT = ROOT / 'expr_mm/evqa_api_graph_baseline'
OUTPUT = OUTPUT_ROOT / 'E-VQA'
MAX_CALLS = 20
MAX_TOKENS = 4096
COMPATIBLE_PREVIOUS_RUNNER = '154b49e843acaf954a54751fa021a42a9c07348333e3a1bd046d14185ca58351'


def checked_api_output(record):
    content = record.get('output') or ''
    if not content.strip() or record.get('finish_reason') == 'length':
        raise RuntimeError('empty or truncated API response; graph expansion stopped')
    return content


def compatible_owner_migration(previous, current, report):
    changed = {key for key in set(previous) | set(current)
               if previous.get(key) != current.get(key)}
    return (changed == {'runner_sha256'}
            and previous.get('runner_sha256') == COMPATIBLE_PREVIOUS_RUNNER
            and report.get('extraction_complete') is True)


def index_artifacts_present(record):
    paths = [value for key, value in (record or {}).items() if key.endswith('_path')]
    return bool(paths) and all(Path(path).is_file() for path in paths)


def load_api_config():
    from dotenv import dotenv_values
    env_file = Path(__file__).resolve().parents[1] / '.env'
    config = dotenv_values(env_file)
    key = config.get('OPENAI_API_KEY')
    url = config.get('OPENAI_BASE_URL')
    model = config.get('GRAPH_LLM_MODEL') or config.get('OPENAI_MODEL')
    if not all((key, url, model)):
        raise RuntimeError('project .env lacks API key, base URL or graph model')
    # Set only this process's graph configuration; never change .env or judges.
    os.environ.update(OPENAI_API_KEY=key, OPENAI_BASE_URL=url,
                      OPENAI_MODEL=model, GRAPH_LLM_MODEL=model)
    return key, url, model


def source_bundle(snapshot):
    from evograph_mm.kb.store import RecordBundle, text_embedding_id, visual_embedding_id
    subset = ROOT / 'datasets_mm/E-VQA/subsets' / SUBSET
    manifest = json.loads((subset / 'manifest.json').read_text())
    images = {r['image_id']: r for r in manifest['copied_images']}
    if len(snapshot['documents']) != 4:
        raise RuntimeError('expected exactly four source documents')
    docs, visual, links = [], [], []
    for original in snapshot['documents']:
        doc = dict(original)
        doc['data_id'] = doc['text_doc_id'].removeprefix('text::')
        doc['split'] = 'train'
        record = images[doc['image_id']]
        path = ROOT / record['subset_path']
        if not image_info(path) or sha256(path) != record['sha256']:
            raise RuntimeError('source image decoding/hash check failed')
        doc['image_path'] = str(path)
        docs.append(doc)
        visual.append({'visual_record_id': f"visual::{doc['image_id']}",
                       'image_id': doc['image_id'], 'image_path': str(path),
                       'data_id': doc['data_id'], 'split': 'train',
                       'source_metadata': doc['source_metadata'], 'image_missing': False})
        for source, target, relation in [
            (doc['data_id'], doc['text_doc_id'], 'has_text_document'),
            (doc['text_doc_id'], text_embedding_id(doc['text_doc_id']), 'has_text_embedding'),
            (f"visual::{doc['image_id']}", visual_embedding_id(doc['image_id']), 'has_visual_embedding'),
        ]:
            links.append({'source_id': source, 'target_id': target,
                          'relation': relation, 'data_id': doc['data_id'], 'split': 'train'})
    return RecordBundle(docs, visual, links)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', action='store_true', help='build CPU GME indexes after API extraction')
    args = parser.parse_args()
    key, url, model_name = load_api_config()
    os.environ.update(CUDA_VISIBLE_DEVICES='', MM_EMBED_DEVICE='cpu',
                      EVOGRAPH_MM_EMBED_RUNTIME_DEVICE='cpu', EVOGRAPH_MM_ENABLE_BGE_TEXT='0',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                      TIKTOKEN_CACHE_DIR=str(GME.parent / '.tiktoken_cache'))
    snapshot_file = OUTPUT_ROOT / 'source_snapshot.json'
    snapshot = json.loads(snapshot_file.read_text())
    bundle = source_bundle(snapshot)
    import graphr1.prompt as prompts
    owner = {'source_sha256': sha256(snapshot_file), 'graph_llm': f'api/{model_name}',
             'original_prompt_sha256': sha256(Path(prompts.__file__)),
             'runner_sha256': sha256(Path(__file__)), 'gleaning_passes': 2,
             'max_calls': MAX_CALLS, 'max_output_tokens': MAX_TOKENS,
             'scope': snapshot['scope'], 'no_gold_qa_in_kb': True,
             'local_extraction_model_loaded': False, 'embedding': 'local GME on CPU'}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / '.build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report_file = OUTPUT_ROOT / 'report.json'
        report = json.loads(report_file.read_text()) if report_file.exists() else {}
        if (OUTPUT / 'owner.json').exists() and json.loads((OUTPUT / 'owner.json').read_text()) != owner:
            previous = json.loads((OUTPUT / 'owner.json').read_text())
            if not compatible_owner_migration(previous, owner, report):
                raise RuntimeError('existing API graph configuration mismatch; retained')
            atomic_json(OUTPUT_ROOT / 'runner_manifest_fix_migration.json', {
                'previous_owner': previous, 'current_owner': owner,
                'reason': 'add missing retrieval build report and reuse completed indexes',
                'extraction_reused': True, 'no_new_extraction_requested': True})
        atomic_json(OUTPUT / 'owner.json', owner)
        if report.get('status') == 'complete' and (not args.index or report.get('indexes_complete')):
            print(json.dumps(report, ensure_ascii=False), flush=True)
            return
        report.update(status='running', pid=os.getpid(), output_dir=str(OUTPUT),
                      model=model_name, training_started=False, local_extraction_model_loaded=False,
                      factual_consistency_verified=False, original_prompt=True,
                      gleaning_passes=2, embedding_device='cpu')
        report.pop('error', None)
        report.pop('error_type', None)
        started = time.perf_counter()
        calls_file = OUTPUT / 'api_calls.json'
        calls = json.loads(calls_file.read_text()) if calls_file.exists() else {}
        def checkpoint(phase, **values):
            report.update(phase=phase, **values)
            atomic_json(report_file, report)
            print(json.dumps({'phase': phase, **values}, ensure_ascii=False), flush=True)
        try:
            import torch
            torch.set_num_threads(4)
            import faiss
            faiss.omp_set_num_threads(4)
            from openai import AsyncOpenAI
            from graphr1 import GraphR1
            from graphr1.utils import EmbeddingFunc
            from evograph_mm.kb.layout import build_layout
            from evograph_mm.kb.store import write_store, text_embedding_id, visual_embedding_id
            from evograph_mm.kb.mm_graph import build_mm_graph_records, write_mm_graph
            os.chdir(OUTPUT)
            logging.getLogger('httpx').setLevel(logging.WARNING)
            logging.getLogger('openai').setLevel(logging.WARNING)
            client = AsyncOpenAI(api_key=key, base_url=url, timeout=120, max_retries=0)
            async def api_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
                messages = ([{'role': 'system', 'content': system_prompt}] if system_prompt else [])
                messages += list(history_messages or []) + [{'role': 'user', 'content': prompt}]
                digest = hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()
                if digest in calls and calls[digest].get('status') == 'complete':
                    return checked_api_output(calls[digest])
                if digest in calls:
                    raise RuntimeError('previous API attempt is incomplete; manual check required before retry')
                if len(calls) >= MAX_CALLS:
                    raise RuntimeError('API request budget reached; no further calls made')
                calls[digest] = {'messages': messages, 'model': model_name, 'status': 'requested'}
                atomic_json(calls_file, calls)
                before = time.perf_counter()
                response = await client.chat.completions.create(
                    model=model_name, messages=messages, temperature=0, max_tokens=MAX_TOKENS,
                    extra_body={'enable_thinking': False})
                choice = response.choices[0]
                content = choice.message.content or ''
                calls[digest].update(status='complete', output=content,
                                     finish_reason=choice.finish_reason,
                                     usage=response.usage.model_dump() if response.usage else None,
                                     seconds=round(time.perf_counter()-before, 3))
                atomic_json(calls_file, calls)
                checkpoint('api_response_saved', api_calls=len(calls), finish_reason=choice.finish_reason)
                return checked_api_output(calls[digest])
            async def no_implicit_embedding(texts):
                raise RuntimeError('unexpected embedding API request during extraction')
            rag = GraphR1(working_dir=str(OUTPUT), llm_model_func=api_llm,
                          llm_model_name=model_name, llm_model_max_async=1,
                          embedding_func_max_async=1,
                          embedding_func=EmbeddingFunc(1536, 1800, no_implicit_embedding, concurrent_limit=1),
                          entity_extract_max_gleaning=2, entity_summary_to_max_tokens=500,
                          enable_llm_cache=False, addon_params={'example_number': 1})
            layout = build_layout(ROOT, 'E-VQA', SUBSET, OUTPUT_ROOT)
            atomic_json(OUTPUT / '.graphr1_seeded.json', {'seeded': False, 'copied': [], 'source': 'api'})
            report['store_counts'] = write_store(layout, bundle)
            write_mm_graph(OUTPUT / 'mm_store/graph', build_mm_graph_records(
                text_documents=bundle.text_documents, image_records=bundle.visual_records))
            if not report.get('extraction_complete'):
                for i, doc in enumerate(bundle.text_documents):
                    checkpoint('api_native_graph_extraction', document=i+1, total_documents=4)
                    rag.insert(doc['contents'])
                smoke.OUTPUT = OUTPUT
                report['graph_counts'] = smoke.graph_provenance(bundle)
                checkpoint('api_graph_built', extraction_complete=True, graph_counts=report['graph_counts'])
            report['api_calls'] = len(calls)
            report['api_usage'] = {
                field: sum((c.get('usage') or {}).get(field, 0) or 0 for c in calls.values())
                for field in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
            if args.index:
                from evograph_mm.kb.build import build_text_graphr1_indexes
                from evograph_mm.kb.indexing.encoders import GMEQwen2VLEncoder, GME_MODEL_REPO_ID
                from evograph_mm.kb.indexing.faiss_store import write_vector_index
                checkpoint('cpu_gme_indexing')
                encoder = GMEQwen2VLEncoder(GME, batch_size=1)
                graph_indexes = report.get('graph_indexes') or {}
                if not all(index_artifacts_present(graph_indexes.get(k)) for k in ('entity', 'hyperedge')):
                    report['graph_indexes'] = build_text_graphr1_indexes(
                        output_dir=OUTPUT, encoder=encoder, model_path=GME,
                        model_repo_id=GME_MODEL_REPO_ID, encoder_mode='real_gme')
                if not index_artifacts_present(report.get('image_index')):
                    report['image_index'] = write_vector_index(
                        layout.indexing_store_root, 'image',
                        [visual_embedding_id(r['image_id']) for r in bundle.visual_records],
                        encoder.encode_images([r['image_path'] for r in bundle.visual_records]),
                        GME, GME_MODEL_REPO_ID, 'GME default image prompt', 'real_gme')
                if not index_artifacts_present(report.get('text_index')):
                    report['text_index'] = write_vector_index(
                        layout.indexing_store_root, 'text',
                        [text_embedding_id(d['text_doc_id']) for d in bundle.text_documents],
                        encoder.encode_texts([d['contents'] for d in bundle.text_documents]),
                        GME, GME_MODEL_REPO_ID, 'GME default document prompt', 'real_gme')
                atomic_json(OUTPUT / 'metadata.json', {'dataset': 'E-VQA', 'subset': SUBSET,
                    'embedding_dimension': 1536, 'scope': snapshot['scope'],
                    'model_path': str(GME), 'encoder_mode': 'real_gme'})
                atomic_json(OUTPUT / 'build_report.json', {
                    'status': 'complete', 'scope': snapshot['scope'],
                    'graph_counts': report['graph_counts'], 'mock_llm': False,
                    'mock_encoder': False, 'graph_llm': f'api/{model_name}',
                    'factual_consistency_verified': False})
                from fastapi.testclient import TestClient
                from evograph_mm.kb.api import create_app
                app = create_app(working_dir=OUTPUT, model_path=GME, dataset='E-VQA', subset=SUBSET,
                                 encoder_factory=lambda *a, **kw: encoder, rag_factory=lambda *a: rag,
                                 reload_interval=0)
                status = app.state.mm_api.status()
                if status['status'] != 'ready':
                    raise RuntimeError('new API-built graph indexes are not ready: ' + str(status['blockers']))
                with TestClient(app) as test_client:
                    result = test_client.post('/search', json={
                        'queries': ['Newport Castle'], 'entity_top_k': 2, 'hyperedge_top_k': 2,
                        'rag_top_k': 0, 'image_top_k': 0})
                    if result.status_code != 200 or not json.loads(result.json()[0]).get('results'):
                        raise RuntimeError('new graph retrieval returned no results')
                    atomic_json(OUTPUT / 'api_retrieval_test.json', result.json())
                checkpoint('indexes_verified', indexes_complete=True, retrieval_test_passed=True)
            subset = ROOT / 'datasets_mm/E-VQA/subsets' / SUBSET
            report['raw_qa_unchanged'] = all(sha256(subset / f'qa_{s}.csv') == digest
                                           for s, digest in snapshot['qa_hashes'].items())
            if not report['raw_qa_unchanged']:
                raise RuntimeError('source QA files changed')
            report.update(status='complete', elapsed_seconds=round(time.perf_counter()-started, 3),
                          limitation='Four source text passages only; not full multimodal extraction or training.')
            checkpoint('finished')
        except Exception as error:
            message = sanitized(error).replace(key, '[REDACTED]')
            report.update(status='failed', error_type=type(error).__name__, error=message)
            checkpoint('stopped')
        print(json.dumps(report, ensure_ascii=False), flush=True)
        if report['status'] != 'complete':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
