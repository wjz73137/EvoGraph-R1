#!/usr/bin/env python3
"""Four image-only API calls; independent image-grounded hypergraph and real CPU indexes."""
from __future__ import annotations

import argparse
import base64
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evograph_mm.kb.gldv2_subset import atomic_json, image_info
from evograph_mm.kb.visual_scene import VISUAL_PROMPT, normalized_label, parse_scene, apply_source_review
from scripts.clean_evqa_api_graph import TARGET_ROOT as PARENT_ROOT
from scripts.run_evqa_api_graph import load_api_config
from scripts.run_evqa_gpu_smoke import ROOT, SUBSET
from scripts.run_evqa_retrieval_smoke import GME

PARENT = PARENT_ROOT / 'E-VQA'
OUTPUT_ROOT = ROOT / 'expr_mm/evqa_api_visual_graph_v1'
OUTPUT = OUTPUT_ROOT / 'E-VQA'
MAX_CALLS = 4
MAX_TOKENS = 2048
MIN_CONFIDENCE = 7.0
FUSION_THRESHOLD = 0.55


def reviewed_scene(image_id, record):
    path = OUTPUT_ROOT / 'vision_source_review.json'
    reviews = read_json(path) if path.exists() else {}
    return apply_source_review(record, reviews.get(image_id))


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def graph_hashes(path):
    return {str(p.relative_to(path)): sha256(p) for p in sorted(path.rglob('*'))
            if p.is_file() and p.name not in {'graphr1.log', 'kv_store_llm_response_cache.json'}}


def digest_id(name, prefix):
    return prefix + hashlib.md5(name.encode()).hexdigest()


def image_manifest():
    records = [json.loads(line) for line in (PARENT / 'mm_store/visual/image_records.jsonl').read_text().splitlines()]
    if len(records) != 4 or len({r['image_id'] for r in records}) != 4:
        raise RuntimeError('expected exactly four distinct baseline images')
    manifest = []
    for record in records:
        path = Path(record['image_path'])
        if path.is_symlink() or not path.resolve().is_relative_to((ROOT / 'datasets_mm/E-VQA/subsets' / SUBSET / 'images').resolve()):
            raise RuntimeError('unexpected image path')
        if not image_info(path) or path.stat().st_size > 10 * 1024 * 1024:
            raise RuntimeError('image invalid or exceeds bounded request size')
        manifest.append({'image_id': record['image_id'], 'image_path': str(path),
                         'image_sha256': sha256(path), 'data_id': record['data_id'], 'mime_type': 'image/jpeg'})
    return manifest


def request_messages(image_bytes):
    """Payload contains image bytes and fixed prompt ONLY, never titles, filenames or QA."""
    return [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(image_bytes).decode()}},
        {'type': 'text', 'text': VISUAL_PROMPT}]}]


def provenance(image, model, kind):
    return {'source': 'api_image_extraction', 'ai_model': model, 'mod': 'vis',
            'image_id': image['image_id'], 'image_path': image['image_path'],
            'image_sha256': image['image_sha256'], 'evidence_basis': 'image_only',
            'kind': kind, 'verification_status': 'pending', 'human_reviewed': False,
            'confidence_note': 'uncalibrated model self-assessment, not truth probability'}


def add_visual_records(entities, edges, graph, sidecar, scenes, images, model, resolutions=None):
    """Anchors are mandatory; dataset article association is separate from visual identity."""
    entities, edges, graph, sidecar = copy.deepcopy((entities, edges, graph, sidecar))
    resolutions = resolutions or {}
    audit = {'images': [], 'quarantined_relations': [], 'quarantined_objects': [], 'fused_objects': []}
    chunks, docs = {}, {}
    canonical_lookup = read_json(PARENT / 'mm_store/graph/image_anchor_lookup.json')
    for image in images:
        image_id = image['image_id']
        scene = scenes[image_id]
        anchor = f'image::{image_id}'
        chunk_id, doc_id = f'visualchunk::{image_id}', f'visualdoc::{image_id}'
        mapping = {'source_chunk_ids': [chunk_id], 'source_image_ids': [image_id],
                   'image_paths': [image['image_path']], 'evidence_basis': 'image_only',
                   'associated_data_ids': [image['data_id']],
                   'wikipedia_urls': [], 'wikipedia_titles': [], 'visual_identity_verified': False}
        anchor_meta = provenance(image, model, 'image_anchor')
        anchor_meta.update(entity_type='IMAGE', source_id=chunk_id)
        entities[digest_id(anchor, 'ent-')] = {'entity_name': anchor, 'content': anchor + '\n' + scene.scene_description, 'metadata': anchor_meta}
        graph.add_node(anchor, role='image_anchor', entity_type='IMAGE', mod='vis', description=scene.scene_description,
                       source_id=chunk_id, metadata=json.dumps(anchor_meta, ensure_ascii=False))
        sidecar['entity'][anchor] = copy.deepcopy(mapping)
        object_names = {}
        for obj in scene.objects:
            if obj.confidence < MIN_CONFIDENCE:
                audit['quarantined_objects'].append({'image_id': image_id, 'object': obj.model_dump(),
                    'reason': 'low uncalibrated confidence; not assumed false'})
                continue
            name = f'"VISUAL::{image_id}::{obj.id}::{obj.name.upper()}"'
            description = obj.description
            target_name = resolutions.get((image_id, obj.id))
            if target_name:
                name = target_name
                record = entities[digest_id(name, 'ent-')]
                record['content'] += '\nVisual observation: ' + description
                record['metadata'].setdefault('visual_observations', []).append({**provenance(image, model, 'resolved_object'),
                    'bbox': obj.bbox, 'description': description, 'object_name': obj.name, 'confidence': obj.confidence})
                record['metadata']['mod'] = 'cross_modal'
                graph.nodes[name]['description'] += '<SEP>Visual observation: ' + description
                graph.nodes[name]['source_id'] += '<SEP>' + chunk_id
                graph.nodes[name]['mod'] = 'cross_modal'
                graph.nodes[name]['metadata'] = json.dumps(record['metadata'], ensure_ascii=False)
                sidecar['entity'][name].setdefault('source_image_ids', []).append(image_id)
                sidecar['entity'][name]['source_chunk_ids'].append(chunk_id)
                audit['fused_objects'].append({'image_id': image_id, 'object_id': obj.id, 'canonical_node': name})
            else:
                metadata = provenance(image, model, 'visual_object')
                metadata.update(entity_type=obj.entity_type, source_id=chunk_id, bbox=obj.bbox, confidence=obj.confidence)
                entities[digest_id(name, 'ent-')] = {'entity_name': name, 'content': obj.name + '\n' + description, 'metadata': metadata}
                graph.add_node(name, role='entity', entity_type=obj.entity_type, mod='vis', description=description,
                               source_id=chunk_id, metadata=json.dumps(metadata, ensure_ascii=False))
                sidecar['entity'][name] = copy.deepcopy(mapping)
            object_names[obj.id] = name

        def add_edge(statement, names, kind, confidence=None):
            name = '<hyperedge>"[IMAGE ' + image_id + '] ' + statement + '"'
            record_id = digest_id(name, 'rel-')
            if record_id in edges:
                raise RuntimeError('duplicate generated visual hyperedge')
            metadata = provenance(image, model, kind)
            metadata.update(anchor_node=anchor)
            if confidence is not None:
                metadata['confidence'] = confidence
            else:
                metadata['confidence_origin'] = 'not_requested; graph weight is neutral, not model confidence'
            if kind == 'dataset_association':
                metadata.update(evidence_basis='dataset_image_article_association', mod='cross_modal',
                                visual_identity_verified=False)
            edges[record_id] = {'content': name, 'hyperedge_name': name, 'metadata': metadata}
            weight = 1.0 if confidence is None else float(confidence)
            graph.add_node(name, role='hyperedge', mod=metadata['mod'], source_id=chunk_id,
                           weight=weight, metadata=json.dumps(metadata, ensure_ascii=False))
            for entity_name in dict.fromkeys([anchor, *names]):
                graph.add_edge(name, entity_name, weight=weight, source_id=chunk_id,
                               relation='grounded_in_image' if entity_name == anchor else 'has_participant')
            sidecar['hyperedge'][name] = {**copy.deepcopy(mapping), 'kind': kind,
                'evidence_basis': metadata['evidence_basis'], 'anchor_node': anchor}
            return name

        scene_node = add_edge(scene.scene_description, list(object_names.values()), 'scene')
        accepted = []
        for relation in scene.relations:
            referenced = [o for o in scene.objects if o.id in relation.object_ids]
            if relation.confidence < MIN_CONFIDENCE or any(o.confidence < MIN_CONFIDENCE for o in referenced):
                audit['quarantined_relations'].append({'image_id': image_id, 'relation': relation.model_dump(),
                                                      'reason': 'low uncalibrated confidence; not assumed false'})
                continue
            accepted.append(add_edge(relation.statement, [object_names[i] for i in relation.object_ids],
                                     'visual_relation', relation.confidence))
        canonical = canonical_lookup[image_id]['canonical_entity']
        canonical_node = '"' + canonical + '"'
        if canonical_node not in graph:
            raise RuntimeError('dataset association target is missing from source graph')
        association = add_edge('The dataset associates this image with the article ' + canonical +
            '; this association is not a visual identification of that landmark.', [canonical_node], 'dataset_association')
        docs[doc_id] = {'content': scene.scene_description, 'metadata': anchor_meta}
        chunks[chunk_id] = {'content': scene.scene_description, 'full_doc_id': doc_id,
                           'tokens': len(scene.scene_description.split()), 'chunk_order_index': 0, 'metadata': anchor_meta}
        audit['images'].append({'image_id': image_id, 'anchor_node': anchor, 'scene_node': scene_node,
                               'visual_object_nodes': list(object_names.values()), 'visual_relation_nodes': accepted,
                               'association_node': association, 'article_association_is_identity': False})
    return entities, edges, graph, sidecar, audit, chunks, docs


def choose_resolutions(scenes, images, entities, sidecar, encoder):
    """Same-source, exact normalized label AND embedding gate; never merge by resemblance alone."""
    import numpy as np
    from evograph_mm.kb.indexing.encoders import TEXT_DOCUMENT_INSTRUCTION
    docs = [json.loads(s) for s in (PARENT / 'graphr1_text/text_documents.jsonl').read_text().splitlines()]
    source_urls = {d['image_id']: d['source_metadata'].get('wikipedia_url') or
                   d['source_metadata']['source_row']['wikipedia_url'] for d in docs}
    metadata = read_json(PARENT / 'entity_index_metadata.json')
    vectors = np.load(PARENT / 'corpus_entity.npy', allow_pickle=False)
    positions = {name: i for i, name in enumerate(metadata['ids'])}
    proposals, audit = {}, []
    for image in images:
        for obj in scenes[image['image_id']].objects:
            candidates = [v for v in entities.values()
                if normalized_label(v['entity_name']) == normalized_label(obj.name)
                and source_urls[image['image_id']] in sidecar['entity'][v['entity_name']].get('wikipedia_urls', [])]
            if len(candidates) != 1 or obj.confidence < MIN_CONFIDENCE:
                continue
            candidate = candidates[0]
            query = encoder.encode_texts([obj.name + '\n' + obj.description], instruction=TEXT_DOCUMENT_INSTRUCTION)[0]
            target = vectors[positions[candidate['entity_name']]]
            similarity = float(np.dot(query, target) / (np.linalg.norm(query) * np.linalg.norm(target)))
            accepted = similarity >= FUSION_THRESHOLD
            audit.append({'image_id': image['image_id'], 'object_id': obj.id, 'candidate': candidate['entity_name'],
                          'normalized_label_exact_match': True, 'same_source_article': True,
                          'embedding_similarity': similarity, 'threshold': FUSION_THRESHOLD, 'accepted': accepted})
            if accepted:
                proposals[(image['image_id'], obj.id)] = candidate['entity_name']
    return proposals, audit


def build_graph(images, calls, report):
    import networkx as nx
    import numpy as np
    import torch
    import faiss
    from evograph_mm.kb.build import _load_entity_records, _load_hyperedge_records, _write_root_graphr1_index
    from evograph_mm.kb.indexing.encoders import GMEQwen2VLEncoder, TEXT_DOCUMENT_INSTRUCTION
    torch.set_num_threads(4)
    faiss.omp_set_num_threads(4)
    scenes = {image['image_id']: reviewed_scene(image['image_id'], calls[image['image_id']]) for image in images}
    entities = read_json(PARENT / 'kv_store_entities.json')
    edges = read_json(PARENT / 'kv_store_hyperedges.json')
    graph = nx.read_graphml(PARENT / 'graph_chunk_entity_relation.graphml')
    sidecar = read_json(PARENT / 'mm_store/graph/graphr1_hit_source_sidecar.json')
    if OUTPUT.exists():
        raise RuntimeError('existing visual graph retained; review incomplete build before retry')
    encoder = GMEQwen2VLEncoder(GME, batch_size=1)
    resolutions, resolution_audit = choose_resolutions(scenes, images, entities, sidecar, encoder)
    entities, edges, graph, sidecar, audit, chunks, docs = add_visual_records(
        entities, edges, graph, sidecar, scenes, images, report['model'], resolutions)
    if any(p.is_symlink() for p in PARENT.rglob('*')):
        raise RuntimeError('unexpected symlink in parent graph')
    shutil.copytree(PARENT, OUTPUT, ignore=shutil.ignore_patterns('graphr1.log', 'kv_store_llm_response_cache.json', 'api_retrieval_test.json'))
    owner = read_json(OUTPUT / 'owner.json')
    atomic_json(OUTPUT / 'owner.json', {**owner, 'visual_model': 'api/' + report['model'],
        'parent_graph': str(PARENT), 'visual_manifest_sha256': sha256(OUTPUT_ROOT / 'owner.json'),
        'visual_prompt_kind': 'custom implementation of paper section 3.3, not a published original visual prompt'})
    atomic_json(OUTPUT / 'kv_store_entities.json', entities)
    atomic_json(OUTPUT / 'kv_store_hyperedges.json', edges)
    for filename, additions in [('kv_store_text_chunks.json', chunks), ('kv_store_full_docs.json', docs)]:
        atomic_json(OUTPUT / filename, {**read_json(OUTPUT / filename), **additions})
    atomic_json(OUTPUT / 'mm_store/graph/graphr1_hit_source_sidecar.json', sidecar)
    nx.write_graphml(graph, OUTPUT / 'graph_chunk_entity_relation.graphml')
    audit['resolution_candidates'] = resolution_audit
    atomic_json(OUTPUT_ROOT / 'visual_graph_audit.json', audit)
    atomic_json(OUTPUT / 'mm_store/visual/scene_records.json', {k: v.model_dump() for k, v in scenes.items()})
    mm_graph = nx.read_graphml(OUTPUT / 'mm_store/graph/graph_mm_entity_relation.graphml')
    visual_nodes = {i['anchor_node'] for i in audit['images']}
    visual_nodes |= {n for i in audit['images'] for n in [i['scene_node'], i['association_node'], *i['visual_object_nodes'], *i['visual_relation_nodes']]}
    for node in visual_nodes:
        mm_graph.add_node(node, **graph.nodes[node])
    for item in audit['images']:
        for node in [item['scene_node'], item['association_node'], *item['visual_relation_nodes']]:
            for neighbor, attrs in graph[node].items():
                mm_graph.add_node(neighbor, **graph.nodes[neighbor])
                mm_graph.add_edge(node, neighbor, **attrs)
    nx.write_graphml(mm_graph, OUTPUT / 'mm_store/graph/graph_mm_entity_relation.graphml')
    mm_meta = read_json(OUTPUT / 'mm_store/graph/graph_metadata.json')
    atomic_json(OUTPUT / 'mm_store/graph/graph_metadata.json', {**mm_meta,
        'node_count': len(mm_graph), 'edge_count': mm_graph.number_of_edges(), 'visual_scene_count': 4})
    report['graph_indexes'] = {}
    for kind, loader, filename in [('entity', _load_entity_records, 'kv_store_entities.json'), ('hyperedge', _load_hyperedge_records, 'kv_store_hyperedges.json')]:
        records = loader(OUTPUT / filename)
        old_meta = read_json(PARENT / f'{kind}_index_metadata.json')
        old_vectors = np.load(PARENT / f'corpus_{kind}.npy', allow_pickle=False)
        old_rows = {name: i for i, name in enumerate(old_meta['ids'])}
        matrix = np.empty((len(records), old_vectors.shape[1]), dtype=np.float32)
        to_encode = []
        for row, record in enumerate(records):
            old_row = old_rows.get(record['id'])
            if old_row is not None and old_meta['contents'][old_row] == record['content']:
                matrix[row] = old_vectors[old_row]
            else:
                to_encode.append(row)
        print(json.dumps({'phase': 'cpu_gme_indexing', 'kind': kind, 'new_or_changed_vectors': len(to_encode)}), flush=True)
        matrix[to_encode] = encoder.encode_texts([records[i]['content'] for i in to_encode], instruction=TEXT_DOCUMENT_INSTRUCTION)
        if not np.isfinite(matrix).all() or np.any(np.linalg.norm(matrix, axis=1) == 0):
            raise RuntimeError('invalid real embedding vectors')
        index_report = _write_root_graphr1_index(output_dir=OUTPUT, namespace=kind,
            ids=[r['id'] for r in records], contents=[r['content'] for r in records], embeddings=matrix,
            corpus_file=f'corpus_{kind}.npy', index_file=f'index_{kind}.bin', metadata_file=f'{kind}_index_metadata.json',
            model_path=GME, model_repo_id=old_meta['model_repo_id'], encoder_mode=old_meta['encoder_mode'])
        # Give generic search responses image-grounding provenance, not just vector IDs.
        by_name = {v['entity_name' if kind == 'entity' else 'hyperedge_name']: v for v in (entities if kind == 'entity' else edges).values()}
        index_meta = read_json(OUTPUT / f'{kind}_index_metadata.json')
        index_meta['records'] = [{'image_id': by_name[r['id']]['metadata'].get('image_id'),
                                 'image_path': by_name[r['id']]['metadata'].get('image_path'),
                                 'source_metadata': by_name[r['id']]['metadata']} for r in records]
        atomic_json(OUTPUT / f'{kind}_index_metadata.json', index_meta)
        index_report.update(reused_vectors=len(records)-len(to_encode), encoded_vectors=len(to_encode))
        report['graph_indexes'][kind] = index_report
        atomic_json(OUTPUT_ROOT / 'report.json', report)
    counts = {'entities': len(entities), 'hyperedges': len(edges), 'nodes': len(graph), 'edges': graph.number_of_edges(),
              'full_docs': len(read_json(OUTPUT / 'kv_store_full_docs.json')), 'text_chunks': len(read_json(OUTPUT / 'kv_store_text_chunks.json'))}
    atomic_json(OUTPUT / 'build_report.json', {**read_json(OUTPUT / 'build_report.json'), 'graph_counts': counts,
        'visual_scene_extraction': True, 'visual_image_count': 4, 'factual_consistency_verified': False})
    from evograph_mm.kb.api import create_app
    from fastapi.testclient import TestClient
    os.chdir(OUTPUT)
    app = create_app(working_dir=OUTPUT, model_path=GME, dataset='E-VQA', subset=SUBSET,
                     encoder_factory=lambda *a, **kw: encoder, reload_interval=0)
    if app.state.mm_api.status()['status'] != 'ready':
        raise RuntimeError('visual graph retrieval not ready')
    tests = []
    with TestClient(app) as client:
        for image in images:
            scene = scenes[image['image_id']]
            response = client.post('/search', json={'queries': [scene.scene_description], 'entity_top_k': 0,
                'hyperedge_top_k': 3, 'rag_top_k': 0, 'image_top_k': 0})
            if response.status_code != 200:
                raise RuntimeError('visual hyperedge search failed')
            results = json.loads(response.json()[0])['results']
            if not any(v['id'] == next(i for i in audit['images'] if i['image_id'] == image['image_id'])['scene_node'] for v in results):
                raise RuntimeError('scene hyperedge is not retrievable with its own description')
            tests.append({'image_id': image['image_id'], 'kind': 'scene_description_index_sanity', 'response': response.json()})
        result = client.post('/search', json={'queries': ['boats floating on turquoise water under a cloudy sky'],
            'entity_top_k': 0, 'hyperedge_top_k': 3, 'rag_top_k': 0, 'image_top_k': 0})
        if result.status_code != 200 or not json.loads(result.json()[0])['results']:
            raise RuntimeError('free text visual search failed')
        tests.append({'kind': 'free_text_visual_query_smoke_not_accuracy', 'response': result.json()})
    for item in audit['images']:
        for node in [item['scene_node'], *item['visual_relation_nodes']]:
            if not graph.has_edge(node, item['anchor_node']):
                raise RuntimeError('visual hyperedge missing image anchor')
    original_files = ['api_calls.json', 'graphr1_text/text_documents.jsonl', 'mm_store/visual/image_records.jsonl']
    original_files += [str(p.relative_to(PARENT)) for p in (PARENT / 'mm_store/indexing').iterdir() if p.is_file()]
    if not all(sha256(PARENT / name) == sha256(OUTPUT / name) for name in original_files):
        raise RuntimeError('original extraction or image/document indexes changed')
    atomic_json(OUTPUT / 'api_retrieval_test.json', tests)
    report.update(graph_counts=counts, indexes_complete=True, retrieval_test_passed=True,
        visual_scenes=4, accepted_visual_relations=sum(len(i['visual_relation_nodes']) for i in audit['images']),
        quarantined_visual_relations=len(audit['quarantined_relations']), resolved_objects=len(audit['fused_objects']),
        quarantined_visual_objects=len(audit['quarantined_objects']),
        fusion_policy='exact normalized label + same source article + embedding threshold; no landmark identity guessed',
        original_document_and_image_indexes_unchanged=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extract', action='store_true', help='up to four paid API image requests, no automatic retry')
    parser.add_argument('--build', action='store_true', help='build independent graph/indexes on CPU from completed requests')
    args = parser.parse_args()
    key, base_url, model = load_api_config()
    parent_report = read_json(PARENT_ROOT / 'report.json')
    parent_owner = read_json(PARENT / 'owner.json')
    if parent_report['status'] != 'complete' or parent_owner['graph_llm'] != 'api/' + model:
        raise RuntimeError('parent graph incomplete or configured API model changed')
    images = image_manifest()
    print(json.dumps({'model': model, 'images': 4, 'max_new_calls': MAX_CALLS if args.extract else 0,
                     'dry_run': not (args.extract or args.build), 'device': 'cpu', 'output': str(OUTPUT_ROOT)}), flush=True)
    if not args.extract and not args.build:
        return
    OUTPUT_ROOT.mkdir(exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES='', MM_EMBED_DEVICE='cpu', EVOGRAPH_MM_EMBED_RUNTIME_DEVICE='cpu',
        EVOGRAPH_MM_ENABLE_BGE_TEXT='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    with (OUTPUT_ROOT / '.build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current_owner = {'parent_graph': str(PARENT), 'parent_artifact_hashes': graph_hashes(PARENT_ROOT),
            'model': model, 'images': images, 'prompt_sha256': hashlib.sha256(VISUAL_PROMPT.encode()).hexdigest(),
            'no_titles_or_qa_sent': True, 'max_calls': MAX_CALLS, 'max_output_tokens': MAX_TOKENS}
        if (OUTPUT_ROOT / 'owner.json').exists() and read_json(OUTPUT_ROOT / 'owner.json') != current_owner:
            raise RuntimeError('existing visual extraction ownership/source/config mismatch; retained')
        atomic_json(OUTPUT_ROOT / 'owner.json', current_owner)
        report = read_json(OUTPUT_ROOT / 'report.json') if (OUTPUT_ROOT / 'report.json').exists() else {}
        if report.get('status') == 'complete' and (not args.build or report.get('indexes_complete')):
            print(json.dumps(report, ensure_ascii=False)); return
        report.update(status='running', model=model, output_dir=str(OUTPUT), visual_prompt_original=False,
            scope='four images; custom image-only implementation of paper section 3.3', cpu_threads=4, gpu_used=False,
            local_extraction_model_loaded=False, all_facts_verified=False, qa_accuracy_evaluated=False,
            training_started=False, strict_paper_replication=False)
        calls_path = OUTPUT_ROOT / 'vision_api_calls.json'
        calls = read_json(calls_path) if calls_path.exists() else {}
        try:
            if args.extract:
                from openai import OpenAI
                client = OpenAI(api_key=key, base_url=base_url, timeout=120, max_retries=0)
                for image in images:
                    image_id = image['image_id']
                    if image_id in calls:
                        reviewed_scene(image_id, calls[image_id]); continue
                    if len(calls) >= MAX_CALLS:
                        raise RuntimeError('visual API request budget reached')
                    calls[image_id] = {'status': 'requested', 'model': model, 'image_sha256': image['image_sha256'],
                        'request': {'image_only': True, 'prompt': VISUAL_PROMPT, 'image_bytes_not_duplicated_in_log': True}}
                    atomic_json(calls_path, calls)
                    print(json.dumps({'phase': 'vision_api_request', 'image_id': image_id, 'call': len(calls)}), flush=True)
                    started = time.monotonic()
                    response = client.chat.completions.create(model=model, messages=request_messages(Path(image['image_path']).read_bytes()),
                        temperature=0, max_tokens=MAX_TOKENS, response_format={'type': 'json_object'},
                        extra_body={'enable_thinking': False})
                    choice = response.choices[0]
                    calls[image_id].update(status='complete', output=choice.message.content or '', finish_reason=choice.finish_reason,
                        usage=response.usage.model_dump() if response.usage else {}, elapsed_seconds=round(time.monotonic()-started, 3))
                    atomic_json(calls_path, calls)
                    reviewed_scene(image_id, calls[image_id])
                    print(json.dumps({'phase': 'vision_api_saved', 'image_id': image_id}), flush=True)
            if set(calls) != {i['image_id'] for i in images}:
                raise RuntimeError('four completed image API records required before build')
            for image_id, record in calls.items():
                reviewed_scene(image_id, record)
            report.update(extraction_complete=True, api_calls=len(calls), api_usage={
                field: sum((c.get('usage') or {}).get(field, 0) or 0 for c in calls.values())
                for field in ['prompt_tokens', 'completion_tokens', 'total_tokens']})
            atomic_json(OUTPUT_ROOT / 'report.json', report)
            if args.build:
                build_graph(images, calls, report)
            if graph_hashes(PARENT_ROOT) != current_owner['parent_artifact_hashes']:
                raise RuntimeError('parent graph changed during visual build')
            snapshot = read_json(PARENT_ROOT / 'source_snapshot.json')
            subset = ROOT / 'datasets_mm/E-VQA/subsets' / SUBSET
            if not all(sha256(subset / f'qa_{s}.csv') == d for s, d in snapshot['qa_hashes'].items()):
                raise RuntimeError('QA files changed')
            report.update(status='complete', parent_graph_unchanged=True, raw_qa_unchanged=True)
            if (OUTPUT_ROOT / 'vision_source_review.json').exists():
                report['source_review_sha256'] = sha256(OUTPUT_ROOT / 'vision_source_review.json')
                report['source_review_kind'] = 'agent image check; originals retained; not human verification'
            report.pop('error', None)
            report.pop('error_type', None)
            atomic_json(OUTPUT_ROOT / 'report.json', report)
            print(json.dumps(report, ensure_ascii=False), flush=True)
        except Exception as exc:
            report.update(status='failed', error_type=type(exc).__name__,
                          error='visual task stopped; see bounded phase and cached response, no automatic retry', api_calls=len(calls))
            atomic_json(OUTPUT_ROOT / 'report.json', report)
            # Do not print API errors or headers which might expose credentials.
            print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__, 'api_calls': len(calls)}), flush=True)
            raise SystemExit(1) from None


if __name__ == '__main__':
    main()
