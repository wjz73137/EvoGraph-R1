#!/usr/bin/env python3
"""Source-scoped, reversible cleanup of the four-document API graph; no API calls."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evograph_mm.kb.gldv2_subset import atomic_json
from scripts.run_evqa_api_graph import OUTPUT as BASE, OUTPUT_ROOT as BASE_ROOT
from scripts.run_evqa_gpu_smoke import ROOT, SUBSET
from scripts.run_evqa_retrieval_smoke import GME

TARGET_ROOT = ROOT / 'expr_mm/evqa_api_graph_cleaned_v1'
KV_NAMES = {'entity': 'kv_store_entities.json', 'hyperedge': 'kv_store_hyperedges.json'}
SOURCE_SHA256 = '10027097ad107443ef11d6b8526d4f6ff96e39631c87a5e7df3e18d8b132f5b8'
QUARANTINE_EDGES = {
    'rel-71f82d5914353c8f6f74517417163965': 'Nicosia is absent from supplied passage; external verification needed, not declared false.',
    'rel-5ca0a490ac4f89db79aee9290f398b20': 'Unfinished inauguration fragment.',
    'rel-9a8822acbb3f9e82fe5bb8c3d2850da8': 'Unfinished inauguration fragment.',
    'rel-c6e7347e9ca5f92466ac543cfcce737d': 'Separate probable-A claim loses original probably-A-or-B scope.',
    'rel-9eb38c7717825b450ac8e4ecde10a614': 'Separate probable-B claim loses original probably-A-or-B scope.',
    'rel-4607067c3bfb7b10a6c5a118f0cdeb2d': 'Conflates arts-centre composition with other inn amenities; held for review, not declared false.',
}
QUARANTINE_ENTITIES = {
    'ent-710adf1fa5aa6b9b0f4d6e0aad8cec22': 'Specific capital name is not supplied by source.',
    'ent-89b48f4fa4e9b99e2cd5797f1ac44abc': 'Entity from unfinished inauguration fragment.',
    'ent-9f7089b1df617604a2619b3fc595184e': 'Then is part of an unfinished sentence, not a usable time entity.',
}
# Explicitly compared statements only. Never merge using vector similarity alone.
MERGES = {
    'rel-f3b6bab8d897dd5d582f9ac992f6ac03': ['rel-538e3412daf3b5dfd218ec1ff6a90f05'],
    'rel-f6a746f93493140aaa69e21163b10b27': ['rel-4b8541640cecdbcd264f5433ef87fffc'],
    'rel-5b7bda88cf7eaba634fad4e054c8f713': ['rel-6b3a0164f7b27e071988ac9f2f9af4f5'],
    'rel-5bbf6f77532eccbfb635841a53eed035': ['rel-b7dfb39c6d8422166baf9296f4780e08'],
    'rel-5f8fed1ece712c89794b6003362f6ecb': ['rel-9190e223c9213fbc97068815afa90d7c', 'rel-95c1689b7f2b17cfa8bf43ecda78cfc4'],
    'rel-1c3b570c585793cd516efcfbc7807e2b': ['rel-a285eac08c2dfc2d2a00c75eca36d800'],
    'rel-55779f0bfd42d5496c9930319fe200b5': ['rel-1087a88388c05d0dc205575afece1834'],
    'rel-eb908f25c8e550de690027ce2b8eab2f': ['rel-5b6961e16f153ddec1c494f039cf16f2'],
}
BUILDER_QUOTE = 'probably by Hugh de Audley, 1st Earl of Gloucester or his son-in-law, Ralph, Earl of Stafford'
BUILDER_DESCRIPTION = 'Newport Castle was probably built by Hugh de Audley, 1st Earl of Gloucester or his son-in-law, Ralph, Earl of Stafford.'
DESCRIPTIONS = {
    'ent-4e64b4bbe0d1b01dba637f06f5ac767c': BUILDER_DESCRIPTION + ' Hugh de Audley is the father-in-law of Ralph, Earl of Stafford.',
    'ent-0b22aabc1254f98ff5b4e0e53ee0501a': BUILDER_DESCRIPTION + ' Ralph, Earl of Stafford is the son-in-law of Hugh de Audley.',
    'ent-2ad2e45bf8a5dfb89428ea1458d94ade': '1571 is derived as 1572 minus one year for the Ottoman seizure of Cyprus from the Venetians. The passage states that Büyük Han was built in 1572, the year after that seizure; it does not explicitly print 1571.',
}
# A declared project schema normalization, not a claim that PRODUCT is factually false.
LOCATION_NAMES = {
    'NEWPORT CASTLE', 'CASTELL CASNEWYDD', 'WELSH: CASTELL CASNEWYDD',
    'CHARLES WILLIAM JONES HOUSE', 'JOHN B. JONES', 'BÜYÜK HAN',
    'MEGÁLO PANDOCHEÍO', 'GREAT INN', 'LIT. GREAT INN', 'PAMBAN BRIDGE',
    'ANNAI INDIRA GANDHI ROAD BRIDGE', 'BANDRA-WORLI SEA LINK',
}
ALIASES = {
    'CASTELL CASNEWYDD': 'NEWPORT CASTLE', 'WELSH: CASTELL CASNEWYDD': 'NEWPORT CASTLE',
    'JOHN B. JONES': 'CHARLES WILLIAM JONES HOUSE',
    'MEGÁLO PANDOCHEÍO': 'BÜYÜK HAN', 'GREAT INN': 'BÜYÜK HAN', 'LIT. GREAT INN': 'BÜYÜK HAN',
}


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_hashes(root):
    # Runtime logging is not graph/index state and can change during searches.
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.name != 'graphr1.log'}


def note(record, reason, **extra):
    record.setdefault('metadata', {}).update(
        cleanup={'tool': 'clean_evqa_api_graph_v1', 'review': 'agent_source_check',
                 'reason': reason, **extra}, human_reviewed=False)


def union_provenance(target, other):
    for key in ('wikipedia_urls', 'wikipedia_titles', 'source_chunk_ids'):
        target[key] = sorted(set(target.get(key, [])) | set(other.get(key, [])))


def clean_records(entities, edges, graph, sidecar):
    """Pure transformation: original dictionaries/graph remain untouched."""
    entities, edges, graph, sidecar = copy.deepcopy((entities, edges, graph, sidecar))
    required_entities = set(QUARANTINE_ENTITIES) | set(DESCRIPTIONS) | {'ent-3b1869f9c4e60352fb498eed3dcab29d'}
    required_edges = set(QUARANTINE_EDGES) | set(MERGES) | {i for ids in MERGES.values() for i in ids} | {'rel-f77c505e541366e9c989b7f62156847e'}
    if not required_entities <= entities.keys() or not required_edges <= edges.keys():
        raise RuntimeError('cleanup plan does not match the baseline records')
    audit = {'quarantined': {'entity': {}, 'hyperedge': {}}, 'merged': [],
             'modified_entities': {}, 'derived_annotations': []}
    for kind, records, plan, name_key in [
        ('hyperedge', edges, QUARANTINE_EDGES, 'hyperedge_name'),
        ('entity', entities, QUARANTINE_ENTITIES, 'entity_name'),
    ]:
        for record_id, reason in plan.items():
            record = records.pop(record_id)
            name = record[name_key]
            audit['quarantined'][kind][record_id] = {
                'reason': reason, 'record': record, 'node': dict(graph.nodes[name]),
                'incident_links': [(a, b, dict(v)) for a, b, v in graph.edges(name, data=True)],
                'source_mapping': sidecar[kind].pop(name),
            }
            graph.remove_node(name)
    for keep, aliases in MERGES.items():
        canonical = edges[keep]['hyperedge_name']
        lineage = []
        for alias in aliases:
            record = edges.pop(alias)
            name = record['hyperedge_name']
            lineage.append({'id': alias, 'record': record, 'node': dict(graph.nodes[name]),
                            'incident_links': [(a, b, dict(v)) for a, b, v in graph.edges(name, data=True)],
                            'source_mapping': sidecar['hyperedge'][name]})
            # Duplicate support does not increase confidence or support weight.
            for neighbor, attrs in list(graph[name].items()):
                if neighbor not in graph[canonical]:
                    graph.add_edge(canonical, neighbor, **copy.deepcopy(attrs))
                else:
                    current = graph[canonical][neighbor]
                    current['weight'] = max(float(current.get('weight', 0)), float(attrs.get('weight', 0)))
                    current['source_id'] = '<SEP>'.join(sorted(set(current.get('source_id', '').split('<SEP>')) | set(attrs.get('source_id', '').split('<SEP>'))))
            graph.nodes[canonical]['source_id'] = '<SEP>'.join(sorted(set(graph.nodes[canonical]['source_id'].split('<SEP>')) | set(graph.nodes[name]['source_id'].split('<SEP>'))))
            union_provenance(sidecar['hyperedge'][canonical], sidecar['hyperedge'].pop(name))
            graph.remove_node(name)
        note(edges[keep], 'Explicit source-scoped paraphrase deduplication.', merged_record_ids=aliases)
        graph.nodes[canonical]['metadata'] = json.dumps(edges[keep]['metadata'], ensure_ascii=False)
        audit['merged'].append({'canonical_id': keep, 'original_duplicates': lineage})
    for record_id, record in entities.items():
        name = record['entity_name']
        plain_name = name.strip('"')
        changes = []
        original = copy.deepcopy(record)
        if record_id in DESCRIPTIONS:
            description = DESCRIPTIONS[record_id]
            graph.nodes[name]['description'] = description
            record['content'] = name + description
            changes.append('Restore disjunction/probability scope.' if record_id != 'ent-2ad2e45bf8a5dfb89428ea1458d94ade' else 'Make derived date explicit in searchable description.')
        if plain_name in LOCATION_NAMES:
            record['metadata']['entity_type'] = '"LOCATION"'
            graph.nodes[name]['entity_type'] = '"LOCATION"'
            changes.append('Declared schema: named fixed structures and their aliases use LOCATION; not a factual correction.')
        if plain_name in ALIASES:
            record['metadata']['alias_of'] = ALIASES[plain_name]
            changes.append('Preserve source-stated building alias; do not interpret as a person.')
        if changes:
            note(record, ' '.join(changes))
            graph.nodes[name]['metadata'] = json.dumps(record['metadata'], ensure_ascii=False)
            audit['modified_entities'][record_id] = {'before': original, 'changes': changes}
    for record_id, records, name_key in [
        ('ent-2ad2e45bf8a5dfb89428ea1458d94ade', entities, 'entity_name'),
        ('ent-3b1869f9c4e60352fb498eed3dcab29d', entities, 'entity_name'),
        ('rel-f77c505e541366e9c989b7f62156847e', edges, 'hyperedge_name'),
    ]:
        record = records[record_id]
        original = copy.deepcopy(record)
        record['metadata'].update(claim_origin='derived_date', derivation='1572 - 1 = 1571',
                                  source_quote='built by the Ottomans in 1572, the year after they had seized Cyprus from the Venetians')
        graph.nodes[record[name_key]]['metadata'] = json.dumps(record['metadata'], ensure_ascii=False)
        audit['derived_annotations'].append({'id': record_id, 'before': original})
    expected_nodes = {v['entity_name'] for v in entities.values()} | {v['hyperedge_name'] for v in edges.values()}
    if set(graph) != expected_nodes:
        raise RuntimeError('graph and active KV records diverged')
    if any(graph.degree(v['hyperedge_name']) == 0 for v in edges.values()):
        raise RuntimeError('cleanup created an orphan hyperedge')
    return entities, edges, graph, sidecar, audit


def select_vectors(metadata, matrix, records):
    """Return same-ID/same-content vectors, identifying every text needing re-encoding."""
    import numpy as np
    if matrix.ndim != 2 or len(metadata['ids']) != len(matrix) or len(metadata['contents']) != len(matrix):
        raise ValueError('baseline vector alignment mismatch')
    if len(set(metadata['ids'])) != len(matrix):
        raise ValueError('duplicate baseline embedding ids')
    positions = {record_id: i for i, record_id in enumerate(metadata['ids'])}
    vectors = np.empty((len(records), matrix.shape[1]), dtype=np.float32)
    changed = []
    for row, record in enumerate(records):
        old_row = positions.get(record['id'])
        if old_row is None:
            raise ValueError('unexpected new embedding id')
        if metadata['contents'][old_row] == record['content']:
            vectors[row] = matrix[old_row]
        else:
            changed.append(row)
    return vectors, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='create a new cleaned graph and verify real CPU retrieval')
    parser.add_argument('--resume', action='store_true', help='resume only a matching, incomplete cleanup; never overwrite a completed graph')
    args = parser.parse_args()
    if args.resume and not args.apply:
        parser.error('--resume requires --apply')
    owner = read_json(BASE / 'owner.json')
    if sha256(BASE_ROOT / 'source_snapshot.json') != SOURCE_SHA256 or owner['source_sha256'] != SOURCE_SHA256 or not owner['graph_llm'].startswith('api/'):
        raise RuntimeError('source/ownership mismatch; no cleanup applied')
    import networkx as nx
    entities, edges = read_json(BASE / 'kv_store_entities.json'), read_json(BASE / 'kv_store_hyperedges.json')
    if len(entities) != 94 or len(edges) != 66:
        raise RuntimeError('baseline record counts changed; cleanup plan needs review')
    snapshot = read_json(BASE_ROOT / 'source_snapshot.json')
    if not any(BUILDER_QUOTE in d['contents'] for d in snapshot['documents']):
        raise RuntimeError('expected builder disjunction missing from source')
    result = clean_records(entities, edges, nx.read_graphml(BASE / 'graph_chunk_entity_relation.graphml'),
                           read_json(BASE / 'mm_store/graph/graphr1_hit_source_sidecar.json'))
    active_entities, active_edges, graph, sidecar, audit = result
    summary = {'entities_before': 94, 'entities_after': len(active_entities),
               'hyperedges_before': 66, 'hyperedges_after': len(active_edges),
               'quarantined_entities': len(QUARANTINE_ENTITIES), 'quarantined_hyperedges': len(QUARANTINE_EDGES),
               'merged_duplicate_hyperedges': sum(map(len, MERGES.values())),
               'schema_normalized_entities': len(LOCATION_NAMES), 'changed_search_descriptions': len(DESCRIPTIONS),
               'new_api_calls': 0, 'gpu_used': False, 'training_started': False,
               'all_facts_verified': False, 'training_ready': False}
    print(json.dumps({'dry_run': not args.apply, **summary}, ensure_ascii=False), flush=True)
    if not args.apply:
        return
    target = TARGET_ROOT / 'E-VQA'
    report_path = TARGET_ROOT / 'report.json'
    baseline_hashes = file_hashes(BASE_ROOT)
    if args.resume:
        previous_owner = read_json(target / 'owner.json')
        if (TARGET_ROOT.is_symlink() or target.is_symlink()
                or read_json(report_path)['status'] != 'running'
                or read_json(TARGET_ROOT / 'baseline_hashes.json') != baseline_hashes
                or previous_owner.get('parent_graph') != str(BASE)
                or previous_owner.get('cleanup_version') != 1
                or any(previous_owner.get(k) != owner[k] for k in owner)
                or read_json(target / KV_NAMES['entity']) != active_entities
                or read_json(target / KV_NAMES['hyperedge']) != active_edges
                or read_json(target / 'mm_store/graph/graphr1_hit_source_sidecar.json') != sidecar
                or not nx.utils.graphs_equal(nx.read_graphml(target / 'graph_chunk_entity_relation.graphml'), graph)):
            raise RuntimeError('incomplete cleanup does not match current plan/baseline; retained without overwrite')
        atomic_json(TARGET_ROOT / 'resume_manifest.json', {'previous_runner_sha256': previous_owner['cleanup_runner_sha256'],
                    'current_runner_sha256': sha256(Path(__file__)), 'same_active_records': True,
                    'same_graph_and_source_mappings': True, 'baseline_unchanged': True})
    else:
        # Exclusive creation: never overwrite baseline or a prior cleanup.
        TARGET_ROOT.mkdir(exist_ok=False)
        atomic_json(report_path, {'status': 'running', **summary})
        atomic_json(TARGET_ROOT / 'baseline_hashes.json', baseline_hashes)
        if any(p.is_symlink() for p in BASE.rglob('*')):
            raise RuntimeError('unexpected source symlink; refusing to copy')
        shutil.copytree(BASE, target, ignore=shutil.ignore_patterns('graphr1.log', 'api_retrieval_test.json'))
        shutil.copy2(BASE_ROOT / 'source_snapshot.json', TARGET_ROOT / 'source_snapshot.json')
    atomic_json(TARGET_ROOT / 'cleanup_audit.json', {**audit, 'policy': {
        'scope': 'only the pinned four-document baseline', 'reviewer': 'agent_source_check',
        'human_reviewed': False, 'all_facts_verified': False,
        'schema': 'named fixed structures and their aliases use LOCATION',
        'quarantine_policy': 'not served in active graph/index; originals retained; not automatically false',
        'baseline_retained_at': str(BASE_ROOT), 'raw_api_output_unchanged': True}})
    atomic_json(target / 'kv_store_entities.json', active_entities)
    atomic_json(target / 'kv_store_hyperedges.json', active_edges)
    atomic_json(target / 'mm_store/graph/graphr1_hit_source_sidecar.json', sidecar)
    nx.write_graphml(graph, target / 'graph_chunk_entity_relation.graphml')
    atomic_json(target / 'owner.json', {**owner, 'cleanup_version': 1,
                'parent_graph': str(BASE), 'cleanup_runner_sha256': sha256(Path(__file__))})
    os.environ.update(CUDA_VISIBLE_DEVICES='', MM_EMBED_DEVICE='cpu', EVOGRAPH_MM_EMBED_RUNTIME_DEVICE='cpu',
                      EVOGRAPH_MM_ENABLE_BGE_TEXT='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      TOKENIZERS_PARALLELISM='false')
    import torch
    import faiss
    import numpy as np
    torch.set_num_threads(4)
    faiss.omp_set_num_threads(4)
    from evograph_mm.kb.build import _load_entity_records, _load_hyperedge_records, _write_root_graphr1_index
    from evograph_mm.kb.indexing.encoders import GMEQwen2VLEncoder, TEXT_DOCUMENT_INSTRUCTION
    encoder = GMEQwen2VLEncoder(GME, batch_size=1)
    index_reports = {}
    for kind, loader in [('entity', _load_entity_records), ('hyperedge', _load_hyperedge_records)]:
        records = loader(target / KV_NAMES[kind])
        old_metadata = read_json(BASE / f'{kind}_index_metadata.json')
        matrix, changed = select_vectors(old_metadata, np.load(BASE / f'corpus_{kind}.npy', allow_pickle=False), records)
        if changed:
            matrix[changed] = encoder.encode_texts([records[i]['content'] for i in changed], instruction=TEXT_DOCUMENT_INSTRUCTION)
        if not np.isfinite(matrix).all() or np.any(np.linalg.norm(matrix, axis=1) == 0):
            raise RuntimeError('invalid real embedding vectors')
        index_reports[kind] = _write_root_graphr1_index(
            output_dir=target, namespace=kind, ids=[v['id'] for v in records], contents=[v['content'] for v in records],
            embeddings=matrix, corpus_file=f'corpus_{kind}.npy', index_file=f'index_{kind}.bin',
            metadata_file=f'{kind}_index_metadata.json', model_path=GME,
            model_repo_id=old_metadata['model_repo_id'], encoder_mode=old_metadata['encoder_mode'])
        index_reports[kind].update(reused_vectors=len(records)-len(changed), reencoded_vectors=len(changed))
    counts = {'entities': len(active_entities), 'hyperedges': len(active_edges), 'nodes': len(graph),
              'edges': graph.number_of_edges(), 'full_docs': 4, 'text_chunks': 4}
    build_report = read_json(target / 'build_report.json')
    atomic_json(target / 'build_report.json', {**build_report, 'graph_counts': counts,
                'cleanup_version': 1, 'quarantined_records_excluded': True})
    os.chdir(target)
    from evograph_mm.kb.api import create_app
    from fastapi.testclient import TestClient
    app = create_app(working_dir=target, model_path=GME, dataset='E-VQA', subset=SUBSET,
                     encoder_factory=lambda *a, **kw: encoder, reload_interval=0)
    if app.state.mm_api.status()['status'] != 'ready':
        raise RuntimeError('cleaned retrieval is not ready')
    checks = []
    with TestClient(app) as client:
        for query in ['Who probably built Newport Castle?', 'John B. Jones', 'Büyük Han', 'Annai Indira Gandhi Road Bridge']:
            response = client.post('/search', json={'queries': [query], 'entity_top_k': 5,
                'hyperedge_top_k': 5, 'rag_top_k': 0, 'image_top_k': 0})
            if response.status_code != 200 or not json.loads(response.json()[0]).get('results'):
                raise RuntimeError('cleaned graph retrieval failed')
            payload = response.json()
            if 'Nicosia' in json.dumps(payload, ensure_ascii=False) or '[text cuts off]' in json.dumps(payload):
                raise RuntimeError('quarantined material remains in active search')
            checks.append({'query': query, 'result': payload})
    atomic_json(target / 'api_retrieval_test.json', checks)
    # No extraction or image/text re-indexing: those byte-identical artifacts remain immutable.
    immutable = ['api_calls.json', 'kv_store_full_docs.json', 'kv_store_text_chunks.json',
                 'graphr1_text/text_documents.jsonl', 'mm_store/visual/image_records.jsonl']
    immutable += [str(p.relative_to(BASE)) for p in (BASE / 'mm_store/indexing').iterdir() if p.is_file()]
    if not all(sha256(BASE / name) == sha256(target / name) for name in immutable):
        raise RuntimeError('original API/source or document/image index was unexpectedly modified')
    if file_hashes(BASE_ROOT) != baseline_hashes:
        raise RuntimeError('baseline changed during cleanup')
    subset = ROOT / 'datasets_mm/E-VQA/subsets' / SUBSET
    if not all(sha256(subset / f'qa_{split}.csv') == digest for split, digest in snapshot['qa_hashes'].items()):
        raise RuntimeError('QA files changed')
    atomic_json(report_path, {'status': 'complete', **summary, 'graph_counts': counts,
                'graph_indexes': index_reports, 'indexes_complete': True, 'retrieval_test_passed': True,
                'baseline_unchanged': True, 'raw_api_output_unchanged': True, 'raw_qa_unchanged': True,
                'source_snapshot_sha256': SOURCE_SHA256, 'output_dir': str(target),
                'completed_at': datetime.now(timezone.utc).isoformat()})
    print(json.dumps(read_json(report_path), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
