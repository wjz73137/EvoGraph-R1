#!/usr/bin/env python3
"""Record API provenance checks and explicit manual-review warnings, without new calls."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evograph_mm.kb.gldv2_subset import atomic_json
from scripts.run_evqa_api_graph import OUTPUT, OUTPUT_ROOT


def main():
    owner = json.loads((OUTPUT / 'owner.json').read_text())
    calls = json.loads((OUTPUT / 'api_calls.json').read_text())
    snapshot = json.loads((OUTPUT_ROOT / 'source_snapshot.json').read_text())
    entities = json.loads((OUTPUT / 'kv_store_entities.json').read_text())
    edges = json.loads((OUTPUT / 'kv_store_hyperedges.json').read_text())
    sidecar = json.loads((OUTPUT / 'mm_store/graph/graphr1_hit_source_sidecar.json').read_text())
    model = owner['graph_llm'].removeprefix('api/')
    assert owner['graph_llm'].startswith('api/')
    assert all(c['status'] == 'complete' and c['model'] == model
               and c['finish_reason'] != 'length' for c in calls.values())
    assert all(v['metadata']['ai_model'] == model for v in [*entities.values(), *edges.values()])
    assert len(sidecar['entity']) == len(entities)
    assert len(sidecar['hyperedge']) == len(edges)
    assert all(v['source_chunk_ids'] for kind in sidecar.values() for v in kind.values())
    texts = [d['contents'] for d in snapshot['documents']]
    warnings = []
    for edge_id, record in edges.items():
        statement = record['content']
        if 'Nicosia' in statement and not any('Nicosia' in text for text in texts):
            warnings.append({'edge_id': edge_id, 'statement': statement,
                             'reason': 'specific capital name is not stated in supplied source; needs external verification, not assumed false'})
        if '[text cuts off]' in statement:
            warnings.append({'edge_id': edge_id, 'statement': statement,
                             'reason': 'unfinished source fragment was encoded as a hyperedge'})
    duplicate_examples = [
        [v['content'] for v in edges.values()
         if 'Newport Castle' in v['content'] and 'Civil War' in v['content']],
        [v['content'] for v in edges.values()
         if 'Newport Castle' in v['content'] and 'Listed building since 1951' in v['content']],
    ]
    audit = {'api_model': model, 'api_calls': len(calls),
             'api_provenance_checks_passed': True, 'source_mapping_passed': True,
             'entities': len(entities), 'hyperedges': len(edges),
             'warnings': warnings, 'duplicate_examples': duplicate_examples,
             'additional_review_notes': [
                 'Gleaning duplicates paraphrased facts; 66 hyperedges are not 66 distinct verified facts.',
                 'The original probably-A-or-B builder claim also appears as separate probable-A and probable-B claims; disjunction must be reviewed.',
                 'John B. Jones is correctly described as a house alias, but its PRODUCT entity type still needs schema review.',
                 '1571 was derived from 1572 minus one year; distinguish source-explicit and derived dates.',
             ],
             'all_facts_verified': False, 'graph_modified_by_audit': False,
             'new_api_calls': 0, 'training_ready': False}
    target = OUTPUT_ROOT / 'quality_audit.json'
    if target.exists():
        raise RuntimeError('existing quality audit retained')
    atomic_json(target, audit)
    print(json.dumps({'provenance_passed': True, 'warning_records': len(warnings),
                      'all_facts_verified': False, 'audit': str(target)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
