#!/usr/bin/env python3
"""Verify image grounding/integrity and disclose raw API schema failures; no API calls."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evograph_mm.kb.gldv2_subset import atomic_json
from evograph_mm.kb.visual_scene import parse_scene, apply_source_review
from scripts import run_evqa_api_visual_graph as visual


def main():
    import networkx as nx
    import numpy as np
    root, output = visual.OUTPUT_ROOT, visual.OUTPUT
    owner = visual.read_json(root / 'owner.json')
    report = visual.read_json(root / 'report.json')
    calls = visual.read_json(root / 'vision_api_calls.json')
    reviews = visual.read_json(root / 'vision_source_review.json')
    if report['status'] != 'complete' or not report['indexes_complete']:
        raise RuntimeError('incomplete visual build')
    assert len(calls) == 4
    assert visual.graph_hashes(visual.PARENT_ROOT) == owner['parent_artifact_hashes']
    raw_failures, reviewed = [], {}
    for image_id, record in calls.items():
        try:
            parse_scene(record)
        except ValueError:
            raw_failures.append(image_id)
        reviewed[image_id] = apply_source_review(record, reviews.get(image_id))
    entities = visual.read_json(output / 'kv_store_entities.json')
    edges = visual.read_json(output / 'kv_store_hyperedges.json')
    graph = nx.read_graphml(output / 'graph_chunk_entity_relation.graphml')
    mm_graph = nx.read_graphml(output / 'mm_store/graph/graph_mm_entity_relation.graphml')
    sidecar = visual.read_json(output / 'mm_store/graph/graphr1_hit_source_sidecar.json')
    assert set(graph) == {r['entity_name'] for r in entities.values()} | {r['hyperedge_name'] for r in edges.values()}
    assert len(sidecar['entity']) == len(entities) and len(sidecar['hyperedge']) == len(edges)
    scene_records = visual.read_json(output / 'mm_store/visual/scene_records.json')
    assert scene_records == {k: v.model_dump() for k, v in reviewed.items()}
    assert len([r for r in entities.values() if r['metadata'].get('kind') == 'image_anchor']) == 4
    for record in edges.values():
        kind = record['metadata'].get('kind')
        if kind in {'scene', 'visual_relation', 'dataset_association'}:
            anchor = record['metadata']['anchor_node']
            name = record['hyperedge_name']
            assert graph.has_edge(name, anchor) and mm_graph.has_edge(name, anchor)
            assert sidecar['hyperedge'][name]['source_image_ids'] == [record['metadata']['image_id']]
            if kind == 'dataset_association':
                assert record['metadata']['visual_identity_verified'] is False
                assert record['metadata']['evidence_basis'] == 'dataset_image_article_association'
    for kind, records, name_key in [('entity', entities, 'entity_name'), ('hyperedge', edges, 'hyperedge_name')]:
        meta = visual.read_json(output / f'{kind}_index_metadata.json')
        vectors = np.load(output / f'corpus_{kind}.npy', allow_pickle=False)
        assert len(meta['ids']) == len(records) == len(vectors)
        assert vectors.shape[1] == 1536 and np.isfinite(vectors).all()
        assert set(meta['ids']) == {v[name_key] for v in records.values()}
        assert not any('Nicosia' in s or '[text cuts off]' in s for s in meta['contents'])
    excluded_objects = sum(len(r.get('excluded_object_ids', [])) for r in reviews.values())
    omitted_boxes = sum(len(r.get('omit_box_object_ids', [])) for r in reviews.values())
    excluded_relations = 0
    for image_id, review in reviews.items():
        removed = set(review.get('excluded_object_ids', []))
        raw = json.loads(calls[image_id]['output'])
        excluded_relations += sum(bool(removed & set(r['object_ids'])) for r in raw['relations'])
    # The actual key is checked in memory; its value is never printed or recorded.
    from dotenv import dotenv_values
    key = dotenv_values(Path(__file__).resolve().parents[1] / '.env')['OPENAI_API_KEY']
    assert all(key not in p.read_text() for p in root.rglob('*') if p.is_file()
               and p.suffix in {'.json', '.jsonl', '.graphml', '.log'})
    result = {'api_calls': 4, 'api_usage': report['api_usage'], 'new_audit_api_calls': 0,
        'raw_schema_failure_images': raw_failures, 'raw_schema_failure_count': len(raw_failures),
        'source_reviewed_images': len(reviews), 'source_review_excluded_objects': excluded_objects,
        'source_review_excluded_dependent_relations': excluded_relations, 'source_review_omitted_boxes': omitted_boxes,
        'source_review_normalized_1000_scale_images': sum('bbox_scale' in r for r in reviews.values()),
        'graph_image_anchors_verified': True, 'graph_and_index_alignment_verified': True,
        'parent_graph_unchanged': True, 'original_responses_retained': True, 'api_key_absent_from_artifacts': True,
        'source_review_sha256': visual.sha256(root / 'vision_source_review.json'),
        'entities': len(entities), 'hyperedges': len(edges), 'visual_objects': sum(len(s.objects) for s in reviewed.values()),
        'scene_hyperedges': 4, 'visual_relation_hyperedges': report['accepted_visual_relations'],
        'dataset_association_hyperedges': 4, 'resolved_visual_text_objects': report['resolved_objects'],
        'all_facts_verified': False, 'human_reviewed': False, 'qa_accuracy_evaluated': False,
        'box_geometry_verified_by_detector': False,
        'limitations': ['Raw API responses all required explicit source review/schema handling; do not claim strict schema success.',
            'Confidence-threshold quarantine counts in build report do not include source-review exclusions, reported separately here.',
            'Approximate bounding boxes are annotations, not detector-validated localizations.',
            'No same-label resolved visual/text entities were found; shared image/article association is not landmark recognition.',
            'Index self-description checks and a free-text smoke query are not VQA accuracy evaluation.']}
    target = root / 'quality_audit.json'
    if target.exists():
        raise RuntimeError('existing audit retained')
    atomic_json(target, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
