import copy
import json

import networkx as nx
import pytest

from evograph_mm.kb.visual_scene import VISUAL_PROMPT, normalized_label, parse_scene, apply_source_review
from scripts import run_evqa_api_visual_graph as visual


def scene_output():
    return {'image_only': True, 'scene_description': 'Small boats float on blue water under a cloudy sky.',
        'objects': [
            {'id': 'o1', 'name': 'boats', 'entity_type': 'OBJECT', 'description': 'Small visible boats on the water.',
             'bbox': [0.1, 0.3, 0.8, 0.7], 'confidence': 9.0},
            {'id': 'o2', 'name': 'water', 'entity_type': 'LOCATION', 'description': 'An expanse of blue water.',
             'bbox': [0.0, 0.3, 1.0, 1.0], 'confidence': 9.0}],
        'relations': [{'statement': 'Boats float on the water.', 'object_ids': ['o1', 'o2'], 'confidence': 9.0}],
        'visible_text': [], 'uncertainties': ['The location is not identifiable from the image.']}


def record(output=None, **values):
    return {'status': 'complete', 'finish_reason': 'stop', 'output': json.dumps(output or scene_output()), **values}


def source_fixture(monkeypatch):
    entities = {'existing': {'entity_name': '"PAMBAN BRIDGE"', 'content': 'A railway bridge.',
                             'metadata': {'entity_type': 'LOCATION'}}}
    graph = nx.Graph()
    graph.add_node('"PAMBAN BRIDGE"', role='entity', description='A railway bridge.', source_id='real-text-chunk')
    images = [{'image_id': 'img-one', 'image_path': '/data/image.jpg', 'image_sha256': 'hash', 'data_id': 'data-one'}]
    monkeypatch.setattr(visual, 'read_json', lambda path: {'img-one': {'canonical_entity': 'PAMBAN BRIDGE'}})
    sidecar = {'entity': {'"PAMBAN BRIDGE"': {'source_chunk_ids': ['real-text-chunk']}}, 'hyperedge': {}}
    return entities, {}, graph, sidecar, {'img-one': parse_scene(record())}, images, 'api-test-model'


def test_payload_has_only_fixed_prompt_and_image_bytes_no_article_or_qa():
    payload = visual.request_messages(b'test-jpeg-bytes')
    assert len(payload) == 1 and len(payload[0]['content']) == 2
    assert payload[0]['content'][1]['text'] == VISUAL_PROMPT
    assert payload[0]['content'][0]['image_url']['url'].startswith('data:image/jpeg;base64,')
    assert 'Pamban' not in json.dumps(payload) and 'Newport' not in json.dumps(payload)
    assert 'filename' not in payload[0]


@pytest.mark.parametrize('finish_reason,status', [('length', 'complete'), ('stop', 'requested'), ('content_filter', 'complete')])
def test_incomplete_visual_outputs_never_enter_graph(finish_reason, status):
    with pytest.raises(ValueError, match='incomplete'):
        parse_scene(record(finish_reason=finish_reason, status=status))


@pytest.mark.parametrize('mutation', ['unknown_object', 'duplicate_object', 'bad_box', 'inverted_box', 'extra_field', 'wrong_modality'])
def test_schema_fails_closed_on_invalid_grounding(mutation):
    output = scene_output()
    if mutation == 'unknown_object':
        output['relations'][0]['object_ids'].append('o6')
    elif mutation == 'duplicate_object':
        output['objects'][1]['id'] = 'o1'
    elif mutation == 'bad_box':
        output['objects'][0]['bbox'][0] = -0.1
    elif mutation == 'inverted_box':
        output['objects'][0]['bbox'] = [0.8, 0.5, 0.1, 0.2]
    elif mutation == 'extra_field':
        output['article_title'] = 'Pamban Bridge'
    else:
        output['image_only'] = False
    with pytest.raises(ValueError):
        parse_scene(record(output))


def test_all_scene_and_relation_hyperedges_include_real_image_anchor(monkeypatch):
    fixture = source_fixture(monkeypatch)
    entities, edges, graph, sidecar, audit, chunks, docs = visual.add_visual_records(*fixture)
    item = audit['images'][0]
    for node in [item['scene_node'], *item['visual_relation_nodes']]:
        assert graph.has_edge(node, item['anchor_node'])
        assert sidecar['hyperedge'][node]['source_image_ids'] == ['img-one']
        assert sidecar['hyperedge'][node]['source_chunk_ids'] == ['visualchunk::img-one']
    assert 'visualchunk::img-one' in chunks and 'visualdoc::img-one' in docs
    expected_nodes = {v['entity_name'] for v in entities.values()} | {v['hyperedge_name'] for v in edges.values()}
    assert set(graph) == expected_nodes
    assert fixture[1] == {}  # Baseline input was not mutated.
    assert not fixture[2].has_node(item['scene_node'])


def test_dataset_bridge_association_is_never_promoted_to_visual_bridge_detection(monkeypatch):
    result = visual.add_visual_records(*source_fixture(monkeypatch))
    _, edges, graph, sidecar, audit, _, _ = result
    item = audit['images'][0]
    assert 'bridge' not in item['scene_node'].casefold()
    assert all('bridge' not in node.casefold() for node in item['visual_object_nodes'])
    association = item['association_node']
    assert graph.has_edge(association, '"PAMBAN BRIDGE"')
    assert 'not a visual identification' in association
    assert sidecar['hyperedge'][association]['evidence_basis'] == 'dataset_image_article_association'
    assert sidecar['hyperedge'][association]['visual_identity_verified'] is False
    assert all(v['metadata']['human_reviewed'] is False for v in edges.values())


def test_low_confidence_objects_and_relations_are_quarantined(monkeypatch):
    fixture = list(source_fixture(monkeypatch))
    output = scene_output(); output['objects'][0]['confidence'] = 4.0
    fixture[4] = {'img-one': parse_scene(record(output))}
    _, edges, _, _, audit, _, _ = visual.add_visual_records(*fixture)
    assert len(audit['quarantined_objects']) == 1
    assert len(audit['quarantined_relations']) == 1
    assert audit['images'][0]['visual_relation_nodes'] == []
    assert not any('::o1::' in str(v) for v in audit['images'][0]['visual_object_nodes'])
    assert len(edges) == 2  # Scene anchor and explicitly qualified dataset association only.


def test_normalization_is_not_fuzzy_identity_resolution():
    assert normalized_label('"OPEN COURTYARD"') == normalized_label('open courtyard')
    assert normalized_label('castle') != normalized_label('Newport Castle')
    assert normalized_label('boats') != normalized_label('ships')
    assert normalized_label('sea') != normalized_label('Palk Strait')


def test_neutral_scene_weight_is_not_fabricated_api_confidence(monkeypatch):
    _, edges, _, _, audit, _, _ = visual.add_visual_records(*source_fixture(monkeypatch))
    scene = audit['images'][0]['scene_node']
    metadata = next(v['metadata'] for v in edges.values() if v['hyperedge_name'] == scene)
    assert 'confidence' not in metadata
    assert metadata['confidence_origin'].startswith('not_requested')


def test_hash_pinned_review_quarantines_invalid_box_and_dependencies_without_repair():
    import hashlib
    output = scene_output()
    output['objects'][0]['bbox'][0] = 320.0
    raw = record(output)
    review = {'raw_output_sha256': hashlib.sha256(raw['output'].encode()).hexdigest(),
              'reviewer': 'agent_image_source_check', 'human_reviewed': False,
              'reason': 'Invalid normalized box; preserve original and exclude record.',
              'excluded_object_ids': ['o1']}
    with pytest.raises(ValueError):
        parse_scene(raw)
    scene = apply_source_review(raw, review)
    assert len(scene.objects) == 1 and scene.objects[0].id == 'o2'
    assert scene.relations == []
    assert json.loads(raw['output'])['objects'][0]['bbox'][0] == 320.0
    with pytest.raises(ValueError, match='hash mismatch'):
        apply_source_review(raw, {**review, 'raw_output_sha256': 'wrong'})


def test_review_cannot_silently_modify_boxes_or_complete_truncated_output():
    import hashlib
    raw = record()
    review = {'raw_output_sha256': hashlib.sha256(raw['output'].encode()).hexdigest(),
              'reviewer': 'agent_image_source_check', 'human_reviewed': False,
              'bbox_fixes': {'o1': [0, 0, 1, 1]}}
    with pytest.raises(ValueError, match='provenance'):
        apply_source_review(raw, review)
    with pytest.raises(ValueError, match='incomplete'):
        apply_source_review({**raw, 'finish_reason': 'length'}, review)


def test_explicit_reviewed_1000_scale_is_not_guessed_for_mixed_boxes():
    import hashlib
    output = scene_output()
    for obj in output['objects']:
        obj['bbox'] = [x * 1000 for x in obj['bbox']]
    raw = record(output)
    review = {'raw_output_sha256': hashlib.sha256(raw['output'].encode()).hexdigest(),
              'reviewer': 'agent_image_source_check', 'human_reviewed': False,
              'reason': 'All boxes inspected and confirmed as the consistent 0..1000 convention.', 'bbox_scale': 1000}
    assert apply_source_review(raw, review).objects[0].bbox == scene_output()['objects'][0]['bbox']
    output['objects'][1]['bbox'] = [0.0, 0.3, 1.0, 1.0]
    raw = record(output); review['raw_output_sha256'] = hashlib.sha256(raw['output'].encode()).hexdigest()
    with pytest.raises(ValueError, match='ambiguous'):
        apply_source_review(raw, review)


def test_missing_modality_and_alias_need_explicit_hash_pinned_attestation():
    import hashlib
    output = scene_output()
    output['image_description'] = output.pop('scene_description')
    output.pop('image_only')
    raw = record(output)
    with pytest.raises(ValueError):
        parse_scene(raw)
    review = {'raw_output_sha256': hashlib.sha256(raw['output'].encode()).hexdigest(),
              'reviewer': 'agent_image_source_check', 'human_reviewed': False,
              'field_aliases': {'image_description': 'scene_description'}, 'attest_image_only_request': True}
    assert apply_source_review(raw, review).image_only is True
    output['image_only'] = False
    raw = record(output); review['raw_output_sha256'] = hashlib.sha256(raw['output'].encode()).hexdigest()
    with pytest.raises(ValueError, match='cannot override'):
        apply_source_review(raw, review)


def test_mixed_box_can_be_explicitly_omitted_without_inventing_coordinates():
    import hashlib
    output = scene_output()
    output['objects'][0]['bbox'] = [0.0, 450.0, 1.0, 1.0]
    raw = record(output)
    review = {'raw_output_sha256': hashlib.sha256(raw['output'].encode()).hexdigest(),
              'reviewer': 'agent_image_source_check', 'human_reviewed': False,
              'omit_box_object_ids': ['o1'], 'reason': 'Visible object retained with image-only, not region, grounding.'}
    scene = apply_source_review(raw, review)
    assert scene.objects[0].bbox is None and len(scene.relations) == 1
    assert json.loads(raw['output'])['objects'][0]['bbox'] == [0.0, 450.0, 1.0, 1.0]


def test_scene_endpoint_preserves_source_identity_limit_and_unknown_ids(tmp_path):
    from scripts.serve_evqa_api_graph import visual_scene_payload
    output = tmp_path / 'E-VQA'; visual_dir = output / 'mm_store/visual'; visual_dir.mkdir(parents=True)
    graph_dir = output / 'mm_store/graph'; graph_dir.mkdir()
    (visual_dir / 'scene_records.json').write_text(json.dumps({'img-one': scene_output()}))
    (tmp_path / 'visual_graph_audit.json').write_text(json.dumps({'images': [{'image_id': 'img-one',
        'anchor_node': 'image::img-one', 'scene_node': 'scene-one', 'visual_relation_nodes': ['relation-one']}]}))
    (output / 'owner.json').write_text(json.dumps({'visual_model': 'api/test'}))
    (graph_dir / 'image_anchor_lookup.json').write_text(json.dumps({'img-one': {'canonical_entity': 'PAMBAN BRIDGE'}}))
    payload = visual_scene_payload(output, 'img-one')
    assert payload['scene'] == scene_output()
    assert payload['article_association_is_visual_identity'] is False
    assert payload['human_reviewed'] is False and payload['all_facts_verified'] is False
    assert visual_scene_payload(output, '../../.env') is None


def test_scene_endpoint_is_unavailable_on_text_only_baseline(tmp_path):
    from scripts.serve_evqa_api_graph import visual_scene_payload
    assert visual_scene_payload(tmp_path, 'img-one') is None
