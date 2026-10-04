import copy

import networkx as nx
import numpy as np
import pytest

from scripts import clean_evqa_api_graph as clean


def fixture_graph():
    entity_ids = set(clean.QUARANTINE_ENTITIES) | set(clean.DESCRIPTIONS) | {'ent-3b1869f9c4e60352fb498eed3dcab29d'}
    edge_ids = set(clean.QUARANTINE_EDGES) | set(clean.MERGES) | {i for ids in clean.MERGES.values() for i in ids} | {'rel-f77c505e541366e9c989b7f62156847e', 'composite-retained'}
    entities, edges, graph, sidecar = {}, {}, nx.Graph(), {'entity': {}, 'hyperedge': {}}
    for record_id in sorted(entity_ids):
        name = '"' + record_id + '"'
        entities[record_id] = {'entity_name': name, 'content': name + 'old description',
                               'metadata': {'entity_type': '"PERSON"', 'ai_model': 'api-extractor', 'human_reviewed': False}}
        graph.add_node(name, role='entity', description='old description', entity_type='"PERSON"')
        sidecar['entity'][name] = {'source_chunk_ids': ['chunk-real']}
    for record_id in sorted(edge_ids):
        name = '<hyperedge>"' + record_id + '"'
        edges[record_id] = {'hyperedge_name': name, 'content': name, 'metadata': {'ai_model': 'api-extractor'}}
        graph.add_node(name, role='hyperedge', source_id='chunk-real', weight=1)
        graph.add_edge(name, entities[next(iter(clean.DESCRIPTIONS))]['entity_name'], source_id='chunk-real', weight=1)
        sidecar['hyperedge'][name] = {'source_chunk_ids': ['chunk-real'], 'wikipedia_urls': ['https://example.invalid/source']}
    return entities, edges, graph, sidecar


def test_quarantine_removes_kv_graph_and_source_sidecar_but_keeps_audit():
    fixture = fixture_graph()
    before = copy.deepcopy(fixture)
    entities, edges, graph, sidecar, audit = clean.clean_records(*fixture)
    for record_id in clean.QUARANTINE_ENTITIES:
        name = before[0][record_id]['entity_name']
        assert record_id not in entities and name not in graph and name not in sidecar['entity']
        assert audit['quarantined']['entity'][record_id]['record'] == before[0][record_id]
    for record_id in clean.QUARANTINE_EDGES:
        name = before[1][record_id]['hyperedge_name']
        assert record_id not in edges and name not in graph and name not in sidecar['hyperedge']
    assert fixture[0] == before[0] and fixture[1] == before[1]
    assert nx.utils.graphs_equal(fixture[2], before[2])


def test_explicit_merges_preserve_neighbors_provenance_and_do_not_merge_composites():
    entities, edges, graph, sidecar = fixture_graph()
    keep, aliases = next(iter(clean.MERGES.items()))
    canonical, alias = edges[keep]['hyperedge_name'], edges[aliases[0]]['hyperedge_name']
    extra_neighbor = entities['ent-3b1869f9c4e60352fb498eed3dcab29d']['entity_name']
    graph.add_edge(alias, extra_neighbor, source_id='chunk-extra', weight=4)
    sidecar['hyperedge'][alias]['source_chunk_ids'].append('chunk-extra')
    _, active, merged, mapping, audit = clean.clean_records(entities, edges, graph, sidecar)
    assert aliases[0] not in active and alias not in merged
    assert merged.has_edge(canonical, extra_neighbor)
    assert mapping['hyperedge'][canonical]['source_chunk_ids'] == ['chunk-extra', 'chunk-real']
    assert 'composite-retained' in active
    assert len(audit['merged']) == len(clean.MERGES)
    assert merged[canonical][entities[next(iter(clean.DESCRIPTIONS))]['entity_name']]['weight'] == 1


def test_builder_probability_disjunction_and_date_derivation_are_preserved():
    entities, _, graph, _, _ = clean.clean_records(*fixture_graph())
    for record_id in list(clean.DESCRIPTIONS)[:2]:
        record = entities[record_id]
        assert clean.BUILDER_DESCRIPTION in record['content']
        assert 'probably' in graph.nodes[record['entity_name']]['description']
        assert record['metadata']['human_reviewed'] is False
        assert record['metadata']['ai_model'] == 'api-extractor'
    date = entities['ent-2ad2e45bf8a5dfb89428ea1458d94ade']
    assert 'does not explicitly print 1571' in date['content']
    assert date['metadata']['claim_origin'] == 'derived_date'


def test_building_alias_type_normalization_does_not_rename_or_invent_a_person():
    entities, edges, graph, sidecar = fixture_graph()
    record_id, name = 'house-alias', '"JOHN B. JONES"'
    entities[record_id] = {'entity_name': name, 'content': name + 'alias for house',
                          'metadata': {'entity_type': '"PRODUCT"'}}
    graph.add_node(name, role='entity', entity_type='"PRODUCT"')
    sidecar['entity'][name] = {'source_chunk_ids': ['chunk-real']}
    result, _, active, _, audit = clean.clean_records(entities, edges, graph, sidecar)
    assert result[record_id]['metadata']['alias_of'] == 'CHARLES WILLIAM JONES HOUSE'
    assert result[record_id]['metadata']['entity_type'] == '"LOCATION"'
    assert result[record_id]['content'] == entities[record_id]['content']
    assert active.nodes[name]['entity_type'] == '"LOCATION"'
    assert audit['modified_entities'][record_id]['before'] == entities[record_id]


def test_vectors_reuse_only_same_id_and_same_content_in_new_order():
    matrix = np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    metadata = {'ids': ['a', 'b', 'c'], 'contents': ['A', 'B', 'C']}
    vectors, changed = clean.select_vectors(metadata, matrix, [
        {'id': 'c', 'content': 'C'}, {'id': 'a', 'content': 'A fixed'}])
    np.testing.assert_array_equal(vectors[0], matrix[2])
    assert changed == [1]  # This row MUST be replaced by a real re-encoded vector.


def test_vector_alignment_and_unknown_ids_fail_closed():
    metadata = {'ids': ['a'], 'contents': ['A']}
    with pytest.raises(ValueError, match='alignment'):
        clean.select_vectors(metadata, np.ones((2, 3)), [])
    with pytest.raises(ValueError, match='new embedding id'):
        clean.select_vectors(metadata, np.ones((1, 3)), [{'id': 'b', 'content': 'B'}])
    with pytest.raises(ValueError, match='duplicate'):
        clean.select_vectors({'ids': ['a', 'a'], 'contents': ['A', 'A']}, np.ones((2, 3)), [])


def test_cleanup_plan_refuses_missing_records():
    fixture = fixture_graph()
    fixture[1].pop(next(iter(clean.QUARANTINE_EDGES)))
    with pytest.raises(RuntimeError, match='does not match'):
        clean.clean_records(*fixture)


def test_index_loaders_use_actual_kv_filenames(tmp_path):
    import json
    from evograph_mm.kb.build import _load_entity_records, _load_hyperedge_records
    entities, edges, _, _ = fixture_graph()
    for kind, records, loader in [('entity', entities, _load_entity_records), ('hyperedge', edges, _load_hyperedge_records)]:
        path = tmp_path / clean.KV_NAMES[kind]
        path.write_text(json.dumps(records))
        assert len(loader(path)) == len(records)
    assert clean.KV_NAMES['entity'] == 'kv_store_entities.json'
