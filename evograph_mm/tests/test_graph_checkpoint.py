import json

import pytest

from evograph_mm.kb.graph_checkpoint import save_graph_checkpoint


def test_graph_checkpoint_copies_and_hashes_all_independent_graphs(tmp_path):
    root = tmp_path / 'graphs'
    for name in ('train_0', 'train_1', 'train_2', 'train_3', 'val_0', 'val_1'):
        graph = root / name
        graph.mkdir(parents=True)
        (graph / 'metadata.json').write_text('{}')
        (graph / 'kv_store_entities.json').write_text('{"entity": "' + name + '"}')
    target = tmp_path / 'checkpoint'
    result = save_graph_checkpoint(target, root)
    assert len(result) == 6
    assert json.loads((target / 'graph_state/manifest.json').read_text()) == result
    for name in result:
        source = root / name / 'kv_store_entities.json'
        snapshot = target / 'graph_state' / name / 'kv_store_entities.json'
        assert source.read_bytes() == snapshot.read_bytes()
        assert source.stat().st_ino != snapshot.stat().st_ino
    with pytest.raises(RuntimeError, match='checkpoint exists'):
        save_graph_checkpoint(target, root)
