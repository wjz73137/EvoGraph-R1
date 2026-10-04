import json

import numpy as np

from agent.tool.tools.hyperedge_index_sync import iter_searchable_hyperedge_contents
from agent.tool.tools.hyperedge_state_sync import build_hyperedge_lookup_payload
from evograph_mm.kb.graph_edit import (
    _bge_graph_corpora,
    apply_controlled_hyperedge_hide,
    capture_graph_text_index_state,
    create_edit_checkpoint,
    ensure_edit_working_dir,
    prepare_edit_working_dir,
    rebuild_graph_text_indexes,
    rebuild_root_text_indexes,
    restore_edit_checkpoint,
)
from evograph_mm.kb.retrieval import MMKBRetriever


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_second_generation_edit_copy_keeps_original_base_and_uses_own_output_dir(
    tmp_path,
):
    approved = tmp_path / "approved"
    approved.mkdir()
    _write_json(approved / "metadata.json", {"output_dir": str(approved)})

    controlled = prepare_edit_working_dir(approved, tmp_path / "controlled")
    isolated = prepare_edit_working_dir(controlled, tmp_path / "isolated")
    metadata = json.loads((isolated / "metadata.json").read_text(encoding="utf-8"))

    assert metadata["output_dir"] == str(isolated)
    assert metadata["base_output_dir"] == str(approved)
    assert metadata["graph_edit_copy"] is True
    assert ensure_edit_working_dir(approved, isolated) == isolated


def test_edit_checkpoint_restores_bge_graph_sidecars(tmp_path):
    graph_dir = tmp_path / "mm_store" / "bge_graph"
    graph_dir.mkdir(parents=True)
    paths = [
        graph_dir / "index_entity.bin",
        graph_dir / "corpus_entity.npy",
        graph_dir / "index_hyperedge.bin",
        graph_dir / "corpus_hyperedge.npy",
        graph_dir / "metadata.json",
    ]
    for index, path in enumerate(paths):
        path.write_bytes(f"before-{index}".encode())

    checkpoint = create_edit_checkpoint(tmp_path)
    for index, path in enumerate(paths):
        path.write_bytes(f"after-{index}".encode())
    restore_edit_checkpoint(tmp_path, checkpoint)

    assert [path.read_bytes() for path in paths] == [
        f"before-{index}".encode() for index in range(len(paths))
    ]


def test_controlled_hide_is_distinct_from_soft_delete_and_all_indexes_exclude_it(tmp_path):
    hyperedges = {
        "rel-visible": {"content": "Visible fact", "deleted": False},
        "rel-hidden": {"content": "Target fact", "deleted": False},
        "rel-soft-deleted": {"content": "Old fact", "deleted": True},
    }
    _write_json(tmp_path / "kv_store_hyperedges.json", hyperedges)
    _write_json(tmp_path / "kv_store_entities.json", {})

    original_vectors = np.asarray(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    np.save(tmp_path / "corpus_hyperedge.npy", original_vectors)
    _write_json(
        tmp_path / "hyperedge_index_metadata.json",
        {
            "ids": ["rel-visible", "rel-hidden", "rel-soft-deleted"],
            "contents": ["Visible fact", "Target fact", "Old fact"],
            "records": [{}, {}, {}],
        },
    )

    result = apply_controlled_hyperedge_hide(
        working_dir=tmp_path,
        target_id="rel-hidden",
        perturbation_id="missing-001",
    )
    updated = json.loads((tmp_path / "kv_store_hyperedges.json").read_text(encoding="utf-8"))

    assert result["hidden_id"] == "rel-hidden"
    assert updated["rel-hidden"]["searchable"] is False
    assert updated["rel-hidden"]["deleted"] is False
    assert updated["rel-hidden"]["controlled_perturbation"]["id"] == "missing-001"

    lookup = build_hyperedge_lookup_payload(updated)
    assert "Target fact" not in lookup["active"]
    assert "Target fact" not in lookup["searchable"]
    assert lookup["all"]["Target fact"] == "rel-hidden"
    assert list(iter_searchable_hyperedge_contents(updated)) == ["Visible fact", "Old fact"]
    assert _bge_graph_corpora(tmp_path)[2] == ["Visible fact"]

    rebuilt = rebuild_root_text_indexes(working_dir=tmp_path)
    metadata = json.loads(
        (tmp_path / "hyperedge_index_metadata.json").read_text(encoding="utf-8")
    )
    rebuilt_vectors = np.load(tmp_path / "corpus_hyperedge.npy")

    assert rebuilt["hyperedge"]["new_embeddings"] == 0
    assert metadata["ids"] == ["rel-visible", "rel-soft-deleted"]
    assert metadata["contents"] == ["Visible fact", "Old fact"]
    np.testing.assert_array_equal(rebuilt_vectors, original_vectors[[0, 2]])


def test_controlled_hide_prunes_bge_index_without_reencoding(tmp_path, monkeypatch):
    hyperedges = {
        "rel-visible": {"content": "Visible fact", "deleted": False},
        "rel-hidden": {
            "content": "Target fact",
            "deleted": False,
            "searchable": False,
        },
        "rel-deleted": {"content": "Old fact", "deleted": True},
    }
    _write_json(tmp_path / "kv_store_hyperedges.json", hyperedges)
    _write_json(tmp_path / "kv_store_entities.json", {})
    graph_dir = tmp_path / "mm_store" / "bge_graph"
    graph_dir.mkdir(parents=True)
    old_vectors = np.zeros((2, 1024), dtype=np.float32)
    old_vectors[0, 0] = 1.0
    old_vectors[1, 1] = 1.0
    np.save(graph_dir / "corpus_hyperedge.npy", old_vectors)

    def fail_if_encoded(*args, **kwargs):
        raise AssertionError("unchanged BGE vectors should be reused")

    monkeypatch.setattr(
        "agent.tool.tools.bge_model_manager.encode_texts_safe",
        fail_if_encoded,
    )
    previous = {
        key: {field: value for field, value in record.items() if field != "searchable"}
        for key, record in hyperedges.items()
    }
    result = rebuild_graph_text_indexes(
        working_dir=tmp_path,
        include_entities=False,
        include_hyperedges=True,
        previous_hyperedges=previous,
    )

    assert result["hyperedge"]["new_embeddings"] == 0
    np.testing.assert_array_equal(
        np.load(graph_dir / "corpus_hyperedge.npy"),
        old_vectors[[0]],
    )


def test_graph_index_rebuild_only_encodes_new_entity_and_hyperedge(tmp_path, monkeypatch):
    _write_json(
        tmp_path / "kv_store_entities.json",
        {"old": {"entity_name": "OLD", "content": "Old entity description"}},
    )
    _write_json(
        tmp_path / "kv_store_hyperedges.json",
        {"old": {"content": "Old fact", "deleted": False}},
    )
    graph_dir = tmp_path / "mm_store" / "bge_graph"
    graph_dir.mkdir(parents=True)
    old_entity_vector = np.full((1, 1024), 0.25, dtype=np.float32)
    old_hyperedge_vector = np.full((1, 1024), 0.5, dtype=np.float32)
    np.save(graph_dir / "corpus_entity.npy", old_entity_vector)
    np.save(graph_dir / "corpus_hyperedge.npy", old_hyperedge_vector)
    previous_state = capture_graph_text_index_state(tmp_path)

    entities = json.loads((tmp_path / "kv_store_entities.json").read_text())
    entities["new"] = {"entity_name": "NEW", "content": "New entity description"}
    _write_json(tmp_path / "kv_store_entities.json", entities)
    hyperedges = json.loads((tmp_path / "kv_store_hyperedges.json").read_text())
    hyperedges["new"] = {"content": "New fact", "deleted": False}
    _write_json(tmp_path / "kv_store_hyperedges.json", hyperedges)

    encoded = []

    def encode_only_new(texts, *, target_dimension):
        encoded.extend(texts)
        return np.ones((len(texts), target_dimension), dtype=np.float32)

    monkeypatch.setattr(
        "agent.tool.tools.bge_model_manager.encode_texts_safe",
        encode_only_new,
    )
    result = rebuild_graph_text_indexes(
        working_dir=tmp_path,
        previous_state=previous_state,
    )

    assert encoded == ["New entity description", "New fact"]
    assert result["entity"]["new_embeddings"] == 1
    assert result["hyperedge"]["new_embeddings"] == 1
    np.testing.assert_array_equal(
        np.load(graph_dir / "corpus_entity.npy")[0], old_entity_vector[0]
    )
    np.testing.assert_array_equal(
        np.load(graph_dir / "corpus_hyperedge.npy")[0], old_hyperedge_vector[0]
    )


def test_controlled_hide_filters_graphr1_context_leak(tmp_path):
    retriever = MMKBRetriever(working_dir=tmp_path, model_path=tmp_path)
    retriever.unsearchable_knowledge = {"target fact"}

    results = retriever._filter_unsearchable_results(
        [
            {"<knowledge>": '"Target fact"', "<coherence>": 1.0},
            {"<knowledge>": '"Visible fact"', "<coherence>": 0.9},
        ]
    )

    assert results == [{"<knowledge>": '"Visible fact"', "<coherence>": 0.9}]
