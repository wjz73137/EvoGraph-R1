"""GraphR1-native edit helpers for multimodal KB working directories."""

from __future__ import annotations

import json
import shutil
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from agent.tool.tools.batch.unified_batch_queue import find_hyperedge_by_content
from agent.tool.tools.hyperedge_state_sync import (
    append_recent_hyperedge_mutations,
    ensure_hyperedge_state_sidecars,
    load_hyperedge_lookup,
)
from agent.tool.tools.hyperedge_sync import (
    get_graph_weight,
    get_hyperedge_name,
    persist_hyperedge_graph,
    prepare_delete_record,
    prepare_update_record,
    run_async,
    sync_hyperedge_graph,
)
from graphr1 import GraphR1


DEFAULT_BGE_TEXT_DIMENSION = 1024

ROOT_KB_FILES = (
    "metadata.json",
    "build_report.json",
    "kv_store_entities.json",
    "kv_store_hyperedges.json",
    "kv_store_full_docs.json",
    "kv_store_text_chunks.json",
    "kv_store_chunks.json",
    "kv_store_llm_response_cache.json",
    "graph_chunk_entity_relation.graphml",
    "corpus_entity.npy",
    "corpus_hyperedge.npy",
    "index_entity.bin",
    "index_hyperedge.bin",
    "entity_index_metadata.json",
    "hyperedge_index_metadata.json",
    "hyperedge_content_lookup.json",
    "hyperedge_recent_mutations.json",
)
ROOT_KB_DIRS = ("graphr1_text", "mm_store")
MUTABLE_FILES = (
    "kv_store_entities.json",
    "kv_store_hyperedges.json",
    "kv_store_full_docs.json",
    "kv_store_text_chunks.json",
    "kv_store_chunks.json",
    "kv_store_llm_response_cache.json",
    "vdb_entities.json",
    "vdb_hyperedges.json",
    "vdb_chunks.json",
    "graph_chunk_entity_relation.graphml",
    "corpus_entity.npy",
    "corpus_hyperedge.npy",
    "index_entity.bin",
    "index_hyperedge.bin",
    "entity_index_metadata.json",
    "hyperedge_index_metadata.json",
    "hyperedge_content_lookup.json",
    "hyperedge_recent_mutations.json",
    "mm_store/graph/graphr1_hit_source_sidecar.json",
    "mm_store/bge_graph/index_entity.bin",
    "mm_store/bge_graph/corpus_entity.npy",
    "mm_store/bge_graph/index_hyperedge.bin",
    "mm_store/bge_graph/corpus_hyperedge.npy",
    "mm_store/bge_graph/metadata.json",
)


def prepare_edit_working_dir(
    base_working_dir: str | Path,
    target_working_dir: str | Path | None = None,
) -> Path:
    """Copy an MM KB once into a separate GraphR1-native edit working dir."""
    base_path = Path(base_working_dir)
    if not base_path.is_dir():
        raise FileNotFoundError(f"missing base MM KB working directory: {base_path}")
    target = Path(target_working_dir) if target_working_dir is not None else _new_edit_dir(base_path)
    if _same_path(base_path, target):
        raise ValueError("edit working_dir must be different from the base MM KB")
    target.mkdir(parents=True, exist_ok=False)
    try:
        _copy_kb_artifacts(base_path, target)
        _rewrite_metadata_for_edit_copy(target, base_path)
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return target


def ensure_edit_working_dir(base_working_dir: str | Path, active_working_dir: str | Path) -> Path:
    """Return the active edit dir only when it was prepared before editing."""
    base_path = Path(base_working_dir)
    active_path = Path(active_working_dir)
    if not active_path.is_dir():
        raise FileNotFoundError(f"missing active MM KB working directory: {active_path}")
    if _same_path(base_path, active_path):
        raise ValueError(
            "MM graph edits require a prepared edit working_dir; copy the KB "
            "and reload or start the API with that working_dir before editing"
        )
    return active_path


def create_edit_checkpoint(working_dir: str | Path) -> dict[str, bytes | None]:
    working_path = Path(working_dir)
    checkpoint: dict[str, bytes | None] = {}
    for relative in MUTABLE_FILES:
        path = working_path / relative
        checkpoint[relative] = path.read_bytes() if path.is_file() else None
    return checkpoint


def restore_edit_checkpoint(working_dir: str | Path, checkpoint: dict[str, bytes | None]) -> None:
    working_path = Path(working_dir)
    for relative, payload in checkpoint.items():
        path = working_path / relative
        if payload is None:
            if path.is_file():
                path.unlink()
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def apply_graph_insert(
    *,
    working_dir: str | Path,
    contents: list[str],
    image_id: str | None = None,
    image_path: str | None = None,
    data_id: str | None = None,
    source_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Insert through GraphR1.insert so entity extraction and graph wiring are native."""
    working_path = Path(working_dir)
    before_entities = _read_json_object(working_path / "kv_store_entities.json")
    before_hyperedges = _read_json_object(working_path / "kv_store_hyperedges.json")
    rag = GraphR1(working_dir=str(working_path))
    rag.insert(contents)
    after_entities = _read_json_object(working_path / "kv_store_entities.json")
    after_hyperedges = _read_json_object(working_path / "kv_store_hyperedges.json")
    new_entity_ids = [key for key in after_entities if key not in before_entities]
    new_hyperedge_ids = [key for key in after_hyperedges if key not in before_hyperedges]
    if new_hyperedge_ids:
        for hyperedge_id in new_hyperedge_ids:
            _merge_mm_provenance(
                after_hyperedges[hyperedge_id],
                image_id=image_id,
                image_path=image_path,
                data_id=data_id,
                source_metadata=source_metadata,
            )
        _write_json_object(working_path / "kv_store_hyperedges.json", after_hyperedges)
        ensure_hyperedge_state_sidecars(str(working_path), after_hyperedges)
    append_recent_hyperedge_mutations(
        str(working_path),
        [
            {
                "hyperedge_id": "",
                "action": "insert",
                "content": content,
                "active": True,
                "searchable": True,
            }
            for content in contents
        ],
    )
    return {
        "ids": new_hyperedge_ids,
        "entity_ids": new_entity_ids,
        "entities_added": len(new_entity_ids),
        "hyperedges_added": len(new_hyperedge_ids),
        "native_graph_insert": True,
    }


def apply_hyperedge_update(
    *,
    working_dir: str | Path,
    content: str,
    new_content: str,
    target_id: str | None = None,
    image_id: str | None = None,
    image_path: str | None = None,
    data_id: str | None = None,
    source_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    working_path = Path(working_dir)
    hyperedges = _read_json_object(working_path / "kv_store_hyperedges.json")
    hyperedge_id = _resolve_hyperedge_id(
        working_path,
        hyperedges,
        content=content,
        target_id=target_id,
        skip_deleted=False,
    )
    if hyperedge_id is None:
        raise ValueError(f"No knowledge item found with content containing: '{content}'")
    hyperedge = deepcopy(hyperedges[hyperedge_id])
    was_deleted = bool(hyperedge.get("deleted", False))
    prepared = prepare_update_record(
        hyperedge,
        new_content=new_content,
        global_config={},
        tool_name="mm_graph_edit",
    )
    updated_hyperedge = prepared["hyperedge"]
    _merge_mm_provenance(
        updated_hyperedge,
        image_id=image_id,
        image_path=image_path,
        data_id=data_id,
        source_metadata=source_metadata,
    )
    hyperedges[hyperedge_id] = updated_hyperedge
    incident_edge_weight = updated_hyperedge.get("weight") if was_deleted else None
    rag = _sync_hyperedge_if_graph_exists(
        working_path,
        old_hyperedge_name=prepared["old_hyperedge_name"],
        new_hyperedge_name=prepared["new_hyperedge_name"],
        hyperedge=updated_hyperedge,
        incident_edge_weight=incident_edge_weight,
    )
    _persist_hyperedge_changes(
        working_path,
        rag,
        hyperedges,
        [
            {
                "hyperedge_id": hyperedge_id,
                "action": "update",
                "content": prepared["new_hyperedge_name"],
                "active": True,
                "searchable": True,
            }
        ],
    )
    return {"ids": [hyperedge_id], "updated_id": hyperedge_id}


def apply_hyperedge_soft_delete(
    *,
    working_dir: str | Path,
    content: str,
    target_id: str | None = None,
) -> dict[str, Any]:
    working_path = Path(working_dir)
    hyperedges = _read_json_object(working_path / "kv_store_hyperedges.json")
    hyperedge_id = _resolve_hyperedge_id(
        working_path,
        hyperedges,
        content=content,
        target_id=target_id,
        skip_deleted=True,
    )
    if hyperedge_id is None:
        raise ValueError(f"No active knowledge item found with content containing: '{content}'")
    hyperedge = deepcopy(hyperedges[hyperedge_id])
    hyperedge_name = get_hyperedge_name(hyperedge)
    rag = GraphR1(working_dir=str(working_path)) if _has_graph(working_path) else None
    current_weight = (
        run_async(get_graph_weight(rag, hyperedge_name, hyperedge.get("weight", 1.0)))
        if rag is not None
        else hyperedge.get("weight", 1.0)
    )
    prepared = prepare_delete_record(
        hyperedge,
        current_weight=current_weight,
        global_config={},
        tool_name="mm_graph_edit",
    )
    hyperedges[hyperedge_id] = prepared["hyperedge"]
    rag = _sync_hyperedge_if_graph_exists(
        working_path,
        old_hyperedge_name=prepared["hyperedge_name"],
        new_hyperedge_name=prepared["hyperedge_name"],
        hyperedge=prepared["hyperedge"],
        incident_edge_weight=prepared["demoted_weight"],
    )
    _persist_hyperedge_changes(
        working_path,
        rag,
        hyperedges,
        [
            {
                "hyperedge_id": hyperedge_id,
                "action": "delete",
                "content": prepared["hyperedge_name"],
                "active": False,
                "searchable": True,
            }
        ],
    )
    return {"ids": [hyperedge_id], "deleted_id": hyperedge_id}


def apply_controlled_hyperedge_hide(
    *,
    working_dir: str | Path,
    target_id: str,
    perturbation_id: str,
    reason: str = "controlled_missing_fact",
) -> dict[str, Any]:
    """Hide one fact from retrieval in an isolated experimental KB copy.

    This is deliberately separate from the paper's soft-delete operation.  It
    leaves the graph record intact for provenance and restoration, but excludes
    it from every text retrieval index through an explicit ``searchable=False``
    marker.
    """
    working_path = Path(working_dir)
    hyperedges = _read_json_object(working_path / "kv_store_hyperedges.json")
    hyperedge = hyperedges.get(target_id)
    if not isinstance(hyperedge, dict):
        raise ValueError(f"Unknown controlled perturbation target: {target_id}")
    if hyperedge.get("searchable", True) is False:
        raise ValueError(f"Hyperedge is already hidden from retrieval: {target_id}")

    updated = deepcopy(hyperedge)
    content = str(updated.get("content") or updated.get("hyperedge_name") or "").strip()
    if not content:
        raise ValueError(f"Hyperedge has no searchable content: {target_id}")
    updated["searchable"] = False
    updated["controlled_perturbation"] = {
        "id": perturbation_id,
        "type": reason,
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "original_searchable": bool(hyperedge.get("searchable", True)),
    }
    hyperedges[target_id] = updated
    _persist_hyperedge_changes(
        working_path,
        None,
        hyperedges,
        [
            {
                "hyperedge_id": target_id,
                "action": "controlled_hide",
                "content": content,
                "active": False,
                "searchable": False,
            }
        ],
    )
    return {
        "ids": [target_id],
        "hidden_id": target_id,
        "content": content,
        "perturbation_id": perturbation_id,
    }


def rebuild_root_text_indexes(
    *,
    working_dir: str | Path,
    include_entities: bool = False,
    include_hyperedges: bool = True,
    model_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Rebuild root FAISS sidecars while reusing every unchanged embedding."""
    working_path = Path(working_dir)
    results: dict[str, dict[str, Any]] = {}
    resolved_model_path = Path(model_path) if model_path is not None else None
    if include_entities:
        results["entity"] = _rebuild_incremental_root_index(
            output_dir=working_path,
            namespace="entity",
            records=_load_entity_records(working_path / "kv_store_entities.json"),
            encoder=None,
            corpus_file="corpus_entity.npy",
            index_file="index_entity.bin",
            metadata_file="entity_index_metadata.json",
            model_path=resolved_model_path,
        )
    if include_hyperedges:
        results["hyperedge"] = _rebuild_incremental_root_index(
            output_dir=working_path,
            namespace="hyperedge",
            records=_load_searchable_hyperedge_records(
                working_path / "kv_store_hyperedges.json"
            ),
            encoder=None,
            corpus_file="corpus_hyperedge.npy",
            index_file="index_hyperedge.bin",
            metadata_file="hyperedge_index_metadata.json",
            model_path=resolved_model_path,
        )
    return results


def rebuild_graph_text_indexes(
    *,
    working_dir: str | Path,
    encoder: Any | None = None,
    model_path: str | Path | None = None,
    include_entities: bool = True,
    include_hyperedges: bool = True,
    previous_hyperedges: dict[str, Any] | None = None,
    previous_state: dict[str, list[str]] | None = None,
) -> dict[str, dict[str, Any]]:
    del encoder, model_path
    working_path = Path(working_dir)
    graph_dir = working_path / "mm_store" / "bge_graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    entity_corpus, entity_embedding_texts, hyperedge_corpus = _bge_graph_corpora(working_path)
    metadata = _read_bge_graph_metadata(graph_dir / "metadata.json")
    results: dict[str, dict[str, Any]] = {}
    if include_entities:
        previous_entity_texts = (
            previous_state.get("entity_embedding_texts")
            if previous_state is not None
            else None
        )
        results["entity"] = _write_bge_graph_index(
            graph_dir=graph_dir,
            namespace="entity",
            texts=entity_embedding_texts,
            corpus=entity_corpus,
            previous_texts=previous_entity_texts,
        )
        metadata["entity_count"] = len(entity_corpus)
    else:
        metadata.setdefault("entity_count", len(entity_corpus))
    if include_hyperedges:
        previous_hyperedge_corpus = (
            previous_state.get("hyperedge_corpus")
            if previous_state is not None
            else (
                _active_hyperedge_contents(previous_hyperedges)
                if previous_hyperedges is not None
                else None
            )
        )
        results["hyperedge"] = _write_bge_graph_index(
            graph_dir=graph_dir,
            namespace="hyperedge",
            texts=hyperedge_corpus,
            corpus=hyperedge_corpus,
            previous_texts=previous_hyperedge_corpus,
        )
        metadata["hyperedge_count"] = len(hyperedge_corpus)
    else:
        metadata.setdefault("hyperedge_count", len(hyperedge_corpus))
    metadata.update(
        {
            "provider": "bge",
            "dimension": DEFAULT_BGE_TEXT_DIMENSION,
            "model": _bge_model_label(),
            "entity_source": "kv_store_entities.json[*].content",
            "hyperedge_source": "active kv_store_hyperedges.json[*].content",
            "layout": "2wiki_compatible_sidecar",
            "encoder_mode": "bge",
        }
    )
    _write_json_object(graph_dir / "metadata.json", metadata)
    return results


def capture_graph_text_index_state(
    working_dir: str | Path,
) -> dict[str, list[str]]:
    """Capture the corpus order needed to reuse BGE vectors after an edit."""
    entity_corpus, entity_embedding_texts, hyperedge_corpus = _bge_graph_corpora(
        Path(working_dir)
    )
    return {
        "entity_corpus": entity_corpus,
        "entity_embedding_texts": entity_embedding_texts,
        "hyperedge_corpus": hyperedge_corpus,
    }


def _bge_graph_corpora(working_path: Path) -> tuple[list[str], list[str], list[str]]:
    entities = _read_json_object(working_path / "kv_store_entities.json")
    hyperedges = _read_json_object(working_path / "kv_store_hyperedges.json")
    entity_corpus = []
    entity_embedding_texts = []
    if isinstance(entities, dict):
        for entity in entities.values():
            if not isinstance(entity, dict):
                continue
            entity_name = entity.get("entity_name")
            content = entity.get("content")
            if isinstance(entity_name, str) and entity_name.strip():
                entity_corpus.append(entity_name)
                entity_embedding_texts.append(
                    content if isinstance(content, str) and content.strip() else entity_name
                )
    hyperedge_corpus = _active_hyperedge_contents(hyperedges)
    return entity_corpus, entity_embedding_texts, hyperedge_corpus


def _active_hyperedge_contents(hyperedges: dict[str, Any]) -> list[str]:
    contents = []
    if not isinstance(hyperedges, dict):
        return contents
    for hyperedge in hyperedges.values():
        if (
            not isinstance(hyperedge, dict)
            or hyperedge.get("deleted", False)
            or hyperedge.get("searchable", True) is False
        ):
            continue
        content = hyperedge.get("content") or hyperedge.get("hyperedge_name")
        if isinstance(content, str) and content.strip():
            contents.append(content)
    return contents


def _write_bge_graph_index(
    *,
    graph_dir: Path,
    namespace: str,
    texts: list[str],
    corpus: list[str],
    previous_texts: list[str] | None = None,
) -> dict[str, Any]:
    import faiss
    from agent.tool.tools.bge_model_manager import encode_texts_safe

    old_corpus_path = graph_dir / f"corpus_{namespace}.npy"
    reusable: dict[str, list[np.ndarray]] = {}
    if previous_texts is not None and old_corpus_path.is_file():
        old_vectors = np.load(old_corpus_path)
        if old_vectors.ndim == 2 and old_vectors.shape[0] == len(previous_texts):
            for text, vector in zip(previous_texts, old_vectors):
                reusable.setdefault(text, []).append(np.asarray(vector, dtype=np.float32))

    rows: list[np.ndarray | None] = []
    missing_texts = []
    for text in texts:
        candidates = reusable.get(text)
        if candidates:
            rows.append(candidates.pop(0))
        else:
            rows.append(None)
            missing_texts.append(text)
    new_vectors = _encode_bge_texts(missing_texts, encode_texts_safe)
    new_index = 0
    for row_index, row in enumerate(rows):
        if row is None:
            rows[row_index] = new_vectors[new_index]
            new_index += 1
    vectors = (
        np.ascontiguousarray(np.vstack(rows), dtype=np.float32)
        if rows
        else np.zeros((0, DEFAULT_BGE_TEXT_DIMENSION), dtype=np.float32)
    )
    index = faiss.index_factory(
        DEFAULT_BGE_TEXT_DIMENSION,
        "Flat",
        faiss.METRIC_INNER_PRODUCT,
    )
    if vectors.shape[0] > 0:
        index.add(vectors)
    index_path = graph_dir / f"index_{namespace}.bin"
    corpus_path = old_corpus_path
    faiss.write_index(index, str(index_path))
    np.save(corpus_path, vectors)
    return {
        "namespace": namespace,
        "provider": "bge",
        "vector_count": int(vectors.shape[0]),
        "embedding_dimension": DEFAULT_BGE_TEXT_DIMENSION,
        "index_path": str(index_path),
        "corpus_path": str(corpus_path),
        "new_embeddings": len(missing_texts),
    }


def _encode_bge_texts(texts: list[str], encode_texts_safe: Any) -> np.ndarray:
    if not texts:
        return np.zeros((0, DEFAULT_BGE_TEXT_DIMENSION), dtype=np.float32)
    batch_size = 4
    vectors = []
    for start in range(0, len(texts), batch_size):
        vectors.append(
            encode_texts_safe(
                texts[start : start + batch_size],
                target_dimension=DEFAULT_BGE_TEXT_DIMENSION,
            )
        )
    return np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)


def _read_bge_graph_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = _read_json_object(path)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _bge_model_label() -> str:
    import os

    return os.getenv("BGE_MODEL_PATH") or os.getenv("BGE_MODEL_NAME") or "BAAI/bge-large-en-v1.5"


def _persist_hyperedge_changes(
    working_path: Path,
    rag: Any,
    hyperedges: dict[str, Any],
    events: list[dict[str, Any]],
) -> None:
    _write_json_object(working_path / "kv_store_hyperedges.json", hyperedges)
    ensure_hyperedge_state_sidecars(str(working_path), hyperedges)
    append_recent_hyperedge_mutations(str(working_path), events)
    if rag is not None:
        run_async(persist_hyperedge_graph(rag))


def _sync_hyperedge_if_graph_exists(
    working_path: Path,
    *,
    old_hyperedge_name: str,
    new_hyperedge_name: str,
    hyperedge: dict[str, Any],
    incident_edge_weight: float | None = None,
) -> Any:
    if not _has_graph(working_path):
        return None
    rag = GraphR1(working_dir=str(working_path))
    run_async(
        sync_hyperedge_graph(
            rag,
            old_hyperedge_name=old_hyperedge_name,
            new_hyperedge_name=new_hyperedge_name,
            hyperedge=hyperedge,
            incident_edge_weight=incident_edge_weight,
        )
    )
    return rag


def _has_graph(working_path: Path) -> bool:
    return (working_path / "graph_chunk_entity_relation.graphml").is_file()


def _resolve_hyperedge_id(
    working_path: Path,
    hyperedges: dict[str, Any],
    *,
    content: str,
    target_id: str | None,
    skip_deleted: bool,
) -> str | None:
    if target_id and target_id in hyperedges:
        hyperedge = hyperedges[target_id]
        if isinstance(hyperedge, dict) and (not skip_deleted or not hyperedge.get("deleted", False)):
            return target_id
    lookup = load_hyperedge_lookup(str(working_path), skip_deleted=skip_deleted)
    return find_hyperedge_by_content(
        hyperedges,
        content,
        skip_deleted=skip_deleted,
        lookup=lookup,
    )


def _load_entity_records(path: Path) -> list[dict[str, Any]]:
    data = _read_json_object(path)
    records = []
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        if value.get("searchable", True) is False:
            continue
        entity_id = str(value.get("entity_name") or key)
        content = str(value.get("content") or entity_id).strip()
        if content:
            records.append({"id": entity_id, "content": content})
    if not records:
        raise ValueError(f"no entity records found in {path}")
    return records


def _load_searchable_hyperedge_records(path: Path) -> list[dict[str, Any]]:
    data = _read_json_object(path)
    records = []
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        if value.get("searchable", True) is False:
            continue
        content = _hyperedge_embedding_content(
            value.get("content") or value.get("hyperedge_name") or ""
        )
        if not content:
            continue
        deleted = bool(value.get("deleted", False))
        record: dict[str, Any] = {
            "id": str(value.get("hyperedge_name") or key),
            "content": content,
            "deleted": deleted,
            "active": not deleted,
            "weight": _coerce_float(value.get("weight"), 1.0),
        }
        if "original_weight" in value:
            record["original_weight"] = _coerce_float(value.get("original_weight"), 1.0)
        deleted_at = value.get("deleted_at")
        if isinstance(deleted_at, str) and deleted_at:
            record["deleted_at"] = deleted_at
        for field in ("image_id", "image_path", "data_id"):
            if isinstance(value.get(field), str) and value.get(field):
                record[field] = value[field]
        source_metadata = value.get("source_metadata")
        if isinstance(source_metadata, dict):
            record["source_metadata"] = dict(source_metadata)
        records.append(record)
    if not records:
        raise ValueError(f"no searchable hyperedge records found in {path}")
    return records


def _rebuild_incremental_root_index(
    *,
    output_dir: Path,
    namespace: str,
    records: list[dict[str, Any]],
    encoder: Any,
    corpus_file: str,
    index_file: str,
    metadata_file: str,
    model_path: Path | None,
) -> dict[str, Any]:
    del encoder
    ids = [str(record["id"]) for record in records]
    contents = [str(record["content"]) for record in records]
    metadata_path = output_dir / metadata_file
    corpus_path = output_dir / corpus_file
    old_vectors: dict[tuple[str, str], list[np.ndarray]] = {}
    old_dimension = None
    if metadata_path.is_file() and corpus_path.is_file():
        try:
            old_metadata = _read_json_object(metadata_path)
            old_ids = old_metadata.get("ids", [])
            old_contents = old_metadata.get("contents", [])
            old_corpus = np.load(corpus_path)
            if (
                isinstance(old_ids, list)
                and isinstance(old_contents, list)
                and old_corpus.ndim == 2
                and len(old_ids) == len(old_contents) == old_corpus.shape[0]
            ):
                old_dimension = int(old_corpus.shape[1])
                for row_id, row_content, vector in zip(old_ids, old_contents, old_corpus):
                    old_vectors.setdefault((str(row_id), str(row_content)), []).append(
                        np.asarray(vector, dtype=np.float32)
                    )
        except Exception:
            old_vectors = {}
            old_dimension = None

    rows: list[np.ndarray | None] = []
    missing_positions = []
    missing_contents = []
    for index, (row_id, content) in enumerate(zip(ids, contents)):
        reusable = old_vectors.get((row_id, content))
        if reusable:
            rows.append(reusable.pop(0))
            continue
        rows.append(None)
        missing_positions.append(index)
        missing_contents.append(content)

    if missing_contents:
        from agent.tool.tools.bge_model_manager import encode_texts_safe

        encoded = np.asarray(
            encode_texts_safe(
                missing_contents,
                target_dimension=DEFAULT_BGE_TEXT_DIMENSION,
            ),
            dtype=np.float32,
        )
        if encoded.ndim != 2 or encoded.shape[0] != len(missing_contents):
            raise ValueError(f"{namespace} encoder returned invalid shape")
        if old_dimension is not None and int(encoded.shape[1]) != old_dimension:
            raise ValueError(
                f"{namespace} embedding dimension changed: old={old_dimension}, new={encoded.shape[1]}"
            )
        for encoded_index, position in enumerate(missing_positions):
            rows[position] = encoded[encoded_index]

    if any(row is None for row in rows):
        raise ValueError(f"{namespace} index rebuild has missing vectors")
    embeddings = np.ascontiguousarray(np.vstack(rows), dtype=np.float32)
    return _write_root_index(
        output_dir=output_dir,
        namespace=namespace,
        ids=ids,
        contents=contents,
        records=records,
        embeddings=embeddings,
        corpus_file=corpus_file,
        index_file=index_file,
        metadata_file=metadata_file,
        model_path=model_path,
        new_embeddings=len(missing_contents),
    )


def _write_root_index(
    *,
    output_dir: Path,
    namespace: str,
    ids: list[str],
    contents: list[str],
    records: list[dict[str, Any]],
    embeddings: np.ndarray,
    corpus_file: str,
    index_file: str,
    metadata_file: str,
    model_path: Path,
    new_embeddings: int,
) -> dict[str, Any]:
    import faiss

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError(f"{namespace} embeddings must be a non-empty 2D array")
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / corpus_file
    index_path = output_dir / index_file
    metadata_path = output_dir / metadata_file
    np.save(corpus_path, embeddings)
    index = faiss.index_factory(embeddings.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
    index.add(embeddings)
    faiss.write_index(index, str(index_path))
    metadata = {
        "namespace": namespace,
        "vector_count": int(embeddings.shape[0]),
        "embedding_dimension": int(embeddings.shape[1]),
        "index_type": "IndexFlatIP",
        "metric": "inner_product",
        "corpus_file": corpus_path.name,
        "index_file": index_path.name,
        "model_path": str(model_path) if model_path is not None else _bge_model_label(),
        "model_repo_id": _bge_model_label(),
        "encoder_mode": "bge",
        "ids": ids,
        "contents": contents,
        "records": records,
    }
    _write_json_object(metadata_path, metadata)
    return {
        "namespace": namespace,
        "vector_count": int(embeddings.shape[0]),
        "embedding_dimension": int(embeddings.shape[1]),
        "index_path": str(index_path),
        "corpus_path": str(corpus_path),
        "metadata_path": str(metadata_path),
        "new_embeddings": int(new_embeddings),
    }


def _merge_mm_provenance(
    hyperedge: dict[str, Any],
    *,
    image_id: str | None,
    image_path: str | None,
    data_id: str | None,
    source_metadata: dict[str, Any] | None,
) -> None:
    if image_id:
        hyperedge["image_id"] = image_id
    if image_path:
        hyperedge["image_path"] = image_path
    if data_id:
        hyperedge["data_id"] = data_id
    if source_metadata:
        merged = dict(hyperedge.get("source_metadata") or {})
        merged.update(source_metadata)
        hyperedge["source_metadata"] = merged


def _copy_kb_artifacts(base_path: Path, target: Path) -> None:
    for name in ROOT_KB_FILES:
        source = base_path / name
        if source.is_file():
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    for name in ROOT_KB_DIRS:
        source = base_path / name
        if source.is_dir():
            shutil.copytree(source, target / name, ignore=_copy_ignore)


def _copy_ignore(directory: str, names: list[str]) -> set[str]:
    del directory
    return {name for name in names if name == "__pycache__" or name.endswith(".pyc")}


def _rewrite_metadata_for_edit_copy(target: Path, base_path: Path) -> None:
    metadata_path = target / "metadata.json"
    metadata = _read_json_object(metadata_path) if metadata_path.is_file() else {}
    original_base = base_path
    inherited_base = metadata.get("base_output_dir")
    if metadata.get("graph_edit_copy") is True and isinstance(inherited_base, str):
        inherited_base = inherited_base.strip()
        if inherited_base:
            original_base = Path(inherited_base)
    metadata["output_dir"] = str(target)
    metadata["base_output_dir"] = str(original_base)
    metadata["graph_edit_copy"] = True
    metadata["graph_edit_copy_created_at"] = datetime.now(timezone.utc).isoformat()
    _write_json_object(metadata_path, metadata)


def _new_edit_dir(base_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_path.parent / f"{base_path.name}__graph_edit_{timestamp}_{uuid.uuid4().hex[:8]}"


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except FileNotFoundError:
        return left.absolute() == right.absolute()


def _hyperedge_embedding_content(value: Any) -> str:
    content = str(value or "").strip()
    if content.startswith("<hyperedge>"):
        content = content[len("<hyperedge>") :].strip()
    return content.strip().strip('"').strip()


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _write_json_object(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


__all__ = [
    "apply_graph_insert",
    "apply_hyperedge_soft_delete",
    "apply_hyperedge_update",
    "create_edit_checkpoint",
    "ensure_edit_working_dir",
    "prepare_edit_working_dir",
    "capture_graph_text_index_state",
    "rebuild_graph_text_indexes",
    "restore_edit_checkpoint",
]
