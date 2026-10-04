#!/usr/bin/env python3
"""Build the complete 1,891-row E-VQA strict-Max knowledge hypergraph."""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evograph_mm.kb.gldv2_subset import atomic_json, image_info


DATA_ROOT = Path("/home/data/dataset/wjz/EvoGraph-R1")
SUBSET = "E-VQA-GLDv2-1898-61-seed0"
SUBSET_ROOT = DATA_ROOT / f"datasets_mm/E-VQA/subsets/{SUBSET}"
STRICT_ROOT = DATA_ROOT / "expr_mm/evqa_api_strict_max_extraction_full1891_v1"
GRAPH_ROOT = DATA_ROOT / "expr_mm/evqa_api_strict_max_graph_full1891_v1"
OUTPUT = GRAPH_ROOT / "E-VQA"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_graphml_atomic(graph, target: Path) -> None:
    import networkx as nx
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".graphml", dir=target.parent)
    os.close(fd)
    temp = Path(name)
    try:
        nx.write_graphml(graph, temp)
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)


def build_bundle(source_manifest: dict, documents_by_id: dict):
    from evograph_mm.kb.store import RecordBundle, text_embedding_id, visual_embedding_id

    subset_manifest = json.loads((SUBSET_ROOT / "manifest.json").read_text())
    image_manifest = {item["image_id"]: item for item in subset_manifest["copied_images"]}
    text_documents, visual_records, links = [], [], []
    seen_images = set()
    for association in source_manifest["row_associations"]:
        row_position = int(association["row_position"])
        document = documents_by_id[association["document_id"]]
        image_id = association["image_id"]
        image_record = image_manifest.get(image_id)
        if not image_record:
            raise RuntimeError(f"subset manifest lacks image {image_id}")
        image_path = DATA_ROOT / image_record["subset_path"]
        if image_id not in seen_images:
            if not image_info(image_path) or sha256(image_path) != image_record["sha256"]:
                raise RuntimeError(f"image decoding/hash check failed: {image_id}")
        data_id = f"evqa_train_row::{row_position:06d}"
        text_doc_id = f"text::{data_id}"
        visual_id = f"visual::{image_id}"
        metadata = {
            "wikipedia_title": document["title"],
            "wikipedia_url": document["wikipedia_url"],
            "wikipedia_section": document["section_title"],
            "source_document_id": document["document_id"],
            "source_row": {"wikipedia_title": document["title"], "row_position": row_position},
        }
        text_documents.append({
            "text_doc_id": text_doc_id, "data_id": data_id, "split": "train",
            "question": "", "question_original": "", "context": "", "answer": "",
            "golden_answers": [], "image_id": image_id, "image_path": str(image_path),
            "source_metadata": metadata, "contents": document["contents"],
        })
        links.extend([
            {"source_id": data_id, "target_id": text_doc_id,
             "relation": "has_text_document", "data_id": data_id, "split": "train"},
            {"source_id": data_id, "target_id": visual_id,
             "relation": "has_image", "data_id": data_id, "split": "train"},
            {"source_id": text_doc_id, "target_id": text_embedding_id(text_doc_id),
             "relation": "has_text_embedding", "data_id": data_id, "split": "train"},
        ])
        if image_id not in seen_images:
            visual_records.append({
                "visual_record_id": visual_id, "image_id": image_id,
                "image_path": str(image_path), "data_id": data_id, "split": "train",
                "source_metadata": metadata, "image_missing": False,
            })
            links.append({
                "source_id": visual_id, "target_id": visual_embedding_id(image_id),
                "relation": "has_visual_embedding", "data_id": data_id, "split": "train",
            })
            seen_images.add(image_id)
    if len(text_documents) != 1891 or len(visual_records) != 1746:
        raise RuntimeError("multimodal bundle does not cover the full available subset")
    return RecordBundle(text_documents, visual_records, links)


def main() -> None:
    source_file = STRICT_ROOT / "source_manifest.json"
    records_file = STRICT_ROOT / "strict_records.json"
    owner_file = STRICT_ROOT / "owner.json"
    deterministic_file = STRICT_ROOT / "deterministic_audit.json"
    semantic_file = STRICT_ROOT / "semantic_audit_report.json"
    extraction_report = json.loads((STRICT_ROOT / "report.json").read_text())
    source_manifest = json.loads(source_file.read_text())
    records = json.loads(records_file.read_text())
    extraction_owner = json.loads(owner_file.read_text())
    deterministic = json.loads(deterministic_file.read_text())
    semantic = json.loads(semantic_file.read_text())
    documents = source_manifest["documents"]
    documents_by_id = {item["document_id"]: item for item in documents}
    if extraction_report.get("status") != "complete" or len(records) != 1237:
        raise RuntimeError("full strict extraction is incomplete")
    if extraction_report.get("exact_evidence_count") != extraction_report.get("fact_count"):
        raise RuntimeError("full extraction contains non-exact evidence")
    if extraction_owner.get("model") != "api/qwen3.7-max-2026-06-08":
        raise RuntimeError("full extraction was not produced by the selected API Max model")
    if deterministic.get("status") != "pass":
        raise RuntimeError("deterministic full-corpus audit did not pass")
    if semantic.get("status") != "complete" or semantic.get("documents") != 1237:
        raise RuntimeError("dual-model semantic audit is incomplete")
    excluded = set(semantic.get("excluded_sample_ids") or [])

    graph_owner = {
        "pipeline": "strict-source-grounded-max-graph-full1891-v1",
        "dataset": "E-VQA", "subset": SUBSET,
        "scope": "all 1,891 available train rows, 1,746 images, 1,237 unique Wikipedia articles",
        "graph_llm": extraction_owner["model"],
        "source_manifest_sha256": sha256(source_file),
        "strict_records_sha256": sha256(records_file),
        "strict_extraction_owner_sha256": sha256(owner_file),
        "deterministic_audit_sha256": sha256(deterministic_file),
        "semantic_audit_sha256": sha256(semantic_file),
        "extraction_fact_count": extraction_report["fact_count"],
        "semantically_excluded_sample_count": len(excluded),
        "evidence_policy": "exact contiguous source span required",
        "semantic_policy": "exclude sampled facts lacking dual-model SUPPORTED consensus",
        "qa_labels_in_graph": False, "local_extraction_model_loaded": False,
        "gpu_used": False, "embedding": "local real GME; indexed separately",
    }
    GRAPH_ROOT.mkdir(parents=True, exist_ok=True)
    with (GRAPH_ROOT / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report_file = GRAPH_ROOT / "report.json"
        if report_file.exists() and json.loads(report_file.read_text()).get("status") == "complete":
            print(report_file.read_text())
            return
        OUTPUT.mkdir(parents=True, exist_ok=True)
        atomic_json(OUTPUT / "owner.json", graph_owner)
        report = {**graph_owner, "status": "running", "training_started": False}
        atomic_json(report_file, report)
        started = time.perf_counter()
        try:
            import networkx as nx
            from graphr1.prompt import GRAPH_FIELD_SEP
            from graphr1.utils import compute_mdhash_id, encode_string_by_tiktoken
            from evograph_mm.kb.layout import build_layout
            from evograph_mm.kb.mm_graph import build_mm_graph_records, write_mm_graph
            from evograph_mm.kb.store import write_store

            bundle = build_bundle(source_manifest, documents_by_id)
            layout = build_layout(DATA_ROOT, "E-VQA", SUBSET, GRAPH_ROOT)
            store_counts = write_store(layout, bundle)
            write_mm_graph(OUTPUT / "mm_store/graph", build_mm_graph_records(
                text_documents=bundle.text_documents, image_records=bundle.visual_records,
            ))

            full_docs, chunks, document_to_chunk = {}, {}, {}
            for document in documents:
                content = document["contents"].strip()
                doc_id = compute_mdhash_id(content, prefix="doc-")
                chunk_id = compute_mdhash_id(content, prefix="chunk-")
                full_docs.setdefault(doc_id, {"content": content})
                chunks.setdefault(chunk_id, {
                    "tokens": len(encode_string_by_tiktoken(content)), "content": content,
                    "chunk_order_index": 0, "full_doc_id": doc_id,
                })
                document_to_chunk[document["document_id"]] = chunk_id

            occurrences = defaultdict(list)
            for document_id, record in records.items():
                document = documents_by_id[document_id]
                for fact_index, fact in enumerate(record["facts"]):
                    sample_id = f"{document_id}#{fact_index}"
                    if sample_id in excluded:
                        continue
                    if not fact.get("evidence_is_exact_source_span") or \
                            fact["evidence"] not in document["contents"]:
                        raise RuntimeError(f"fact lost exact evidence: {sample_id}")
                    key = " ".join(fact["statement"].casefold().split())
                    occurrences[key].append((document, fact_index, fact))
            if len(occurrences) <= 1000:
                raise RuntimeError("full graph unexpectedly has no more than 1,000 unique hyperedges")

            graph = nx.Graph()
            hyperedges, entity_accumulator = {}, defaultdict(lambda: {
                "type_counts": Counter(), "statements": [], "chunk_ids": set(),
                "titles": set(), "urls": set(),
            })
            sidecar = {"entity": {}, "hyperedge": {}}
            for group in occurrences.values():
                statement = group[0][2]["statement"]
                hyperedge_name = "<hyperedge>" + json.dumps(statement, ensure_ascii=False)
                hyperedge_id = compute_mdhash_id(hyperedge_name, prefix="rel-")
                chunk_ids = sorted({document_to_chunk[item[0]["document_id"]] for item in group})
                metadata = {
                    "source": "api_strict_source_grounded",
                    "ai_model": extraction_owner["model"].removeprefix("api/"),
                    "verification_status": "exact_evidence_and_dual_model_risk_sample_gate",
                    "human_reviewed": False,
                    "source_id": GRAPH_FIELD_SEP.join(chunk_ids),
                    "source_document_count": len({item[0]["document_id"] for item in group}),
                }
                hyperedges[hyperedge_id] = {
                    "content": hyperedge_name, "hyperedge_name": hyperedge_name,
                    "metadata": metadata,
                }
                graph.add_node(hyperedge_name, role="hyperedge", weight=10.0,
                               source_id=metadata["source_id"],
                               metadata=json.dumps(metadata, ensure_ascii=False))
                sidecar["hyperedge"][hyperedge_name] = {
                    "wikipedia_urls": sorted({item[0]["wikipedia_url"] for item in group}),
                    "wikipedia_titles": sorted({item[0]["title"] for item in group}),
                    "source_chunk_ids": chunk_ids,
                    "source_document_ids": sorted({item[0]["document_id"] for item in group}),
                    "evidence": [item[2]["evidence"] for item in group],
                    "fact_indexes": [item[1] for item in group],
                }
                for document, fact_index, fact in group:
                    chunk_id = document_to_chunk[document["document_id"]]
                    for entity in fact["entities"]:
                        entity_name = json.dumps(entity["name"].upper(), ensure_ascii=False)
                        value = entity_accumulator[entity_name]
                        value["type_counts"][entity["type"]] += 1
                        value["statements"].append(statement)
                        value["chunk_ids"].add(chunk_id)
                        value["titles"].add(document["title"])
                        value["urls"].add(document["wikipedia_url"])
                        graph.add_edge(hyperedge_name, entity_name, weight=100.0,
                                       source_id=chunk_id,
                                       metadata=json.dumps(metadata, ensure_ascii=False))

            entities = {}
            conflict_count = 0
            for entity_name, value in entity_accumulator.items():
                ranked_types = sorted(value["type_counts"].items(), key=lambda item: (-item[1], item[0]))
                entity_type = ranked_types[0][0]
                conflict_count += int(len(ranked_types) > 1)
                statements = list(dict.fromkeys(value["statements"]))
                source_ids = GRAPH_FIELD_SEP.join(sorted(value["chunk_ids"]))
                description = GRAPH_FIELD_SEP.join(statements)
                metadata = {
                    "source": "api_strict_source_grounded",
                    "ai_model": extraction_owner["model"].removeprefix("api/"),
                    "verification_status": "exact_evidence_and_dual_model_risk_sample_gate",
                    "human_reviewed": False, "entity_type": json.dumps(entity_type),
                    "entity_type_votes": dict(value["type_counts"]), "source_id": source_ids,
                }
                entity_id = compute_mdhash_id(entity_name, prefix="ent-")
                entities[entity_id] = {"content": entity_name + description,
                                       "entity_name": entity_name, "metadata": metadata}
                graph.add_node(entity_name, role="entity", entity_type=json.dumps(entity_type),
                               description=description, source_id=source_ids,
                               metadata=json.dumps(metadata, ensure_ascii=False))
                sidecar["entity"][entity_name] = {
                    "wikipedia_urls": sorted(value["urls"]),
                    "wikipedia_titles": sorted(value["titles"]),
                    "source_chunk_ids": sorted(value["chunk_ids"]),
                    "entity_type_votes": dict(value["type_counts"]),
                }

            atomic_json(OUTPUT / "kv_store_full_docs.json", full_docs)
            atomic_json(OUTPUT / "kv_store_text_chunks.json", chunks)
            atomic_json(OUTPUT / "kv_store_chunks.json", {})
            atomic_json(OUTPUT / "kv_store_entities.json", entities)
            atomic_json(OUTPUT / "kv_store_hyperedges.json", hyperedges)
            atomic_json(OUTPUT / "mm_store/graph/graphr1_hit_source_sidecar.json", sidecar)
            atomic_json(OUTPUT / ".graphr1_seeded.json", {
                "seeded": False, "source": "api_strict_source_grounded", "copied": [],
            })
            write_graphml_atomic(graph, OUTPUT / "graph_chunk_entity_relation.graphml")
            atomic_json(OUTPUT / "metadata.json", {
                "dataset": "E-VQA", "subset": SUBSET, "scope": graph_owner["scope"],
                "graph_llm": extraction_owner["model"], "strict_source_grounding": True,
                "indexes_built": False,
            })
            counts = {
                "available_train_rows": 1891, "unique_images": 1746,
                "source_documents": len(documents), "unique_content_documents": len(full_docs),
                "text_chunks": len(chunks), "entities": len(entities),
                "hyperedges": len(hyperedges), "nodes": graph.number_of_nodes(),
                "links": graph.number_of_edges(),
            }
            report.update(
                status="complete", store_counts=store_counts, graph_counts=counts,
                elapsed_seconds=round(time.perf_counter() - started, 3),
                all_fact_evidence_exact=True, all_entities_linked_to_facts=True,
                entity_type_conflicts_resolved=conflict_count, indexes_built=False,
                output_dir=str(OUTPUT),
            )
            atomic_json(OUTPUT / "build_report.json", report)
            atomic_json(report_file, report)
            print(json.dumps(report, ensure_ascii=False), flush=True)
        except Exception as exc:
            report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            atomic_json(report_file, report)
            raise


if __name__ == "__main__":
    main()
