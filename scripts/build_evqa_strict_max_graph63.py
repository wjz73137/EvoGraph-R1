#!/usr/bin/env python3
"""Build the audited 63-document E-VQA strict-Max knowledge hypergraph."""
from __future__ import annotations

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
SUBSET = "paper_64_16_seed0"
SUBSET_ROOT = DATA_ROOT / f"datasets_mm/E-VQA/subsets/{SUBSET}"
STRICT_ROOT = DATA_ROOT / "expr_mm/evqa_api_strict_max_extraction_63_v1"
OUTPUT_ROOT = DATA_ROOT / "expr_mm/evqa_api_strict_max_graph_63_v1"
OUTPUT = OUTPUT_ROOT / "E-VQA"


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


def build_bundle(documents: list[dict]):
    from evograph_mm.kb.store import RecordBundle, text_embedding_id, visual_embedding_id

    subset_manifest = json.loads((SUBSET_ROOT / "manifest.json").read_text())
    image_manifest = {item["image_id"]: item for item in subset_manifest["copied_images"]}
    text_documents, visual_records, links = [], [], []
    seen_images = set()
    for document in documents:
        image_id = document["image_id"]
        image_record = image_manifest.get(image_id)
        if not image_record:
            raise RuntimeError(f"subset manifest lacks image {image_id}")
        image_path = DATA_ROOT / image_record["subset_path"]
        if not image_info(image_path) or sha256(image_path) != image_record["sha256"]:
            raise RuntimeError(f"image decoding/hash check failed: {image_id}")
        data_id = document["document_id"]
        text_doc_id = f"text::{data_id}"
        source_metadata = {
            "wikipedia_title": document["title"],
            "wikipedia_url": document["wikipedia_url"],
            "wikipedia_section": document["section_title"],
            "source_row": {"wikipedia_title": document["title"]},
        }
        text_documents.append({
            "text_doc_id": text_doc_id,
            "data_id": data_id,
            "split": "train",
            "question": "",
            "question_original": "",
            "context": "",
            "answer": "",
            "golden_answers": [],
            "image_id": image_id,
            "image_path": str(image_path),
            "source_metadata": source_metadata,
            "contents": document["contents"],
        })
        if image_id not in seen_images:
            visual_records.append({
                "visual_record_id": f"visual::{image_id}",
                "image_id": image_id,
                "image_path": str(image_path),
                "data_id": data_id,
                "split": "train",
                "source_metadata": source_metadata,
                "image_missing": False,
            })
            seen_images.add(image_id)
        for source, target, relation in [
            (data_id, text_doc_id, "has_text_document"),
            (text_doc_id, text_embedding_id(text_doc_id), "has_text_embedding"),
            (f"visual::{image_id}", visual_embedding_id(image_id), "has_visual_embedding"),
        ]:
            links.append({"source_id": source, "target_id": target, "relation": relation,
                          "data_id": data_id, "split": "train"})
    return RecordBundle(text_documents, visual_records, links)


def main() -> None:
    source_file = STRICT_ROOT / "source_manifest.json"
    records_file = STRICT_ROOT / "strict_records.json"
    owner_file = STRICT_ROOT / "owner.json"
    extraction_report_file = STRICT_ROOT / "report.json"
    deterministic_audit_file = STRICT_ROOT / "deterministic_audit.json"
    semantic_audit_file = STRICT_ROOT / "semantic_audit_report.json"
    source_manifest = json.loads(source_file.read_text())
    records = json.loads(records_file.read_text())
    extraction_owner = json.loads(owner_file.read_text())
    extraction_report = json.loads(extraction_report_file.read_text())
    deterministic_audit = json.loads(deterministic_audit_file.read_text())
    semantic_audit = json.loads(semantic_audit_file.read_text())
    documents = source_manifest["documents"]
    documents_by_id = {document["document_id"]: document for document in documents}

    if extraction_report.get("status") != "complete" or extraction_report.get("fact_count") != 912:
        raise RuntimeError("63-document strict extraction is incomplete")
    if extraction_report.get("exact_evidence_count") != 912:
        raise RuntimeError("strict extraction contains non-exact evidence")
    if extraction_owner.get("model") != "api/qwen3.7-max-2026-06-08":
        raise RuntimeError("strict extraction was not produced by the selected Max model")
    if deterministic_audit.get("status") != "pass":
        raise RuntimeError("deterministic extraction audit did not pass")
    if semantic_audit.get("status") != "complete" or semantic_audit.get("disagreement_count") != 0:
        raise RuntimeError("semantic extraction audit is incomplete or disputed")
    if semantic_audit.get("consensus_label_counts") != {"SUPPORTED": 170}:
        raise RuntimeError("semantic extraction audit lacks the expected supported consensus")
    if len(documents) != 63 or set(records) != set(documents_by_id):
        raise RuntimeError("strict records do not match the frozen 63-document source manifest")

    graph_owner = {
        "pipeline": "strict-source-grounded-max-graph63-v1",
        "dataset": "E-VQA",
        "subset": SUBSET,
        "scope": "63 unique training Wikipedia articles; complete first non-empty section",
        "graph_llm": extraction_owner["model"],
        "source_manifest_sha256": sha256(source_file),
        "strict_records_sha256": sha256(records_file),
        "strict_extraction_owner_sha256": sha256(owner_file),
        "deterministic_audit_sha256": sha256(deterministic_audit_file),
        "semantic_audit_sha256": sha256(semantic_audit_file),
        "fact_count": 912,
        "evidence_policy": "exact contiguous source span required",
        "qa_labels_in_graph": False,
        "local_extraction_model_loaded": False,
        "gpu_used": False,
        "embedding": "local real GME; indexed separately",
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        graph_owner_file = OUTPUT / "owner.json"
        if graph_owner_file.exists() and json.loads(graph_owner_file.read_text()) != graph_owner:
            raise RuntimeError("existing 63-document graph has a different owner; retained")
        report_file = OUTPUT_ROOT / "report.json"
        if report_file.exists() and json.loads(report_file.read_text()).get("status") == "complete":
            print(report_file.read_text())
            return
        OUTPUT.mkdir(parents=True, exist_ok=True)
        atomic_json(graph_owner_file, graph_owner)
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

            bundle = build_bundle(documents)
            layout = build_layout(DATA_ROOT, "E-VQA", SUBSET, OUTPUT_ROOT)
            store_counts = write_store(layout, bundle)
            write_mm_graph(OUTPUT / "mm_store/graph", build_mm_graph_records(
                text_documents=bundle.text_documents, image_records=bundle.visual_records,
            ))

            full_docs, chunks, document_to_chunk = {}, {}, {}
            for document in documents:
                content = document["contents"].strip()
                doc_id = compute_mdhash_id(content, prefix="doc-")
                chunk_id = compute_mdhash_id(content, prefix="chunk-")
                if doc_id in full_docs or chunk_id in chunks:
                    raise RuntimeError("two source documents produced the same content hash")
                full_docs[doc_id] = {"content": content}
                chunks[chunk_id] = {
                    "tokens": len(encode_string_by_tiktoken(content)),
                    "content": content,
                    "chunk_order_index": 0,
                    "full_doc_id": doc_id,
                }
                document_to_chunk[document["document_id"]] = chunk_id

            graph = nx.Graph()
            hyperedges: dict[str, dict] = {}
            entity_accumulator: dict[str, dict] = {}
            sidecar = {"entity": {}, "hyperedge": {}}
            seen_statements = set()
            for document_id, record in records.items():
                document = documents_by_id[document_id]
                source, title = document["contents"], document["title"]
                chunk_id = document_to_chunk[document_id]
                mapping = {
                    "wikipedia_urls": [document["wikipedia_url"]],
                    "wikipedia_titles": [title],
                    "source_chunk_ids": [chunk_id],
                }
                for fact_index, fact in enumerate(record["facts"]):
                    if not fact.get("evidence_is_exact_source_span") or fact["evidence"] not in source:
                        raise RuntimeError(f"fact lost exact evidence: {title} #{fact_index}")
                    statement = fact["statement"]
                    normalized = " ".join(statement.casefold().split())
                    if normalized in seen_statements:
                        raise RuntimeError(f"duplicate strict hyperedge: {statement}")
                    seen_statements.add(normalized)
                    hyperedge_name = "<hyperedge>" + json.dumps(statement, ensure_ascii=False)
                    hyperedge_id = compute_mdhash_id(hyperedge_name, prefix="rel-")
                    metadata = {
                        "source": "api_strict_source_grounded",
                        "ai_model": extraction_owner["model"].removeprefix("api/"),
                        "verification_status": "exact_evidence_and_cross_model_sample_audit",
                        "human_reviewed": False,
                        "source_id": chunk_id,
                        "source_title": title,
                        "source_document_id": document_id,
                        "fact_index": fact_index,
                        "evidence": fact["evidence"],
                    }
                    hyperedges[hyperedge_id] = {
                        "content": hyperedge_name,
                        "hyperedge_name": hyperedge_name,
                        "metadata": metadata,
                    }
                    graph.add_node(hyperedge_name, role="hyperedge", weight=10.0,
                                   source_id=chunk_id,
                                   metadata=json.dumps(metadata, ensure_ascii=False))
                    sidecar["hyperedge"][hyperedge_name] = {
                        **copy.deepcopy(mapping), "evidence": fact["evidence"],
                        "fact_index": fact_index, "source_document_id": document_id,
                    }
                    for entity in fact["entities"]:
                        if entity["name"].casefold() not in statement.casefold():
                            raise RuntimeError("entity name is no longer a fact substring")
                        entity_name = json.dumps(entity["name"].upper(), ensure_ascii=False)
                        current = entity_accumulator.setdefault(entity_name, {
                            "type": entity["type"], "statements": [], "chunk_ids": set(),
                            "titles": set(), "urls": set(),
                        })
                        if current["type"] != entity["type"]:
                            raise RuntimeError(f"entity type conflict for {entity_name}")
                        current["statements"].append(statement)
                        current["chunk_ids"].add(chunk_id)
                        current["titles"].add(title)
                        current["urls"].add(document["wikipedia_url"])
                        if graph.has_node(entity_name):
                            existing = set(graph.nodes[entity_name]["source_id"].split(GRAPH_FIELD_SEP))
                            graph.nodes[entity_name]["source_id"] = GRAPH_FIELD_SEP.join(
                                sorted(existing | {chunk_id})
                            )
                        else:
                            graph.add_node(entity_name, role="entity",
                                           entity_type=json.dumps(entity["type"]),
                                           description=statement, source_id=chunk_id,
                                           metadata=json.dumps(metadata, ensure_ascii=False))
                        graph.add_edge(hyperedge_name, entity_name, weight=100.0,
                                       source_id=chunk_id,
                                       metadata=json.dumps(metadata, ensure_ascii=False))

            entities = {}
            for entity_name, value in entity_accumulator.items():
                statements = list(dict.fromkeys(value["statements"]))
                source_ids = GRAPH_FIELD_SEP.join(sorted(value["chunk_ids"]))
                description = GRAPH_FIELD_SEP.join(statements)
                metadata = {
                    "source": "api_strict_source_grounded",
                    "ai_model": extraction_owner["model"].removeprefix("api/"),
                    "verification_status": "exact_evidence_and_cross_model_sample_audit",
                    "human_reviewed": False,
                    "entity_type": json.dumps(value["type"]),
                    "source_id": source_ids,
                }
                entity_id = compute_mdhash_id(entity_name, prefix="ent-")
                entities[entity_id] = {"content": entity_name + description,
                                       "entity_name": entity_name, "metadata": metadata}
                graph.nodes[entity_name]["description"] = description
                graph.nodes[entity_name]["metadata"] = json.dumps(metadata, ensure_ascii=False)
                sidecar["entity"][entity_name] = {
                    "wikipedia_urls": sorted(value["urls"]),
                    "wikipedia_titles": sorted(value["titles"]),
                    "source_chunk_ids": sorted(value["chunk_ids"]),
                }

            if len(hyperedges) != 912 or not entities or graph.number_of_edges() != 1886:
                raise RuntimeError("63-document strict graph counts are incomplete")
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
                "documents": len(full_docs), "text_chunks": len(chunks),
                "entities": len(entities), "hyperedges": len(hyperedges),
                "nodes": graph.number_of_nodes(), "links": graph.number_of_edges(),
            }
            report.update(
                status="complete", store_counts=store_counts, graph_counts=counts,
                elapsed_seconds=round(time.perf_counter() - started, 3),
                all_fact_evidence_exact=True, all_entities_linked_to_facts=True,
                entity_type_conflicts=0, indexes_built=False, output_dir=str(OUTPUT),
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
