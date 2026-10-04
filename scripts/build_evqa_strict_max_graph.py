#!/usr/bin/env python3
"""Build a four-document source-grounded graph from strict Max extraction records."""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evograph_mm.kb.gldv2_subset import atomic_json


PROJECT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/home/data/dataset/wjz/EvoGraph-R1")
SOURCE_ROOT = DATA_ROOT / "expr_mm/evqa_api_graph_baseline"
STRICT_ROOT = DATA_ROOT / "expr_mm/evqa_api_strict_max_extraction_v2"
OUTPUT_ROOT = DATA_ROOT / "expr_mm/evqa_api_strict_max_graph_v1"
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
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".graphml",
                                dir=target.parent)
    os.close(fd)
    temp = Path(name)
    try:
        nx.write_graphml(graph, temp)
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)


def main() -> None:
    source_snapshot_file = SOURCE_ROOT / "source_snapshot.json"
    strict_records_file = STRICT_ROOT / "strict_records.json"
    strict_report_file = STRICT_ROOT / "report.json"
    strict_owner_file = STRICT_ROOT / "owner.json"
    strict_report = json.loads(strict_report_file.read_text())
    strict_owner = json.loads(strict_owner_file.read_text())
    if strict_report.get("status") != "complete" or strict_report.get("fact_count") != 49:
        raise RuntimeError("strict Max extraction v2 is incomplete")
    if strict_report.get("exact_evidence_count") != strict_report["fact_count"]:
        raise RuntimeError("strict extraction still contains non-exact evidence")
    if strict_owner.get("model") != "api/qwen3.7-max-2026-06-08":
        raise RuntimeError("strict extraction was not produced by the selected Max model")

    snapshot = json.loads(source_snapshot_file.read_text())
    records = json.loads(strict_records_file.read_text())
    documents = {
        item["source_metadata"]["wikipedia_title"]: item
        for item in snapshot["documents"]
    }
    if set(records) != set(documents):
        raise RuntimeError("strict records do not match the source snapshot")
    owner = {
        "pipeline": "strict-source-grounded-max-graph-v1",
        "dataset": "E-VQA",
        "scope": snapshot["scope"],
        "graph_llm": strict_owner["model"],
        "source_snapshot_sha256": sha256(source_snapshot_file),
        "strict_records_sha256": sha256(strict_records_file),
        "strict_extraction_owner_sha256": sha256(strict_owner_file),
        "fact_count": 49,
        "evidence_policy": "exact contiguous source span required",
        "local_extraction_model_loaded": False,
        "gpu_used": False,
        "embedding": "local real GME; indexed separately",
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owner_file = OUTPUT / "owner.json"
        if owner_file.exists() and json.loads(owner_file.read_text()) != owner:
            raise RuntimeError("existing strict Max graph has a different owner; retained")
        report_file = OUTPUT_ROOT / "report.json"
        if report_file.exists():
            report = json.loads(report_file.read_text())
            if report.get("status") == "complete":
                print(json.dumps(report, ensure_ascii=False))
                return
        OUTPUT.mkdir(parents=True, exist_ok=True)
        atomic_json(owner_file, owner)
        report = {**owner, "status": "running", "training_started": False}
        atomic_json(report_file, report)
        started = time.perf_counter()
        try:
            import networkx as nx
            from graphr1.prompt import GRAPH_FIELD_SEP
            from graphr1.utils import compute_mdhash_id, encode_string_by_tiktoken
            from evograph_mm.kb.layout import build_layout
            from evograph_mm.kb.mm_graph import build_mm_graph_records, write_mm_graph
            from evograph_mm.kb.store import write_store
            from scripts.run_evqa_api_graph import source_bundle

            bundle = source_bundle(snapshot)
            layout = build_layout(DATA_ROOT, "E-VQA", "paper_64_16_seed0", OUTPUT_ROOT)
            store_counts = write_store(layout, bundle)
            write_mm_graph(OUTPUT / "mm_store/graph", build_mm_graph_records(
                text_documents=bundle.text_documents,
                image_records=bundle.visual_records,
            ))

            full_docs, chunks = {}, {}
            title_to_chunk = {}
            for title, document in documents.items():
                content = document["contents"].strip()
                doc_id = compute_mdhash_id(content, prefix="doc-")
                chunk_id = compute_mdhash_id(content, prefix="chunk-")
                full_docs[doc_id] = {"content": content}
                chunks[chunk_id] = {
                    "tokens": len(encode_string_by_tiktoken(content)),
                    "content": content,
                    "chunk_order_index": 0,
                    "full_doc_id": doc_id,
                }
                title_to_chunk[title] = chunk_id

            graph = nx.Graph()
            hyperedges: dict[str, dict] = {}
            entity_accumulator: dict[str, dict] = {}
            sidecar = {"entity": {}, "hyperedge": {}}
            for title, record in records.items():
                document = documents[title]
                source = document["contents"]
                chunk_id = title_to_chunk[title]
                mapping = {
                    "wikipedia_urls": [document["source_metadata"]["wikipedia_url"]],
                    "wikipedia_titles": [title],
                    "source_chunk_ids": [chunk_id],
                }
                for fact_index, fact in enumerate(record["facts"]):
                    if not fact.get("evidence_is_exact_source_span") or fact["evidence"] not in source:
                        raise RuntimeError(f"fact lost exact evidence: {title} #{fact_index}")
                    statement = fact["statement"]
                    hyperedge_name = "<hyperedge>" + json.dumps(statement, ensure_ascii=False)
                    hyperedge_id = compute_mdhash_id(hyperedge_name, prefix="rel-")
                    if hyperedge_id in hyperedges:
                        raise RuntimeError(f"duplicate strict hyperedge: {statement}")
                    metadata = {
                        "source": "api_strict_source_grounded",
                        "ai_model": strict_owner["model"].removeprefix("api/"),
                        "verification_status": "exact_source_evidence",
                        "human_reviewed": False,
                        "source_id": chunk_id,
                        "source_title": title,
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
                        **copy.deepcopy(mapping),
                        "evidence": fact["evidence"],
                        "fact_index": fact_index,
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
                        current["urls"].add(document["source_metadata"]["wikipedia_url"])
                        if graph.has_node(entity_name):
                            existing = set(graph.nodes[entity_name]["source_id"].split(GRAPH_FIELD_SEP))
                            graph.nodes[entity_name]["source_id"] = GRAPH_FIELD_SEP.join(
                                sorted(existing | {chunk_id}))
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
                    "ai_model": strict_owner["model"].removeprefix("api/"),
                    "verification_status": "exact_source_evidence",
                    "human_reviewed": False,
                    "entity_type": json.dumps(value["type"]),
                    "source_id": source_ids,
                }
                entity_id = compute_mdhash_id(entity_name, prefix="ent-")
                entities[entity_id] = {
                    "content": entity_name + description,
                    "entity_name": entity_name,
                    "metadata": metadata,
                }
                graph.nodes[entity_name]["description"] = description
                graph.nodes[entity_name]["metadata"] = json.dumps(metadata, ensure_ascii=False)
                sidecar["entity"][entity_name] = {
                    "wikipedia_urls": sorted(value["urls"]),
                    "wikipedia_titles": sorted(value["titles"]),
                    "source_chunk_ids": sorted(value["chunk_ids"]),
                }

            if len(hyperedges) != 49 or not entities or graph.number_of_edges() == 0:
                raise RuntimeError("strict graph counts are incomplete")
            atomic_json(OUTPUT / "kv_store_full_docs.json", full_docs)
            atomic_json(OUTPUT / "kv_store_text_chunks.json", chunks)
            atomic_json(OUTPUT / "kv_store_chunks.json", {})
            atomic_json(OUTPUT / "kv_store_entities.json", entities)
            atomic_json(OUTPUT / "kv_store_hyperedges.json", hyperedges)
            atomic_json(OUTPUT / "mm_store/graph/graphr1_hit_source_sidecar.json", sidecar)
            atomic_json(OUTPUT / ".graphr1_seeded.json", {
                "seeded": False, "source": "api_strict_source_grounded", "copied": []})
            write_graphml_atomic(graph, OUTPUT / "graph_chunk_entity_relation.graphml")
            atomic_json(OUTPUT / "metadata.json", {
                "dataset": "E-VQA", "subset": "paper_64_16_seed0",
                "scope": snapshot["scope"], "graph_llm": strict_owner["model"],
                "strict_source_grounding": True, "indexes_built": False,
            })
            counts = {
                "documents": len(full_docs), "text_chunks": len(chunks),
                "entities": len(entities), "hyperedges": len(hyperedges),
                "nodes": graph.number_of_nodes(), "links": graph.number_of_edges(),
            }
            report.update(
                status="complete", store_counts=store_counts, graph_counts=counts,
                elapsed_seconds=round(time.perf_counter() - started, 3),
                all_fact_evidence_exact=True,
                all_entities_linked_to_facts=True,
                entity_type_conflicts=0,
                indexes_built=False,
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
