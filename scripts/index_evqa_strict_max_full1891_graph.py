#!/usr/bin/env python3
"""Build real GME indexes for the complete 1,891-row strict-Max graph."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MM_EMBED_DEVICE", "cuda")
os.environ.setdefault("EVOGRAPH_MM_EMBED_RUNTIME_DEVICE", "cuda")
os.environ.setdefault("EVOGRAPH_MM_ENABLE_BGE_TEXT", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evograph_mm.kb.gldv2_subset import atomic_json


DATA_ROOT = Path("/home/data/dataset/wjz/EvoGraph-R1")
SUBSET = "E-VQA-GLDv2-1898-61-seed0"
GRAPH_ROOT = DATA_ROOT / "expr_mm/evqa_api_strict_max_graph_full1891_v1"
OUTPUT = GRAPH_ROOT / "E-VQA"
SOURCE_MANIFEST = DATA_ROOT / "expr_mm/evqa_api_strict_max_extraction_full1891_v1/source_manifest.json"
STRICT_RECORDS = DATA_ROOT / "expr_mm/evqa_api_strict_max_extraction_full1891_v1/strict_records.json"
SEMANTIC_AUDIT = DATA_ROOT / "expr_mm/evqa_api_strict_max_extraction_full1891_v1/semantic_audit_report.json"
GME = DATA_ROOT.parent / "models/gme-Qwen2-VL-2B-Instruct"


def artifact_present(record: dict | None) -> bool:
    paths = [Path(value) for key, value in (record or {}).items() if key.endswith("_path")]
    return bool(paths) and all(path.is_file() and path.stat().st_size > 0 for path in paths)


def main() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if len([part for part in visible.split(",") if part.strip()]) != 1:
        raise RuntimeError("full index build must be pinned to exactly one approved GPU")
    text_batch_size = int(os.environ.get("STRICT_TEXT_BATCH", "8"))
    image_batch_size = int(os.environ.get("STRICT_IMAGE_BATCH", "2"))
    if not 1 <= text_batch_size <= 16 or not 1 <= image_batch_size <= 4:
        raise RuntimeError("index batch sizes exceed the validated RTX 3090 limits")
    report_file = GRAPH_ROOT / "report.json"
    report = json.loads(report_file.read_text())
    counts = report.get("graph_counts", {})
    if report.get("status") != "complete" or counts.get("hyperedges", 0) <= 1000:
        raise RuntimeError("complete full strict-Max graph is unavailable")
    with (GRAPH_ROOT / ".index.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if report.get("index_status") == "complete" and all(
            artifact_present(report.get(key))
            for key in ("entity_index", "hyperedge_index", "image_index", "text_index")
        ):
            print(json.dumps(report, ensure_ascii=False))
            return
        report.update(index_status="running", index_device=f"cuda:0 (physical GPU {visible})",
                      index_text_batch_size=text_batch_size,
                      index_image_batch_size=image_batch_size, index_cpu_threads=4)
        report.pop("index_error", None)
        atomic_json(report_file, report)
        started = time.perf_counter()
        try:
            import faiss
            import torch
            torch.set_num_threads(4)
            torch.set_num_interop_threads(2)
            faiss.omp_set_num_threads(4)
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError("the pinned index process does not see exactly one CUDA device")
            free_bytes, _ = torch.cuda.mem_get_info(0)
            if free_bytes < 18 * 1024**3:
                raise RuntimeError("the selected GPU no longer has at least 18 GiB free")

            from evograph_mm.kb.build import build_text_graphr1_indexes
            from evograph_mm.kb.indexing.encoders import GMEQwen2VLEncoder, GME_MODEL_REPO_ID
            from evograph_mm.kb.indexing.faiss_store import write_vector_index
            from evograph_mm.kb.layout import build_layout
            from evograph_mm.kb.store import text_embedding_id, visual_embedding_id
            from scripts.build_evqa_strict_max_full1891_graph import build_bundle

            source = json.loads(SOURCE_MANIFEST.read_text())
            documents = {item["document_id"]: item for item in source["documents"]}
            bundle = build_bundle(source, documents)
            layout = build_layout(DATA_ROOT, "E-VQA", SUBSET, GRAPH_ROOT)
            encoder = GMEQwen2VLEncoder(GME, batch_size=text_batch_size)
            if encoder.device != "cuda":
                raise RuntimeError(f"GME resolved unexpected device: {encoder.device!r}")
            mode = (f"real_gme_gpu{visible}_textbatch{text_batch_size}_"
                    f"imagebatch{image_batch_size}_full1891")
            if not (artifact_present(report.get("entity_index")) and
                    artifact_present(report.get("hyperedge_index"))):
                graph_indexes = build_text_graphr1_indexes(
                    output_dir=OUTPUT, encoder=encoder, model_path=GME,
                    model_repo_id=GME_MODEL_REPO_ID, encoder_mode=mode,
                )
                report["entity_index"] = graph_indexes["entity"]
                report["hyperedge_index"] = graph_indexes["hyperedge"]
                atomic_json(report_file, report)
            print(json.dumps({"phase": "graph_indexes", "entity": counts["entities"],
                              "hyperedge": counts["hyperedges"]}), flush=True)
            encoder.batch_size = image_batch_size
            if not artifact_present(report.get("image_index")):
                report["image_index"] = write_vector_index(
                    layout.indexing_store_root, "image",
                    [visual_embedding_id(item["image_id"]) for item in bundle.visual_records],
                    encoder.encode_images([item["image_path"] for item in bundle.visual_records]),
                    GME, GME_MODEL_REPO_ID, "GME default image prompt", mode,
                )
                atomic_json(report_file, report)
            print(json.dumps({"phase": "image_index", "images": len(bundle.visual_records)}), flush=True)
            encoder.batch_size = text_batch_size
            if not artifact_present(report.get("text_index")):
                report["text_index"] = write_vector_index(
                    layout.indexing_store_root, "text",
                    [text_embedding_id(item["text_doc_id"]) for item in bundle.text_documents],
                    encoder.encode_texts([item["contents"] for item in bundle.text_documents]),
                    GME, GME_MODEL_REPO_ID, "GME default document prompt", mode,
                )
                atomic_json(report_file, report)
            print(json.dumps({"phase": "text_index", "texts": len(bundle.text_documents)}), flush=True)

            from fastapi.testclient import TestClient
            from evograph_mm.kb.api import create_app
            app = create_app(
                working_dir=OUTPUT, model_path=GME, dataset="E-VQA", subset=SUBSET,
                encoder_factory=lambda *args, **kwargs: encoder, reload_interval=0,
            )
            status = app.state.mm_api.status()
            if status["status"] != "ready":
                raise RuntimeError("full graph retrieval is not ready: " + str(status["blockers"]))
            records = json.loads(STRICT_RECORDS.read_text())
            excluded = set(json.loads(SEMANTIC_AUDIT.read_text()).get("excluded_sample_ids") or [])
            document_list = source["documents"]
            positions = [0, 137, 274, 411, 548, 685, 822, 959, 1096, 1236]
            checks = []
            with TestClient(app) as test_client:
                for position in positions:
                    for offset in range(len(document_list)):
                        document = document_list[(position + offset) % len(document_list)]
                        document_id = document["document_id"]
                        candidates = [
                            item for index, item in enumerate(records[document_id]["facts"])
                            if f"{document_id}#{index}" not in excluded
                        ]
                        if candidates:
                            fact = candidates[0]
                            break
                    else:
                        raise RuntimeError("semantic screening excluded every fact")
                    response = test_client.post("/search", json={
                        "queries": [fact["statement"]], "entity_top_k": 5,
                        "hyperedge_top_k": 5, "rag_top_k": 0, "image_top_k": 0,
                    })
                    if response.status_code != 200:
                        raise RuntimeError(f"retrieval query failed: {response.status_code}")
                    payload = json.loads(response.json()[0])
                    if fact["statement"].casefold() not in json.dumps(payload, ensure_ascii=False).casefold():
                        raise RuntimeError(f"exact-statement retrieval missed {document['title']}")
                    checks.append({"title": document["title"], "query": fact["statement"],
                                   "response": payload})
            validation_file = OUTPUT / "strict_retrieval_validation.json"
            atomic_json(validation_file, {"status": "passed", "device": report["index_device"],
                                          "checks": checks, "mutation_endpoints_used": False})
            metadata = json.loads((OUTPUT / "metadata.json").read_text())
            metadata.update(embedding_dimension=1536, embedding_model=str(GME),
                            embedding_model_repo_id=GME_MODEL_REPO_ID,
                            encoder_mode=mode, indexes_built=True)
            atomic_json(OUTPUT / "metadata.json", metadata)
            report.update(
                index_status="complete", indexes_built=True, retrieval_status=status,
                retrieval_checks=len(checks), retrieval_validation=str(validation_file),
                index_elapsed_seconds=round(time.perf_counter() - started, 3),
                cuda_peak_allocated_mib=round(torch.cuda.max_memory_allocated(0) / 1024**2, 1),
            )
            atomic_json(OUTPUT / "build_report.json", report)
            atomic_json(report_file, report)
            print(json.dumps({"index_status": "complete", "device": report["index_device"],
                              "indexes": {key: report[key]["vector_count"] for key in
                                          ("entity_index", "hyperedge_index", "image_index", "text_index")},
                              "retrieval_checks": len(checks),
                              "seconds": report["index_elapsed_seconds"],
                              "cuda_peak_allocated_mib": report["cuda_peak_allocated_mib"]},
                             ensure_ascii=False), flush=True)
        except Exception as exc:
            report.update(index_status="failed", index_error_type=type(exc).__name__,
                          index_error=str(exc), indexes_built=False)
            atomic_json(report_file, report)
            raise


if __name__ == "__main__":
    main()
