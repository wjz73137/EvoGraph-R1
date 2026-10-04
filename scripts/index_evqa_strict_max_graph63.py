#!/usr/bin/env python3
"""Build real single-GPU GME indexes for the audited 63-document graph."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault("MM_EMBED_DEVICE", "cuda")
os.environ.setdefault("EVOGRAPH_MM_EMBED_RUNTIME_DEVICE", "cuda")
os.environ.setdefault("EVOGRAPH_MM_ENABLE_BGE_TEXT", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evograph_mm.kb.gldv2_subset import atomic_json


DATA_ROOT = Path("/home/data/dataset/wjz/EvoGraph-R1")
SUBSET = "paper_64_16_seed0"
GRAPH_ROOT = DATA_ROOT / "expr_mm/evqa_api_strict_max_graph_63_v1"
OUTPUT = GRAPH_ROOT / "E-VQA"
SOURCE_MANIFEST = DATA_ROOT / "expr_mm/evqa_api_strict_max_extraction_63_v1/source_manifest.json"
STRICT_RECORDS = DATA_ROOT / "expr_mm/evqa_api_strict_max_extraction_63_v1/strict_records.json"
EXPECTED_PHYSICAL_GPU = "2"


def index_artifacts_present(record: dict | None) -> bool:
    paths = [Path(value) for key, value in (record or {}).items() if key.endswith("_path")]
    return bool(paths) and all(path.is_file() and path.stat().st_size > 0 for path in paths)


def main() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != EXPECTED_PHYSICAL_GPU:
        raise RuntimeError(
            f"63-document index build must be pinned to physical GPU 2; got {visible!r}"
        )
    owner_file = OUTPUT / "owner.json"
    report_file = GRAPH_ROOT / "report.json"
    report = json.loads(report_file.read_text())
    owner = json.loads(owner_file.read_text())
    counts = report.get("graph_counts", {})
    if report.get("status") != "complete" or counts.get("hyperedges") != 912:
        raise RuntimeError("63-document strict-Max graph is incomplete")
    if owner.get("embedding") != "local real GME; indexed separately":
        raise RuntimeError("graph owner lacks the expected embedding configuration")

    with (GRAPH_ROOT / ".index.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (report.get("index_status") == "complete" and all(
            index_artifacts_present(report.get(key))
            for key in ("entity_index", "hyperedge_index", "image_index", "text_index")
        )):
            print(json.dumps(report, ensure_ascii=False))
            return
        report.update(index_status="running", index_device="cuda:0 (physical GPU 2)",
                      index_batch_size=1, index_cpu_threads=4)
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
                raise RuntimeError("the pinned indexing process does not see exactly one CUDA device")
            free_bytes, _ = torch.cuda.mem_get_info(0)
            if free_bytes < 18 * 1024**3:
                raise RuntimeError("physical GPU 2 no longer has at least 18 GiB free")

            from evograph_mm.kb.build import build_text_graphr1_indexes
            from evograph_mm.kb.indexing.encoders import GMEQwen2VLEncoder, GME_MODEL_REPO_ID
            from evograph_mm.kb.indexing.faiss_store import write_vector_index
            from evograph_mm.kb.layout import build_layout
            from evograph_mm.kb.store import text_embedding_id, visual_embedding_id
            from scripts.build_evqa_strict_max_graph63 import build_bundle
            from scripts.run_evqa_retrieval_smoke import GME

            documents = json.loads(SOURCE_MANIFEST.read_text())["documents"]
            records = json.loads(STRICT_RECORDS.read_text())
            bundle = build_bundle(documents)
            layout = build_layout(DATA_ROOT, "E-VQA", SUBSET, GRAPH_ROOT)
            encoder = GMEQwen2VLEncoder(GME, batch_size=1)
            if encoder.device != "cuda":
                raise RuntimeError(f"GME resolved unexpected device: {encoder.device!r}")

            graph_indexes = build_text_graphr1_indexes(
                output_dir=OUTPUT, encoder=encoder, model_path=GME,
                model_repo_id=GME_MODEL_REPO_ID, encoder_mode="real_gme_gpu2_batch63",
            )
            report["entity_index"] = graph_indexes["entity"]
            report["hyperedge_index"] = graph_indexes["hyperedge"]
            atomic_json(report_file, report)
            print(json.dumps({"phase": "graph_indexes", "entity": 913, "hyperedge": 912}),
                  flush=True)
            report["image_index"] = write_vector_index(
                layout.indexing_store_root, "image",
                [visual_embedding_id(item["image_id"]) for item in bundle.visual_records],
                encoder.encode_images([item["image_path"] for item in bundle.visual_records]),
                GME, GME_MODEL_REPO_ID, "GME default image prompt", "real_gme_gpu2_batch63",
            )
            atomic_json(report_file, report)
            print(json.dumps({"phase": "image_index", "images": len(bundle.visual_records)}),
                  flush=True)
            report["text_index"] = write_vector_index(
                layout.indexing_store_root, "text",
                [text_embedding_id(item["text_doc_id"]) for item in bundle.text_documents],
                encoder.encode_texts([item["contents"] for item in bundle.text_documents]),
                GME, GME_MODEL_REPO_ID, "GME default document prompt", "real_gme_gpu2_batch63",
            )
            atomic_json(report_file, report)
            print(json.dumps({"phase": "text_index", "texts": len(bundle.text_documents)}),
                  flush=True)

            from fastapi.testclient import TestClient
            from evograph_mm.kb.api import create_app
            app = create_app(
                working_dir=OUTPUT, model_path=GME, dataset="E-VQA", subset=SUBSET,
                encoder_factory=lambda *args, **kwargs: encoder, reload_interval=0,
            )
            status = app.state.mm_api.status()
            if status["status"] != "ready":
                raise RuntimeError("63-document graph retrieval is not ready: " + str(status["blockers"]))
            positions = [0, 8, 16, 24, 32, 40, 48, 56, 62]
            checks = []
            for position in positions:
                document = documents[position]
                fact = records[document["document_id"]]["facts"][0]
                checks.append((document["title"], fact["statement"]))
            results = []
            with TestClient(app) as test_client:
                for title, statement in checks:
                    response = test_client.post("/search", json={
                        "queries": [statement], "entity_top_k": 5,
                        "hyperedge_top_k": 5, "rag_top_k": 0, "image_top_k": 0,
                    })
                    if response.status_code != 200:
                        raise RuntimeError(f"retrieval query failed: {response.status_code}")
                    payload = json.loads(response.json()[0])
                    rendered = json.dumps(payload, ensure_ascii=False)
                    if statement.casefold() not in rendered.casefold():
                        raise RuntimeError(f"exact-statement retrieval missed source fact for {title}")
                    results.append({"title": title, "query": statement, "response": payload})
            validation_file = OUTPUT / "strict_retrieval_validation.json"
            atomic_json(validation_file, {
                "status": "passed", "device": "cuda:0 (physical GPU 2)",
                "checks": results, "mutation_endpoints_used": False,
            })
            metadata = json.loads((OUTPUT / "metadata.json").read_text())
            metadata.update(
                embedding_dimension=1536, embedding_model=str(GME),
                embedding_model_repo_id=GME_MODEL_REPO_ID,
                encoder_mode="real_gme_gpu2_batch63", indexes_built=True,
            )
            atomic_json(OUTPUT / "metadata.json", metadata)
            report.update(
                index_status="complete", indexes_built=True, retrieval_status=status,
                retrieval_checks=len(checks), retrieval_validation=str(validation_file),
                index_elapsed_seconds=round(time.perf_counter() - started, 3),
                cuda_peak_allocated_mib=round(torch.cuda.max_memory_allocated(0) / 1024**2, 1),
            )
            atomic_json(OUTPUT / "build_report.json", report)
            atomic_json(report_file, report)
            print(json.dumps({
                "index_status": "complete", "device": report["index_device"],
                "indexes": {key: report[key].get("vector_count") for key in (
                    "entity_index", "hyperedge_index", "image_index", "text_index"
                )},
                "retrieval_checks": len(checks), "seconds": report["index_elapsed_seconds"],
                "cuda_peak_allocated_mib": report["cuda_peak_allocated_mib"],
            }, ensure_ascii=False), flush=True)
        except Exception as exc:
            report.update(index_status="failed", index_error_type=type(exc).__name__,
                          index_error=str(exc), indexes_built=False)
            atomic_json(report_file, report)
            raise


if __name__ == "__main__":
    main()
