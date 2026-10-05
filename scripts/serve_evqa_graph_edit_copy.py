#!/usr/bin/env python3
"""Serve one prepared E-VQA graph-edit copy on localhost.

The server refuses the immutable baseline graph.  ``--working-dir`` must be a
copy created by ``evograph_mm.kb.graph_edit.prepare_edit_working_dir`` whose
metadata points back to the approved API-extracted full graph.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evograph_mm.kb.gldv2_subset import atomic_json
from scripts.run_evqa_gpu_smoke import ROOT
from scripts.run_evqa_retrieval_smoke import GME


APPROVED_BASE = (
    ROOT / "expr_mm/evqa_api_strict_max_graph_full1891_v1/E-VQA"
).resolve()


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _validate_edit_copy(working_dir: Path) -> tuple[dict, dict, dict]:
    target = working_dir.resolve()
    if target == APPROVED_BASE:
        raise RuntimeError("refusing to expose the immutable baseline as writable")
    metadata = _load_json(target / "metadata.json")
    if metadata.get("graph_edit_copy") is not True:
        raise RuntimeError("working-dir is not a prepared graph-edit copy")
    base_raw = metadata.get("base_output_dir")
    if not isinstance(base_raw, str) or Path(base_raw).resolve() != APPROVED_BASE:
        raise RuntimeError("graph-edit copy does not point to the approved API baseline")

    report = _load_json(APPROVED_BASE.parent / "report.json")
    indexes_ready = report.get("indexes_complete") is True or (
        report.get("indexes_built") is True
        and report.get("index_status") == "complete"
    )
    if report.get("status") != "complete" or not indexes_ready:
        raise RuntimeError("approved API baseline graph/indexes are incomplete")
    owner = _load_json(APPROVED_BASE / "owner.json")
    if not str(owner.get("graph_llm", "")).startswith("api/"):
        raise RuntimeError("refusing a graph that was not extracted by the configured API")
    return metadata, report, owner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--working-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8006)
    parser.add_argument("--runtime-encoder", choices=("gme", "cached"), default="gme",
                        help="cached uses existing image vectors and BGE; no runtime GME fallback")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")

    target = args.working_dir.resolve()
    metadata, report, owner = _validate_edit_copy(target)
    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        MM_EMBED_DEVICE="cpu",
        EVOGRAPH_MM_EMBED_RUNTIME_DEVICE="cpu",
        EVOGRAPH_MM_ENABLE_BGE_TEXT="1",
        BGE_MODEL_PATH=str(GME.parent / "bge-large-en-v1.5"),
        BGE_DEVICE="cpu",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        EVOGRAPH_MM_ENABLE_RUNTIME_ENCODER="1" if args.runtime_encoder == "gme" else "0",
    )

    service_id = hashlib.sha256(str(target).encode()).hexdigest()[:12]
    lock_path = ROOT / f"logs/.evqa_graph_edit_{service_id}.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with socket.socket() as check:
            check.bind(("127.0.0.1", args.port))

        import faiss
        import torch
        import uvicorn

        from evograph_mm.kb.api import create_app
        from evograph_mm.kb.indexing.encoders import GMEQwen2VLEncoder

        torch.set_num_threads(4)
        faiss.omp_set_num_threads(4)
        os.chdir(target)
        app = create_app(
            working_dir=target,
            model_path=GME,
            dataset="E-VQA",
            subset=report.get("subset"),
            encoder_factory=(lambda *a, **kw: GMEQwen2VLEncoder(
                GME,
                batch_size=8,
            )) if args.runtime_encoder == "gme" else None,
            reload_interval=0,
        )
        status = app.state.mm_api.status()
        if status.get("status") != "ready":
            raise RuntimeError(f"graph-edit copy is not ready: {status.get('blockers')}")
        if args.runtime_encoder == "cached" and not status.get("bge_graph_index", {}).get("loaded"):
            raise RuntimeError("cached mode requires a ready BGE graph index")
        service = {
            "pid": os.getpid(),
            "host": "127.0.0.1",
            "port": args.port,
            "device": "cpu",
            "cpu_threads": 4,
            "runtime_encoder": args.runtime_encoder,
            "read_only": False,
            "working_dir": str(target),
            "base_working_dir": metadata["base_output_dir"],
            "graph_llm": owner["graph_llm"],
            "status": status,
        }
        atomic_json(target.parent / "graph_edit_service_status.json", service)
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=args.port,
            workers=1,
            access_log=False,
        )


if __name__ == "__main__":
    main()
