#!/usr/bin/env python3
"""Cache label-free fused query vectors for the 16-row held-out GRPO eval."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time

os.environ.setdefault("MM_EMBED_DEVICE", "cuda")
os.environ.setdefault("EVOGRAPH_MM_EMBED_RUNTIME_DEVICE", "cuda")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd

from evograph_mm.kb.gldv2_subset import atomic_json
from evograph_mm.kb.indexing.encoders import (
    FUSED_DOCUMENT_INSTRUCTION,
    GMEQwen2VLEncoder,
)


DATA_ROOT = Path("/home/data/dataset/wjz/EvoGraph-R1")
GRAPH = DATA_ROOT / "expr_mm/evqa_api_strict_max_graph_full1891_v1/E-VQA"
DATASET = DATA_ROOT / "datasets_mm/E-VQA/processed/paper_grpo_stage1_64_16_v1/test.parquet"
MODEL = Path("/home/data/dataset/wjz/models/gme-Qwen2-VL-2B-Instruct")
OUTPUT = GRAPH / "mm_store/query_fused"


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npy", dir=path.parent)
    os.close(fd)
    temp = Path(name)
    try:
        with temp.open("wb") as stream:
            np.save(stream, array)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def main() -> None:
    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if len(visible) != 1:
        raise RuntimeError("query cache build must be pinned to exactly one approved GPU")

    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the pinned cache process must see exactly one CUDA device")
    free_bytes, _ = torch.cuda.mem_get_info(0)
    if free_bytes < 18 * 1024**3:
        raise RuntimeError("the selected GPU has less than 18 GiB free")

    frame = pd.read_parquet(DATASET)
    if len(frame) != 16:
        raise RuntimeError(f"expected 16 held-out rows, got {len(frame)}")

    records = []
    items = []
    for position, row in frame.iterrows():
        image_id = str(row["image_id"])
        image_path = str(row["image_path"])
        question = str(row["extra_info"]["question"])
        if not Path(image_path).is_file():
            raise RuntimeError(f"missing eval image: {image_path}")
        records.append({
            "embedding_index": position,
            "image_id": image_id,
            "question": question,
            "image_path": image_path,
            "label_fields_used": [],
        })
        items.append({"text": question, "image": image_path})

    started = time.perf_counter()
    encoder = GMEQwen2VLEncoder(MODEL, batch_size=1)
    if encoder.device != "cuda":
        raise RuntimeError(f"unexpected embedding device: {encoder.device}")
    vectors = encoder.encode_fused(items, instruction=FUSED_DOCUMENT_INSTRUCTION)
    if vectors.shape != (16, 1536) or not np.isfinite(vectors).all():
        raise RuntimeError(f"invalid query cache matrix: {vectors.shape}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_npy(OUTPUT / "qa_fused_embeddings.npy", vectors.astype(np.float32, copy=False))
    records_text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    records_path = OUTPUT / "qa_fused_records.jsonl"
    fd, name = tempfile.mkstemp(prefix=f".{records_path.name}.", dir=OUTPUT)
    os.close(fd)
    temp = Path(name)
    try:
        temp.write_text(records_text, encoding="utf-8")
        os.replace(temp, records_path)
    finally:
        temp.unlink(missing_ok=True)

    report = {
        "status": "complete",
        "rows": 16,
        "dimension": 1536,
        "device": f"cuda:0 (physical GPU {visible[0]})",
        "model": str(MODEL),
        "dataset": str(DATASET),
        "instruction": FUSED_DOCUMENT_INSTRUCTION,
        "label_fields_used": [],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "cuda_peak_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 1024**2, 1),
    }
    atomic_json(OUTPUT / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
