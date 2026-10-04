#!/usr/bin/env python3
"""Create a four-row, data-disk-only GRPO smoke subset from audited examples."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import pandas as pd

from evograph_mm.kb.gldv2_subset import atomic_json, image_info


DATA_ROOT = Path("/home/data/dataset/wjz/EvoGraph-R1")
SOURCE = DATA_ROOT / "datasets_mm/E-VQA/processed/paper_64_16_seed0/train.parquet"
OUTPUT = DATA_ROOT / "datasets_mm/E-VQA/processed/paper_grpo_smoke63_v1"
TRAIN_ROWS = [1, 3]
VAL_ROWS = [9, 17]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_parquet_atomic(dataframe: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".parquet", dir=target.parent)
    os.close(fd)
    temp = Path(name)
    try:
        dataframe.to_parquet(temp, index=False)
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)


def main() -> None:
    dataframe = pd.read_parquet(SOURCE)
    if len(dataframe) != 64:
        raise RuntimeError(f"expected 64 source rows, got {len(dataframe)}")
    train = dataframe.iloc[TRAIN_ROWS].reset_index(drop=True)
    val = dataframe.iloc[VAL_ROWS].reset_index(drop=True)
    for split, frame in (("train", train), ("test", val)):
        for position, row in frame.iterrows():
            if not image_info(Path(row["image_path"])):
                raise RuntimeError(f"{split} row {position} has an invalid image")
            if not row["reward_model"].get("ground_truth"):
                raise RuntimeError(f"{split} row {position} lacks a ground truth")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_parquet_atomic(train, OUTPUT / "train.parquet")
    write_parquet_atomic(val, OUTPUT / "test.parquet")
    manifest = {
        "version": "evqa-grpo-smoke63-v1",
        "source": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "train_source_rows": TRAIN_ROWS,
        "validation_source_rows": VAL_ROWS,
        "train_rows": len(train),
        "validation_rows": len(val),
        "purpose": "one-step two-GPU GRPO integration smoke; not an accuracy experiment",
        "graph_service": "http://127.0.0.1:8005/search",
    }
    atomic_json(OUTPUT / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
