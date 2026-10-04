#!/usr/bin/env python3
"""Create the prompt-corrected 64/16 E-VQA GRPO stage-one dataset."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import pandas as pd

from evograph_mm.kb.gldv2_subset import atomic_json, image_info
from evograph_mm.kb.vqa_prompt import build_vqa_user_prompt


DATA_ROOT = Path("/home/data/dataset/wjz/EvoGraph-R1")
SOURCE_ROOT = DATA_ROOT / "datasets_mm/E-VQA/processed/paper_64_16_seed0"
OUTPUT_ROOT = DATA_ROOT / "datasets_mm/E-VQA/processed/paper_grpo_stage1_64_16_v1"


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


def rewrite_prompt(row: pd.Series):
    extra = row["extra_info"]
    question = str(extra["question"])
    return [{"role": "user", "content": build_vqa_user_prompt(question=question)}]


def main() -> None:
    manifest = {
        "version": "evqa-grpo-stage1-64-16-v1",
        "purpose": "prompt-corrected multi-turn GRPO stage-one train/eval",
        "graph_service": "http://127.0.0.1:8005/search",
        "splits": {},
    }
    for split, expected in (("train", 64), ("test", 16)):
        source = SOURCE_ROOT / f"{split}.parquet"
        frame = pd.read_parquet(source)
        if len(frame) != expected:
            raise RuntimeError(f"expected {expected} {split} rows, got {len(frame)}")
        for position, row in frame.iterrows():
            if not image_info(Path(row["image_path"])):
                raise RuntimeError(f"{split} row {position} has an invalid image")
            ground_truth = row["reward_model"].get("ground_truth")
            if ground_truth is None or len(ground_truth) == 0:
                raise RuntimeError(f"{split} row {position} lacks ground truth")
        frame = frame.copy()
        frame["prompt"] = frame.apply(rewrite_prompt, axis=1)
        target = OUTPUT_ROOT / f"{split}.parquet"
        write_parquet_atomic(frame, target)
        manifest["splits"][split] = {
            "rows": len(frame),
            "source": str(source),
            "source_sha256": sha256(source),
            "output": str(target),
            "output_sha256": sha256(target),
        }
    atomic_json(OUTPUT_ROOT / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
