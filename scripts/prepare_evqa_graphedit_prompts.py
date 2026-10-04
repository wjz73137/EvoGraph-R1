#!/usr/bin/env python3
"""Create an E-VQA parquet variant with GraphEdit-compatible user prompts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evograph_mm.kb.vqa_prompt import build_graph_edit_user_prompt


def _rewrite_split(source: Path, destination: Path, subset: str) -> int:
    table = pq.read_table(source)
    rows = table.to_pylist()
    for row in rows:
        extra = row.get("extra_info") or {}
        question = str(extra.get("question") or "").strip()
        if not question:
            raise ValueError(f"missing question in {source}")
        row["prompt"] = [
            {"role": "user", "content": build_graph_edit_user_prompt(question=question)}
        ]
        row["data_source"] = f"E-VQA/{subset}"
        data_id = str(row.get("data_id") or "")
        if data_id:
            parts = data_id.split(":")
            if len(parts) >= 3:
                parts[1] = subset
                row["data_id"] = ":".join(parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rewritten = pa.Table.from_pylist(rows, schema=table.schema)
    pq.write_table(rewritten, destination, compression="zstd")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subset", required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"output-dir already exists: {args.output_dir}")
    counts = {
        split: _rewrite_split(
            args.source_dir / f"{split}.parquet",
            args.output_dir / f"{split}.parquet",
            args.subset,
        )
        for split in ("train", "test")
    }
    print({"subset": args.subset, **counts})


if __name__ == "__main__":
    main()
