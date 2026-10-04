#!/usr/bin/env python3
"""Build a two-row GraphEdit smoke dataset from the audited held-out cases."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")

    source = pq.read_table(args.source)
    rows = source.to_pylist()
    expected = {
        "How many stars does this hotel have?",
        "In which country is this lake located?",
    }
    questions = {str((row.get("extra_info") or {}).get("question")) for row in rows}
    if len(rows) != 2 or questions != expected:
        raise SystemExit(f"unexpected targeted source rows: {questions}")

    train_rows = []
    for index, row in enumerate(rows):
        row = dict(row)
        extra = dict(row.get("extra_info") or {})
        extra["split"] = "train"
        extra["targeted_graph_edit_smoke"] = True
        row["extra_info"] = extra
        row["data_id"] = f"E-VQA/graphedit_targeted_smoke:train:{index}:{extra['image_id']}"
        row["data_source"] = "E-VQA/graphedit_targeted_smoke"
        train_rows.append(row)

    args.output.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(train_rows, schema=source.schema),
        args.output / "train.parquet",
        compression="zstd",
    )
    pq.write_table(source, args.output / "test.parquet", compression="zstd")
    print({"train": len(train_rows), "test": len(rows), "questions": sorted(questions)})


if __name__ == "__main__":
    main()
