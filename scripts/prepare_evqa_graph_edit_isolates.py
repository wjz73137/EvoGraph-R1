#!/usr/bin/env python3
"""Create independent writable graph copies for GraphEdit trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evograph_mm.kb.graph_edit import prepare_edit_working_dir


ROOT = Path("/home/data/dataset/wjz/EvoGraph-R1")
DEFAULT_SOURCE = (
    ROOT / "expr_mm/evqa_graphedit_controlled_missing_v1/E-VQA"
)
DEFAULT_OUTPUT = (
    ROOT / "expr_mm/evqa_graphedit_controlled_missing_v1/isolated_smoke_v4"
)
COPY_NAMES = ("train_0", "train_1", "train_2", "train_3", "val_0", "val_1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--train-source",
        type=Path,
        help="optional graph source for train_0..train_3",
    )
    parser.add_argument(
        "--val-source",
        type=Path,
        help="optional graph source for val_0..val_1",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source = args.source.resolve()
    train_source = (args.train_source or source).resolve()
    val_source = (args.val_source or source).resolve()
    output = args.output.resolve()
    for split, split_source in (("train", train_source), ("val", val_source)):
        if not split_source.is_dir():
            raise SystemExit(f"{split} source graph is missing: {split_source}")
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)

    created = []
    try:
        for name in COPY_NAMES:
            copy_source = train_source if name.startswith("train_") else val_source
            target = prepare_edit_working_dir(copy_source, output / name)
            metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
            created.append(
                {
                    "name": name,
                    "source": str(copy_source),
                    "working_dir": str(target),
                    "output_dir": metadata.get("output_dir"),
                    "base_output_dir": metadata.get("base_output_dir"),
                }
            )
    except Exception:
        # Keep completed copies for diagnosis; the script never overwrites them.
        raise

    manifest = {
        "source": str(source),
        "train_source": str(train_source),
        "val_source": str(val_source),
        "output": str(output),
        "copies": created,
    }
    (output / "isolation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
