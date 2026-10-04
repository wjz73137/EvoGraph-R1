#!/usr/bin/env python3
"""Prepare an isolated, auditable missing-fact GraphEdit benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from evograph_mm.kb.graph_edit import (
    apply_controlled_hyperedge_hide,
    prepare_edit_working_dir,
    rebuild_graph_text_indexes,
    rebuild_root_text_indexes,
)


ROOT = Path("/home/data/dataset/wjz/EvoGraph-R1")
BASE = ROOT / "expr_mm/evqa_api_strict_max_graph_full1891_v1/E-VQA"
SOURCE_DATA = ROOT / "datasets_mm/E-VQA/processed/paper_grpo_graph_edit_v1/test.parquet"
DEFAULT_OUTPUT = ROOT / "expr_mm/evqa_graphedit_controlled_missing_v1"

# These facts are unique answer-supporting rows in the 16-row held-out seed.
# Keeping the mapping explicit makes the perturbation auditable and reproducible.
SPECS = {
    0: ("rel-892cdad773433cf3975863c20a49cde0", "modern"),
    6: ("rel-0607d7c92211728ecd3cac83bbaa8670", "China"),
    7: ("rel-cc6a9b45b811b2bb0e98353b8edb53f4", "the main building"),
    9: ("rel-a7bc6e396620f03bd272fb1246cd7445", "Arabic"),
    10: ("rel-9310a450895bc37de3e95afccb558d68", "holy"),
    12: ("rel-88664a6e7dc5b2250b45c8a85efdc80b", "Nockamixon Cliffs"),
    13: ("rel-f0e4ac5a8100149b49e7834525096684", "five"),
    15: ("rel-c944005434955d1ec05377965a0632f4", "Finland"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_parquet(source: pd.DataFrame, rows: list[int], target: Path, split: str, manifest: dict):
    frame = source.iloc[rows].copy()
    frame["_controlled_source_row"] = rows
    frame = frame.reset_index(drop=True)
    data_source = "E-VQA/paper_grpo_graph_edit_controlled_missing_v1"
    for index in range(len(frame)):
        extra = dict(frame.at[index, "extra_info"])
        spec = manifest[str(frame.at[index, "_controlled_source_row"])]
        extra.update(
            {
                "controlled_perturbation": "missing_fact",
                "controlled_perturbation_id": spec["perturbation_id"],
                "hidden_hyperedge_id": spec["hyperedge_id"],
                "hidden_fact": spec["fact"],
                "split": split,
            }
        )
        frame.at[index, "extra_info"] = extra
        frame.at[index, "data_source"] = data_source
        frame.at[index, "data_id"] = (
            f"{data_source}:{split}:{index}:{extra.get('image_id', '')}"
        )
    frame = frame.drop(columns=["_controlled_source_row"])
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    if not BASE.is_dir() or not SOURCE_DATA.is_file():
        raise SystemExit("approved full graph or source parquet is missing")

    target = output / "E-VQA"
    prepare_edit_working_dir(BASE, target)
    hyperedges_before = json.loads((BASE / "kv_store_hyperedges.json").read_text())
    source = pd.read_parquet(SOURCE_DATA)
    manifest: dict[str, dict] = {}
    for row_index, (hyperedge_id, fact) in SPECS.items():
        if row_index >= len(source):
            raise SystemExit(f"source row is missing: {row_index}")
        record = hyperedges_before.get(hyperedge_id)
        if not isinstance(record, dict):
            raise SystemExit(f"hyperedge is missing: {hyperedge_id}")
        content = str(record.get("content") or record.get("hyperedge_name") or "")
        if fact.casefold() not in content.casefold():
            raise SystemExit(f"answer is not present in {hyperedge_id}: {fact}")
        perturbation_id = f"missing-test-{row_index:03d}"
        apply_controlled_hyperedge_hide(
            working_dir=target,
            target_id=hyperedge_id,
            perturbation_id=perturbation_id,
        )
        manifest[str(row_index)] = {
            "row_index": row_index,
            "hyperedge_id": hyperedge_id,
            "fact": fact,
            "content": content,
            "perturbation_id": perturbation_id,
        }

    rebuild_root_text_indexes(working_dir=target)
    rebuild_graph_text_indexes(
        working_dir=target,
        include_entities=False,
        include_hyperedges=True,
        previous_hyperedges=hyperedges_before,
    )

    output_data = output / "datasets"
    output_data.mkdir(parents=True, exist_ok=True)
    rows = sorted(SPECS)
    write_parquet(source, rows[:6], output_data / "train.parquet", "train", manifest)
    write_parquet(source, rows[6:], output_data / "test.parquet", "test", manifest)

    hidden = json.loads((target / "kv_store_hyperedges.json").read_text())
    manifest_payload = {
        "version": 1,
        "base_working_dir": str(BASE),
        "working_dir": str(target),
        "source_parquet": str(SOURCE_DATA),
        "base_hyperedges_sha256": sha256(BASE / "kv_store_hyperedges.json"),
        "controlled_hyperedges": manifest,
        "hidden_count": sum(
            1
            for item in hidden.values()
            if isinstance(item, dict) and item.get("searchable", True) is False
        ),
        "dataset_files": {
            "train": str(output_data / "train.parquet"),
            "test": str(output_data / "test.parquet"),
        },
    }
    (output / "controlled_missing_manifest.json").write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "README.txt").write_text(
        "This is an isolated controlled-missing GraphEdit benchmark.\n"
        "The approved API graph is never modified.\n"
        "Each hidden fact remains in KV provenance with searchable=false and is\n"
        "excluded from root FAISS, BGE graph FAISS, and lookup sidecars.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
