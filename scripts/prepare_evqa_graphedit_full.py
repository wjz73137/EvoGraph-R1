#!/usr/bin/env python3
"""Prepare the full E-VQA GraphEdit stage with isolated fixed validation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

import pandas as pd

from evograph_mm.kb.gldv2_subset import atomic_json, image_info
from evograph_mm.kb.schema import build_multimodal_qa_record, parse_e_vqa_answers
from evograph_mm.kb.vqa_prompt import build_graph_edit_user_prompt


DATA_ROOT = Path("/home/data/dataset/wjz/EvoGraph-R1")
SUBSET = "E-VQA-GLDv2-1898-61-seed0"
SUBSET_ROOT = DATA_ROOT / f"datasets_mm/E-VQA/subsets/{SUBSET}"
SOURCE_MANIFEST = (
    DATA_ROOT
    / "expr_mm/evqa_api_strict_max_extraction_full1891_v1/source_manifest.json"
)
VALIDATION_MANIFEST = (
    DATA_ROOT
    / "datasets_mm/E-VQA/processed/paper_grpo_graph_covered_full_v1/manifest.json"
)
OUTPUT = DATA_ROOT / "datasets_mm/E-VQA/processed/paper_grpo_graph_edit_full1891_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_parquet_atomic(frame: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".parquet", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite existing output: {OUTPUT}")

    rows = list(
        csv.DictReader(
            (SUBSET_ROOT / "qa_train.csv").open(newline="", encoding="utf-8")
        )
    )
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    validation = json.loads(VALIDATION_MANIFEST.read_text(encoding="utf-8"))
    associations = source["row_associations"]
    documents = {item["document_id"]: item for item in source["documents"]}
    if len(rows) != 1891 or len(associations) != 1891:
        raise RuntimeError("full E-VQA source is not the expected 1,891 rows")

    validation_positions = {
        int(item["row_position"])
        for item in validation["splits"]["test"]["records"]
    }
    validation_documents = {
        associations[position]["document_id"] for position in validation_positions
    }
    validation_images = {
        associations[position]["image_id"] for position in validation_positions
    }
    train_positions = [
        position
        for position, association in enumerate(associations)
        if position not in validation_positions
        and association["document_id"] not in validation_documents
        and association["image_id"] not in validation_images
    ]
    removed_positions = [
        position
        for position, association in enumerate(associations)
        if position not in validation_positions
        and (
            association["document_id"] in validation_documents
            or association["image_id"] in validation_images
        )
    ]
    if len(train_positions) % 2:
        raise RuntimeError("full GraphEdit train split must be divisible by batch size two")

    split_positions = {
        "train": train_positions,
        "test": sorted(validation_positions),
    }
    split_manifest: dict[str, dict[str, object]] = {}
    for split, positions in split_positions.items():
        records = []
        provenance = []
        for position in positions:
            row = rows[position]
            association = associations[position]
            document = documents[association["document_id"]]
            image_id = association["image_id"]
            image_path = next(
                path
                for candidate_id, path in zip(
                    document["image_ids"], document["image_paths"]
                )
                if candidate_id == image_id
            )
            if not image_info(Path(image_path)):
                raise RuntimeError(f"invalid image for source row {position}: {image_path}")
            answers = parse_e_vqa_answers(row.get("answer"))
            record = build_multimodal_qa_record(
                data_source="E-VQA/paper_grpo_graph_edit_full1891_v1",
                data_id=f"E-VQA:graph-edit-full:{split}:{position}:{image_id}",
                dataset="E-VQA",
                split=split,
                index=position,
                question=row["question"],
                question_original=row.get("question_original") or row["question"],
                answers=answers,
                raw_answer=row.get("answer"),
                image_id=image_id,
                image_path=image_path,
                image_missing=False,
                subset_root=str(SUBSET_ROOT),
                source_path=image_path,
                source_csv="qa_train.csv",
                row_index=position,
                source_row=row,
                context=[
                    row.get("wikipedia_title", ""),
                    row.get("wikipedia_url", ""),
                    row.get("evidence_section_title", ""),
                ],
                original_metadata={
                    "graph_document_id": association["document_id"],
                    "selection": "full_available_e_vqa_with_validation_isolation",
                    "expected_graph_state": (
                        "natural_api_graph" if split == "train" else "controlled_validation"
                    ),
                },
            )
            record["prompt"] = [
                {
                    "role": "user",
                    "content": build_graph_edit_user_prompt(question=row["question"]),
                }
            ]
            records.append(record)
            provenance.append(
                {
                    "row_position": position,
                    "document_id": association["document_id"],
                    "image_id": image_id,
                }
            )

        target = OUTPUT / f"{split}.parquet"
        write_parquet_atomic(pd.DataFrame(records), target)
        split_manifest[split] = {
            "rows": len(records),
            "unique_documents": len({item["document_id"] for item in provenance}),
            "unique_images": len({item["image_id"] for item in provenance}),
            "parquet": str(target),
            "sha256": sha256(target),
            "records": provenance,
        }

    payload = {
        "version": "evqa-graphedit-full1891-v1",
        "source_subset": SUBSET,
        "source_rows": len(rows),
        "split_policy": (
            "fixed 16-row graph-covered validation; exclude every training row sharing "
            "its graph document or image"
        ),
        "runtime_label_policy": (
            "gold answers are used only by the reward; prompts, retrieval, websearch, "
            "and graph-edit commit validation do not receive them"
        ),
        "train_graph_policy": "unaltered immutable API graph copied per service shard",
        "validation_graph_policy": "eight audited facts hidden in isolated service copies",
        "removed_for_validation_isolation": len(removed_positions),
        "removed_row_positions": removed_positions,
        "splits": split_manifest,
    }
    atomic_json(OUTPUT / "manifest.json", payload)
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "train_rows": len(train_positions),
                "validation_rows": len(validation_positions),
                "removed_for_validation_isolation": len(removed_positions),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
