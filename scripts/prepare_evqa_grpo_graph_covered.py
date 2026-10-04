#!/usr/bin/env python3
"""Prepare deterministic 64/16 GRPO splits whose answers exist in the graph."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import re
import tempfile

import pandas as pd

from evograph_mm.kb.gldv2_subset import atomic_json, image_info
from evograph_mm.kb.schema import build_multimodal_qa_record, parse_e_vqa_answers


DATA_ROOT = Path("/home/data/dataset/wjz/EvoGraph-R1")
SUBSET = "E-VQA-GLDv2-1898-61-seed0"
SUBSET_ROOT = DATA_ROOT / f"datasets_mm/E-VQA/subsets/{SUBSET}"
EXTRACTION = DATA_ROOT / "expr_mm/evqa_api_strict_max_extraction_full1891_v1"
OUTPUT = DATA_ROOT / "datasets_mm/E-VQA/processed/paper_grpo_graph_covered_64_16_v1"


def normalized(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def write_parquet_atomic(frame: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".parquet", dir=target.parent)
    os.close(fd)
    temp = Path(name)
    try:
        frame.to_parquet(temp, index=False)
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)


def main() -> None:
    rows = list(csv.DictReader((SUBSET_ROOT / "qa_train.csv").open(newline="", encoding="utf-8")))
    source = json.loads((EXTRACTION / "source_manifest.json").read_text())
    strict = json.loads((EXTRACTION / "strict_records.json").read_text())
    semantic = json.loads((EXTRACTION / "semantic_audit_report.json").read_text())
    excluded = set(semantic.get("excluded_sample_ids") or [])
    associations = source["row_associations"]
    documents = {item["document_id"]: item for item in source["documents"]}
    if len(rows) != 1891 or len(associations) != 1891:
        raise RuntimeError("full graph source rows are not the expected 1,891")

    eligible = []
    for row_position, (row, association) in enumerate(zip(rows, associations)):
        document_id = association["document_id"]
        facts = [
            fact
            for fact_index, fact in enumerate(strict[document_id]["facts"])
            if f"{document_id}#{fact_index}" not in excluded
        ]
        fact_text = normalized(
            " ".join(
                str(fact.get(field, ""))
                for fact in facts
                for field in ("statement", "evidence")
            )
        )
        answers = parse_e_vqa_answers(row.get("answer"))
        matched_answers = [
            answer for answer in answers
            if normalized(answer) and normalized(answer) in fact_text
        ]
        if matched_answers:
            eligible.append({
                "row_position": row_position,
                "row": row,
                "document_id": document_id,
                "image_id": association["image_id"],
                "matched_answers": matched_answers,
            })

    selected = {"train": [], "test": []}
    used_documents: set[str] = set()
    used_images: set[str] = set()
    for split, count in (("train", 64), ("test", 16)):
        for item in eligible:
            if item["document_id"] in used_documents or item["image_id"] in used_images:
                continue
            selected[split].append(item)
            used_documents.add(item["document_id"])
            used_images.add(item["image_id"])
            if len(selected[split]) == count:
                break
        if len(selected[split]) != count:
            raise RuntimeError(f"could only select {len(selected[split])}/{count} {split} rows")

    manifest_splits = {}
    for split, items in selected.items():
        records = []
        provenance = []
        for item in items:
            row = item["row"]
            row_position = item["row_position"]
            document = documents[item["document_id"]]
            image_id = item["image_id"]
            image_path = next(
                path for candidate_id, path in zip(document["image_ids"], document["image_paths"])
                if candidate_id == image_id
            )
            if not image_info(Path(image_path)):
                raise RuntimeError(f"invalid graph-covered image: {image_path}")
            answers = parse_e_vqa_answers(row.get("answer"))
            record = build_multimodal_qa_record(
                data_source="E-VQA/paper_grpo_graph_covered_64_16_v1",
                data_id=f"E-VQA:graph-covered:{split}:{row_position}:{image_id}",
                dataset="E-VQA",
                split=split,
                index=row_position,
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
                row_index=row_position,
                source_row=row,
                context=[row.get("wikipedia_title", ""), row.get("wikipedia_url", ""), row.get("evidence_section_title", "")],
                original_metadata={
                    "graph_document_id": item["document_id"],
                    "selection": "answer_exactly_present_in_semantically_accepted_graph_fact",
                },
            )
            records.append(record)
            provenance.append({
                "row_position": row_position,
                "document_id": item["document_id"],
                "image_id": image_id,
                "matched_answers": item["matched_answers"],
            })
        frame = pd.DataFrame(records)
        write_parquet_atomic(frame, OUTPUT / f"{split}.parquet")
        manifest_splits[split] = provenance

    manifest = {
        "version": "evqa-grpo-graph-covered-64-16-v1",
        "source_subset": SUBSET,
        "eligible_rows": len(eligible),
        "eligibility_policy": (
            "at least one normalized gold answer occurs in a semantically accepted "
            "strict graph fact statement or exact evidence span"
        ),
        "split_policy": (
            "CSV order; unique image and graph document across both splits; "
            "64 train then 16 validation"
        ),
        "label_use": "selection and reward only; labels are not included in prompts or retrieval requests",
        "splits": manifest_splits,
    }
    atomic_json(OUTPUT / "manifest.json", manifest)
    print(json.dumps({
        "output": str(OUTPUT),
        "eligible_rows": len(eligible),
        "train_rows": len(selected["train"]),
        "validation_rows": len(selected["test"]),
        "train_documents": len({item["document_id"] for item in selected["train"]}),
        "validation_documents": len({item["document_id"] for item in selected["test"]}),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
