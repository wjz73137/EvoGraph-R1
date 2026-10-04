#!/usr/bin/env python3
"""Prepare the full graph-covered E-VQA GRPO train split and fixed validation."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import pandas as pd

from evograph_mm.kb.gldv2_subset import atomic_json, image_info
from evograph_mm.kb.schema import build_multimodal_qa_record
from evograph_mm.kb.schema import parse_e_vqa_answers


DATA_ROOT = Path("/home/data/dataset/wjz/EvoGraph-R1")
SUBSET = "E-VQA-GLDv2-1898-61-seed0"
SUBSET_ROOT = DATA_ROOT / f"datasets_mm/E-VQA/subsets/{SUBSET}"
EXTRACTION = DATA_ROOT / "expr_mm/evqa_api_strict_max_extraction_full1891_v1"
REFERENCE = (
    DATA_ROOT
    / "datasets_mm/E-VQA/processed/paper_grpo_graph_covered_64_16_v1/manifest.json"
)
OUTPUT = DATA_ROOT / "datasets_mm/E-VQA/processed/paper_grpo_graph_covered_full_v1"


def normalized(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_parquet_atomic(frame: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".parquet", dir=target.parent
    )
    os.close(fd)
    temporary = Path(name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    rows = list(csv.DictReader((SUBSET_ROOT / "qa_train.csv").open(newline="", encoding="utf-8")))
    source = json.loads((EXTRACTION / "source_manifest.json").read_text())
    strict = json.loads((EXTRACTION / "strict_records.json").read_text())
    semantic = json.loads((EXTRACTION / "semantic_audit_report.json").read_text())
    reference = json.loads(REFERENCE.read_text())
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
            answer
            for answer in answers
            if normalized(answer) and normalized(answer) in fact_text
        ]
        if matched_answers:
            eligible.append(
                {
                    "row_position": row_position,
                    "row": row,
                    "document_id": document_id,
                    "image_id": association["image_id"],
                    "matched_answers": matched_answers,
                }
            )

    validation_positions = {
        int(item["row_position"]) for item in reference["splits"]["test"]
    }
    validation = [item for item in eligible if item["row_position"] in validation_positions]
    if len(validation) != 16:
        raise RuntimeError(f"expected the fixed 16 validation rows, found {len(validation)}")
    validation_documents = {item["document_id"] for item in validation}
    validation_images = {item["image_id"] for item in validation}
    train = [
        item
        for item in eligible
        if item["row_position"] not in validation_positions
        and item["document_id"] not in validation_documents
        and item["image_id"] not in validation_images
    ]

    manifest_splits = {}
    for split, items in (("train", train), ("test", validation)):
        records = []
        provenance = []
        for item in items:
            row = item["row"]
            document = documents[item["document_id"]]
            image_id = item["image_id"]
            image_path = next(
                path
                for candidate_id, path in zip(document["image_ids"], document["image_paths"])
                if candidate_id == image_id
            )
            if not image_info(Path(image_path)):
                raise RuntimeError(f"invalid graph-covered image: {image_path}")
            answers = parse_e_vqa_answers(row.get("answer"))
            records.append(
                build_multimodal_qa_record(
                    data_source="E-VQA/paper_grpo_graph_covered_full_v1",
                    data_id=f"E-VQA:graph-covered-full:{split}:{item['row_position']}:{image_id}",
                    dataset="E-VQA",
                    split=split,
                    index=item["row_position"],
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
                    row_index=item["row_position"],
                    source_row=row,
                    context=[
                        row.get("wikipedia_title", ""),
                        row.get("wikipedia_url", ""),
                        row.get("evidence_section_title", ""),
                    ],
                    original_metadata={
                        "graph_document_id": item["document_id"],
                        "selection": "answer_exactly_present_in_semantically_accepted_graph_fact",
                    },
                )
            )
            provenance.append(
                {
                    "row_position": item["row_position"],
                    "document_id": item["document_id"],
                    "image_id": image_id,
                    "matched_answers": item["matched_answers"],
                }
            )
        padding_rows = 0
        if split == "train" and len(records) % 2:
            # The trainer intentionally drops incomplete batches.  Add one
            # explicitly recorded duplicate so all 649 unique source rows are
            # consumed with the proven-stable global batch size of two.
            padded_record = json.loads(json.dumps(records[0]))
            padded_record["data_id"] = f"{padded_record['data_id']}:batch-padding"
            padded_record["extra_info"]["original_metadata"]["batch_padding"] = True
            records.append(padded_record)
            padded_provenance = dict(provenance[0])
            padded_provenance["batch_padding"] = True
            provenance.append(padded_provenance)
            padding_rows = 1
        target = OUTPUT / f"{split}.parquet"
        write_parquet_atomic(pd.DataFrame(records), target)
        manifest_splits[split] = {
            "rows": len(records),
            "unique_source_rows": len(items),
            "batch_padding_rows": padding_rows,
            "parquet": str(target),
            "sha256": sha256(target),
            "records": provenance,
        }

    removed_for_validation_isolation = len(eligible) - len(train) - len(validation)
    manifest = {
        "version": "evqa-grpo-graph-covered-full-v1",
        "source_subset": SUBSET,
        "eligible_rows": len(eligible),
        "eligibility_policy": (
            "at least one normalized gold answer occurs in a semantically accepted "
            "strict graph fact statement or exact evidence span"
        ),
        "split_policy": (
            "retain the fixed graph-covered 16-row validation split; train on every "
            "other eligible row after excluding validation document/image overlap"
        ),
        "removed_for_validation_isolation": removed_for_validation_isolation,
        "label_use": "selection and reward only; labels are absent from prompts and retrieval requests",
        "splits": manifest_splits,
    }
    atomic_json(OUTPUT / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "eligible_rows": len(eligible),
                "train_rows": len(train),
                "validation_rows": len(validation),
                "removed_for_validation_isolation": removed_for_validation_isolation,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
