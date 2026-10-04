#!/usr/bin/env python3
"""Deterministically audit the 63-document strict-Max extraction artifact."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evograph_mm.kb.gldv2_subset import atomic_json
from evograph_mm.kb.strict_extraction import ENTITY_TYPES


DATA_ROOT = Path("/home/data/dataset/wjz/EvoGraph-R1")
OUTPUT_ROOT = DATA_ROOT / "expr_mm/evqa_api_strict_max_extraction_63_v1"
QA_TRAIN = DATA_ROOT / "datasets_mm/E-VQA/subsets/paper_64_16_seed0/qa_train.csv"


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def main() -> None:
    source_manifest = json.loads((OUTPUT_ROOT / "source_manifest.json").read_text())
    records = json.loads((OUTPUT_ROOT / "strict_records.json").read_text())
    calls = json.loads((OUTPUT_ROOT / "api_calls.json").read_text())
    run_report = json.loads((OUTPUT_ROOT / "report.json").read_text())

    errors: list[str] = []
    warnings: list[str] = []
    documents = source_manifest.get("documents", [])
    if len(documents) != 63 or len(records) != 63:
        errors.append(f"expected 63 source documents and records, got {len(documents)}/{len(records)}")
    if run_report.get("status") != "complete":
        errors.append("batch report is not complete")

    source_by_id = {item["document_id"]: item for item in documents}
    if set(source_by_id) != set(records):
        errors.append("record document IDs differ from the frozen source manifest")

    fact_count = 0
    entity_mentions = 0
    exact_evidence = 0
    per_document_duplicates: dict[str, list[str]] = {}
    entity_types: dict[str, set[str]] = defaultdict(set)
    entity_spellings: dict[str, set[str]] = defaultdict(set)
    for document_id, record in records.items():
        source_doc = source_by_id.get(document_id)
        if source_doc is None:
            continue
        if record.get("document") != source_doc:
            errors.append(f"{document_id}: embedded document metadata differs from manifest")
        source = source_doc["contents"]
        if text_sha256(source) != source_doc["contents_sha256"]:
            errors.append(f"{document_id}: source text checksum mismatch")
        seen_statements: set[str] = set()
        duplicates: list[str] = []
        facts = record.get("facts")
        if not isinstance(facts, list) or not facts:
            errors.append(f"{document_id}: missing facts")
            continue
        for index, fact in enumerate(facts):
            fact_count += 1
            statement = fact.get("statement")
            evidence = fact.get("evidence")
            if not isinstance(statement, str) or not statement.strip():
                errors.append(f"{document_id} fact {index}: empty statement")
                continue
            key = normalized(statement)
            if key in seen_statements:
                duplicates.append(statement)
            seen_statements.add(key)
            if not isinstance(evidence, str) or evidence not in source:
                errors.append(f"{document_id} fact {index}: evidence is not an exact source span")
            else:
                exact_evidence += 1
            if fact.get("evidence_is_exact_source_span") is not True:
                errors.append(f"{document_id} fact {index}: exact-evidence flag is not true")
            entities = fact.get("entities")
            if not isinstance(entities, list) or not entities:
                errors.append(f"{document_id} fact {index}: no linked entities")
                continue
            local_entities: set[tuple[str, str]] = set()
            for entity in entities:
                entity_mentions += 1
                name, entity_type = entity.get("name"), entity.get("type")
                if not isinstance(name, str) or name.casefold() not in statement.casefold():
                    errors.append(f"{document_id} fact {index}: non-substring entity {name!r}")
                    continue
                if entity_type not in ENTITY_TYPES:
                    errors.append(f"{document_id} fact {index}: invalid entity type {entity_type!r}")
                    continue
                entity_key = (name.casefold(), entity_type)
                if entity_key in local_entities:
                    errors.append(f"{document_id} fact {index}: duplicate entity {entity_key!r}")
                local_entities.add(entity_key)
                entity_types[name.casefold()].add(entity_type)
                entity_spellings[name.casefold()].add(name)
        if duplicates:
            per_document_duplicates[document_id] = duplicates

    if per_document_duplicates:
        errors.append(f"{len(per_document_duplicates)} documents contain normalized duplicate facts")

    conflicts = []
    for name, types in sorted(entity_types.items()):
        if len(types) > 1:
            conflicts.append({
                "normalized_name": name,
                "spellings": sorted(entity_spellings[name]),
                "types": sorted(types),
            })
    if conflicts:
        warnings.append(f"{len(conflicts)} normalized entity names have multiple types")

    call_statuses = Counter(call.get("status", "missing") for call in calls.values())
    incomplete_calls = [{"digest": digest, "purpose": call.get("purpose"),
                         "status": call.get("status")}
                        for digest, call in calls.items() if call.get("status") != "complete"]
    if len(incomplete_calls) != 1 or incomplete_calls[0]["status"] != "requested" or not (
        incomplete_calls[0]["purpose"] or ""
    ).startswith("anchor_facts:"):
        errors.append("unexpected incomplete API-call records")
    elif any(calls[incomplete_calls[0]["digest"]].get(field)
             for field in ("output", "usage", "finish_reason")):
        errors.append("known gateway-rejected call unexpectedly contains generated output")
    else:
        warnings.append("one anchor request was rejected by the gateway before generation and is retained")

    transmitted = "\n".join(
        str(message.get("content", ""))
        for call in calls.values() for message in call.get("messages", [])
    ).casefold()
    leaked_questions = []
    with QA_TRAIN.open(newline="") as stream:
        for row_number, row in enumerate(csv.DictReader(stream), start=2):
            for field in ("question_original", "question"):
                question = (row.get(field) or "").strip()
                if len(question) >= 12 and question.casefold() in transmitted:
                    leaked_questions.append({"row": row_number, "field": field})
    if leaked_questions:
        errors.append(f"{len(leaked_questions)} QA question strings appear in transmitted messages")

    if fact_count != run_report.get("fact_count"):
        errors.append("recomputed fact count differs from run report")
    if exact_evidence != fact_count:
        errors.append("not every fact has exact evidence")
    if entity_mentions != run_report.get("entity_mentions"):
        errors.append("recomputed entity-mention count differs from run report")

    purpose_counts = Counter(call.get("purpose", "").split(":", 1)[0] for call in calls.values())
    audit = {
        "status": "pass" if not errors else "fail",
        "scope": "deterministic full-corpus structural and provenance audit",
        "documents": len(records),
        "facts": fact_count,
        "exact_evidence": exact_evidence,
        "entity_mentions": entity_mentions,
        "unique_normalized_entities": len(entity_types),
        "entity_type_conflict_count": len(conflicts),
        "entity_type_conflicts": conflicts,
        "duplicate_fact_documents": per_document_duplicates,
        "api_call_statuses": dict(call_statuses),
        "api_call_purposes": dict(purpose_counts),
        "incomplete_calls": incomplete_calls,
        "qa_question_strings_transmitted": leaked_questions,
        "errors": errors,
        "warnings": warnings,
        "semantic_entailment_independently_verified": False,
    }
    atomic_json(OUTPUT_ROOT / "deterministic_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
