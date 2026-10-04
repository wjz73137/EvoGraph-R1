#!/usr/bin/env python3
"""Deterministic full-corpus audit for the 1,891-row strict API extraction."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evograph_mm.kb.gldv2_subset import atomic_json
from evograph_mm.kb.strict_extraction import ENTITY_TYPES


ROOT = Path("/home/data/dataset/wjz/EvoGraph-R1/expr_mm/evqa_api_strict_max_extraction_full1891_v1")


def normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def main() -> None:
    source = json.loads((ROOT / "source_manifest.json").read_text())
    records = json.loads((ROOT / "strict_records.json").read_text())
    calls = json.loads((ROOT / "api_calls.json").read_text())
    report = json.loads((ROOT / "report.json").read_text())
    documents = source["documents"]
    by_id = {item["document_id"]: item for item in documents}
    errors = []
    warnings = []
    if source.get("available_train_row_count") != 1891:
        errors.append("source manifest does not cover all 1,891 available train rows")
    if source.get("row_association_count") != 1891:
        errors.append("row association count is not 1,891")
    if source.get("unique_image_count") != 1746:
        errors.append("unique image count is not 1,746")
    if len(documents) != 1237 or len(records) != 1237:
        errors.append(f"expected 1,237 documents/records, got {len(documents)}/{len(records)}")
    if set(by_id) != set(records):
        errors.append("record IDs differ from the frozen source manifest")
    if report.get("status") != "complete":
        errors.append("extraction report is not complete")

    facts = 0
    exact = 0
    mentions = 0
    duplicate_documents = []
    types_by_name = defaultdict(set)
    for document_id, record in records.items():
        document = by_id.get(document_id)
        if not document:
            continue
        if record.get("document") != document:
            errors.append(f"{document_id}: embedded document differs from manifest")
        if hashlib.sha256(document["contents"].encode()).hexdigest() != document["contents_sha256"]:
            errors.append(f"{document_id}: source checksum mismatch")
        seen = set()
        duplicated = False
        for index, fact in enumerate(record.get("facts") or []):
            facts += 1
            statement = fact.get("statement")
            evidence = fact.get("evidence")
            if not isinstance(statement, str) or not statement.strip():
                errors.append(f"{document_id}#{index}: empty statement")
                continue
            key = normalized(statement)
            if key in seen:
                duplicated = True
            seen.add(key)
            if isinstance(evidence, str) and evidence in document["contents"] and \
                    fact.get("evidence_is_exact_source_span") is True:
                exact += 1
            else:
                errors.append(f"{document_id}#{index}: evidence is not an exact source span")
            entities = fact.get("entities")
            if not isinstance(entities, list) or not entities:
                errors.append(f"{document_id}#{index}: no linked entity")
                continue
            local = set()
            for entity in entities:
                mentions += 1
                name = entity.get("name")
                entity_type = entity.get("type")
                if not isinstance(name, str) or name.casefold() not in statement.casefold():
                    errors.append(f"{document_id}#{index}: entity is not a statement substring")
                if entity_type not in ENTITY_TYPES:
                    errors.append(f"{document_id}#{index}: invalid entity type {entity_type!r}")
                pair = (str(name).casefold(), entity_type)
                if pair in local:
                    errors.append(f"{document_id}#{index}: duplicate entity {pair!r}")
                local.add(pair)
                types_by_name[str(name).casefold()].add(entity_type)
        if duplicated:
            duplicate_documents.append(document_id)
    if duplicate_documents:
        errors.append(f"{len(duplicate_documents)} documents contain duplicate facts")

    call_statuses = Counter(item.get("status") for item in calls.values())
    if call_statuses.get("requested") or set(call_statuses) - {"complete", "failed"}:
        errors.append(f"API call cache has incomplete entries: {dict(call_statuses)}")
    if call_statuses.get("failed"):
        warnings.append(
            f"{call_statuses['failed']} rejected API calls were retained as audit records; "
            "all affected documents have validated fallback records"
        )
    type_conflicts = {name: sorted(types) for name, types in types_by_name.items() if len(types) > 1}
    if type_conflicts:
        warnings.append(
            f"{len(type_conflicts)} cross-document entity labels have type conflicts; "
            "the graph builder resolves them by deterministic majority vote"
        )
    if facts != report.get("fact_count") or exact != facts:
        errors.append("recomputed fact/evidence count differs from the extraction report")
    if mentions != report.get("entity_mentions"):
        errors.append("recomputed entity mention count differs from the extraction report")

    audit = {
        "status": "pass" if not errors else "fail",
        "scope": "full deterministic structural, source-grounding, and provenance audit",
        "available_train_rows": source.get("available_train_row_count"),
        "row_associations": source.get("row_association_count"),
        "unique_images": source.get("unique_image_count"),
        "documents": len(records),
        "facts": facts,
        "exact_evidence": exact,
        "entity_mentions": mentions,
        "unique_normalized_entities": len(types_by_name),
        "cross_document_entity_type_conflicts": len(type_conflicts),
        "duplicate_fact_documents": duplicate_documents,
        "api_call_statuses": dict(call_statuses),
        "api_call_purposes": dict(Counter(
            item.get("purpose", "").split(":", 1)[0] for item in calls.values()
        )),
        "errors": errors,
        "warnings": warnings,
        "semantic_entailment_independently_verified": False,
    }
    atomic_json(ROOT / "deterministic_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
