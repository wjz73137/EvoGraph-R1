#!/usr/bin/env python3
"""Materialize source-validated Max facts and entity links for the pinned E-VQA sample."""
from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evograph_mm.kb.gldv2_subset import atomic_json
from evograph_mm.kb.strict_extraction import (
    ENTITY_LINK_SYSTEM,
    EVIDENCE_REPAIR_SYSTEM,
    STRICT_FACT_SYSTEM,
    apply_evidence_repairs,
    invalid_evidence_indexes,
    parse_json_object,
    validate_entity_links,
    validate_facts,
)


PROJECT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/home/data/dataset/wjz/EvoGraph-R1")
QUALITY_ROOT = DATA_ROOT / "expr_mm/evqa_api_extraction_quality_v1"
SOURCE_SNAPSHOT = DATA_ROOT / "expr_mm/evqa_api_graph_baseline/source_snapshot.json"
OUTPUT_ROOT = DATA_ROOT / "expr_mm/evqa_api_strict_max_extraction_v2"
MAX_CALLS = 9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prompt_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def main() -> None:
    from dotenv import dotenv_values
    config = dotenv_values(PROJECT / ".env")
    key, url = config.get("OPENAI_API_KEY"), config.get("OPENAI_BASE_URL")
    model = config.get("STRICT_EXTRACTION_MODEL") or config.get("QUALITY_JUDGE_MODEL")
    if not all((key, url, model)):
        raise RuntimeError(".env lacks API key, base URL, or strict/quality model")

    quality_report = json.loads((QUALITY_ROOT / "report.json").read_text())
    if quality_report.get("status") != "complete" or quality_report.get("report_version") != 4:
        raise RuntimeError("the pinned extraction comparison is not complete")
    if quality_report.get("comparison_extraction_model") != model:
        raise RuntimeError("configured strict extraction model differs from the evaluated Max model")
    if quality_report["scores"]["grounded_atomic_max"]["overall"]["unsupported_count"] != 0:
        raise RuntimeError("evaluated Max arm contains unsupported facts")
    if sha256(SOURCE_SNAPSHOT) != quality_report["source_snapshot_sha256"]:
        raise RuntimeError("source snapshot changed since evaluation")

    snapshot = json.loads(SOURCE_SNAPSHOT.read_text())
    sources = {
        item["source_metadata"]["wikipedia_title"]: item
        for item in snapshot["documents"]
    }
    evaluated = json.loads((QUALITY_ROOT / "arm_outputs.json").read_text())[
        "grounded_atomic_max"
    ]
    if set(evaluated) != set(sources):
        raise RuntimeError("evaluated output/source title mismatch")

    owner = {
        "pipeline": "strict-source-grounded-max-v2",
        "model": f"api/{model}",
        "source_snapshot_sha256": sha256(SOURCE_SNAPSHOT),
        "quality_report_sha256": sha256(QUALITY_ROOT / "report.json"),
        "evaluated_outputs_sha256": sha256(QUALITY_ROOT / "arm_outputs.json"),
        "fact_prompt_sha256": prompt_sha256(STRICT_FACT_SYSTEM),
        "evidence_repair_prompt_sha256": prompt_sha256(EVIDENCE_REPAIR_SYSTEM),
        "entity_link_prompt_sha256": prompt_sha256(ENTITY_LINK_SYSTEM),
        "facts_imported_from_evaluated_max_arm": True,
        "local_model_loaded": False,
        "gpu_used": False,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    owner_file = OUTPUT_ROOT / "owner.json"
    calls_file = OUTPUT_ROOT / "api_calls.json"
    records_file = OUTPUT_ROOT / "strict_records.json"
    report_file = OUTPUT_ROOT / "report.json"
    with (OUTPUT_ROOT / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if owner_file.exists() and json.loads(owner_file.read_text()) != owner:
            raise RuntimeError("existing strict extraction output has a different owner; retained")
        atomic_json(owner_file, owner)
        if report_file.exists():
            report = json.loads(report_file.read_text())
            if report.get("status") == "complete":
                print(json.dumps(report, ensure_ascii=False))
                return
        calls = json.loads(calls_file.read_text()) if calls_file.exists() else {}
        records = json.loads(records_file.read_text()) if records_file.exists() else {}
        report = {**owner, "status": "running", "training_started": False}
        atomic_json(report_file, report)

        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=url, timeout=180, max_retries=0)

        def call_api(purpose: str, messages: list[dict], max_tokens: int = 5000) -> dict:
            digest = hashlib.sha256(json.dumps(
                {"model": model, "messages": messages}, ensure_ascii=False, sort_keys=True
            ).encode()).hexdigest()
            cached = calls.get(digest)
            if cached and cached.get("status") == "complete":
                return parse_json_object(cached["output"])
            if cached:
                raise RuntimeError(f"incomplete retained API call requires review: {digest}")
            if len(calls) >= MAX_CALLS:
                raise RuntimeError("strict extraction API call budget reached")
            calls[digest] = {"purpose": purpose, "model": model, "messages": messages,
                             "status": "requested"}
            atomic_json(calls_file, calls)
            started = time.perf_counter()
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=0, max_tokens=max_tokens,
                response_format={"type": "json_object"},
                extra_body={"enable_thinking": False},
            )
            choice = response.choices[0]
            content = choice.message.content or ""
            calls[digest].update(
                status="complete", output=content, finish_reason=choice.finish_reason,
                usage=response.usage.model_dump() if response.usage else None,
                seconds=round(time.perf_counter() - started, 3),
            )
            atomic_json(calls_file, calls)
            if not content.strip() or choice.finish_reason == "length":
                raise RuntimeError(f"empty or truncated API response for {purpose}")
            print(json.dumps({"phase": purpose, "api_calls": len(calls)}, ensure_ascii=False),
                  flush=True)
            return parse_json_object(content)

        try:
            for title, document in sources.items():
                if title in records:
                    continue
                source = document["contents"]
                facts = validate_facts({"facts": evaluated[title]}, source, require_exact=False)
                bad_indexes = invalid_evidence_indexes(facts)
                if bad_indexes:
                    repair_input = {
                        "title": title,
                        "passage": source,
                        "items": [{"fact_index": index,
                                   "statement": facts[index]["statement"]}
                                  for index in bad_indexes],
                    }
                    repairs = call_api(
                        f"repair_evidence:{title}",
                        [{"role": "system", "content": EVIDENCE_REPAIR_SYSTEM},
                         {"role": "user", "content": json.dumps(repair_input, ensure_ascii=False)}],
                    )
                    facts = apply_evidence_repairs(facts, repairs, source)
                validate_facts({"facts": facts}, source, require_exact=True)
                link_input = {
                    "title": title,
                    "facts": [{"fact_index": index, "statement": fact["statement"]}
                              for index, fact in enumerate(facts)],
                }
                link_messages = [
                    {"role": "system", "content": ENTITY_LINK_SYSTEM},
                    {"role": "user", "content": json.dumps(link_input, ensure_ascii=False)},
                ]
                raw_links = call_api(f"link_entities:{title}", link_messages)
                try:
                    links = validate_entity_links(raw_links, facts)
                except ValueError as first_error:
                    retry_messages = link_messages + [
                        {"role": "assistant", "content": json.dumps(raw_links, ensure_ascii=False)},
                        {"role": "user", "content": (
                            "Your entity-link output failed deterministic validation: "
                            f"{first_error}. Return the complete corrected JSON object."
                        )},
                    ]
                    raw_links = call_api(f"link_entities_retry:{title}", retry_messages)
                    links = validate_entity_links(raw_links, facts)
                linked_facts = []
                for index, (fact, link) in enumerate(zip(facts, links)):
                    if link["fact_index"] != index:
                        raise RuntimeError("validated entity links lost fact order")
                    linked_facts.append({**fact, "entities": link["entities"]})
                records[title] = {
                    "text_doc_id": document["text_doc_id"],
                    "image_id": document["image_id"],
                    "source_metadata": document["source_metadata"],
                    "facts": linked_facts,
                }
                atomic_json(records_file, records)

            fact_count = sum(len(item["facts"]) for item in records.values())
            entity_mentions = sum(len(fact["entities"]) for item in records.values()
                                  for fact in item["facts"])
            unique_entities = {
                (entity["name"].casefold(), entity["type"])
                for item in records.values() for fact in item["facts"]
                for entity in fact["entities"]
            }
            usage = {field: sum((call.get("usage") or {}).get(field, 0) or 0
                                for call in calls.values())
                     for field in ("prompt_tokens", "completion_tokens", "total_tokens")}
            report.update(
                status="complete",
                documents=len(records),
                fact_count=fact_count,
                exact_evidence_count=sum(
                    fact["evidence_is_exact_source_span"]
                    for item in records.values() for fact in item["facts"]),
                evidence_repairs=sum(
                    1 for call in calls.values()
                    if call["purpose"].startswith("repair_evidence:")),
                entity_mentions=entity_mentions,
                unique_typed_entities=len(unique_entities),
                all_entity_names_are_fact_substrings=True,
                api_calls=len(calls),
                api_usage=usage,
                output=str(records_file),
                graph_built=False,
            )
            atomic_json(report_file, report)
            print(json.dumps(report, ensure_ascii=False), flush=True)
        except Exception as exc:
            report.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                          api_calls=len(calls))
            atomic_json(report_file, report)
            raise


if __name__ == "__main__":
    main()
