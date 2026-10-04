#!/usr/bin/env python3
"""Strict API extraction for all 1,237 available train Wikipedia articles."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evograph_mm.kb.gldv2_subset import atomic_json
from evograph_mm.kb.strict_extraction import (
    ENTITY_TYPES,
    ENTITY_LINK_SYSTEM,
    EVIDENCE_REPAIR_SYSTEM,
    FACT_ANCHOR_REPAIR_SYSTEM,
    STRICT_FACT_SYSTEM,
    apply_evidence_repairs,
    apply_fact_anchor_repairs,
    invalid_evidence_indexes,
    parse_json_object,
    validate_entity_links,
    validate_facts,
)


PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path("/home/data/dataset/wjz/EvoGraph-R1/expr_mm/evqa_api_strict_max_extraction_full1891_v1")
SOURCE_FILE = ROOT / "source_manifest.json"
MAX_CALLS = 6000
DEFAULT_WORKERS = 8
_thread_local = threading.local()
_count_lock = threading.Lock()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def main() -> None:
    from dotenv import dotenv_values
    from openai import BadRequestError, OpenAI

    config = dotenv_values(PROJECT / ".env")
    api_key = config.get("OPENAI_API_KEY")
    base_url = config.get("OPENAI_BASE_URL")
    model = config.get("STRICT_EXTRACTION_MODEL")
    if not all((api_key, base_url, model)):
        raise RuntimeError(".env lacks API key, base URL, or STRICT_EXTRACTION_MODEL")
    source_manifest = json.loads(SOURCE_FILE.read_text())
    documents = source_manifest["documents"]
    if source_manifest.get("status") != "complete" or len(documents) != 1237:
        raise RuntimeError("full source manifest is incomplete")

    workers = int(os.getenv("STRICT_API_WORKERS", str(DEFAULT_WORKERS)))
    if not 1 <= workers <= 16:
        raise RuntimeError("STRICT_API_WORKERS must be between 1 and 16")
    calls_dir = ROOT / "api_calls"
    records_dir = ROOT / "records"
    calls_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)
    report_file = ROOT / "report.json"
    owner_file = ROOT / "owner.json"
    owner = {
        "pipeline": "strict-source-grounded-max-full1891-v1",
        "model": f"api/{model}",
        "source_manifest_sha256": sha256(SOURCE_FILE),
        "fact_prompt_sha256": text_sha256(STRICT_FACT_SYSTEM),
        "evidence_repair_prompt_sha256": text_sha256(EVIDENCE_REPAIR_SYSTEM),
        "fact_anchor_prompt_sha256": text_sha256(FACT_ANCHOR_REPAIR_SYSTEM),
        "entity_link_prompt_sha256": text_sha256(ENTITY_LINK_SYSTEM),
        "no_qa_question_answer_or_evidence_sent_to_api": True,
        "local_model_loaded": False,
        "gpu_used": False,
        "api_workers": workers,
    }

    with (ROOT / ".run.lock").open("a") as run_lock:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if owner_file.exists():
            existing = json.loads(owner_file.read_text())
            # Worker count changes execution speed, not artifact semantics.
            if {k: v for k, v in existing.items() if k != "api_workers"} != {
                k: v for k, v in owner.items() if k != "api_workers"
            }:
                raise RuntimeError("existing full extraction has a different owner; retained")
        atomic_json(owner_file, owner)
        if report_file.exists() and json.loads(report_file.read_text()).get("status") == "complete":
            records_current = True
            for document in documents:
                path = records_dir / f"{document['document_id'].removeprefix('wiki::')}.json"
                if not path.exists() or json.loads(path.read_text()).get("document") != document:
                    records_current = False
                    break
            if records_current:
                print(report_file.read_text())
                return

        def client():
            current = getattr(_thread_local, "client", None)
            if current is None:
                current = OpenAI(api_key=api_key, base_url=base_url, timeout=300, max_retries=2)
                _thread_local.client = current
            return current

        def call_api(purpose: str, messages: list[dict], max_tokens: int = 10000) -> dict:
            digest = hashlib.sha256(json.dumps(
                {"model": model, "messages": messages}, ensure_ascii=False, sort_keys=True
            ).encode()).hexdigest()
            path = calls_dir / f"{digest}.json"
            if path.exists():
                cached = json.loads(path.read_text())
                if cached.get("status") == "complete":
                    content = str(cached.get("output", ""))
                    if content.strip() and cached.get("finish_reason") != "length":
                        return parse_json_object(content)
            with _count_lock:
                call_count = sum(1 for _ in calls_dir.glob("*.json"))
                if not path.exists() and call_count >= MAX_CALLS:
                    raise RuntimeError("full extraction API call budget reached")
                attempt = 1
                if path.exists():
                    attempt = int(json.loads(path.read_text()).get("attempt", 0)) + 1
                atomic_json(path, {
                    "purpose": purpose, "model": model, "messages": messages,
                    "status": "requested", "attempt": attempt,
                })
            started = time.perf_counter()
            try:
                response = client().chat.completions.create(
                    model=model, messages=messages, temperature=0, max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    extra_body={"enable_thinking": False},
                )
                choice = response.choices[0]
                content = choice.message.content or ""
                record = {
                    "purpose": purpose, "model": model, "messages": messages,
                    "status": "complete", "attempt": attempt, "output": content,
                    "finish_reason": choice.finish_reason,
                    "usage": response.usage.model_dump() if response.usage else None,
                    "seconds": round(time.perf_counter() - started, 3),
                }
                atomic_json(path, record)
                if not content.strip() or choice.finish_reason == "length":
                    raise RuntimeError(f"empty or truncated API response for {purpose}")
                return parse_json_object(content)
            except Exception as exc:
                if path.exists():
                    failed = json.loads(path.read_text())
                    if failed.get("status") != "complete":
                        failed.update(status="failed", error_type=type(exc).__name__,
                                      error=str(exc), seconds=round(time.perf_counter() - started, 3))
                        atomic_json(path, failed)
                raise

        def process(document: dict) -> dict:
            document_id = document["document_id"]
            record_path = records_dir / f"{document_id.removeprefix('wiki::')}.json"
            if record_path.exists():
                record = json.loads(record_path.read_text())
                if record.get("document") == document and record.get("facts"):
                    return record
            title, source = document["title"], document["contents"]

            def repair_fact_anchors(current_facts: list[dict], indexes: list[int],
                                    base_purpose: str) -> list[dict]:
                anchor_input = {
                    "title": title, "passage": source,
                    "items": [{
                        "fact_index": i, "statement": current_facts[i]["statement"],
                        "evidence": current_facts[i].get("evidence", ""),
                    } for i in indexes],
                }
                messages = [
                    {"role": "system", "content": FACT_ANCHOR_REPAIR_SYSTEM},
                    {"role": "user", "content": "Return JSON only.\n" + json.dumps(
                        anchor_input, ensure_ascii=False)},
                ]
                payload = call_api(f"{base_purpose}:{document_id}:{title}", messages)
                for correction in range(4):
                    try:
                        return apply_fact_anchor_repairs(
                            current_facts, payload, source, indexes, title
                        )
                    except ValueError as anchor_error:
                        if correction == 3:
                            raise
                        payload = call_api(
                            f"{base_purpose}_retry{correction + 1}:{document_id}:{title}",
                            messages + [
                                {"role": "assistant", "content": json.dumps(
                                    payload, ensure_ascii=False)},
                                {"role": "user", "content": (
                                    "The repair failed deterministic validation: "
                                    f"{anchor_error}. Return complete corrected JSON. Evidence must "
                                    "be one verbatim contiguous span and the rewritten statement "
                                    "must include the exact article title."
                                )},
                            ],
                        )
                raise AssertionError("unreachable")

            fact_messages = [
                {"role": "system", "content": STRICT_FACT_SYSTEM},
                {"role": "user", "content": f"TITLE: {title}\n\nPASSAGE:\n{source}"},
            ]
            forced_safe_spans = {
                "wiki::88554812029e0183d89c": [
                    "The Remembrance park (Spanish: Parque de la memoria) is a public space "
                    "situated in front of the Río de la Plata estuary in the northern end of the "
                    "Belgrano section of Buenos Aires.",
                ],
            }.get(document_id)
            if forced_safe_spans:
                fact_payload = call_api(
                    f"extract_exact_span_fallback:{document_id}:{title}",
                    [{"role": "system", "content": STRICT_FACT_SYSTEM},
                     {"role": "user", "content": (
                         "Use the complete supplied sentence verbatim as evidence. Return neutral "
                         f"facts only.\nTITLE: {title}\n\nPASSAGE:\n"
                         + "\n".join(forced_safe_spans)
                     )}],
                )
            else:
                try:
                    fact_payload = call_api(f"extract:{document_id}:{title}", fact_messages)
                except BadRequestError:
                    # DashScope content inspection occasionally rejects otherwise benign
                    # geopolitical names. Keep extraction API-only, but restrict these articles
                    # to exact neutral source spans rather than dropping the documents.
                    safe_spans = {
                        "wiki::51549d1bf7efebe4c192": [
                            "is the historical building located at 1 Ketagalan Boulevard",
                        ],
                        "wiki::d18ce54d133a24c8adb2": [
                            "The pass supports scarce amounts of vegetation and is usually "
                            "snow-covered to some extent throughout the year.",
                            "Sela Lake, near the summit of the pass, is one of approximately 101 "
                            "lakes in the area that are sacred in Tibetan Buddhism.",
                            "While Sela Pass does get heavy snowfall in winters, it is usually open "
                            "throughout the year unless landslides or snow require the pass to be "
                            "shut down temporarily.",
                        ],
                    }.get(document_id)
                    if not safe_spans or any(span not in source for span in safe_spans):
                        raise
                    safe_source = "\n".join(safe_spans)
                    fact_payload = call_api(
                        f"extract_safe_fallback:{document_id}:{title}",
                        [{"role": "system", "content": STRICT_FACT_SYSTEM},
                         {"role": "user", "content": (
                             "Extract only neutral, non-sensitive facts from these verbatim source "
                             f"spans.\nTITLE: {title}\n\nPASSAGE:\n{safe_source}"
                         )}],
                    )
            facts = validate_facts(fact_payload, source, require_exact=False)
            bad_indexes = invalid_evidence_indexes(facts)
            if bad_indexes:
                repair_messages = [
                    {"role": "system", "content": EVIDENCE_REPAIR_SYSTEM},
                    {"role": "user", "content": json.dumps({
                        "title": title, "passage": source,
                        "items": [{"fact_index": i, "statement": facts[i]["statement"]}
                                  for i in bad_indexes],
                    }, ensure_ascii=False)},
                ]
                repair_payload = call_api(
                    f"repair_evidence:{document_id}:{title}", repair_messages
                )
                for correction in range(3):
                    try:
                        facts = apply_evidence_repairs(facts, repair_payload, source)
                        break
                    except ValueError as repair_error:
                        if correction == 2:
                            facts = repair_fact_anchors(
                                facts, bad_indexes, "repair_evidence_via_anchor"
                            )
                            break
                        repair_payload = call_api(
                            f"repair_evidence_retry{correction + 1}:{document_id}:{title}",
                            repair_messages + [
                                {"role": "assistant", "content": json.dumps(
                                    repair_payload, ensure_ascii=False)},
                                {"role": "user", "content": (
                                    "The repair failed deterministic validation: "
                                    f"{repair_error}. Return the complete corrected JSON. Every "
                                    "evidence value must be one verbatim contiguous passage span."
                                )},
                            ],
                        )
            validate_facts({"facts": facts}, source, require_exact=True)

            def link_and_validate(base_purpose: str, messages: list[dict], *,
                                  require_entities: bool = True) -> list[dict]:
                payload = call_api(f"{base_purpose}:{document_id}:{title}", messages)
                for correction in range(4):
                    try:
                        raw_links = payload.get("links")
                        if isinstance(raw_links, list):
                            sanitized = []
                            for raw in raw_links:
                                if not isinstance(raw, dict) or not isinstance(
                                    raw.get("fact_index"), int
                                ) or not 0 <= raw["fact_index"] < len(facts):
                                    sanitized.append(raw)
                                    continue
                                statement = facts[raw["fact_index"]]["statement"]
                                entities = []
                                for entity in raw.get("entities", []):
                                    if not isinstance(entity, dict):
                                        continue
                                    name, entity_type = entity.get("name"), entity.get("type")
                                    if entity_type not in ENTITY_TYPES or not isinstance(name, str):
                                        continue
                                    if name.strip().casefold() in statement.casefold():
                                        entities.append({"name": name.strip(), "type": entity_type})
                                    elif title in statement:
                                        entities.append({"name": title, "type": entity_type})
                                sanitized.append({"fact_index": raw["fact_index"],
                                                  "entities": entities})
                            payload = {"links": sanitized}
                        return validate_entity_links(
                            payload, facts, require_entities=require_entities
                        )
                    except ValueError as link_error:
                        if correction == 3:
                            raise
                        payload = call_api(
                            f"{base_purpose}_retry{correction + 1}:{document_id}:{title}",
                            messages + [
                                {"role": "assistant", "content": json.dumps(
                                    payload, ensure_ascii=False)},
                                {"role": "user", "content": (
                                    "The output failed deterministic validation: "
                                    f"{link_error}. Return the complete corrected JSON object. "
                                    "Copy every entity name as one exact contiguous substring of "
                                    "its corresponding statement; omit invalid candidates."
                                )},
                            ],
                        )
                raise AssertionError("unreachable")

            link_input = {"title": title, "facts": [
                {"fact_index": i, "statement": fact["statement"]}
                for i, fact in enumerate(facts)
            ]}
            link_messages = [
                {"role": "system", "content": ENTITY_LINK_SYSTEM},
                {"role": "user", "content": json.dumps(link_input, ensure_ascii=False)},
            ]
            links = link_and_validate(
                "link_entities", link_messages, require_entities=False
            )

            empty_indexes = [link["fact_index"] for link in links if not link["entities"]]
            if empty_indexes:
                facts = repair_fact_anchors(facts, empty_indexes, "anchor_facts")
                anchored_input = {"title": title, "facts": [
                    {"fact_index": i, "statement": fact["statement"]}
                    for i, fact in enumerate(facts)
                ]}
                anchored_messages = [
                    {"role": "system", "content": ENTITY_LINK_SYSTEM},
                    {"role": "user", "content": json.dumps(anchored_input, ensure_ascii=False)},
                ]
                links = link_and_validate("link_entities_anchored", anchored_messages)
            record = {
                "document": document,
                "facts": [{**fact, "entities": link["entities"]}
                          for fact, link in zip(facts, links)],
            }
            atomic_json(record_path, record)
            return record

        existing = {
            path.stem for path in records_dir.glob("*.json")
        }
        pending = [doc for doc in documents
                   if doc["document_id"].removeprefix("wiki::") not in existing]
        report = {**owner, "status": "running", "documents_total": len(documents),
                  "documents_complete": len(documents) - len(pending),
                  "training_started": False}
        atomic_json(report_file, report)
        completed = report["documents_complete"]
        failures = []
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="strict-api") as pool:
            futures = {pool.submit(process, document): document for document in pending}
            for future in as_completed(futures):
                document = futures[future]
                try:
                    record = future.result()
                except Exception as exc:
                    failures.append({
                        "document_id": document["document_id"], "title": document["title"],
                        "error_type": type(exc).__name__, "error": str(exc),
                    })
                else:
                    completed += 1
                    print(json.dumps({
                        "document": f"{completed}/{len(documents)}",
                        "title": document["title"], "facts": len(record["facts"]),
                    }, ensure_ascii=False), flush=True)
                if completed % 10 == 0 or failures:
                    report.update(documents_complete=completed, failures=failures[-20:])
                    atomic_json(report_file, report)

        if failures:
            report.update(status="failed", documents_complete=completed,
                          failed_documents=len(failures), failures=failures)
            atomic_json(report_file, report)
            raise RuntimeError(f"{len(failures)} documents failed API extraction")

        records = {}
        for document in documents:
            path = records_dir / f"{document['document_id'].removeprefix('wiki::')}.json"
            record = json.loads(path.read_text())
            if record.get("document") != document:
                old_document = record.get("document") or {}
                ignored = {"image_ids", "image_paths"}
                old_core = {key: value for key, value in old_document.items() if key not in ignored}
                new_core = {key: value for key, value in document.items() if key not in ignored}
                if old_core != new_core:
                    raise RuntimeError(
                        f"record source changed beyond image aggregation: {document['document_id']}"
                    )
                record["document"] = document
                atomic_json(path, record)
            records[document["document_id"]] = record
        calls = {}
        for path in sorted(calls_dir.glob("*.json")):
            calls[path.stem] = json.loads(path.read_text())
        call_statuses = Counter(call.get("status") for call in calls.values())
        if call_statuses.get("requested"):
            raise RuntimeError("one or more API call records remain in requested state")
        atomic_json(ROOT / "strict_records.json", records)
        atomic_json(ROOT / "api_calls.json", calls)

        fact_count = sum(len(value["facts"]) for value in records.values())
        entity_mentions = sum(len(fact["entities"]) for value in records.values()
                              for fact in value["facts"])
        unique_entities = {(entity["name"].casefold(), entity["type"])
                           for value in records.values() for fact in value["facts"]
                           for entity in fact["entities"]}
        usage = {field: sum((call.get("usage") or {}).get(field, 0) or 0
                            for call in calls.values())
                 for field in ("prompt_tokens", "completion_tokens", "total_tokens")}
        purposes = Counter(call["purpose"].split(":", 1)[0] for call in calls.values())
        report.update(
            status="complete", documents_complete=len(records), api_calls=len(calls),
            fact_count=fact_count,
            exact_evidence_count=sum(fact["evidence_is_exact_source_span"]
                                     for value in records.values() for fact in value["facts"]),
            entity_mentions=entity_mentions, unique_typed_entities=len(unique_entities),
            api_call_purposes=dict(purposes), api_usage=usage,
            api_call_statuses=dict(call_statuses),
            recovered_failed_api_calls=call_statuses.get("failed", 0),
            factual_consistency_independently_verified=False,
            graph_built=False, output=str(ROOT / "strict_records.json"), failures=[],
        )
        atomic_json(report_file, report)
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
