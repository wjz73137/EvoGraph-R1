#!/usr/bin/env python3
"""Extract the first non-empty section of 63 train articles with strict Max."""
from __future__ import annotations

import csv
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evograph_mm.kb.gldv2_subset import atomic_json, image_info
from evograph_mm.kb.strict_extraction import (
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
DATA_ROOT = Path("/home/data/dataset/wjz/EvoGraph-R1")
SUBSET = "paper_64_16_seed0"
SUBSET_ROOT = DATA_ROOT / f"datasets_mm/E-VQA/subsets/{SUBSET}"
QA_TRAIN = SUBSET_ROOT / "qa_train.csv"
KB_FILE = DATA_ROOT / f"datasets_mm/E-VQA/raw/kb/{SUBSET}_wiki_pages.json"
OUTPUT_ROOT = DATA_ROOT / "expr_mm/evqa_api_strict_max_extraction_63_v1"
MAX_CALLS = 190


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def prepare_sources() -> dict:
    kb = json.loads(KB_FILE.read_text())
    if not kb.get("complete") or kb.get("missing_urls"):
        raise RuntimeError("small-subset Wikipedia page cache is incomplete")
    subset_manifest = json.loads((SUBSET_ROOT / "manifest.json").read_text())
    images = {item["image_id"]: item for item in subset_manifest["copied_images"]}
    rows = list(csv.DictReader(QA_TRAIN.open(newline="")))
    selected, seen = [], set()
    for row in rows:
        # Deliberately read only article identity and image association. Questions,
        # answers and evidence labels do not affect source selection or extraction.
        url = row["wikipedia_url"].split("|")[0].strip()
        if url in seen:
            continue
        page = kb["pages"].get(url)
        if not page:
            raise RuntimeError(f"Wikipedia page cache lacks {url}")
        if len(page["section_texts"]) != len(page["section_titles"]):
            raise RuntimeError(f"section arrays are misaligned for {url}")
        section_id = next((index for index, value in enumerate(page["section_texts"])
                           if value.strip()), None)
        if section_id is None:
            raise RuntimeError(f"Wikipedia page has no non-empty section: {url}")
        image_id = row["dataset_image_ids"].split(",")[0].strip()
        image = images.get(image_id)
        if not image:
            raise RuntimeError(f"subset manifest lacks image {image_id}")
        image_path = DATA_ROOT / image["subset_path"]
        if not image_info(image_path):
            raise RuntimeError(f"subset image cannot be decoded: {image_id}")
        title = page["title"]
        section_title = page["section_titles"][section_id]
        passage = page["section_texts"][section_id].strip()
        contents = f'"{title}"\nSection: {section_title}\n{passage}'
        document_id = "wiki::" + hashlib.sha256(
            f"{url}\0{section_id}".encode()).hexdigest()[:20]
        selected.append({
            "document_id": document_id,
            "title": title,
            "wikipedia_url": url,
            "section_id": section_id,
            "section_title": section_title,
            "image_id": image_id,
            "image_path": str(image_path),
            "contents": contents,
            "contents_sha256": text_sha256(contents),
        })
        seen.add(url)
    if len(rows) != 64 or len(selected) != 63:
        raise RuntimeError(f"expected 64 train rows and 63 unique articles, got {len(rows)}/{len(selected)}")
    return {
        "version": "evqa-train63-first-section-v1",
        "selection_policy": (
            "unique training-article URLs in CSV row order; complete first non-empty section; "
            "questions, answers, question types and evidence labels are neither selected on nor transmitted"
        ),
        "source_pages_sha256": sha256(KB_FILE),
        "qa_train_sha256": sha256(QA_TRAIN),
        "train_row_count": len(rows),
        "document_count": len(selected),
        "documents": selected,
    }


def main() -> None:
    from dotenv import dotenv_values
    config = dotenv_values(PROJECT / ".env")
    key, url = config.get("OPENAI_API_KEY"), config.get("OPENAI_BASE_URL")
    model = config.get("STRICT_EXTRACTION_MODEL")
    if not all((key, url, model)):
        raise RuntimeError(".env lacks API key, base URL, or STRICT_EXTRACTION_MODEL")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    source_manifest = prepare_sources()
    source_file = OUTPUT_ROOT / "source_manifest.json"
    if source_file.exists() and json.loads(source_file.read_text()) != source_manifest:
        raise RuntimeError("existing batch source manifest differs; retained")
    atomic_json(source_file, source_manifest)
    owner = {
        "pipeline": "strict-source-grounded-max-batch63-v1",
        "model": f"api/{model}",
        "source_manifest_sha256": sha256(source_file),
        "fact_prompt_sha256": text_sha256(STRICT_FACT_SYSTEM),
        "evidence_repair_prompt_sha256": text_sha256(EVIDENCE_REPAIR_SYSTEM),
        "fact_anchor_prompt_sha256": text_sha256(FACT_ANCHOR_REPAIR_SYSTEM),
        "entity_link_prompt_sha256": text_sha256(ENTITY_LINK_SYSTEM),
        "no_qa_question_answer_or_evidence_sent_to_api": True,
        "local_model_loaded": False,
        "gpu_used": False,
    }
    owner_file = OUTPUT_ROOT / "owner.json"
    calls_file = OUTPUT_ROOT / "api_calls.json"
    records_file = OUTPUT_ROOT / "strict_records.json"
    report_file = OUTPUT_ROOT / "report.json"
    with (OUTPUT_ROOT / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if owner_file.exists():
            existing_owner = json.loads(owner_file.read_text())
            legacy_owner = {key: value for key, value in owner.items()
                            if key != "fact_anchor_prompt_sha256"}
            if existing_owner not in (owner, legacy_owner):
                raise RuntimeError("existing batch extraction has a different owner; retained")
        atomic_json(owner_file, owner)
        if report_file.exists():
            existing = json.loads(report_file.read_text())
            if existing.get("status") == "complete":
                print(json.dumps(existing, ensure_ascii=False))
                return
        calls = json.loads(calls_file.read_text()) if calls_file.exists() else {}
        records = json.loads(records_file.read_text()) if records_file.exists() else {}
        report = {**owner, "status": "running", "documents_total": 63,
                  "documents_complete": len(records), "training_started": False}
        atomic_json(report_file, report)

        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=url, timeout=240, max_retries=0)

        def call_api(purpose: str, messages: list[dict], max_tokens: int = 10000) -> dict:
            digest = hashlib.sha256(json.dumps(
                {"model": model, "messages": messages}, ensure_ascii=False, sort_keys=True
            ).encode()).hexdigest()
            cached = calls.get(digest)
            if cached and cached.get("status") == "complete":
                if not str(cached.get("output", "")).strip() or cached.get("finish_reason") == "length":
                    raise RuntimeError(
                        f"retained empty or truncated API call requires inspection: {digest}"
                    )
                return parse_json_object(cached["output"])
            if cached:
                raise RuntimeError(f"retained incomplete API call requires inspection: {digest}")
            if len(calls) >= MAX_CALLS:
                raise RuntimeError("batch extraction API call budget reached")
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
            return parse_json_object(content)

        try:
            for position, document in enumerate(source_manifest["documents"], start=1):
                document_id = document["document_id"]
                if document_id in records:
                    continue
                title, source = document["title"], document["contents"]
                fact_payload = call_api(
                    f"extract:{document_id}:{title}",
                    [{"role": "system", "content": STRICT_FACT_SYSTEM},
                     {"role": "user", "content": f"TITLE: {title}\n\nPASSAGE:\n{source}"}],
                )
                facts = validate_facts(fact_payload, source, require_exact=False)
                bad_indexes = invalid_evidence_indexes(facts)
                if bad_indexes:
                    repair_input = {
                        "title": title, "passage": source,
                        "items": [{"fact_index": index, "statement": facts[index]["statement"]}
                                  for index in bad_indexes],
                    }
                    repair_payload = call_api(
                        f"repair_evidence:{document_id}:{title}",
                        [{"role": "system", "content": EVIDENCE_REPAIR_SYSTEM},
                         {"role": "user", "content": json.dumps(repair_input, ensure_ascii=False)}],
                    )
                    facts = apply_evidence_repairs(facts, repair_payload, source)
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
                raw_links = call_api(f"link_entities:{document_id}:{title}", link_messages)
                try:
                    links = validate_entity_links(raw_links, facts, require_entities=False)
                except ValueError as first_error:
                    retry_messages = link_messages + [
                        {"role": "assistant", "content": json.dumps(raw_links, ensure_ascii=False)},
                        {"role": "user", "content": (
                            "Your entity-link output failed deterministic validation: "
                            f"{first_error}. Return the complete corrected JSON object."
                        )},
                    ]
                    raw_links = call_api(f"link_entities_retry:{document_id}:{title}", retry_messages)
                    links = validate_entity_links(raw_links, facts, require_entities=False)
                empty_indexes = [link["fact_index"] for link in links if not link["entities"]]
                if empty_indexes:
                    anchor_input = {
                        "title": title,
                        "passage": source,
                        "items": [{
                            "fact_index": index,
                            "statement": facts[index]["statement"],
                            "evidence": facts[index]["evidence"],
                        } for index in empty_indexes],
                    }
                    anchor_messages = [
                        {"role": "system", "content": FACT_ANCHOR_REPAIR_SYSTEM},
                        {"role": "user", "content": (
                            "Return JSON only.\n" + json.dumps(anchor_input, ensure_ascii=False)
                        )},
                    ]
                    anchor_payload = call_api(
                        f"anchor_facts:{document_id}:{title}",
                        anchor_messages,
                    )
                    try:
                        facts = apply_fact_anchor_repairs(
                            facts, anchor_payload, source, empty_indexes, title
                        )
                    except ValueError as anchor_error:
                        anchor_retry_messages = anchor_messages + [
                            {"role": "assistant", "content": json.dumps(
                                anchor_payload, ensure_ascii=False
                            )},
                            {"role": "user", "content": (
                                "The repair failed deterministic validation: "
                                f"{anchor_error}. Return complete corrected JSON. Evidence must be one "
                                "verbatim contiguous span; never join passages with ellipses."
                            )},
                        ]
                        anchor_payload = call_api(
                            f"anchor_facts_retry:{document_id}:{title}", anchor_retry_messages
                        )
                        facts = apply_fact_anchor_repairs(
                            facts, anchor_payload, source, empty_indexes, title
                        )
                    anchored_input = {
                        "title": title,
                        "facts": [{"fact_index": index, "statement": fact["statement"]}
                                  for index, fact in enumerate(facts)],
                    }
                    anchored_messages = [
                        {"role": "system", "content": ENTITY_LINK_SYSTEM},
                        {"role": "user", "content": json.dumps(anchored_input, ensure_ascii=False)},
                    ]
                    raw_links = call_api(
                        f"link_entities_anchored:{document_id}:{title}", anchored_messages
                    )
                    try:
                        links = validate_entity_links(raw_links, facts)
                    except ValueError as anchored_error:
                        anchored_retry = anchored_messages + [
                            {"role": "assistant", "content": json.dumps(raw_links, ensure_ascii=False)},
                            {"role": "user", "content": (
                                "Your entity-link output failed deterministic validation: "
                                f"{anchored_error}. Return the complete corrected JSON object."
                            )},
                        ]
                        raw_links = call_api(
                            f"link_entities_anchored_retry:{document_id}:{title}", anchored_retry
                        )
                        links = validate_entity_links(raw_links, facts)
                records[document_id] = {
                    "document": document,
                    "facts": [{**fact, "entities": link["entities"]}
                              for fact, link in zip(facts, links)],
                }
                atomic_json(records_file, records)
                report.update(
                    documents_complete=len(records),
                    api_calls=len(calls),
                    fact_count=sum(len(value["facts"]) for value in records.values()),
                    last_document={"position": position, "document_id": document_id, "title": title},
                )
                atomic_json(report_file, report)
                print(json.dumps({
                    "document": f"{len(records)}/63", "title": title,
                    "facts": len(facts), "api_calls": len(calls),
                    "evidence_repairs": len(bad_indexes),
                }, ensure_ascii=False), flush=True)

            fact_count = sum(len(value["facts"]) for value in records.values())
            entity_mentions = sum(len(fact["entities"]) for value in records.values()
                                  for fact in value["facts"])
            unique_entities = {(entity["name"].casefold(), entity["type"])
                               for value in records.values() for fact in value["facts"]
                               for entity in fact["entities"]}
            usage = {field: sum((call.get("usage") or {}).get(field, 0) or 0
                                for call in calls.values())
                     for field in ("prompt_tokens", "completion_tokens", "total_tokens")}
            report.update(
                status="complete", documents_complete=len(records), fact_count=fact_count,
                exact_evidence_count=sum(fact["evidence_is_exact_source_span"]
                                         for value in records.values() for fact in value["facts"]),
                entity_mentions=entity_mentions, unique_typed_entities=len(unique_entities),
                evidence_repair_calls=sum(call["purpose"].startswith("repair_evidence:")
                                          for call in calls.values()),
                entity_retry_calls=sum(call["purpose"].startswith("link_entities_retry:")
                                       for call in calls.values()),
                api_calls=len(calls), api_usage=usage,
                factual_consistency_independently_verified=False,
                graph_built=False, output=str(records_file),
            )
            atomic_json(report_file, report)
            print(json.dumps(report, ensure_ascii=False), flush=True)
        except Exception as exc:
            report.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                          documents_complete=len(records), api_calls=len(calls))
            atomic_json(report_file, report)
            raise


if __name__ == "__main__":
    main()
