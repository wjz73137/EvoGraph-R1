#!/usr/bin/env python3
"""Dual-model semantic audit of the full strict-Max extraction."""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evograph_mm.kb.gldv2_subset import atomic_json
from evograph_mm.kb.strict_extraction import parse_json_object


PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path("/home/data/dataset/wjz/EvoGraph-R1/expr_mm/evqa_api_strict_max_extraction_full1891_v1")
BATCH_DOCUMENTS = 16
SYSTEM = """You are a strict semantic auditor for source-grounded knowledge extraction.
For every supplied item, decide whether its entire statement is entailed by the supplied passage.
Preserve the force of dates, quantities, negation, alternatives, comparisons, and uncertainty.
The quoted evidence is a locator; use its surrounding passage only to resolve the article subject.
Do not use outside knowledge. Labels:
- SUPPORTED: the complete statement follows from the passage with no material addition or distortion.
- PARTIAL: the central claim is supported, but a material qualifier, relation, subject, or scope is added,
  omitted, or insufficiently established.
- UNSUPPORTED: the central claim is contradicted by or does not follow from the passage.
Return JSON only, with exactly one judgment per sample_id and no extra IDs:
{"judgments":[{"sample_id":"...","label":"SUPPORTED|PARTIAL|UNSUPPORTED","reason":"at most 18 words"}]}"""
_thread_local = threading.local()


def main() -> None:
    from dotenv import dotenv_values
    from openai import BadRequestError, OpenAI

    config = dotenv_values(PROJECT / ".env")
    api_key, base_url = config.get("OPENAI_API_KEY"), config.get("OPENAI_BASE_URL")
    extractor = config.get("STRICT_EXTRACTION_MODEL")
    judge_models = [config.get("QUALITY_JUDGE_MODEL"), config.get("OPENAI_MODEL")]
    if not api_key or not base_url or not extractor or not all(judge_models):
        raise RuntimeError(".env lacks semantic audit configuration")
    if judge_models[0] == judge_models[1]:
        raise RuntimeError("semantic audit requires two distinct judge models")
    workers = int(os.getenv("STRICT_AUDIT_WORKERS", "8"))
    records = json.loads((ROOT / "strict_records.json").read_text())
    calls = json.loads((ROOT / "api_calls.json").read_text())
    document_ids = list(records)

    tags = defaultdict(set)
    for call in calls.values():
        prefix = call.get("purpose", "").split(":", 1)[0]
        if prefix not in {"repair_evidence", "anchor_facts", "anchor_facts_retry"}:
            continue
        purpose = call["purpose"]
        matches = [document_id for document_id in document_ids if f":{document_id}:" in purpose]
        if len(matches) != 1:
            raise RuntimeError(f"cannot map extraction call to a document: {purpose}")
        for repair in parse_json_object(call["output"]).get("repairs", []):
            fact_index = int(repair["fact_index"])
            if 0 <= fact_index < len(records[matches[0]]["facts"]):
                tags[(matches[0], fact_index)].add(prefix)
    for document_id, record in records.items():
        untouched = [i for i in range(len(record["facts"])) if (document_id, i) not in tags]
        candidates = untouched or list(range(len(record["facts"])))
        digest = int(hashlib.sha256(document_id.encode()).hexdigest(), 16)
        tags[(document_id, candidates[digest % len(candidates)])].add("per_document_sample")

    selected = defaultdict(list)
    for (document_id, fact_index), sample_tags in sorted(tags.items()):
        fact = records[document_id]["facts"][fact_index]
        selected[document_id].append({
            "sample_id": f"{document_id}#{fact_index}",
            "fact_index": fact_index,
            "selection_tags": sorted(sample_tags),
            "statement": fact["statement"], "evidence": fact["evidence"],
        })
    batches = []
    for offset in range(0, len(document_ids), BATCH_DOCUMENTS):
        ids = document_ids[offset:offset + BATCH_DOCUMENTS]
        batches.append({
            "batch": len(batches),
            "documents": [{
                "document_id": document_id,
                "title": records[document_id]["document"]["title"],
                "passage": records[document_id]["document"]["contents"],
                "items": selected[document_id],
            } for document_id in ids],
        })
    manifest = {
        "version": "strict-max-full1891-semantic-audit-v1",
        "selection": "all repaired/anchored facts plus one deterministic untouched fact per article",
        "extractor_model": extractor, "judge_models": judge_models,
        "document_count": len(document_ids), "sample_count": len(tags),
        "batch_count": len(batches), "batches": batches,
    }
    atomic_json(ROOT / "semantic_audit_manifest.json", manifest)
    calls_dir = ROOT / "semantic_audit_calls"
    calls_dir.mkdir(exist_ok=True)

    def client():
        current = getattr(_thread_local, "client", None)
        if current is None:
            current = OpenAI(api_key=api_key, base_url=base_url, timeout=300, max_retries=2)
            _thread_local.client = current
        return current

    def call_judge(model: str, batch: dict) -> tuple[str, int, dict]:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(batch, ensure_ascii=False)},
        ]
        digest = hashlib.sha256(json.dumps(
            {"model": model, "messages": messages}, ensure_ascii=False, sort_keys=True
        ).encode()).hexdigest()
        path = calls_dir / f"{digest}.json"

        def safe_fallback() -> tuple[str, int, dict]:
            safe_batch = {
                **batch,
                "documents": [{
                    **document,
                    "passage": "\n".join(dict.fromkeys(
                        item["evidence"] for item in document["items"]
                    )),
                    "passage_scope": "selected exact evidence spans only",
                } for document in batch["documents"]],
            }
            safe_messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": (
                    "The passage field contains only the selected verbatim evidence spans. "
                    "Judge entailment using those spans only.\n" +
                    json.dumps(safe_batch, ensure_ascii=False)
                )},
            ]
            safe_digest = hashlib.sha256(json.dumps(
                {"model": model, "messages": safe_messages},
                ensure_ascii=False, sort_keys=True,
            ).encode()).hexdigest()
            safe_path = calls_dir / f"{safe_digest}.json"
            if safe_path.exists():
                cached_safe = json.loads(safe_path.read_text())
                if cached_safe.get("status") == "complete":
                    return model, batch["batch"], parse_json_object(cached_safe["output"])
            safe_attempt = int(json.loads(safe_path.read_text()).get("attempt", 0)) + 1 \
                if safe_path.exists() else 1
            atomic_json(safe_path, {
                "model": model, "batch": batch["batch"], "messages": safe_messages,
                "status": "requested", "attempt": safe_attempt,
                "input_fallback": "selected_exact_evidence_spans",
            })
            safe_started = time.perf_counter()
            try:
                response = client().chat.completions.create(
                    model=model, messages=safe_messages, temperature=0, max_tokens=12000,
                    response_format={"type": "json_object"},
                    extra_body={"enable_thinking": False},
                )
                choice = response.choices[0]
                content = choice.message.content or ""
                atomic_json(safe_path, {
                    "model": model, "batch": batch["batch"], "messages": safe_messages,
                    "status": "complete", "attempt": safe_attempt, "output": content,
                    "finish_reason": choice.finish_reason,
                    "usage": response.usage.model_dump() if response.usage else None,
                    "seconds": round(time.perf_counter() - safe_started, 3),
                    "input_fallback": "selected_exact_evidence_spans",
                })
                if not content.strip() or choice.finish_reason == "length":
                    raise RuntimeError(f"truncated safe audit response for {model}")
                return model, batch["batch"], parse_json_object(content)
            except Exception as exc:
                failed_safe = json.loads(safe_path.read_text())
                if failed_safe.get("status") != "complete":
                    failed_safe.update(status="failed", error_type=type(exc).__name__,
                                       error=str(exc))
                    atomic_json(safe_path, failed_safe)
                raise

        if path.exists():
            cached = json.loads(path.read_text())
            if cached.get("status") == "complete":
                return model, batch["batch"], parse_json_object(cached["output"])
            if cached.get("status") == "failed" and \
                    "DataInspectionFailed" in str(cached.get("error", "")):
                return safe_fallback()
        attempt = int(json.loads(path.read_text()).get("attempt", 0)) + 1 if path.exists() else 1
        atomic_json(path, {"model": model, "batch": batch["batch"], "messages": messages,
                           "status": "requested", "attempt": attempt})
        started = time.perf_counter()
        try:
            response = client().chat.completions.create(
                model=model, messages=messages, temperature=0, max_tokens=12000,
                response_format={"type": "json_object"}, extra_body={"enable_thinking": False},
            )
            choice = response.choices[0]
            content = choice.message.content or ""
            record = {"model": model, "batch": batch["batch"], "messages": messages,
                      "status": "complete", "attempt": attempt, "output": content,
                      "finish_reason": choice.finish_reason,
                      "usage": response.usage.model_dump() if response.usage else None,
                      "seconds": round(time.perf_counter() - started, 3)}
            atomic_json(path, record)
            if not content.strip() or choice.finish_reason == "length":
                raise RuntimeError(f"truncated audit response for {model} batch {batch['batch']}")
            return model, batch["batch"], parse_json_object(content)
        except BadRequestError as exc:
            failed = json.loads(path.read_text())
            if failed.get("status") != "complete":
                failed.update(status="failed", error_type=type(exc).__name__, error=str(exc))
                atomic_json(path, failed)
            if "DataInspectionFailed" in str(exc):
                return safe_fallback()
            raise
        except Exception as exc:
            failed = json.loads(path.read_text())
            if failed.get("status") != "complete":
                failed.update(status="failed", error_type=type(exc).__name__, error=str(exc))
                atomic_json(path, failed)
            raise

    expected = {batch["batch"]: {
        item["sample_id"] for document in batch["documents"] for item in document["items"]
    } for batch in batches}
    batches_by_index = {batch["batch"]: batch for batch in batches}

    def correct_judgments(model: str, batch_index: int, payload: dict,
                          validation_error: str, correction: int) -> dict:
        correction_messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(
                batches_by_index[batch_index], ensure_ascii=False)},
            {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)},
            {"role": "user", "content": (
                f"Validation failed: {validation_error}. Return corrected JSON with exactly these "
                "sample_id values, each exactly once: " +
                json.dumps(sorted(expected[batch_index]), ensure_ascii=False)
            )},
        ]
        digest = hashlib.sha256(json.dumps(
            {"model": model, "messages": correction_messages},
            ensure_ascii=False, sort_keys=True,
        ).encode()).hexdigest()
        path = calls_dir / f"{digest}.json"
        if path.exists():
            cached = json.loads(path.read_text())
            if cached.get("status") == "complete":
                return parse_json_object(cached["output"])
        attempt = int(json.loads(path.read_text()).get("attempt", 0)) + 1 \
            if path.exists() else 1
        atomic_json(path, {
            "model": model, "batch": batch_index, "messages": correction_messages,
            "status": "requested", "attempt": attempt,
            "output_correction": correction,
        })
        started = time.perf_counter()
        response = client().chat.completions.create(
            model=model, messages=correction_messages, temperature=0, max_tokens=12000,
            response_format={"type": "json_object"}, extra_body={"enable_thinking": False},
        )
        choice = response.choices[0]
        content = choice.message.content or ""
        atomic_json(path, {
            "model": model, "batch": batch_index, "messages": correction_messages,
            "status": "complete", "attempt": attempt, "output": content,
            "finish_reason": choice.finish_reason,
            "usage": response.usage.model_dump() if response.usage else None,
            "seconds": round(time.perf_counter() - started, 3),
            "output_correction": correction,
        })
        if not content.strip() or choice.finish_reason == "length":
            raise RuntimeError(f"truncated corrected audit response for {model} batch {batch_index}")
        return parse_json_object(content)

    judgments = {model: {} for model in judge_models}
    jobs = [(model, batch) for model in judge_models for batch in batches]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="semantic-audit") as pool:
        futures = [pool.submit(call_judge, model, batch) for model, batch in jobs]
        for future in as_completed(futures):
            model, batch_index, payload = future.result()
            for correction in range(4):
                raw = payload.get("judgments")
                validation_error = ""
                by_id = {}
                if not isinstance(raw, list):
                    validation_error = "response lacks a judgments list"
                else:
                    for item in raw:
                        if not isinstance(item, dict):
                            validation_error = "a judgment is not an object"
                            break
                        sample_id = item.get("sample_id")
                        label, reason = item.get("label"), item.get("reason")
                        if sample_id in by_id or label not in {
                            "SUPPORTED", "PARTIAL", "UNSUPPORTED"
                        } or not isinstance(reason, str):
                            validation_error = "duplicate ID, invalid label, or invalid reason"
                            break
                        by_id[sample_id] = {"label": label, "reason": reason}
                    if not validation_error and set(by_id) != expected[batch_index]:
                        validation_error = "response returned the wrong sample IDs"
                if not validation_error:
                    break
                if correction == 3:
                    raise ValueError(
                        f"{model} batch {batch_index} remained invalid: {validation_error}"
                    )
                payload = correct_judgments(
                    model, batch_index, payload, validation_error, correction + 1
                )
            judgments[model].update(by_id)
            print(json.dumps({"model": model, "batch": batch_index + 1,
                              "batches": len(batches), "samples": len(by_id)}), flush=True)

    sample_lookup = {item["sample_id"]: item for values in selected.values() for item in values}
    agreements, disagreements, exclusions = [], [], []
    for sample_id in sorted(sample_lookup):
        item = {**sample_lookup[sample_id], "judgments": {
            model: judgments[model][sample_id] for model in judge_models
        }}
        labels = [item["judgments"][model]["label"] for model in judge_models]
        if len(set(labels)) == 1:
            agreements.append(item)
        else:
            disagreements.append(item)
        if labels != ["SUPPORTED", "SUPPORTED"]:
            exclusions.append(sample_id)
    audit_calls = {path.stem: json.loads(path.read_text()) for path in calls_dir.glob("*.json")}
    atomic_json(ROOT / "semantic_audit_calls.json", audit_calls)
    report = {
        "status": "complete", "documents": len(document_ids), "samples": len(sample_lookup),
        "selection_tag_counts": dict(Counter(tag for values in tags.values() for tag in values)),
        "judge_models": judge_models, "extractor_model": extractor,
        "cross_model_judge_present": any(model != extractor for model in judge_models),
        "label_counts": {model: dict(Counter(
            item["label"] for item in judgments[model].values()
        )) for model in judge_models},
        "agreement_count": len(agreements), "disagreement_count": len(disagreements),
        "consensus_label_counts": dict(Counter(
            item["judgments"][judge_models[0]]["label"] for item in agreements
        )),
        "excluded_sample_ids": exclusions,
        "non_supported_agreements": [item for item in agreements
                                     if item["judgments"][judge_models[0]]["label"] != "SUPPORTED"],
        "disagreements": disagreements,
        "api_calls": len(audit_calls),
        "api_usage": {model: {field: sum(
            (call.get("usage") or {}).get(field, 0) or 0 for call in audit_calls.values()
            if call.get("model") == model
        ) for field in ("prompt_tokens", "completion_tokens", "total_tokens")}
                      for model in judge_models},
        "note": "Model judgments are a quality screen, not independent human verification.",
    }
    atomic_json(ROOT / "semantic_audit_report.json", report)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in {"non_supported_agreements", "disagreements"}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
