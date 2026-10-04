#!/usr/bin/env python3
"""Cross-model semantic audit of strict-Max facts, emphasizing repaired facts."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evograph_mm.kb.gldv2_subset import atomic_json
from evograph_mm.kb.strict_extraction import parse_json_object


PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path("/home/data/dataset/wjz/EvoGraph-R1/expr_mm/evqa_api_strict_max_extraction_63_v1")
BATCH_DOCUMENTS = 16
MAX_CALLS = 12
SYSTEM = """You are a strict semantic auditor for source-grounded knowledge extraction.
For every supplied item, decide whether its entire statement is entailed by the supplied passage.
Preserve the force of dates, quantities, negation, alternatives, comparisons, and uncertainty.
The quoted evidence is a locator; you may use its surrounding passage context, especially to resolve the
article subject. Do not use outside knowledge. Labels:
- SUPPORTED: the complete statement follows from the passage with no material addition or distortion.
- PARTIAL: the central claim is supported, but a material qualifier, relation, subject, or scope is added,
  omitted, or insufficiently established.
- UNSUPPORTED: the central claim is contradicted by or does not follow from the passage.
Return JSON only, with exactly one judgment per sample_id and no extra IDs:
{"judgments":[{"sample_id":"...","label":"SUPPORTED|PARTIAL|UNSUPPORTED","reason":"at most 18 words"}]}"""


def get_document_id(purpose: str, document_ids: list[str]) -> str:
    matches = [document_id for document_id in document_ids if f":{document_id}:" in purpose]
    if len(matches) != 1:
        raise RuntimeError(f"cannot uniquely map API purpose to document: {purpose}")
    return matches[0]


def main() -> None:
    from dotenv import dotenv_values
    from openai import OpenAI

    config = dotenv_values(PROJECT / ".env")
    key, url = config.get("OPENAI_API_KEY"), config.get("OPENAI_BASE_URL")
    extractor = config.get("STRICT_EXTRACTION_MODEL")
    judge_models = [config.get("QUALITY_JUDGE_MODEL"), config.get("OPENAI_MODEL")]
    if not key or not url or not extractor or not all(judge_models):
        raise RuntimeError(".env lacks API or semantic-audit model configuration")
    if judge_models[0] == judge_models[1]:
        raise RuntimeError("semantic audit requires two distinct configured judge models")

    records = json.loads((ROOT / "strict_records.json").read_text())
    extraction_calls = json.loads((ROOT / "api_calls.json").read_text())
    document_ids = list(records)
    tags: dict[tuple[str, int], set[str]] = defaultdict(set)
    for call in extraction_calls.values():
        prefix = call.get("purpose", "").split(":", 1)[0]
        if call.get("status") != "complete" or prefix not in {
            "repair_evidence", "anchor_facts", "anchor_facts_retry"
        }:
            continue
        document_id = get_document_id(call["purpose"], document_ids)
        for repair in parse_json_object(call["output"]).get("repairs", []):
            tags[(document_id, repair["fact_index"])].add(prefix)

    for document_id, record in records.items():
        untouched = [index for index in range(len(record["facts"]))
                     if (document_id, index) not in tags]
        candidates = untouched or list(range(len(record["facts"])))
        digest = int(hashlib.sha256(document_id.encode()).hexdigest(), 16)
        tags[(document_id, candidates[digest % len(candidates)])].add("per_document_sample")

    selected_by_document: dict[str, list[dict]] = defaultdict(list)
    for (document_id, fact_index), sample_tags in sorted(tags.items()):
        fact = records[document_id]["facts"][fact_index]
        selected_by_document[document_id].append({
            "sample_id": f"{document_id}#{fact_index}",
            "fact_index": fact_index,
            "selection_tags": sorted(sample_tags),
            "statement": fact["statement"],
            "evidence": fact["evidence"],
        })

    batches = []
    for offset in range(0, len(document_ids), BATCH_DOCUMENTS):
        batch_ids = document_ids[offset:offset + BATCH_DOCUMENTS]
        batches.append({
            "batch": len(batches),
            "documents": [{
                "document_id": document_id,
                "title": records[document_id]["document"]["title"],
                "passage": records[document_id]["document"]["contents"],
                "items": selected_by_document[document_id],
            } for document_id in batch_ids],
        })
    manifest = {
        "version": "strict-max-batch63-semantic-audit-v1",
        "selection": (
            "all facts touched by evidence repair or fact anchoring, plus one deterministic untouched "
            "fact per document"
        ),
        "extractor_model": extractor,
        "judge_models": judge_models,
        "document_count": len(document_ids),
        "sample_count": len(tags),
        "risk_sample_count": sum("per_document_sample" not in value or len(value) > 1
                                 for value in tags.values()),
        "batch_count": len(batches),
        "batches": batches,
    }
    atomic_json(ROOT / "semantic_audit_manifest.json", manifest)

    calls_file = ROOT / "semantic_audit_calls.json"
    calls = json.loads(calls_file.read_text()) if calls_file.exists() else {}
    client = OpenAI(api_key=key, base_url=url, timeout=240, max_retries=0)

    def call_judge(model: str, batch: dict) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(batch, ensure_ascii=False)},
        ]
        digest = hashlib.sha256(json.dumps(
            {"model": model, "messages": messages}, ensure_ascii=False, sort_keys=True
        ).encode()).hexdigest()
        cached = calls.get(digest)
        if cached and cached.get("status") == "complete":
            return parse_json_object(cached["output"])
        if cached:
            raise RuntimeError(f"retained incomplete semantic-audit call: {digest}")
        if len(calls) >= MAX_CALLS:
            raise RuntimeError("semantic-audit API call budget reached")
        calls[digest] = {"model": model, "batch": batch["batch"], "messages": messages,
                         "status": "requested"}
        atomic_json(calls_file, calls)
        started = time.perf_counter()
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=0, max_tokens=12000,
            response_format={"type": "json_object"}, extra_body={"enable_thinking": False},
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
            raise RuntimeError(f"empty or truncated semantic audit: {model} batch {batch['batch']}")
        return parse_json_object(content)

    expected_by_batch = {
        batch["batch"]: {
            item["sample_id"] for document in batch["documents"] for item in document["items"]
        } for batch in batches
    }
    judgments: dict[str, dict[str, dict]] = {model: {} for model in judge_models}
    for model in judge_models:
        for batch in batches:
            payload = call_judge(model, batch)
            raw = payload.get("judgments")
            if not isinstance(raw, list):
                raise ValueError(f"{model} batch {batch['batch']} lacks judgments")
            by_id = {}
            for item in raw:
                sample_id, label, reason = item.get("sample_id"), item.get("label"), item.get("reason")
                if sample_id in by_id or label not in {"SUPPORTED", "PARTIAL", "UNSUPPORTED"}:
                    raise ValueError(f"{model} batch {batch['batch']} has invalid judgment")
                if not isinstance(reason, str):
                    raise ValueError(f"{model} batch {batch['batch']} has invalid reason")
                by_id[sample_id] = {"label": label, "reason": reason}
            if set(by_id) != expected_by_batch[batch["batch"]]:
                raise ValueError(f"{model} batch {batch['batch']} returned wrong sample IDs")
            judgments[model].update(by_id)
            print(json.dumps({
                "model": model, "batch": f"{batch['batch'] + 1}/{len(batches)}",
                "samples": len(by_id), "labels": dict(Counter(x["label"] for x in by_id.values())),
            }, ensure_ascii=False), flush=True)

    sample_lookup = {item["sample_id"]: item for items in selected_by_document.values()
                     for item in items}
    label_counts = {model: dict(Counter(item["label"] for item in result.values()))
                    for model, result in judgments.items()}
    agreements, disagreements = [], []
    for sample_id in sorted(sample_lookup):
        labels = {model: judgments[model][sample_id]["label"] for model in judge_models}
        item = {**sample_lookup[sample_id], "judgments": {
            model: judgments[model][sample_id] for model in judge_models
        }}
        (agreements if len(set(labels.values())) == 1 else disagreements).append(item)
    consensus_counts = Counter(
        item["judgments"][judge_models[0]]["label"] for item in agreements
    )
    usage = {model: {
        field: sum((call.get("usage") or {}).get(field, 0) or 0 for call in calls.values()
                   if call.get("model") == model)
        for field in ("prompt_tokens", "completion_tokens", "total_tokens")
    } for model in judge_models}
    report = {
        "status": "complete",
        "documents": len(document_ids),
        "samples": len(tags),
        "selection_tag_counts": dict(Counter(tag for values in tags.values() for tag in values)),
        "judge_models": judge_models,
        "extractor_model": extractor,
        "extractor_self_judge_present": extractor in judge_models,
        "cross_model_judge_present": any(model != extractor for model in judge_models),
        "independent_provider_or_model_family": False,
        "label_counts": label_counts,
        "agreement_count": len(agreements),
        "disagreement_count": len(disagreements),
        "consensus_label_counts": dict(consensus_counts),
        "non_supported_agreements": [item for item in agreements
                                     if item["judgments"][judge_models[0]]["label"] != "SUPPORTED"],
        "disagreements": disagreements,
        "api_calls": len(calls),
        "api_usage": usage,
        "note": "Model judgments are a quality screen, not independent human verification.",
    }
    atomic_json(ROOT / "semantic_audit_report.json", report)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"non_supported_agreements", "disagreements"}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
