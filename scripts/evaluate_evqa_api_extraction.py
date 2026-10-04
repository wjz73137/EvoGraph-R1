#!/usr/bin/env python3
"""Compare three API-only extraction arms on the pinned four-document E-VQA sample."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evograph_mm.kb.gldv2_subset import atomic_json
from evograph_mm.kb.strict_extraction import STRICT_FACT_SYSTEM


PROJECT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/home/data/dataset/wjz/EvoGraph-R1")
BASELINE_ROOT = DATA_ROOT / "expr_mm/evqa_api_graph_baseline"
BASELINE_GRAPH = BASELINE_ROOT / "E-VQA"
OUTPUT_ROOT = DATA_ROOT / "expr_mm/evqa_api_extraction_quality_v1"
GOLD_FILE = PROJECT / "scripts/evqa_extraction_gold_v1.json"
MAX_CALLS = 32
REPORT_VERSION = 4

REVIEW_SYSTEM = """You are the source-grounding review stage for knowledge extraction.
Return JSON only. Compare the proposed facts against the supplied passage, then return a corrected complete list.
Remove unsupported additions and duplicates. Add important omitted facts.
Preserve uncertainty, alternatives, scope, dates, comparisons, and limiting phrases.
Keep shared uncertainty or an A-or-B alternative in one fact. Do not calculate unstated dates.
Do not complete a cut-off sentence or name an unnamed place using outside knowledge.
Output: {"facts":[{"statement":"...","evidence":"an exact contiguous quote from the passage"}]}.
Every evidence value must be copied verbatim and must entail the whole statement."""

JUDGE_SYSTEM = """You audit candidate knowledge facts against a source passage and a fixed gold inventory.
Return JSON only and judge solely from the supplied passage and policy.
A candidate is supported only if every part is directly entailed without outside knowledge, unstated arithmetic,
or completion of cut-off text. A fact may be a faithful paraphrase.
matched_gold_ids contains only gold facts that the candidate expresses completely, not facts it merely overlaps.
Mark duplicate_of as the zero-based index of an earlier semantically equivalent candidate, otherwise null.
Do not penalize a supported composite fact merely because it maps to multiple gold IDs.
Output exactly one evaluation for every candidate index:
{"evaluations":[{"index":0,"supported":true,"matched_gold_ids":["ID"],"duplicate_of":null,"reason":"short reason"}]}"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(cleaned[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("API result is not a JSON object")
    return value


def load_sources() -> dict[str, str]:
    snapshot = json.loads((BASELINE_ROOT / "source_snapshot.json").read_text())
    sources = {
        record["source_metadata"]["wikipedia_title"]: record["contents"]
        for record in snapshot["documents"]
    }
    if len(sources) != 4:
        raise RuntimeError("pinned source snapshot must contain exactly four documents")
    return sources


def load_and_validate_gold(sources: dict[str, str]) -> dict:
    gold = json.loads(GOLD_FILE.read_text())
    if set(gold["documents"]) != set(sources):
        raise RuntimeError("gold/source title mismatch")
    ids: set[str] = set()
    for title, facts in gold["documents"].items():
        for fact in facts:
            if fact["id"] in ids:
                raise RuntimeError(f"duplicate gold ID: {fact['id']}")
            ids.add(fact["id"])
            if fact["evidence"] not in sources[title]:
                raise RuntimeError(f"gold evidence is not an exact source span: {fact['id']}")
    if len(ids) != 50:
        raise RuntimeError(f"expected 50 gold facts, got {len(ids)}")
    return gold


def load_baseline_candidates(sources: dict[str, str]) -> dict[str, list[dict]]:
    edges = json.loads((BASELINE_GRAPH / "kv_store_hyperedges.json").read_text())
    sidecar = json.loads((BASELINE_GRAPH / "mm_store/graph/graphr1_hit_source_sidecar.json").read_text())
    candidates = {title: [] for title in sources}
    for edge_id, record in edges.items():
        statement = record["content"]
        provenance = sidecar["hyperedge"].get(statement)
        if not provenance or len(provenance.get("wikipedia_titles", [])) != 1:
            raise RuntimeError(f"baseline edge lacks unique source mapping: {edge_id}")
        title = provenance["wikipedia_titles"][0]
        if title not in candidates:
            raise RuntimeError(f"unexpected baseline source title: {title}")
        statement = re.sub(r"^<hyperedge>", "", statement).strip()
        if len(statement) >= 2 and statement[0] == statement[-1] == '"':
            statement = statement[1:-1]
        candidates[title].append({"candidate_id": edge_id, "statement": statement})
    if sum(map(len, candidates.values())) != 66:
        raise RuntimeError("baseline must contain the retained 66 raw API hyperedges")
    return candidates


def validate_extraction(value: dict, source: str) -> list[dict]:
    facts = value.get("facts")
    if not isinstance(facts, list):
        raise ValueError("extraction result lacks a facts list")
    result = []
    for index, fact in enumerate(facts):
        if not isinstance(fact, dict):
            raise ValueError(f"fact {index} is not an object")
        statement, evidence = fact.get("statement"), fact.get("evidence")
        if not isinstance(statement, str) or not statement.strip():
            raise ValueError(f"fact {index} has no statement")
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError(f"fact {index} has no evidence text")
        # Keep protocol violations visible instead of silently repairing or
        # discarding an otherwise judgeable candidate statement.
        result.append({"statement": statement.strip(), "evidence": evidence,
                       "evidence_is_exact_source_span": evidence in source})
    return result


def validate_judgment(value: dict, candidates: list[dict], gold_ids: set[str]) -> list[dict]:
    evaluations = value.get("evaluations")
    if not isinstance(evaluations, list) or len(evaluations) != len(candidates):
        raise ValueError("judge did not return exactly one evaluation per candidate")
    by_index = {}
    for item in evaluations:
        index = item.get("index")
        if not isinstance(index, int) or index in by_index or not 0 <= index < len(candidates):
            raise ValueError("judge returned invalid or repeated candidate index")
        supported = item.get("supported")
        matched = item.get("matched_gold_ids")
        duplicate = item.get("duplicate_of")
        if not isinstance(supported, bool) or not isinstance(matched, list):
            raise ValueError(f"judge returned invalid fields for candidate {index}")
        if any(not isinstance(item_id, str) or item_id not in gold_ids for item_id in matched):
            raise ValueError(f"judge returned an unknown gold ID for candidate {index}")
        if not supported and matched:
            raise ValueError(f"unsupported candidate {index} cannot cover gold facts")
        if duplicate is not None and (not isinstance(duplicate, int) or not 0 <= duplicate < index):
            raise ValueError(f"candidate {index} has an invalid duplicate_of value")
        by_index[index] = {
            "index": index,
            "supported": supported,
            "matched_gold_ids": sorted(set(matched)),
            "duplicate_of": duplicate,
            "reason": str(item.get("reason", "")),
        }
    return [by_index[index] for index in range(len(candidates))]


def score(candidates: list[dict], evaluations: list[dict], gold_count: int) -> dict:
    total = len(candidates)
    supported = sum(item["supported"] for item in evaluations)
    duplicates = sum(item["duplicate_of"] is not None for item in evaluations)
    matched = sorted({gold_id for item in evaluations if item["supported"]
                      for gold_id in item["matched_gold_ids"]})
    supported_unique = [item for item in evaluations
                        if item["supported"] and item["duplicate_of"] is None]
    mapping_counts = [len(item["matched_gold_ids"]) for item in supported_unique]
    precision = supported / total if total else 0.0
    recall = len(matched) / gold_count if gold_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    evidence_flags = [item["evidence_is_exact_source_span"] for item in candidates
                      if "evidence_is_exact_source_span" in item]
    return {
        "candidate_count": total,
        "supported_count": supported,
        "unsupported_count": total - supported,
        "duplicate_count": duplicates,
        "covered_gold_count": len(matched),
        "gold_count": gold_count,
        "claim_precision": round(precision, 4),
        "gold_recall": round(recall, 4),
        "f1": round(f1, 4),
        "hallucination_rate": round((total - supported) / total, 4) if total else 0.0,
        "duplicate_rate": round(duplicates / total, 4) if total else 0.0,
        "exact_evidence_count": sum(evidence_flags) if evidence_flags else None,
        "evidence_count": len(evidence_flags) if evidence_flags else None,
        "exact_evidence_rate": round(sum(evidence_flags) / len(evidence_flags), 4)
        if evidence_flags else None,
        "multi_gold_candidate_count": sum(count > 1 for count in mapping_counts),
        "mean_gold_mappings_per_supported_unique_candidate":
            round(sum(mapping_counts) / len(mapping_counts), 4) if mapping_counts else 0.0,
        "matched_gold_ids": matched,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    sources = load_sources()
    gold = load_and_validate_gold(sources)
    baseline = load_baseline_candidates(sources)
    if args.validate_only:
        print(json.dumps({"sources": len(sources), "gold_facts": 50,
                          "baseline_candidates": 66, "validation": "passed"}))
        return

    from dotenv import dotenv_values
    config = dotenv_values(PROJECT / ".env")
    key = config.get("OPENAI_API_KEY")
    url = config.get("OPENAI_BASE_URL")
    extract_model = config.get("GRAPH_LLM_MODEL") or config.get("OPENAI_MODEL")
    judge_model = config.get("QUALITY_JUDGE_MODEL") or config.get("ACCURACY_JUDGE_MODEL")
    if not all((key, url, extract_model, judge_model)):
        raise RuntimeError(".env lacks API key, base URL, extraction model, or judge model")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    report_file = OUTPUT_ROOT / "report.json"
    calls_file = OUTPUT_ROOT / "api_calls.json"
    outputs_file = OUTPUT_ROOT / "arm_outputs.json"
    # v2 preserves the first judge artifact, whose Pamban inventory used a
    # coarser combined fact. Identical non-Pamban calls are still cacheable.
    judgments_file = OUTPUT_ROOT / "judgments_v2.json"
    with (OUTPUT_ROOT / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if report_file.exists():
            existing = json.loads(report_file.read_text())
            if (existing.get("status") == "complete"
                    and existing.get("report_version") == REPORT_VERSION):
                print(json.dumps(existing, ensure_ascii=False))
                return
        calls = json.loads(calls_file.read_text()) if calls_file.exists() else {}
        arms = json.loads(outputs_file.read_text()) if outputs_file.exists() else {
            "original_graphr1": baseline,
            "grounded_atomic": {},
            "grounded_atomic_reviewed": {},
            "grounded_atomic_max": {},
        }
        arms.setdefault("grounded_atomic_max", {})
        judgments = json.loads(judgments_file.read_text()) if judgments_file.exists() else {}
        # Backfill exact-span flags for results cached before this validation
        # field was introduced. This is a deterministic source comparison and
        # does not change the model's statement or evidence text.
        for arm_name in ("grounded_atomic", "grounded_atomic_reviewed", "grounded_atomic_max"):
            for title, candidates in arms[arm_name].items():
                for candidate in candidates:
                    candidate["evidence_is_exact_source_span"] = (
                        candidate.get("evidence") in sources[title])
        atomic_json(outputs_file, arms)
        report = {
            "report_version": REPORT_VERSION,
            "status": "running",
            "scope": "same four pinned source passages",
            "source_snapshot_sha256": sha256(BASELINE_ROOT / "source_snapshot.json"),
            "gold_sha256": sha256(GOLD_FILE),
            "gold_fact_count": 50,
            "extraction_model": extract_model,
            "comparison_extraction_model": judge_model,
            "judge_model": judge_model,
            "api_only": True,
            "local_model_loaded": False,
            "gpu_used": False,
            "human_gold": True,
            "judge_is_api_not_human": True,
            "training_started": False,
        }
        atomic_json(report_file, report)
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=url, timeout=180, max_retries=0)

        def call_api(*, purpose: str, model: str, messages: list[dict], max_tokens: int) -> dict:
            digest = hashlib.sha256(json.dumps({"model": model, "messages": messages},
                                               ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            cached = calls.get(digest)
            if cached and cached.get("status") == "complete":
                return parse_json_object(cached["output"])
            if cached:
                raise RuntimeError(f"retained incomplete API call requires inspection: {digest}")
            if len(calls) >= MAX_CALLS:
                raise RuntimeError("API call budget reached")
            calls[digest] = {"purpose": purpose, "model": model, "messages": messages,
                             "status": "requested"}
            atomic_json(calls_file, calls)
            started = time.perf_counter()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                extra_body={"enable_thinking": False},
            )
            choice = response.choices[0]
            content = choice.message.content or ""
            calls[digest].update(
                status="complete",
                output=content,
                finish_reason=choice.finish_reason,
                usage=response.usage.model_dump() if response.usage else None,
                seconds=round(time.perf_counter() - started, 3),
            )
            atomic_json(calls_file, calls)
            if not content.strip() or choice.finish_reason == "length":
                raise RuntimeError(f"empty or truncated API response for {purpose}")
            print(json.dumps({"phase": purpose, "api_calls": len(calls),
                              "finish_reason": choice.finish_reason}, ensure_ascii=False), flush=True)
            return parse_json_object(content)

        try:
            for title, source in sources.items():
                if title not in arms["grounded_atomic"]:
                    messages = [
                        {"role": "system", "content": STRICT_FACT_SYSTEM},
                        {"role": "user", "content": f"TITLE: {title}\n\nPASSAGE:\n{source}"},
                    ]
                    value = call_api(purpose=f"extract:{title}", model=extract_model,
                                     messages=messages, max_tokens=5000)
                    arms["grounded_atomic"][title] = validate_extraction(value, source)
                    atomic_json(outputs_file, arms)
                if title not in arms["grounded_atomic_reviewed"]:
                    proposal = arms["grounded_atomic"][title]
                    messages = [
                        {"role": "system", "content": REVIEW_SYSTEM},
                        {"role": "user", "content": (
                            f"TITLE: {title}\n\nPASSAGE:\n{source}\n\n"
                            f"PROPOSED FACTS:\n{json.dumps({'facts': proposal}, ensure_ascii=False)}")},
                    ]
                    value = call_api(purpose=f"review:{title}", model=extract_model,
                                     messages=messages, max_tokens=5000)
                    arms["grounded_atomic_reviewed"][title] = validate_extraction(value, source)
                    atomic_json(outputs_file, arms)
                if title not in arms["grounded_atomic_max"]:
                    messages = [
                        {"role": "system", "content": STRICT_FACT_SYSTEM},
                        {"role": "user", "content": f"TITLE: {title}\n\nPASSAGE:\n{source}"},
                    ]
                    value = call_api(purpose=f"extract_max:{title}", model=judge_model,
                                     messages=messages, max_tokens=5000)
                    arms["grounded_atomic_max"][title] = validate_extraction(value, source)
                    atomic_json(outputs_file, arms)

            for arm_name, documents in arms.items():
                judgments.setdefault(arm_name, {})
                for title, source in sources.items():
                    if title in judgments[arm_name]:
                        continue
                    candidates = documents[title]
                    gold_facts = gold["documents"][title]
                    compact_candidates = [
                        {"index": index, "statement": item["statement"]}
                        for index, item in enumerate(candidates)
                    ]
                    payload = {
                        "policy": gold["policy"],
                        "title": title,
                        "passage": source,
                        "gold_facts": [{"id": item["id"], "fact": item["fact"]}
                                       for item in gold_facts],
                        "candidates": compact_candidates,
                    }
                    messages = [
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ]
                    value = call_api(purpose=f"judge:{arm_name}:{title}", model=judge_model,
                                     messages=messages, max_tokens=7000)
                    gold_ids = {item["id"] for item in gold_facts}
                    judgments[arm_name][title] = validate_judgment(value, candidates, gold_ids)
                    atomic_json(judgments_file, judgments)

            scores = {}
            for arm_name, documents in arms.items():
                per_document = {}
                all_candidates, all_evaluations = [], []
                all_gold_ids = set()
                for title in sources:
                    facts = gold["documents"][title]
                    per_document[title] = score(
                        documents[title], judgments[arm_name][title], len(facts))
                    offset = len(all_candidates)
                    all_candidates.extend(documents[title])
                    for item in judgments[arm_name][title]:
                        adjusted = dict(item)
                        adjusted["index"] += offset
                        if adjusted["duplicate_of"] is not None:
                            adjusted["duplicate_of"] += offset
                        all_evaluations.append(adjusted)
                    all_gold_ids.update(item["id"] for item in facts)
                overall = score(all_candidates, all_evaluations, len(all_gold_ids))
                scores[arm_name] = {"overall": overall, "documents": per_document}
            usage = {
                field: sum((call.get("usage") or {}).get(field, 0) or 0 for call in calls.values())
                for field in ("prompt_tokens", "completion_tokens", "total_tokens")
            }
            report.update(
                status="complete",
                api_calls=len(calls),
                api_usage=usage,
                scores=scores,
                metrics={
                    "claim_precision": "supported candidate claims / all candidate claims",
                    "gold_recall": "unique completely expressed gold facts / 50",
                    "hallucination_rate": "unsupported candidate claims / all candidate claims",
                    "duplicate_rate": "claims semantically equivalent to an earlier claim / all candidate claims",
                    "exact_evidence_rate": "exact contiguous source quotes / evidence fields; not applicable to the original GraphR1 format",
                    "multi_gold_candidate_count": "supported nonduplicate candidates that completely express more than one gold fact; a diagnostic proxy for non-atomic aggregation",
                    "f1": "harmonic mean of claim precision and gold recall",
                },
                limitations=[
                    "The fixed gold inventory was manually authored from the four passages, not from dataset QA answers.",
                    "Candidate-to-gold mapping and support decisions were made by the configured API judge, not a human annotator.",
                    "Four short passages are a development comparison, not a dataset-level benchmark.",
                    "The original GraphR1 arm includes two gleaning passes; the two constrained arms use one extraction pass, with the reviewed arm adding one correction pass.",
                ],
                artifacts={
                    "gold": str(GOLD_FILE),
                    "arm_outputs": str(outputs_file),
                    "judgments": str(judgments_file),
                    "api_calls": str(calls_file),
                },
            )
            atomic_json(report_file, report)
            print(json.dumps({"status": "complete", "api_calls": len(calls),
                              "scores": {name: value["overall"] for name, value in scores.items()}},
                             ensure_ascii=False), flush=True)
        except Exception as exc:
            report.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                          api_calls=len(calls))
            atomic_json(report_file, report)
            raise


if __name__ == "__main__":
    main()
