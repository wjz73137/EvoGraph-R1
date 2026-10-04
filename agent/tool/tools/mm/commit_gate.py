"""Evidence-grounded pre-commit validation for multimodal graph edits."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class GraphEditGateDecision:
    allowed: bool
    decision: str
    validity: str
    conflict_type: str
    confidence: float
    reason: str
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def graph_edit_commit_gate_enabled() -> bool:
    mode = os.getenv(
        "EVOGRAPH_GRAPH_EDIT_COMMIT_GATE",
        os.getenv("GRAPH_EDIT_COMMIT_GATE", "off"),
    )
    return str(mode).strip().casefold() in {"1", "true", "yes", "on", "api"}


def validate_graph_edit_commit(
    operation: str,
    args: Mapping[str, Any],
) -> GraphEditGateDecision:
    """Validate an edit against the trajectory evidence before it reaches the KB."""
    if not graph_edit_commit_gate_enabled():
        return GraphEditGateDecision(
            allowed=True,
            decision="PERMANENT",
            validity="support",
            conflict_type="none",
            confidence=1.0,
            reason="pre-commit gate disabled",
        )

    _load_project_env()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", os.getenv("OPENAI_API_BASE", "")).strip()
    model = (
        os.getenv("GRAPH_EDIT_GATE_MODEL", "").strip()
        or os.getenv("QUALITY_JUDGE_MODEL", "").strip()
        or os.getenv("OPENAI_MODEL", "").strip()
    )
    missing = [
        name
        for name, value in (
            ("OPENAI_API_KEY", api_key),
            ("OPENAI_BASE_URL", base_url),
            ("GRAPH_EDIT_GATE_MODEL/QUALITY_JUDGE_MODEL/OPENAI_MODEL", model),
        )
        if not value
    ]
    if missing:
        return _reject(
            reason="pre-commit validator is not configured: " + ", ".join(missing),
            conflict_type="validator_unavailable",
            model=model,
        )

    evidence = _build_evidence_packet(operation, args)
    try:
        from openai import OpenAI

        timeout = _positive_float(os.getenv("GRAPH_EDIT_GATE_TIMEOUT", "120"), 120.0)
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=1,
        )
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(evidence, ensure_ascii=False, indent=2),
                },
            ],
            temperature=0,
            max_tokens=700,
            response_format={"type": "json_object"},
            extra_body={"enable_thinking": False},
        )
        content = completion.choices[0].message.content
        payload = _parse_json_object(content)
        return _normalize_decision(payload, model=model)
    except Exception as exc:
        return _reject(
            reason=f"pre-commit validator failed: {type(exc).__name__}: {exc}",
            conflict_type="validator_unavailable",
            model=model,
        )


_SYSTEM_PROMPT = """You are a conservative knowledge-graph edit validator.
Judge only from the supplied original question, prior tool evidence, and candidate edit.
Never use an unstated answer or guess. In particular, reject facts about a namesake,
nearby entity, different branch, or different location even when the names are similar.
Check: (1) source entailment, (2) entity identity and visual anchor, (3) contradiction
with KB evidence, and (4) time, location, unit, and condition scope.

Return exactly one JSON object with these fields:
- decision: REJECT, TENTATIVE, or PERMANENT
- validity: support, refute, or insufficient
- conflict_type: none, entity_identity_mismatch, source_not_entailing,
  contradicts_kb, scope_mismatch, temporal_mismatch, duplicate, or other
- confidence: number from 0 to 1
- reason: a concise evidence-grounded explanation

PERMANENT is allowed only when the supplied evidence explicitly supports the complete
candidate fact for the same anchored entity and scope. Use TENTATIVE for genuinely
insufficient evidence and REJECT for mismatches or contradictions."""


def _build_evidence_packet(operation: str, args: Mapping[str, Any]) -> dict[str, Any]:
    history = args.get("__trajectory_history", [])
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
        history = []
    compact_history: list[dict[str, Any]] = []
    for call in list(history)[-8:]:
        if not isinstance(call, Mapping):
            continue
        compact_history.append(
            {
                "tool": _clip(call.get("tool", ""), 64),
                "args": _compact_args(call.get("args")),
                "result": _clip(call.get("result", ""), 6000),
            }
        )
    candidate = {
        key: value
        for key, value in args.items()
        if not str(key).startswith("__")
    }
    return {
        "original_question": _clip(args.get("__question", ""), 2000),
        "operation": str(operation),
        "candidate_edit": candidate,
        "prior_tool_evidence_in_chronological_order": compact_history,
    }


def _compact_args(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return _clip(value, 1000)
    return {
        str(key): _clip(item, 1000)
        for key, item in value.items()
        if not str(key).startswith("__")
    }


def _clip(value: Any, limit: int) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(value)
    if len(text) <= limit:
        return text
    head = max(1, limit * 2 // 3)
    tail = max(1, limit - head - 31)
    return text[:head] + "\n...[middle evidence clipped]...\n" + text[-tail:]


def _parse_json_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("validator returned empty content")
    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("validator response is not a JSON object")
    return payload


def _normalize_decision(
    payload: Mapping[str, Any],
    *,
    model: str,
) -> GraphEditGateDecision:
    decision = str(payload.get("decision", "TENTATIVE")).strip().upper()
    if decision not in {"REJECT", "TENTATIVE", "PERMANENT"}:
        decision = "TENTATIVE"
    validity = str(payload.get("validity", "insufficient")).strip().casefold()
    if validity not in {"support", "refute", "insufficient"}:
        validity = "insufficient"
    conflict_type = str(payload.get("conflict_type", "other")).strip().casefold()
    allowed_conflicts = {
        "none",
        "entity_identity_mismatch",
        "source_not_entailing",
        "contradicts_kb",
        "scope_mismatch",
        "temporal_mismatch",
        "duplicate",
        "other",
    }
    if conflict_type not in allowed_conflicts:
        conflict_type = "other"
    try:
        confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    threshold = _positive_float(os.getenv("GRAPH_EDIT_GATE_MIN_CONFIDENCE", "0.85"), 0.85)
    allowed = (
        decision == "PERMANENT"
        and validity == "support"
        and conflict_type == "none"
        and confidence >= threshold
    )
    reason = str(payload.get("reason", "validator provided no reason")).strip()
    if not allowed and decision == "PERMANENT" and confidence < threshold:
        decision = "TENTATIVE"
        reason = f"confidence {confidence:.3f} is below commit threshold {threshold:.3f}; {reason}"
    return GraphEditGateDecision(
        allowed=allowed,
        decision=decision,
        validity=validity,
        conflict_type=conflict_type,
        confidence=confidence,
        reason=reason,
        model=model,
    )


def _reject(*, reason: str, conflict_type: str, model: str) -> GraphEditGateDecision:
    return GraphEditGateDecision(
        allowed=False,
        decision="TENTATIVE",
        validity="insufficient",
        conflict_type=conflict_type,
        confidence=0.0,
        reason=reason,
        model=model,
    )


def _load_project_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    project_root = Path(__file__).resolve().parents[4]
    load_dotenv(project_root / ".env", override=False)


def _positive_float(raw: Any, default: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


__all__ = [
    "GraphEditGateDecision",
    "graph_edit_commit_gate_enabled",
    "validate_graph_edit_commit",
]
