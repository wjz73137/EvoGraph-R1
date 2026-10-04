"""Strict source-grounded API extraction contracts used by production graph builds."""
from __future__ import annotations

import json
import re
from typing import Any


STRICT_FACT_SYSTEM = """You extract knowledge facts from one supplied passage.
Return JSON only. Use no outside knowledge. Every fact must be explicitly supported by the passage.
Preserve uncertainty, alternatives, scope, dates, comparisons, and limiting phrases exactly.
Resolve pronouns only when the passage makes the referent unambiguous.
Use atomic facts, except that a shared uncertainty or A-or-B alternative must remain in one fact.
Do not calculate an unstated date. Do not complete a cut-off sentence.
Avoid semantic duplicates and do not infer a proper name for an unnamed place.
Output: {"facts":[{"statement":"...","evidence":"an exact contiguous quote from the passage"}]}.
The evidence value must be copied verbatim and must entail the whole statement."""

EVIDENCE_REPAIR_SYSTEM = """You repair evidence fields without changing any fact statement.
Return JSON only. For every supplied item, copy one exact contiguous quote from the passage that entails
the complete statement. Do not use ellipses, normalize whitespace, add outside knowledge, edit statements,
or return additional items.
Output: {"repairs":[{"fact_index":0,"evidence":"exact contiguous source quote"}]}"""

FACT_ANCHOR_REPAIR_SYSTEM = """You repair only the supplied knowledge-fact statements.
Each repaired statement must be self-contained and must explicitly contain the supplied article title,
spelled exactly, as its subject or context entity. Preserve the original claim, qualifications, quantities,
and scope; use only the supplied passage. Do not add a new claim. Copy one exact contiguous passage quote
that supports the complete repaired statement. Return exactly one repair per supplied fact index.
Output: {"repairs":[{"fact_index":0,"statement":"...","evidence":"exact source quote"}]}"""

ENTITY_LINK_SYSTEM = """You link already validated facts to explicit entity mentions for a knowledge graph.
Return JSON only. Do not add, remove, merge, split, or rewrite facts.
For every fact index, list all useful named entities and explicit dates/times that occur as exact contiguous
substrings of that fact statement. Entity names must preserve the statement's spelling; never resolve an
unnamed place using outside knowledge. Generic words such as bridge, castle, island, year, and centre are not
entities by themselves. Named physical structures (including houses, castles, bridges, roads, and their
aliases) are LOCATION. Nationality or language adjectives such as Welsh and British are CATEGORY unless the
text names an actual institution. A person-like string used explicitly as a structure alias inherits LOCATION;
do not classify it as PERSON. Allowed types are PERSON, ORGANIZATION, LOCATION, EVENT, TIME, CATEGORY, PRODUCT.
Every fact must have at least one entity. Output:
{"links":[{"fact_index":0,"entities":[{"name":"exact fact substring","type":"LOCATION"}]}]}"""

ENTITY_TYPES = {
    "PERSON", "ORGANIZATION", "LOCATION", "EVENT", "TIME", "CATEGORY", "PRODUCT"
}


def parse_json_object(text: str) -> dict[str, Any]:
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


def validate_facts(payload: dict[str, Any], source: str, *, require_exact: bool) -> list[dict[str, Any]]:
    raw_facts = payload.get("facts")
    if not isinstance(raw_facts, list) or not raw_facts:
        raise ValueError("fact extraction must contain a non-empty facts list")
    facts = []
    for index, raw in enumerate(raw_facts):
        if not isinstance(raw, dict):
            raise ValueError(f"fact {index} is not an object")
        statement, evidence = raw.get("statement"), raw.get("evidence")
        if not isinstance(statement, str) or not statement.strip():
            raise ValueError(f"fact {index} has no statement")
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError(f"fact {index} has no evidence")
        exact = evidence in source
        if require_exact and not exact:
            raise ValueError(f"fact {index} evidence is not an exact contiguous source span")
        facts.append({"statement": statement.strip(), "evidence": evidence,
                      "evidence_is_exact_source_span": exact})
    normalized = [re.sub(r"\s+", " ", fact["statement"]).casefold() for fact in facts]
    if len(set(normalized)) != len(normalized):
        raise ValueError("fact extraction contains exact normalized duplicates")
    return facts


def invalid_evidence_indexes(facts: list[dict[str, Any]]) -> list[int]:
    return [index for index, fact in enumerate(facts)
            if not fact.get("evidence_is_exact_source_span")]


def _recover_unique_whitespace_equivalent_span(source: str, evidence: str) -> str | None:
    """Return the exact source span when only whitespace code points differ.

    Recovery is deliberately conservative: the whitespace-normalized evidence must
    occur exactly once in the normalized source. The returned value is sliced from
    the original source, so downstream storage still satisfies byte-for-byte span
    validation (for example, when Wikipedia contains a non-breaking space).
    """
    normalized_source: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    index = 0
    while index < len(source):
        start = index
        if source[index].isspace():
            while index < len(source) and source[index].isspace():
                index += 1
            normalized_source.append(" ")
        else:
            normalized_source.append(source[index])
            index += 1
        starts.append(start)
        ends.append(index)
    needle = re.sub(r"\s+", " ", evidence)
    haystack = "".join(normalized_source)
    first = haystack.find(needle)
    if first < 0 or haystack.find(needle, first + 1) >= 0:
        return None
    last = first + len(needle) - 1
    return source[starts[first]:ends[last]]


def _recover_unique_case_equivalent_span(source: str, evidence: str) -> str | None:
    """Recover a unique source slice when the API changed only letter case."""
    matches = list(re.finditer(re.escape(evidence), source, flags=re.IGNORECASE))
    if len(matches) != 1:
        return None
    return matches[0].group(0)


def apply_evidence_repairs(
    facts: list[dict[str, Any]], payload: dict[str, Any], source: str,
) -> list[dict[str, Any]]:
    invalid = set(invalid_evidence_indexes(facts))
    repairs = payload.get("repairs")
    if not isinstance(repairs, list) or len(repairs) != len(invalid):
        raise ValueError("evidence repair must return exactly one item per invalid fact")
    seen = set()
    repaired = [dict(fact) for fact in facts]
    for raw in repairs:
        if not isinstance(raw, dict) or not isinstance(raw.get("fact_index"), int):
            raise ValueError("invalid evidence repair item")
        index, evidence = raw["fact_index"], raw.get("evidence")
        if index not in invalid or index in seen:
            raise ValueError("evidence repair returned an unexpected or duplicate fact index")
        if not isinstance(evidence, str):
            raise ValueError(f"evidence repair {index} is not an exact source span")
        exact_evidence = evidence if evidence in source else _recover_unique_whitespace_equivalent_span(
            source, evidence
        ) or _recover_unique_case_equivalent_span(source, evidence)
        if exact_evidence is None:
            raise ValueError(f"evidence repair {index} is not an exact source span")
        repaired[index]["evidence"] = exact_evidence
        repaired[index]["evidence_is_exact_source_span"] = True
        seen.add(index)
    if seen != invalid:
        raise ValueError("evidence repair omitted invalid facts")
    return repaired


def apply_fact_anchor_repairs(
    facts: list[dict[str, Any]], payload: dict[str, Any], source: str,
    indexes: list[int], title: str,
) -> list[dict[str, Any]]:
    expected = set(indexes)
    repairs = payload.get("repairs")
    if not isinstance(repairs, list) or len(repairs) != len(expected):
        raise ValueError("fact-anchor repair must return exactly one item per requested fact")
    repaired = [dict(fact) for fact in facts]
    seen = set()
    for raw in repairs:
        if not isinstance(raw, dict) or not isinstance(raw.get("fact_index"), int):
            raise ValueError("invalid fact-anchor repair item")
        index = raw["fact_index"]
        if index not in expected or index in seen:
            raise ValueError("fact-anchor repair returned an unexpected or duplicate fact index")
        statement, evidence = raw.get("statement"), raw.get("evidence")
        if not isinstance(statement, str) or title not in statement:
            raise ValueError(f"fact-anchor repair {index} does not contain the exact article title")
        if not isinstance(evidence, str):
            raise ValueError(f"fact-anchor repair {index} has invalid evidence")
        exact_evidence = evidence if evidence in source else _recover_unique_whitespace_equivalent_span(
            source, evidence
        ) or _recover_unique_case_equivalent_span(source, evidence)
        if exact_evidence is None:
            raise ValueError(f"fact-anchor repair {index} evidence is not an exact source span")
        repaired[index] = {
            "statement": statement.strip(),
            "evidence": exact_evidence,
            "evidence_is_exact_source_span": True,
        }
        seen.add(index)
    return validate_facts({"facts": repaired}, source, require_exact=True)


def validate_entity_links(
    payload: dict[str, Any], facts: list[dict[str, Any]], *, require_entities: bool = True,
) -> list[dict[str, Any]]:
    raw_links = payload.get("links")
    if not isinstance(raw_links, list) or len(raw_links) != len(facts):
        raise ValueError("entity linker must return exactly one record per fact")
    by_index: dict[int, dict[str, Any]] = {}
    for raw in raw_links:
        if not isinstance(raw, dict) or not isinstance(raw.get("fact_index"), int):
            raise ValueError("invalid entity-link record")
        index = raw["fact_index"]
        if index in by_index or not 0 <= index < len(facts):
            raise ValueError("entity linker returned an invalid or repeated fact index")
        entities = raw.get("entities")
        if not isinstance(entities, list) or (require_entities and not entities):
            raise ValueError(f"fact {index} has no linked entities")
        statement = facts[index]["statement"]
        checked, seen = [], set()
        for entity in entities:
            if not isinstance(entity, dict):
                raise ValueError(f"fact {index} has a non-object entity")
            name, entity_type = entity.get("name"), entity.get("type")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"fact {index} has an empty entity name")
            name = name.strip()
            if name.casefold() not in statement.casefold():
                raise ValueError(f"entity {name!r} is not an exact fact substring")
            if entity_type not in ENTITY_TYPES:
                raise ValueError(f"entity {name!r} has unsupported type {entity_type!r}")
            key = (name.casefold(), entity_type)
            if key not in seen:
                checked.append({"name": name, "type": entity_type})
                seen.add(key)
        by_index[index] = {"fact_index": index, "entities": checked}
    return [by_index[index] for index in range(len(facts))]
