import pytest

from evograph_mm.kb.strict_extraction import (
    apply_evidence_repairs,
    apply_fact_anchor_repairs,
    invalid_evidence_indexes,
    parse_json_object,
    validate_entity_links,
    validate_facts,
)


def test_strict_fact_validation_and_targeted_evidence_repair():
    source = "Büyük Han is located in the capital of Cyprus."
    payload = {"facts": [{
        "statement": "Büyük Han is located in the capital of Cyprus.",
        "evidence": "Büyük Han ... is located in the capital of Cyprus.",
    }]}
    facts = validate_facts(payload, source, require_exact=False)
    assert invalid_evidence_indexes(facts) == [0]
    with pytest.raises(ValueError, match="exact contiguous"):
        validate_facts(payload, source, require_exact=True)
    repaired = apply_evidence_repairs(facts, {"repairs": [{
        "fact_index": 0,
        "evidence": source,
    }]}, source)
    assert repaired[0]["statement"] == payload["facts"][0]["statement"]
    assert repaired[0]["evidence_is_exact_source_span"] is True


def test_evidence_repair_recovers_only_a_unique_whitespace_equivalent_span():
    source = "The bridge is 2.345\u00a0km long."
    facts = validate_facts({"facts": [{
        "statement": "The bridge is 2.345 km long.",
        "evidence": "The bridge is 2.345 km long.",
    }]}, source, require_exact=False)
    repaired = apply_evidence_repairs(facts, {"repairs": [{
        "fact_index": 0,
        "evidence": "The bridge is 2.345 km long.",
    }]}, source)
    assert repaired[0]["evidence"] == source
    assert repaired[0]["evidence"] in source

    repeated_source = f"{source} Again: {source}"
    repeated = validate_facts({"facts": [{
        "statement": "The bridge is 2.345 km long.",
        "evidence": "The bridge is 2.345 km long.",
    }]}, repeated_source, require_exact=False)
    with pytest.raises(ValueError, match="exact source span"):
        apply_evidence_repairs(repeated, {"repairs": [{
            "fact_index": 0,
            "evidence": "The bridge is 2.345 km long.",
        }]}, repeated_source)


def test_entity_links_cannot_rewrite_or_invent_entity_names():
    facts = [{"statement": "Newport Castle was sacked by Owain Glyndŵr in 1402."}]
    links = validate_entity_links({"links": [{
        "fact_index": 0,
        "entities": [
            {"name": "Newport Castle", "type": "LOCATION"},
            {"name": "Owain Glyndŵr", "type": "PERSON"},
            {"name": "1402", "type": "TIME"},
        ],
    }]}, facts)
    assert [item["name"] for item in links[0]["entities"]] == [
        "Newport Castle", "Owain Glyndŵr", "1402"
    ]
    with pytest.raises(ValueError, match="not an exact fact substring"):
        validate_entity_links({"links": [{
            "fact_index": 0,
            "entities": [{"name": "Wales", "type": "LOCATION"}],
        }]}, facts)


def test_empty_entity_link_can_be_detected_then_fact_is_anchored():
    source = "Light from the telescope is directed down through a shaft."
    facts = validate_facts({"facts": [{
        "statement": source,
        "evidence": source,
    }]}, source, require_exact=True)
    links = validate_entity_links({"links": [{
        "fact_index": 0,
        "entities": [],
    }]}, facts, require_entities=False)
    assert links[0]["entities"] == []
    repaired = apply_fact_anchor_repairs(facts, {"repairs": [{
        "fact_index": 0,
        "statement": "In the Einstein Tower, light from the telescope is directed down through a shaft.",
        "evidence": source,
    }]}, source, [0], "Einstein Tower")
    assert repaired[0]["evidence"] in source
    assert "Einstein Tower" in repaired[0]["statement"]


def test_json_fence_is_losslessly_removed():
    assert parse_json_object('```json\n{"facts": []}\n```') == {"facts": []}
