import json

import pytest

from scripts.compare_evqa_extraction_advanced import extract_document, validate_items, window_spans
from scripts.revalidate_evqa_review import normalize_review_response
from scripts.summarize_evqa_extraction_advanced import coverage


def test_inventory_rejects_invented_evidence_and_incomplete_tail():
    source = 'The hall opened in 1900. It was inaugurated by then'
    raw = json.dumps({'items': [
        {'label': 'date', 'evidence': 'The hall opened in 1900.'},
        {'label': 'invented', 'evidence': 'The hall opened in 2000.'},
        {'label': 'tail', 'evidence': 'It was inaugurated by then'}]})
    valid, rejected = validate_items(raw, source)
    assert len(valid) == 1
    assert not valid[0]['label_semantics_verified']
    assert {r['reason'] for r in rejected} == {'inventory_evidence_not_source', 'incomplete_inventory_tail'}


def test_overlapping_windows_preserve_abbreviations_and_skip_tail():
    source = 'The hall (lit. Great Hall) opened. It is in the U.S. state of Iowa. It closed. Fragment'
    windows = [source[a:b] for a, b in window_spans(source)]
    assert len(windows) == 2
    assert 'lit. Great Hall' in windows[0]
    assert 'Fragment' not in windows[1]
    assert len(window_spans('One sentence.')) == 1
    assert window_spans('Fragment') == []


@pytest.mark.parametrize('arm', ['inventory_7b', 'entity_7b'])
def test_two_stage_protocol_uses_only_source_verified_hints(arm):
    source = 'The hall opened in 1900.'
    stages = []
    def generate(stage, rules, user):
        stages.append(stage)
        if stage == 'inventory':
            return json.dumps({'items': [{'label': 'date', 'evidence': source},
                                        {'label': 'secret_untrusted_hint', 'evidence': 'Invented.'}]})
        assert 'secret_untrusted_hint' not in user
        return json.dumps({'facts': [{'statement': source, 'evidence': source}]})
    result = extract_document(arm, 'Hall', source, generate)
    assert stages == ['inventory', 'facts']
    assert len(result['facts']) == 1
    assert not result['facts'][0]['semantic_entailment_verified']


def test_review_is_untrusted_draft_not_automatic_entailment_verification():
    source = 'The hall probably opened in 1900.'
    draft = [{'statement': 'The hall opened in 1900.', 'evidence': source}]
    def generate(stage, rules, user):
        assert stage == 'review'
        assert 'untrusted' in rules.lower()
        assert json.dumps(draft, ensure_ascii=False) in user
        return json.dumps({'facts': [{'statement': source, 'evidence': source}]})
    result = extract_document('review_7b', 'Hall', source, generate, draft)
    assert result['facts'][0]['statement'] == source
    assert not result['facts'][0]['semantic_entailment_verified']
    with pytest.raises(ValueError):
        extract_document('review_7b', 'Hall', source, generate)


def test_window_deduplication_and_parse_failures_are_recorded():
    source = 'First fact. Second fact. Third fact.'
    def generate(stage, rules, user):
        return json.dumps({'facts': [{'statement': 'Second fact.', 'evidence': 'Second fact.'}]})
    result = extract_document('window_7b', 'Article', source, generate)
    assert len(result['facts']) == 1
    assert result['exact_duplicates_removed'] == 1
    failed = extract_document('window_7b', 'Article', source, lambda *args: 'invalid JSON')
    assert not failed['facts']
    assert len(failed['errors']) == 2


def test_review_shape_normalization_is_lossless_and_rejects_other_shapes():
    facts = [{'statement': 'Probably opened.', 'evidence': 'Probably opened.'}]
    assert normalize_review_response(json.dumps(facts)) == (facts, True)
    assert normalize_review_response(json.dumps({'facts': facts})) == (facts, False)
    with pytest.raises(ValueError):
        normalize_review_response('{"items": []}')


def test_coverage_checks_provenance_and_does_not_include_tail():
    doc = {'source': 'One fact. Fragment', 'facts': [
        {'source_start': 0, 'source_end': 9, 'evidence': 'One fact.',
         'semantic_entailment_verified': False}]}
    assert coverage(doc) == (8, 8)
    doc['facts'][0]['evidence'] = 'Other fact.'
    with pytest.raises(ValueError):
        coverage(doc)
