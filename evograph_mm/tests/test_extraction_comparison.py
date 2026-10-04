from scripts.compare_evqa_extraction import locate_evidence, parse_facts, sentence_spans, validate_facts


def test_evidence_offsets_and_no_invented_numbers():
    source = 'The hall was built in 1572, the year after Cyprus was seized.'
    valid = {'statement': 'The hall was built in 1572.', 'evidence': source}
    invalid = {'statement': 'Cyprus was seized in 1571.', 'evidence': source}
    absent = {'statement': 'A fact.', 'evidence': 'Invented evidence.'}
    accepted, rejected = validate_facts([valid, invalid, absent, valid], source)
    assert len(accepted) == 1
    assert source[accepted[0]['source_start']:accepted[0]['source_end']] == source
    assert not accepted[0]['semantic_entailment_verified']
    assert {r['reason'] for r in rejected} == {
        'numbers_not_in_evidence', 'evidence_not_exact_source_span', 'exact_duplicate'}


def test_extracting_quotes_does_not_promote_model_paraphrases():
    source = 'The hall was built in 1572.'
    accepted, _ = validate_facts(
        [{'evidence': source, 'statement': 'The hall was seized in 1572.'}], source, True)
    assert accepted[0]['statement'] == source
    assert accepted[0]['representation'] == 'verbatim_quote'


def test_sentence_split_preserves_initials_and_register_abbreviation():
    source = 'The house is also known as John B. Jones. It is on a register in the U.S. National Register. Last fragment'
    sentences = [x[2] for x in sentence_spans(source)]
    assert len(sentences) == 3
    assert 'John B. Jones.' in sentences[0]
    assert 'U.S. National Register.' in sentences[1]
    assert sentences[2] == 'Last fragment'
    assert parse_facts('```json\n{"facts": []}\n```') == []


def test_parenthesized_abbreviation_is_not_sentence_boundary():
    source = 'The hall (lit. Great Hall) is a building. It was restored.'
    assert len(list(sentence_spans(source))) == 2


def test_whitespace_only_repair_preserves_exact_original_span():
    source = 'The bridge stands between Mandapam  and Pamban.'
    quote = 'The bridge stands between Mandapam and Pamban.'
    assert locate_evidence(quote, source) == (0, len(source), source, True)
    assert locate_evidence('The bridge stands between Mandapam and India.', source) is None


def test_reject_incomplete_tail_and_facts_from_other_sentence():
    source = 'The hall opened in 1900. It was restored in 1990. It was inaugurated by then'
    facts = [{'statement': 'The hall opened in 1900.', 'evidence': 'The hall opened in 1900.'},
             {'statement': 'The hall was inaugurated by then', 'evidence': 'It was inaugurated by then'}]
    accepted, rejected = validate_facts(facts, source, target_span=(24, 47))
    assert not accepted
    assert {x['reason'] for x in rejected} == {
        'evidence_outside_target_sentence'}
    _, rejected = validate_facts(facts, source)
    assert rejected[0]['reason'] == 'incomplete_source_tail'
