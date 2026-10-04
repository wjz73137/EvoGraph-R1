#!/usr/bin/env python3
"""Verify source identity and summarize all eight diagnostic extraction arms."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evograph_mm.kb.gldv2_subset import atomic_json
from scripts.compare_evqa_extraction import BASELINE, ROOT, sentence_spans
from scripts.compare_evqa_extraction_advanced import ARMS, OUTPUT_ROOT

MANUAL_REVIEW = {
    'inventory_7b': {
        'finding': 'The checklist still drops aliases and important clauses. Several intermediate quotes concatenate or paraphrase source text and are rejected. Pamban first-sea/longest-sea statements quote Open instead of Opened and are rejected.',
        'example_rejected_evidence': 'Open on 24 February 1914, it was India\'s first sea bridge, and was the longest sea bridge in India until the opening of the Bandra-Worli Sea Link in 2010.',
    },
    'entity_7b': {
        'finding': 'The Büyük Han entity inventory repeats until the token limit; the subsequent facts call falls back to an empty inventory and must not be counted as a successful two-stage extraction for that document. Inventories often paraphrase rather than quote. Alternative possible builders are split into separate probable-builder claims, losing explicit disjunction.',
        'truncated_document': 'Büyük Han', 'truncated_stage': 'inventory',
    },
    'review_7b': {
        'finding': 'Three JSON array responses are losslessly normalized on CPU; original strict reports retained. Against the prior 37 sentence candidates, the 36 normalized statements add nothing and remove the Newport probable-builders statement. The unqualified conventional-bridge statement is unchanged.',
        'removed_supported_candidate': 'Newport Castle was probably built by Hugh de Audley, 1st Earl of Gloucester or his son-in-law, Ralph, Earl of Stafford.',
        'new_statement_count_relative_to_draft': 0,
    },
    'window_7b': {
        'finding': 'Retains the bridge for-the-most-part qualifier and the 1990s restoration sentence. However, overlapping windows introduce paraphrase duplicates even after seven exact duplicates are removed. A Newport builder statement drops probably; many subjects remain generic or pronouns. More records do not mean more correct atomic facts.',
        'unsupported_strengthening': 'Newport Castle was built by Hugh de Audley, 1st Earl of Gloucester or his son-in-law, Ralph, Earl of Stafford',
        'source_qualifier': 'probably',
    },
}


def coverage(doc):
    source = doc['source']
    required = {i for start, end, text in sentence_spans(source)
                if text[-1:] in '.!?' for i in range(start, end)
                if not source[i].isspace()}
    covered = set()
    for fact in doc['facts']:
        start, end = fact['source_start'], fact['source_end']
        if not (0 <= start < end <= len(source)) or source[start:end] != fact['evidence']:
            raise ValueError('evidence provenance verification failed')
        if fact['semantic_entailment_verified']:
            raise ValueError('these experiments cannot certify entailment')
        covered.update(range(start, end))
    return len(required & covered), len(required)


def main():
    documents = json.loads(BASELINE.read_text())['documents']
    expected = [(d['source_metadata']['wikipedia_title'], d['contents'].split('\n', 2)[2])
                for d in documents]
    previous = ROOT / 'expr_mm/evqa_extraction_comparison_v2'
    paths = [(name, previous / name / 'report.json') for name in
             ('whole_3b', 'whole_7b', 'sentence_7b', 'extractive_7b')]
    paths += [(name, OUTPUT_ROOT / ('review_7b_array_normalized' if name == 'review_7b' else name)
               / 'report.json') for name in ARMS]
    rows = []
    for name, path in paths:
        report = json.loads(path.read_text())
        if report['status'] != 'complete' or len(report['results']) != len(expected):
            raise ValueError(f'incomplete report: {name}')
        if [(d['title'], d['source']) for d in report['results']] != expected:
            raise ValueError(f'comparison input mismatch: {name}')
        if report['training_started'] or report['remote_api_calls']:
            raise ValueError('unexpected experiment scope')
        per_doc = [{'title': d['title'], 'counts': coverage(d), 'records': len(d['facts'])}
                   for d in report['results']]
        numerator = sum(d['counts'][0] for d in per_doc)
        denominator = sum(d['counts'][1] for d in per_doc)
        row = {'arm': name, 'accepted_records': report['accepted_count'],
               'rejected_records': report['rejected_count'],
               'parse_errors': report['parse_error_count'],
               'strict_protocol_parse_errors': report.get('strict_protocol_parse_error_count', report['parse_error_count']),
               'output_shape_deviations': report.get('output_shape_deviations', 0),
               'token_limit_count': report['token_limit_count'],
               'exact_duplicates_removed': report.get('exact_duplicates_removed'),
               'inference_seconds': report['elapsed_seconds'],
               'peak_gpu_allocated_mib': report['peak_gpu_allocated_mib'],
               'calls': report['calls'],
               'evidence_character_coverage_percent': round(100*numerator/denominator, 2),
               'documents': per_doc, 'report': str(path)}
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != 'documents'}), flush=True)
    summary = {'scope': 'same four Wikipedia text passages; eight local extraction arms',
               'arms': rows, 'manual_review': MANUAL_REVIEW,
               'recommended_evidence_backbone': 'extractive_7b',
               'recommended_atomic_candidate_pending_semantic_review': 'sentence_7b',
               'selection_caveat': 'No new arm is accepted for production; this four-passage diagnostic cannot establish general superiority.',
               'metric_caveat': 'Record count is not correct fact count; paraphrases and overlapping quotes can duplicate facts. Evidence character coverage is not semantic accuracy, fact recall, multimodal QA accuracy, or an official paper benchmark.',
               'timing_caveat': 'Inference elapsed time includes output validation and saving, excludes checkpoint loading, and was measured in different concurrent batches. It is not a controlled throughput benchmark. Review time excludes the cost of generating its prior sentence draft.',
               'semantic_quality_passed': False, 'production_graph_replaced': False,
               'training_started': False, 'remote_api_calls': False}
    target = OUTPUT_ROOT / 'summary.json'
    if target.exists():
        raise RuntimeError('existing summary retained')
    atomic_json(target, summary)


if __name__ == '__main__':
    main()
