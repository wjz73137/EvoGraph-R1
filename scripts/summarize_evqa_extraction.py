#!/usr/bin/env python3
"""Summarize measured resource use and evidence preservation, without QA scoring."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evograph_mm.kb.gldv2_subset import atomic_json
from scripts.compare_evqa_extraction import ARMS, OUTPUT_ROOT, sentence_spans


def main():
    rows = []
    for arm in ARMS:
        path = OUTPUT_ROOT / arm / 'report.json'
        report = json.loads(path.read_text())
        if report['status'] != 'complete':
            raise RuntimeError(f'incomplete experiment: {arm}')
        covered_count = required_count = 0
        docs = []
        for doc in report['results']:
            source = doc['source']
            required = {
                i for start, end, text in sentence_spans(source)
                if text[-1:] in '.!?' for i in range(start, end)
                if not source[i].isspace()
            }
            covered = {
                i for fact in doc['facts']
                for i in range(fact['source_start'], fact['source_end'])
            }
            verified = all(
                source[fact['source_start']:fact['source_end']] == fact['evidence']
                for fact in doc['facts']
            )
            if not verified:
                raise RuntimeError(f'source provenance verification failed: {arm}')
            covered_count += len(covered & required)
            required_count += len(required)
            docs.append({'title': doc['title'], 'record_count': len(doc['facts']),
                         'evidence_characters_covered': len(covered & required),
                         'complete_source_characters': len(required),
                         'source_spans_verified': verified})
        rows.append({
            'arm': arm, 'model': ARMS[arm][0], 'mode': ARMS[arm][1],
            'accepted_records': report['accepted_count'],
            'rejected_records': report['rejected_count'],
            'parse_errors': report['parse_error_count'],
            'token_limit_count': report['token_limit_count'],
            'inference_seconds': report['elapsed_seconds'],
            'peak_gpu_allocated_mib': report['peak_gpu_allocated_mib'],
            'model_calls': report['calls'],
            'evidence_character_coverage_percent': round(100*covered_count/required_count, 2),
            'documents': docs, 'report': str(path),
            'reused_inference_outputs': not report.get('new_gpu_inference', True),
        })
    summary = {
        'scope': 'same four existing Wikipedia passages; text construction only',
        'arms': rows,
        'metric_definition': 'Evidence character coverage measures preservation of non-whitespace characters in complete source sentences. It is NOT fact recall, semantic accuracy, QA accuracy, or a paper benchmark score.',
        'manual_review': {
            'whole_3b': 'Statements omit probable builders, aliases, and multiple historical/structural facts even when the cited source paragraph contains them; a bridge description loses the for-the-most-part qualifier.',
            'whole_7b': 'Still omits aliases, mosque/fountain and multiple bridge details; a nonverbatim fabricated quotation and incomplete source tail were rejected.',
            'sentence_7b': 'Recovers mosque/fountain and bridge mechanics, but still omits some aliases, the capital location and the 1990s restoration period; the conventional-bridge statement loses the for-the-most-part qualifier. Pronouns and semantic duplicates also need review. A real quote does not prove the statement it accompanies.',
            'extractive_7b': 'After whitespace-only restoration and removal of the unfinished tail, the 23 verbatim fragments preserve every complete source sentence in these four passages. Fragments include pronouns and multiple facts; they are evidence records rather than validated atomic hyperedges.',
        },
        'recommended_evidence_backbone': 'extractive_7b',
        'recommended_atomic_extraction_candidate': 'sentence_7b',
        'production_graph_replaced': False,
        'training_started': False,
        'remote_api_calls': False,
        'all_factual_quality_passed': False,
    }
    target = OUTPUT_ROOT / 'summary.json'
    if target.exists():
        raise RuntimeError('existing summary retained')
    atomic_json(target, summary)
    for row in rows:
        print(json.dumps({k: v for k, v in row.items() if k != 'documents'}))


if __name__ == '__main__':
    main()
