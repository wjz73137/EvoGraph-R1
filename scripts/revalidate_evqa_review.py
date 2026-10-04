#!/usr/bin/env python3
"""CPU-only JSON shape normalization; never changes statements or quotations."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evograph_mm.kb.gldv2_subset import atomic_json
from scripts.compare_evqa_extraction import validate_facts
from scripts.compare_evqa_extraction_advanced import OUTPUT_ROOT


def normalize_review_response(raw):
    data = json.loads(raw)
    if isinstance(data, list):
        return data, True
    if isinstance(data, dict) and isinstance(data.get('facts'), list):
        return data['facts'], False
    raise ValueError('review response is neither a facts object nor a facts array')


def main():
    origin = OUTPUT_ROOT / 'review_7b'
    output = OUTPUT_ROOT / 'review_7b_array_normalized'
    if output.exists():
        raise RuntimeError('existing normalized results retained')
    report = json.loads((origin / 'report.json').read_text())
    if report['status'] != 'complete':
        raise ValueError('incomplete source report')
    calls = json.loads((origin / 'calls.json').read_text())
    report['strict_protocol_parse_error_count'] = report['parse_error_count']
    deviations = 0
    for doc in report['results']:
        call = next(c for c in calls if c['document'] == doc['title'])
        facts, changed = normalize_review_response(call['raw_output'])
        accepted, rejected = validate_facts(facts, doc['source'])
        doc.update(facts=accepted, rejected=rejected, errors=[],
                   json_shape_normalized=changed)
        deviations += int(changed)
    report.update(accepted_count=sum(len(d['facts']) for d in report['results']),
                  rejected_count=sum(len(d['rejected']) for d in report['results']),
                  parse_error_count=0, output_shape_deviations=deviations,
                  new_gpu_inference=False, validation_replayed_from=str(origin),
                  normalization='wrap bare JSON arrays as facts arrays; content unchanged')
    output.mkdir(parents=True)
    atomic_json(output / 'owner.json', {
        'source_inference_dir': str(origin), 'new_gpu_inference': False,
        'source_report_sha256': hashlib.sha256((origin / 'report.json').read_bytes()).hexdigest(),
        'calls_sha256': hashlib.sha256((origin / 'calls.json').read_bytes()).hexdigest(),
        'normalizer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    atomic_json(output / 'report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'results'}))


if __name__ == '__main__':
    main()
