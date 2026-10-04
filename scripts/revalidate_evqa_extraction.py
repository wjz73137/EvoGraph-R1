#!/usr/bin/env python3
"""Revalidate saved model outputs on CPU; preserve the original inference artifacts."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evograph_mm.kb.gldv2_subset import atomic_json
from scripts.compare_evqa_extraction import (
    OUTPUT_ROOT, ROOT, parse_facts, validate_facts,
)


def main():
    previous = ROOT / 'expr_mm/evqa_extraction_comparison_v1'
    for arm in ['whole_3b', 'whole_7b', 'extractive_7b']:
        origin = previous / arm
        output = OUTPUT_ROOT / arm
        if output.exists():
            raise RuntimeError(f'existing output retained: {output}')
        report = json.loads((origin / 'report.json').read_text())
        calls = json.loads((origin / 'calls.json').read_text())
        for doc in report['results']:
            raw = next(c['raw_output'] for c in calls if c['document'] == doc['title'])
            accepted, rejected = validate_facts(
                parse_facts(raw), doc['source'], arm == 'extractive_7b')
            doc.update(facts=accepted, rejected=rejected)
        report.update(accepted_count=sum(len(d['facts']) for d in report['results']),
                      rejected_count=sum(len(d['rejected']) for d in report['results']),
                      validation_replayed_from=str(origin), new_gpu_inference=False)
        output.mkdir(parents=True)
        atomic_json(output / 'report.json', report)
        atomic_json(output / 'owner.json', {
            'source_inference_dir': str(origin), 'validation_version': 'v2',
            'new_gpu_inference': False})
        print(arm, report['accepted_count'], report['rejected_count'])


if __name__ == '__main__':
    main()
