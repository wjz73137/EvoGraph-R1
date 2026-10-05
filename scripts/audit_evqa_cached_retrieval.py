#!/usr/bin/env python3
"""Check dataset vector coverage and compare live search outputs across encoder modes.

Uses questions and image IDs only, never reference answers. Training must be stopped
between snapshot and compare so that graph edits cannot change the comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import requests


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graphs', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()
    report = {'coverage': {}, 'queries': {}, 'responses': {}, 'health': {}}
    for split, names in [('train', ['train_0', 'train_1', 'train_2', 'train_3']),
                         ('test', ['val_0', 'val_1'])]:
        frame = pd.read_parquet(args.dataset / f'{split}.parquet')
        sample = frame.iloc[[0, len(frame)//3, 2*len(frame)//3, len(frame)-1]]
        visual = {'queries': ['<img>'] * len(sample),
                  'image_ids': sample.image_id.tolist(),
                  'context_queries': [item['question'] for item in sample.extra_info],
                  'visual_entity_top_k': 5}
        text = {'queries': [item['question'] for item in sample.extra_info]}
        for name in names:
            graph = args.graphs / name
            ids = set(json.loads((graph / 'mm_store/indexing/image_ids.json').read_text()))
            missing = [image for image in frame.image_id if 'visual_embedding::' + image not in ids]
            if missing:
                raise RuntimeError(f'{name}: missing indexed images: {missing[:5]}')
            report['coverage'][name] = {'rows': len(frame), 'indexed_images': len(frame), 'missing': 0}
            port = 8010 + (int(name[-1]) if name.startswith('train') else 4 + int(name[-1]))
            base = f'http://127.0.0.1:{port}'
            response = requests.get(base + '/status', timeout=30)
            response.raise_for_status()
            health = response.json()
            report['health'][name] = health
            if args.compare and (health.get('runtime_encoder_enabled') or health.get('model_loaded')):
                raise RuntimeError(f'{name}: runtime GME is still enabled')
            if not health.get('bge_graph_index', {}).get('loaded'):
                raise RuntimeError(f'{name}: BGE graph index unavailable')
            for mode, payload in [('visual', visual), ('text', text)]:
                key = name + '/' + mode
                response = requests.post(base + '/search', json=payload, timeout=300)
                response.raise_for_status()
                results = response.json()
                results = [json.loads(item) if isinstance(item, str) else item for item in results]
                if any('error' in item or not item.get('results') for item in results):
                    raise RuntimeError(f'{key}: empty or failed retrieval')
                report['queries'][key] = payload
                report['responses'][key] = digest(results)
    if args.compare:
        previous = json.loads(args.snapshot.read_text())
        for field in ('coverage', 'queries', 'responses'):
            if previous[field] != report[field]:
                raise RuntimeError(f'retrieval changed after switching encoder mode: {field}')
        report['identical_to_snapshot'] = True
        target = args.snapshot.with_name(args.snapshot.stem + '_comparison.json')
    else:
        target = args.snapshot
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'report': str(target), 'coverage': report['coverage'],
                      'identical_to_snapshot': report.get('identical_to_snapshot')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
