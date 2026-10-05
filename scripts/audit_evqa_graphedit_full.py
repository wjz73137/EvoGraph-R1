#!/usr/bin/env python3
"""Audit full GraphEdit split isolation and writable graph copies before training."""
import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


ROOT = Path('/home/data/dataset/wjz/EvoGraph-R1')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'datasets_mm/E-VQA/processed/paper_grpo_graph_edit_full1891_v1')
    parser.add_argument('--graphs', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.dataset / 'manifest.json').read_text())
    isolates = json.loads((args.graphs / 'isolation_manifest.json').read_text())
    perturbations = json.loads((ROOT / 'expr_mm/evqa_graphedit_controlled_missing_v1/controlled_missing_manifest.json').read_text())
    splits = {}
    for split, expected_count in (('train', 1862), ('test', 16)):
        path = args.dataset / f'{split}.parquet'
        assert sha256(path) == manifest['splits'][split]['sha256'], f'{split} parquet hash'
        rows = pq.read_table(path).to_pylist()
        assert len(rows) == expected_count, f'{split} row count'
        images = set()
        documents = set()
        for row in rows:
            extra = row['extra_info']
            assert row['data_source'] == 'E-VQA/paper_grpo_graph_edit_full1891_v1'
            assert extra['question'] in row['prompt'][0]['content']
            assert row['reward_model']['ground_truth'], 'missing reward answer'
            images.add(extra['image_id'])
            documents.add(extra['original_metadata']['graph_document_id'])
            image_path = Path(extra['image_path'])
            assert image_path.is_file(), str(image_path)
            with Image.open(image_path) as image:
                image.verify()
        splits[split] = {'rows': len(rows), 'images': images, 'documents': documents}
    assert not splits['train']['images'] & splits['test']['images'], 'shared train/validation image'
    assert not splits['train']['documents'] & splits['test']['documents'], 'shared train/validation document'
    copies = []
    core_names = ('kv_store_entities.json', 'kv_store_hyperedges.json',
                  'graph_chunk_entity_relation.graphml', 'index_hyperedge.bin',
                  'corpus_hyperedge.npy', 'mm_store/bge_graph/index_hyperedge.bin',
                  'mm_store/bge_graph/corpus_hyperedge.npy')
    for item in isolates['copies']:
        target = Path(item['working_dir'])
        source = Path(item['source'])
        meta = json.loads((target / 'metadata.json').read_text())
        assert meta['graph_edit_copy'] is True
        assert Path(meta['output_dir']).resolve() == target.resolve()
        assert Path(meta['base_output_dir']).resolve() == Path(perturbations['base_working_dir']).resolve()
        hashes = {}
        for name in core_names:
            assert (target / name).is_file(), name
            assert not (target / name).samefile(source / name), f'copy shares inode: {name}'
            hashes[name] = sha256(target / name)
            assert hashes[name] == sha256(source / name), f'copy differs: {name}'
        edges = json.loads((target / 'kv_store_hyperedges.json').read_text())
        hidden = [key for key, edge in edges.items() if edge.get('searchable', True) is False]
        expected_hidden = 0 if item['name'].startswith('train_') else 8
        assert len(hidden) == expected_hidden, f'{item["name"]}: hidden count'
        for spec in perturbations['controlled_hyperedges'].values():
            assert (spec['hyperedge_id'] in hidden) == bool(expected_hidden)
        copies.append({'name': item['name'], 'working_dir': str(target),
                       'hidden_facts': len(hidden), 'core_sha256': hashes})
    assert len(copies) == 6
    report = {'status': 'passed', 'splits': {name: {'rows': item['rows'],
              'unique_images': len(item['images']), 'unique_documents': len(item['documents'])}
              for name, item in splits.items()}, 'shared_images': 0, 'shared_documents': 0,
              'removed_for_isolation': manifest['removed_for_validation_isolation'], 'copies': copies}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'copies'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
