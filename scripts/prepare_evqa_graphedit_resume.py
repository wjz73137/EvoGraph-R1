#!/usr/bin/env python3
"""Preserve a failed run and seed a new output with checkpoint-retained metrics.

Refuses graph mutations newer than the checkpoint. Keeps the failed journal intact;
only metrics/trajectories at or before the restored model step count in the new run.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys


def copy_graphs(source, output, graph_audit, graph_root=None):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from evograph_mm.kb.graph_edit import prepare_edit_working_dir
    destination = output / 'isolated_graphs'
    graph_root = Path(graph_root) if graph_root else source / 'isolated_graphs'
    if destination.exists():
        raise RuntimeError('resume graph destination already exists')
    for name, audit in graph_audit.items():
        target = prepare_edit_working_dir(graph_root / name, destination / name)
        for filename, expected in audit['core_sha256'].items():
            if hashlib.sha256((target / filename).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'graph copy mismatch: {name}/{filename}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--copy-graphs-only', action='store_true', help='finish graph copies for an already prepared output')
    args = parser.parse_args()
    if args.copy_graphs_only:
        manifest_path = args.output / 'resume_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if manifest['source'] != str(args.source.resolve()) or manifest['next_logged_step'] != args.step + 1:
            raise RuntimeError('resume manifest does not match the requested source/step')
        copy_graphs(args.source, args.output, manifest['graph_audit'], manifest.get('graph_source'))
        manifest['independent_graph_copies'] = True
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        print('Independent resume graph copies verified')
        return
    if args.output.exists():
        raise RuntimeError('resume output already exists; refusing to overwrite it')
    checkpoint = args.source / 'checkpoints' / f'global_step_{args.step}'
    data = checkpoint / 'data.pt'
    for rank in (0, 1):
        for kind in ('model', 'optim', 'extra_state'):
            shard = checkpoint / 'actor' / f'{kind}_world_size_2_rank_{rank}.pt'
            if not shard.is_file() or not shard.stat().st_size:
                raise RuntimeError(f'missing checkpoint shard: {shard}')
    checkpoint_time = data.stat().st_mtime
    graph_root = checkpoint / 'graph_state'
    paired_graph_checkpoint = (graph_root / 'manifest.json').is_file()
    if not paired_graph_checkpoint:
        graph_root = args.source / 'isolated_graphs'
    paired_hashes = json.loads((graph_root / 'manifest.json').read_text()) if paired_graph_checkpoint else None
    graph_audit = {}
    for graph in sorted(graph_root.iterdir()):
        if not graph.is_dir() or not graph.name.startswith(('train_', 'val_')):
            continue
        journal = graph / 'hyperedge_recent_mutations.json'
        mutations = json.loads(journal.read_text()).get('mutations', []) if journal.exists() else []
        for mutation in mutations:
            if datetime.fromisoformat(mutation['timestamp']).timestamp() > checkpoint_time:
                raise RuntimeError(f'{graph.name}: graph has a post-checkpoint mutation')
        graph_audit[graph.name] = {'mutations': len(mutations), 'core_sha256': {
            name: hashlib.sha256((graph / name).read_bytes()).hexdigest()
            for name in ('kv_store_entities.json', 'kv_store_hyperedges.json',
                         'graph_chunk_entity_relation.graphml')}}
        if paired_hashes is not None:
            for filename, actual in graph_audit[graph.name]['core_sha256'].items():
                if paired_hashes.get(graph.name, {}).get(filename) != actual:
                    raise RuntimeError(f'paired graph checkpoint hash mismatch: {graph.name}/{filename}')
    if not graph_audit:
        raise RuntimeError('no graph copies found for resume')
    results = args.source / 'expr_results' / args.experiment
    rows = [json.loads(line) for line in (results / 'evals_training.jsonl').read_text().splitlines() if line.strip()]
    if not paired_graph_checkpoint and any(row.get('trajectory/successful_graph_edit_count/mean', 0) for row in rows
           if int(row['global_steps']) > args.step):
        raise RuntimeError('failed run contains successful graph edits after checkpoint')
    retained = [row for row in rows if int(row['global_steps']) <= args.step]
    if [int(row['global_steps']) for row in retained] != list(range(327, args.step + 1)):
        raise RuntimeError('retained optimizer metric steps are not contiguous')
    # All checks precede creating a destination. Never modify the failed source.
    output_results = args.output / 'expr_results' / args.experiment
    output_results.mkdir(parents=True)
    (output_results / 'evals_training.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in retained))
    for row in retained:
        name = f'train_trajectories_step{int(row["global_steps"])}.json'
        shutil.copy2(results / name, output_results / name)
    manifest = {'source': str(args.source.resolve()), 'checkpoint': str(checkpoint.resolve()),
                'checkpoint_data_sha256': hashlib.sha256(data.read_bytes()).hexdigest(),
                'retained_updates': len(retained), 'discarded_optimizer_updates_to_replay': len(rows)-len(retained),
                'next_logged_step': args.step+1, 'reset_dataloader_on_resume': False,
                'graph_audit': graph_audit, 'graph_source': str(graph_root.resolve()),
                'paired_graph_checkpoint': paired_graph_checkpoint, 'graph_resume_limitation':
                None if paired_graph_checkpoint else
                'No graph snapshot at model checkpoint; mutation journal and zero later successful edits agree. '
                'Failed edit rollback can rewrite mtimes; API cache histories are not rewound.'}
    (args.output / 'resume_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    copy_graphs(args.source, args.output, graph_audit, graph_root)
    manifest['independent_graph_copies'] = True
    (args.output / 'resume_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
