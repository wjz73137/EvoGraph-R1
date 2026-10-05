"""Disk-backed graph snapshots paired with a drained training checkpoint.

Caller must have finished all synchronous tool calls before invoking this helper.
No graph model is loaded and no mutable-file bytes are retained in memory.
"""
import hashlib
import json
from pathlib import Path


def save_graph_checkpoint(checkpoint_dir, graph_root):
    from evograph_mm.kb.graph_edit import prepare_edit_working_dir

    checkpoint_dir = Path(checkpoint_dir)
    graph_root = Path(graph_root)
    destination = checkpoint_dir / 'graph_state'
    names = ('train_0', 'train_1', 'train_2', 'train_3', 'val_0', 'val_1')
    if destination.exists() or not all((graph_root / name).is_dir() for name in names):
        raise RuntimeError('graph checkpoint exists or a service graph is missing')
    manifest = {}
    for name in names:
        target = prepare_edit_working_dir(graph_root / name, destination / name)
        hashes = {}
        for filename in ('kv_store_entities.json', 'kv_store_hyperedges.json',
                         'graph_chunk_entity_relation.graphml', 'hyperedge_recent_mutations.json'):
            path = target / filename
            if path.exists():
                value = hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        value.update(block)
                hashes[filename] = value.hexdigest()
        manifest[name] = hashes
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest
