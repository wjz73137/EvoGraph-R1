#!/usr/bin/env python3
"""Retire only the five identified local-Qwen graph outputs, recoverably."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evograph_mm.kb.gldv2_subset import atomic_json

ROOT = Path('/home/data/dataset/wjz/EvoGraph-R1')
NAMES = ('evqa_native_graph_smoke', 'evqa_native_graph_grounded',
         'evqa_native_graph_grounded_7b', 'evqa_native_graph_grounded_7b_v2',
         'evqa_native_graph_grounded_7b_v3')
SNAPSHOT = ROOT / 'expr_mm/evqa_api_graph_baseline/source_snapshot.json'


def validate_targets():
    parent = ROOT / 'expr_mm'
    targets = [parent / n for n in NAMES]
    owners = []
    for path in targets:
        if path.is_symlink() or not path.is_dir() or path.resolve().parent != parent.resolve():
            raise RuntimeError(f'invalid exact retirement target: {path}')
        owner = json.loads((path / 'E-VQA/owner.json').read_text())
        model = owner.get('graph_llm', '')
        if not model.startswith('local') or 'Qwen2.5-VL' not in model:
            raise RuntimeError(f'not an identified local-Qwen output: {path}')
        owners.append(owner)
    if any(o['documents'] != owners[0]['documents'] for o in owners):
        raise RuntimeError('local graph source mismatch')
    return targets, owners


def stop_owned_old_service():
    status_file = ROOT / 'logs/evqa_native_retrieval_api_status.json'
    status = json.loads(status_file.read_text())
    if status['working_dir'] != str(ROOT / 'expr_mm/evqa_native_graph_smoke/E-VQA'):
        raise RuntimeError('service working directory is not the retired graph')
    pid = int(status['pid'])
    proc = Path('/proc') / str(pid)
    if not proc.exists():
        return {'pid': pid, 'already_stopped': True}
    command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
    required = '/home/wjz/projects/EvoGraph-R1/scripts/run_evqa_native_graph_smoke.py --serve --port 8003'
    if proc.stat().st_uid != os.getuid() or required not in command:
        raise RuntimeError('service PID ownership or command changed; no signal sent')
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 15
    while proc.exists() and time.monotonic() < deadline:
        if (proc / 'stat').read_text().split()[2] == 'Z':
            break
        time.sleep(0.2)
    if proc.exists() and (proc / 'stat').read_text().split()[2] != 'Z':
        raise RuntimeError('old service did not exit; no graph directories moved')
    return {'pid': pid, 'signal': 'SIGTERM', 'stopped': True, 'port': 8003}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    targets, owners = validate_targets()
    if not args.execute:
        print(json.dumps({'exact_targets': [str(p) for p in targets]}, indent=2))
        return
    # Preserve only real source text/provenance, not extracted local graph facts.
    snapshot = {'documents': owners[0]['documents'], 'scope': owners[0]['scope'],
                'source_pages_sha256': owners[0]['source_pages_sha256'],
                'qa_hashes': owners[0]['qa_hashes'],
                'retired_source_owner': str(targets[0] / 'E-VQA/owner.json')}
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    if SNAPSHOT.exists() and json.loads(SNAPSHOT.read_text()) != snapshot:
        raise RuntimeError('existing source snapshot mismatch; retained')
    atomic_json(SNAPSHOT, snapshot)
    service = stop_owned_old_service()
    trash_parent = ROOT / '.trash'
    trash_parent.mkdir(exist_ok=True)
    trash = Path(tempfile.mkdtemp(prefix='local-extraction-', dir=trash_parent))
    manifest = {'recoverable': True, 'service': service, 'moved': [],
                'preserved': ['datasets', 'models', 'non-graph diagnostic outputs'],
                'source_snapshot': str(SNAPSHOT)}
    atomic_json(trash / 'manifest.json', manifest)
    for path in targets:
        destination = trash / path.name
        path.rename(destination)
        manifest['moved'].append({'original': str(path), 'recovery_path': str(destination)})
        atomic_json(trash / 'manifest.json', manifest)
    atomic_json(ROOT / 'logs/local_graph_retirement_report.json', manifest)
    atomic_json(ROOT / 'logs/evqa_native_retrieval_api_retired.json', service)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
