"""Opt-in, read-only memory snapshots at training phase boundaries.

Never record environment values, command lines, prompts or API credentials.
Each process writes its own small JSONL file outside the source tree.
"""
from datetime import datetime
import json
import os
from pathlib import Path
import sys


def _colon_kib(path):
    result = {}
    for line in path.read_text().splitlines():
        if ':' not in line:
            continue
        key, value = line.split(':', 1)
        parts = value.split()
        if len(parts) == 2 and parts[1] == 'kB':
            result[key] = int(parts[0]) * 1024
    return result


def _cgroup_dir(proc_root=Path('/proc'), cgroup_root=Path('/sys/fs/cgroup')):
    for line in (proc_root / 'self/cgroup').read_text().splitlines():
        if line.startswith('0::'):
            relative = line[3:].lstrip('/')
            if '..' in Path(relative).parts:
                raise ValueError('invalid cgroup path')
            return cgroup_root / relative
    return None


def snapshot(stage, step=None, proc_root=Path('/proc'), cgroup_root=Path('/sys/fs/cgroup')):
    record = {'checked_at': datetime.now().astimezone().isoformat(),
              'pid': os.getpid(), 'stage': stage, 'step': step}
    for name, path in [('host', proc_root / 'meminfo'),
                       ('process_status', proc_root / 'self/status'),
                       ('process_smaps', proc_root / 'self/smaps_rollup')]:
        try:
            values = _colon_kib(path)
            if name == 'host':
                keys = ('MemTotal', 'MemAvailable', 'MemFree', 'Cached', 'Shmem',
                        'Unevictable', 'Mlocked', 'SwapTotal', 'SwapFree')
                values = {key: values[key] for key in keys if key in values}
            record[name + '_bytes'] = values
        except (OSError, ValueError):
            record[name + '_unavailable'] = True
    try:
        group = _cgroup_dir(proc_root, cgroup_root)
        if group is not None:
            record['cgroup'] = str(group.relative_to(cgroup_root))
            record['cgroup_memory'] = {}
            for name in ('memory.current', 'memory.peak', 'memory.swap.current',
                         'memory.stat', 'memory.events', 'memory.pressure'):
                try:
                    value = (group / name).read_text().strip()
                    if name in ('memory.stat', 'memory.events'):
                        value = {key: int(count) for key, count in
                                 (line.split() for line in value.splitlines())}
                    elif name != 'memory.pressure':
                        value = int(value)
                    record['cgroup_memory'][name] = value
                except (OSError, ValueError):
                    pass
    except (OSError, ValueError):
        record['cgroup_unavailable'] = True
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        record['cuda_memory_bytes'] = {
            'allocated': torch.cuda.memory_allocated(), 'reserved': torch.cuda.memory_reserved()}
    cumem = sys.modules.get('vllm.device_allocator.cumem')
    allocator = getattr(getattr(cumem, 'CuMemAllocator', None), 'instance', None)
    if allocator is not None:
        record['vllm_cpu_backup_bytes'] = sum(
            item.cpu_backup_tensor.numel() * item.cpu_backup_tensor.element_size()
            for item in allocator.pointer_to_data.values() if item.cpu_backup_tensor is not None)
    return record


def record_host_memory(stage, step=None):
    output = os.getenv('EVOGRAPH_MEMORY_DIAGNOSTICS_DIR', '').strip()
    if not output:
        return
    try:
        destination = Path(output)
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / f'process_{os.getpid()}.jsonl').open('a') as stream:
            stream.write(json.dumps(snapshot(stage, step)) + '\n')
    except Exception as exc:
        # Observability must not make a successful optimizer update fail.
        print(f'HOST_MEMORY_DIAGNOSTIC_UNAVAILABLE: {type(exc).__name__}', flush=True)
