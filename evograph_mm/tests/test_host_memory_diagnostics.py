import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from verl.utils.debug import host_memory


def test_snapshot_captures_pressure_and_separates_shared_memory(tmp_path):
    proc = tmp_path / 'proc'
    (proc / 'self').mkdir(parents=True)
    (proc / 'meminfo').write_text('MemAvailable: 500 kB\nShmem: 100 kB\nSwapFree: 1 kB\n')
    (proc / 'self/status').write_text('Name: worker\nVmRSS: 200 kB\nVmLck: 50 kB\n')
    (proc / 'self/smaps_rollup').write_text('0000-1000 ---p\nPss: 150 kB\nPss_Shmem: 90 kB\n')
    (proc / 'self/cgroup').write_text('0::/user.slice/experiment.service\n')
    group = tmp_path / 'cgroups/user.slice/experiment.service'
    group.mkdir(parents=True)
    (group / 'memory.current').write_text('1024000')
    (group / 'memory.stat').write_text('anon 100\nfile 200\nshmem 90\n')
    (group / 'memory.pressure').write_text('some avg10=1.00 avg60=0.50 avg300=0.10 total=123')
    result = host_memory.snapshot('test', 651, proc, tmp_path / 'cgroups')
    assert result['host_bytes']['Shmem'] == 102400
    assert result['process_status_bytes']['VmLck'] == 51200
    assert result['process_smaps_bytes']['Pss_Shmem'] == 92160
    assert result['cgroup_memory']['memory.stat']['shmem'] == 90
    assert 'avg10=1.00' in result['cgroup_memory']['memory.pressure']


def test_diagnostics_are_opt_in_and_do_not_record_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv('EVOGRAPH_MEMORY_DIAGNOSTICS_DIR', raising=False)
    host_memory.record_host_memory('disabled')
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setenv('EVOGRAPH_MEMORY_DIAGNOSTICS_DIR', str(tmp_path))
    monkeypatch.setenv('OPENAI_API_KEY', 'synthetic-secret-never-record')
    host_memory.record_host_memory('enabled', 651)
    content = (tmp_path / f'process_{os.getpid()}.jsonl').read_text()
    assert 'synthetic-secret-never-record' not in content
    assert json.loads(content)['stage'] == 'enabled'


def test_vllm_backup_is_measured_without_constructing_an_allocator(monkeypatch):
    tensor = SimpleNamespace(numel=lambda: 100, element_size=lambda: 2)
    allocator = SimpleNamespace(pointer_to_data={1: SimpleNamespace(cpu_backup_tensor=tensor),
                                               2: SimpleNamespace(cpu_backup_tensor=None)})
    module = SimpleNamespace(CuMemAllocator=SimpleNamespace(instance=allocator))
    monkeypatch.setitem(sys.modules, 'vllm.device_allocator.cumem', module)
    assert host_memory.snapshot('backup')['vllm_cpu_backup_bytes'] == 200
