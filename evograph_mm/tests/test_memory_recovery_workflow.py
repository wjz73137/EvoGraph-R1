import json
from types import SimpleNamespace

import pytest

from scripts import run_evqa_memory_recovery as workflow


def prepare(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, 'PROBE', tmp_path)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2,3')


def test_workflow_stops_on_probe_failure_without_launching_full(tmp_path, monkeypatch):
    prepare(tmp_path, monkeypatch)
    calls = []
    def run(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return SimpleNamespace(returncode=9)
    monkeypatch.setattr(workflow.subprocess, 'run', run)
    assert workflow.main() == 1
    assert len(calls) == 1
    assert calls[0][1]['env']['EVOGRAPH_CHECKPOINT_MMAP_LOAD'] == 'true'
    assert calls[0][1]['env']['EVOGRAPH_FSDP_CPU_OFFLOAD_NON_BLOCKING'] == 'false'
    assert calls[0][1]['env']['EVOGRAPH_REF_NATIVE_CPU_OFFLOAD'] == 'false'
    assert calls[0][1]['env']['EVOGRAPH_REMOVE_PREVIOUS_CHECKPOINT'] == 'false'
    socket = calls[0][1]['env']['EVOGRAPH_RAY_TMPDIR'] + '/session_2026-10-06_15-47-24_089947_2174513/sockets/plasma_store'
    assert len(socket.encode()) <= 107
    assert json.loads((tmp_path / 'recovery_workflow.json').read_text())['status'] == 'probe_process_failed'


def test_workflow_stops_on_failed_memory_gate(tmp_path, monkeypatch):
    prepare(tmp_path, monkeypatch)
    calls = []
    def run(arguments, **kwargs):
        calls.append(arguments)
        return SimpleNamespace(returncode=0 if len(calls) == 1 else 1)
    monkeypatch.setattr(workflow.subprocess, 'run', run)
    assert workflow.main() == 1
    assert len(calls) == 2
    assert not any(arguments[0] == 'systemctl' for arguments in calls)
    assert json.loads((tmp_path / 'recovery_workflow.json').read_text())['status'] == 'memory_gate_blocked'


def test_workflow_refuses_unapproved_gpu_selection(tmp_path, monkeypatch):
    prepare(tmp_path, monkeypatch)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    with pytest.raises(RuntimeError, match='approved GPUs'):
        workflow.main()
    assert list(tmp_path.iterdir()) == []
