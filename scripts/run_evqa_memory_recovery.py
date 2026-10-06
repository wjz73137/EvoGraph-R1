"""Run 20 real restored updates; continue the full epoch only after the memory gate passes.

Own user units and exact experiment roots only. Never stop unrelated Ray/GPU jobs.
Normal checkpoint 670 is retained: terminal checkpoint 671 has an advanced counter
and must not be used to seed an otherwise contiguous full-stage metric journal.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

PROJECT = Path('/home/wjz/projects/EvoGraph-R1')
PYTHON = '/home/data/env/wjz/evograph-r1/bin/python'
ROOT = Path('/home/data/dataset/wjz/EvoGraph-R1/expr_mm')
PROBE = ROOT / 'evqa_graphedit_full1891_3b_epoch1_memory_probe_v3'
FULL = ROOT / 'evqa_graphedit_full1891_3b_epoch1_memory_resume670_v3'
CHECKPOINT = ROOT / 'evqa_graphedit_full1891_3b_epoch1_resume400_v1/checkpoints/global_step_650'
EXPERIMENT = 'Qwen2.5-VL-3B_E-VQA_GraphEdit_full1891_epoch1_v1'
SERVICES = 'evograph-ge-full1891-memory-probe-v3-services.service'
FULL_SERVICE = 'evograph-ge-full1891-memory-resume670-v3-services'
FULL_TRAIN = 'evograph-ge-full1891-memory-resume670-v3-train'


def run(arguments, **kwargs):
    return subprocess.run(arguments, check=True, **kwargs)


def state(status, **fields):
    report = {'status': status, **fields}
    (PROBE / 'recovery_workflow.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


def main():
    if os.getenv('CUDA_VISIBLE_DEVICES') != '2,3':
        raise RuntimeError('recovery must be restricted to approved GPUs 2,3')
    env = os.environ.copy()
    env.update(EVOGRAPH_GRAPHEDIT_CHECKPOINT=str(CHECKPOINT),
               EVOGRAPH_RESET_DATALOADER_ON_RESUME='false', EVOGRAPH_SAVE_FREQ='10',
               EVOGRAPH_REMOVE_PREVIOUS_CHECKPOINT='false', EVOGRAPH_TOTAL_TRAINING_STEPS='671',
               EVOGRAPH_GRAPHEDIT_OUTPUT_ROOT=str(PROBE),
               EVOGRAPH_GRAPHEDIT_SERVICE_ROOT=str(PROBE / 'isolated_graphs'),
               EVOGRAPH_CHECKPOINT_MMAP_LOAD='true',
               EVOGRAPH_FSDP_CPU_OFFLOAD_NON_BLOCKING='false',
               EVOGRAPH_REF_NATIVE_CPU_OFFLOAD='false',
               EVOGRAPH_MEMORY_DIAGNOSTICS_DIR=str(PROBE / 'memory_diagnostics'),
               EVOGRAPH_RAY_TMPDIR='/home/data/dataset/wjz/.ray/gmp3')
    state('running_20_update_probe', first_step=651, last_step=670)
    result = subprocess.run(['/bin/bash', str(PROJECT / 'scripts/run_evqa_graphedit_full_3b.sh')],
                            env=env, cwd=PROJECT)
    if result.returncode:
        state('probe_process_failed', returncode=result.returncode)
        return 1
    gate = subprocess.run([PYTHON, str(PROJECT / 'scripts/check_evqa_memory_probe.py'),
                           '--output', str(PROBE)], cwd=PROJECT)
    if gate.returncode:
        state('memory_gate_blocked', report=str(PROBE / 'memory_probe_report.json'))
        return 1
    state('memory_gate_accepted_preparing_continuation')
    run([PYTHON, str(PROJECT / 'scripts/prepare_evqa_graphedit_resume.py'), '--source', str(PROBE),
         '--output', str(FULL), '--experiment', EXPERIMENT, '--step', '670'], cwd=PROJECT)
    service = run(['systemctl', '--user', 'show', SERVICES, '--property=ExecStart', '--value'],
                  capture_output=True, text=True).stdout
    if str(PROBE / 'isolated_graphs') not in service or 'run_evqa_graphedit_services.sh' not in service:
        raise RuntimeError('refusing to stop an unexpected retrieval service')
    run(['systemctl', '--user', 'stop', SERVICES])
    run(['systemd-run', '--user', '--unit=' + FULL_SERVICE,
         '--property=WorkingDirectory=' + str(PROJECT), '/usr/bin/env', 'CUDA_VISIBLE_DEVICES=',
         'LD_LIBRARY_PATH=/home/data/env/wjz/evograph-r1/lib', 'OMP_NUM_THREADS=4',
         'MKL_NUM_THREADS=4', 'MAX_JOBS=4', 'EVOGRAPH_SERVICE_RUNTIME_ENCODER=cached',
         str(PROJECT / 'scripts/run_evqa_graphedit_services.sh'), str(FULL / 'isolated_graphs')])
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 180
    while True:
        try:
            for port in range(8010, 8016):
                with opener.open(f'http://127.0.0.1:{port}/health', timeout=5) as response:
                    if json.load(response).get('status') != 'healthy':
                        raise RuntimeError('unhealthy restored retrieval service')
            break
        except Exception:
            if time.monotonic() >= deadline:
                raise RuntimeError('restored retrieval services did not become healthy')
            time.sleep(10)
    run([PYTHON, str(PROJECT / 'scripts/audit_evqa_cached_retrieval.py'), '--graphs',
         str(FULL / 'isolated_graphs'), '--dataset',
         '/home/data/dataset/wjz/EvoGraph-R1/datasets_mm/E-VQA/processed/paper_grpo_graph_edit_full1891_v1',
         '--snapshot', str(FULL / 'retrieval_preflight.json')], cwd=PROJECT)
    run(['systemd-run', '--user', '--unit=' + FULL_TRAIN,
         '--property=WorkingDirectory=' + str(PROJECT),
         '--property=StandardOutput=append:' + str(FULL / 'train.log'),
         '--property=StandardError=append:' + str(FULL / 'train.log'), '/usr/bin/env',
         'CUDA_VISIBLE_DEVICES=2,3', 'EVOGRAPH_GRAPHEDIT_CHECKPOINT=' + str(PROBE / 'checkpoints/global_step_670'),
         'EVOGRAPH_RESET_DATALOADER_ON_RESUME=false', 'EVOGRAPH_SAVE_FREQ=10',
         'EVOGRAPH_TOTAL_TRAINING_STEPS=1258', 'EVOGRAPH_CHECKPOINT_MMAP_LOAD=true',
         'EVOGRAPH_FSDP_CPU_OFFLOAD_NON_BLOCKING=false',
         'EVOGRAPH_REF_NATIVE_CPU_OFFLOAD=false',
         'EVOGRAPH_MEMORY_DIAGNOSTICS_DIR=' + str(FULL / 'memory_diagnostics'),
         'EVOGRAPH_GRAPHEDIT_OUTPUT_ROOT=' + str(FULL),
         'EVOGRAPH_GRAPHEDIT_SERVICE_ROOT=' + str(FULL / 'isolated_graphs'),
         'EVOGRAPH_RAY_TMPDIR=/home/data/dataset/wjz/.ray/gmr3',
         '/bin/bash', str(PROJECT / 'scripts/run_evqa_graphedit_full_reported.sh')])
    run(['systemd-run', '--user', '--unit=evograph-ge-full1891-memory-resume670-v3-status',
         '--on-active=30min', '--on-unit-active=30min', '--timer-property=AccuracySec=1min',
         PYTHON, str(PROJECT / 'scripts/record_evqa_training_status.py'), '--output', str(FULL),
         '--experiment', EXPERIMENT, '--unit', FULL_TRAIN + '.service'])
    state('full_continuation_launched', unit=FULL_TRAIN + '.service', output=str(FULL),
          next_step=671, full_epoch_not_yet_complete=True)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        state('workflow_failed', exception=type(exc).__name__)
        raise
