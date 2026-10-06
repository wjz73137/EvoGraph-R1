"""Require real updates, paired state and conservative memory headroom before continuation."""
import argparse
import json
import math
from pathlib import Path

EXPERIMENT = 'Qwen2.5-VL-3B_E-VQA_GraphEdit_full1891_epoch1_v1'
GIB = 2 ** 30


def assess_probe(output):
    output = Path(output)
    failures = []
    results = output / 'expr_results' / EXPERIMENT
    metrics = [json.loads(line) for line in (results / 'evals_training.jsonl').read_text().splitlines()
               if line.strip()]
    if [int(row['global_steps']) for row in metrics] != list(range(327, 671)):
        failures.append('optimizer metric range is not exactly 327..670')
    new = [row for row in metrics if int(row['global_steps']) >= 651]
    if any(not math.isfinite(float(row.get('actor/grad_norm', float('nan')))) for row in new):
        failures.append('nonfinite or missing actor gradient norm')
    checkpoint = output / 'checkpoints/global_step_670'
    required = [checkpoint / 'actor' / f'{kind}_world_size_2_rank_{rank}.pt'
                for kind in ('model', 'optim', 'extra_state') for rank in (0, 1)]
    required += [checkpoint / 'data.pt', checkpoint / 'graph_state/manifest.json',
                 results / 'evals_step671.json']
    if not all(path.is_file() and path.stat().st_size for path in required):
        failures.append('normal step670 checkpoint or diagnostic final validation is incomplete')
    records, actor_samples = [], []
    for path in sorted((output / 'memory_diagnostics').glob('process_*.jsonl')):
        samples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        records.extend(samples)
        actor = [row for row in samples if row['stage'] == 'actor_update_offloaded']
        if actor:
            actor_samples.append(actor)
    if len(actor_samples) != 2 or any(len(rows) != 20 for rows in actor_samples):
        failures.append('expected 20 post-update samples from each of two GPU workers')
    stages = {row['stage'] for row in records}
    if not {'checkpoint_load_complete', 'trainer_checkpoint_complete', 'rollout_sleep_complete'} <= stages:
        failures.append('missing loading, sleep or checkpoint phase diagnostics')
    available = [r.get('host_bytes', {}).get('MemAvailable') for r in records]
    available = [x for x in available if x is not None]
    unreclaimable, pressure = [], []
    for row in records:
        group = row.get('cgroup_memory', {})
        stat = group.get('memory.stat', {})
        if 'anon' in stat and 'shmem' in stat:
            unreclaimable.append(stat['anon'] + stat['shmem'])
        for line in group.get('memory.pressure', '').splitlines():
            if line.startswith('some '):
                fields = dict(field.split('=', 1) for field in line.split()[1:])
                pressure.append(float(fields['avg10']))
    if not available or min(available) < 40 * GIB:
        failures.append('host memory headroom below 40 GiB or unavailable')
    if not unreclaimable or max(unreclaimable) > 80 * GIB:
        failures.append('training cgroup anonymous+shared memory above 80 GiB or unavailable')
    if not pressure or max(pressure) >= 40:
        failures.append('memory pressure avg10 at least 40 percent or unavailable')
    growth = []
    for samples in actor_samples:
        values = [r.get('process_smaps_bytes', {}).get('Pss_Anon') for r in samples]
        if len(values) < 4 or any(x is None for x in values):
            failures.append('missing anonymous proportional memory samples')
            continue
        delta = max(values[-2:]) - max(values[:2])
        growth.append(delta / GIB)
        if delta > 4 * GIB:
            failures.append('worker anonymous memory grew more than 4 GiB')
    report = {'status': 'accepted' if not failures else 'blocked', 'failures': failures,
              'new_optimizer_updates': len(new), 'profile_records': len(records),
              'minimum_host_available_gib': min(available) / GIB if available else None,
              'maximum_cgroup_anon_shmem_gib': max(unreclaimable) / GIB if unreclaimable else None,
              'maximum_memory_pressure_avg10': max(pressure) if pressure else None,
              'worker_anonymous_growth_gib': growth, 'resume_checkpoint': str(checkpoint),
              'limitation': 'A 20-update check is not proof of long-run stability or answer quality.'}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = assess_probe(args.output)
    (args.output / 'memory_probe_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    raise SystemExit(0 if report['status'] == 'accepted' else 1)


if __name__ == '__main__':
    main()
