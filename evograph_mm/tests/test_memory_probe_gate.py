import json

from scripts.check_evqa_memory_probe import assess_probe, EXPERIMENT, GIB


def probe_fixture(tmp_path):
    results = tmp_path / 'expr_results' / EXPERIMENT
    results.mkdir(parents=True)
    (results / 'evals_training.jsonl').write_text(''.join(
        json.dumps({'global_steps': step, 'actor/grad_norm': 1.0}) + '\n'
        for step in range(327, 671)))
    (results / 'evals_step671.json').write_text('{"score": 0}')
    checkpoint = tmp_path / 'checkpoints/global_step_670'
    (checkpoint / 'actor').mkdir(parents=True)
    for rank in (0, 1):
        for kind in ('model', 'optim', 'extra_state'):
            (checkpoint / 'actor' / f'{kind}_world_size_2_rank_{rank}.pt').write_bytes(b'state')
    (checkpoint / 'data.pt').write_bytes(b'state')
    (checkpoint / 'graph_state').mkdir()
    (checkpoint / 'graph_state/manifest.json').write_text('{}')
    diagnostics = tmp_path / 'memory_diagnostics'
    diagnostics.mkdir()
    def record(stage):
        return {'stage': stage, 'host_bytes': {'MemAvailable': 80 * GIB},
                'process_smaps_bytes': {'Pss_Anon': 20 * GIB},
                'cgroup_memory': {'memory.stat': {'anon': 40 * GIB, 'shmem': 10 * GIB},
                                  'memory.pressure': 'some avg10=0.00 avg60=0.00 total=0'}}
    for rank in (0, 1):
        rows = [record(stage) for stage in ('checkpoint_load_complete', 'trainer_checkpoint_complete',
                                           'rollout_sleep_complete')]
        rows += [record('actor_update_offloaded') for _ in range(20)]
        (diagnostics / f'process_{rank}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return diagnostics


def test_probe_gate_accepts_real_range_and_memory_headroom(tmp_path):
    probe_fixture(tmp_path)
    report = assess_probe(tmp_path)
    assert report['status'] == 'accepted'
    assert report['new_optimizer_updates'] == 20
    assert report['maximum_cgroup_anon_shmem_gib'] == 50


def test_probe_gate_blocks_shared_memory_pressure(tmp_path):
    diagnostics = probe_fixture(tmp_path)
    path = diagnostics / 'process_0.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]['cgroup_memory']['memory.stat']['shmem'] = 70 * GIB
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    report = assess_probe(tmp_path)
    assert report['status'] == 'blocked'
    assert any('anonymous+shared' in reason for reason in report['failures'])


def test_probe_gate_refuses_terminal_counter_as_resume_checkpoint(tmp_path):
    probe_fixture(tmp_path)
    (tmp_path / 'checkpoints/global_step_670').rename(tmp_path / 'checkpoints/global_step_671')
    report = assess_probe(tmp_path)
    assert report['status'] == 'blocked'
    assert any('step670 checkpoint' in reason for reason in report['failures'])
