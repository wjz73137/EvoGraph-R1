import json
from pathlib import Path
import subprocess
import sys


def run_report(root, **options):
    script = Path(__file__).resolve().parents[2] / 'scripts/report_evqa_graph_edit_training.py'
    command = [sys.executable, str(script), '--output', str(root), '--experiment', 'test',
               '--expected-updates', '2', '--expected-final-step', '329', '--process-exit', '0']
    result = subprocess.run(command, capture_output=True, text=True)
    return result, json.loads((root / 'completion_report.json').read_text())


def test_exit_zero_without_updates_is_not_completed_training(tmp_path):
    result, report = run_report(tmp_path)
    assert result.returncode == 1
    assert report['status'] == 'incomplete_or_failed'
    assert report['actual_optimizer_updates'] == 0


def test_report_requires_updates_validation_and_checkpoint_shards(tmp_path):
    results = tmp_path / 'expr_results/test'
    results.mkdir(parents=True)
    metrics = [dict(global_steps=step, **{'critic/rewards/mean': 0.5}) for step in [327, 328]]
    (results / 'evals_training.jsonl').write_text('\n'.join(json.dumps(row) for row in metrics))
    (results / 'evals_step329.json').write_text('{"val/test_score": 0.5}')
    actor = tmp_path / 'checkpoints/global_step_329/actor'
    actor.mkdir(parents=True)
    for kind in ['model', 'optim', 'extra_state']:
        for rank in [0, 1]:
            (actor / f'{kind}_world_size_2_rank_{rank}.pt').write_bytes(b'test')
    result, report = run_report(tmp_path)
    assert result.returncode == 0
    assert report['actual_optimizer_updates'] == 2
    assert report['successful_edit_tool_calls'] == 0
    assert report['mean_training_reward'] == 0.5
