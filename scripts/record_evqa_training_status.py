#!/usr/bin/env python3
"""Read-only, half-hour training checks recorded in the experiment directory."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--experiment', required=True)
    args = parser.parse_args()
    state = subprocess.run([
        'systemctl', '--user', 'show', 'evograph-ge-full1891-train.service',
        '--property=ActiveState,SubState,ExecMainStatus',
    ], capture_output=True, text=True, timeout=10)
    latest = None
    metrics_path = args.output / 'expr_results' / args.experiment / 'evals_training.jsonl'
    if metrics_path.is_file():
        for line in metrics_path.read_text().splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            latest = {key: item[key] for key in (
                'global_steps', 'critic/rewards/mean', 'critic/answer_f1_score/mean',
                'critic/format_score/mean', 'actor/grad_norm',
                'trajectory/successful_graph_edit_count/mean',
            ) if key in item}
    gpu = subprocess.run([
        'nvidia-smi', '-i', '2,3', '--query-gpu=index,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits',
    ], capture_output=True, text=True, timeout=10)
    record = {'checked_at': datetime.now().astimezone().isoformat(),
              'unit_state': state.stdout.strip(), 'latest_training_metrics': latest,
              'gpu_2_3': gpu.stdout.strip(), 'gpu_query_exit': gpu.returncode}
    with (args.output / 'periodic_status.jsonl').open('a') as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(json.dumps(record, ensure_ascii=False))


if __name__ == '__main__':
    main()
