#!/usr/bin/env python3
"""Write an evidence-based completion report; never equate process exit with success."""
import argparse
import json
from pathlib import Path
from statistics import mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--expected-updates', type=int, default=931)
    parser.add_argument('--expected-final-step', type=int, default=1258)
    parser.add_argument('--process-exit', type=int, required=True)
    args = parser.parse_args()
    results = args.output / 'expr_results' / args.experiment
    metrics_path = results / 'evals_training.jsonl'
    updates = {}
    malformed_metric_lines = 0
    if metrics_path.is_file():
        for line in metrics_path.read_text().splitlines():
            if line.strip():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    malformed_metric_lines += 1
                    continue
                updates[int(item['global_steps'])] = item
    val_path = results / f'evals_step{args.expected_final_step}.json'
    validation = json.loads(val_path.read_text()) if val_path.is_file() else {}
    checkpoint = args.output / 'checkpoints' / f'global_step_{args.expected_final_step}'
    required = [checkpoint / 'actor' / f'{kind}_world_size_2_rank_{rank}.pt'
                for kind in ('model', 'optim', 'extra_state') for rank in (0, 1)]
    shards_ready = all(path.is_file() and path.stat().st_size > 0 for path in required)
    files = [path for path in checkpoint.rglob('*') if path.is_file()] if checkpoint.is_dir() else []
    expected_steps = set(range(args.expected_final_step - args.expected_updates, args.expected_final_step))
    complete = (args.process_exit == 0 and set(updates) == expected_steps
                and bool(validation) and shards_ready and not malformed_metric_lines)
    def avg(key):
        values = [float(item[key]) for item in updates.values() if key in item]
        return mean(values) if values else None
    def count(key):
        # The configured train batch is two questions, each sampled twice.
        return round(sum(float(item.get(key, 0)) * 4 for item in updates.values()))
    summary = {
        'status': 'complete' if complete else 'incomplete_or_failed',
        'process_exit': args.process_exit, 'actual_optimizer_updates': len(updates),
        'expected_optimizer_updates': args.expected_updates,
        'malformed_metric_lines': malformed_metric_lines,
        'first_logged_step': min(updates) if updates else None,
        'last_logged_step': max(updates) if updates else None,
        'final_checkpoint_counter': args.expected_final_step,
        'training_question_presentations': len(updates) * 2,
        'training_trajectories': len(updates) * 4,
        'mean_training_reward': avg('critic/rewards/mean'),
        'mean_training_f1': avg('critic/answer_f1_score/mean'),
        'mean_training_em': avg('critic/answer_em_score/mean'),
        'mean_training_format': avg('critic/format_score/mean'),
        'successful_edit_tool_calls': count('trajectory/successful_graph_edit_count/mean'),
        'edits_followed_by_nonempty_kb_query': count('trajectory/verified_graph_edit_count/mean'),
        'websearch_calls': count('trajectory/websearch_count/mean'),
        'final_validation': validation,
        'checkpoint': str(checkpoint), 'checkpoint_shards_ready': shards_ready,
        'checkpoint_files': len(files), 'checkpoint_bytes': sum(path.stat().st_size for path in files),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'completion_report.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    lines = [
        '# 全量 GraphEdit 训练报告', '',
        f"状态：{summary['status']}；训练进程退出码：{args.process_exit}。", '',
        f"实际优化器更新：{len(updates)} / {args.expected_updates}。",
        f"问题呈现次数：{len(updates) * 2}；采样轨迹：{len(updates) * 4}。",
        f"成功编辑工具调用：{summary['successful_edit_tool_calls']}；后续有非空 KB 检索：{summary['edits_followed_by_nonempty_kb_query']}。", '',
        '这些编辑计数不是新增独立事实数量，也不证明每条事实已通过语义核验。',
        '若编辑数为零，不能把本实验称为已学会 GraphEdit，即使所有优化器更新完成。', '',
        f"Checkpoint：{checkpoint}；{len(files)} 个文件；{summary['checkpoint_bytes'] / 2**30:.2f} GiB。", '',
        '配置：Qwen2.5-VL-3B-Instruct，物理 GPU 2、3，BF16 FSDP，vLLM 0.7.3，',
        'native sleep level 1，batch=2，n_repeat=2，学习率 5e-7，1 个全量 epoch。',
        '从原检索阶段 step 326 恢复，不使用定向验证问题训练出的烟测权重。', '',
        '本实验是两张 3090 上的 3B 资源适配复现：论文使用 7B 和四张 80GB A100；',
        'API 建图模型、约束提示词、置信度门控和编辑奖励附加项也不同，不能声称论文指标已复现。', '',
        '详细指标与最终验证见同目录 completion_report.json；逐步操作见项目',
        'docs/GRAPHEDIT_PROGRESS_2026-10-05.md；训练轨迹见 expr_results。',
    ]
    (args.output / 'completion_report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary, ensure_ascii=False))
    raise SystemExit(0 if complete else 1)


if __name__ == '__main__':
    main()
