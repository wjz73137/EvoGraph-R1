#!/usr/bin/env python3
"""Compare native vLLM output and buffers across sleep levels on one GPU."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': 'What is the capital of France? Answer in one short sentence.'}],
        tokenize=False, add_generation_prompt=True,
    )
    engine = LLM(model=args.model, dtype='bfloat16', max_model_len=4096,
                 gpu_memory_utilization=0.45, enforce_eager=True,
                 enable_sleep_mode=True, max_num_seqs=1)
    model = engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
    saved_parameters = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
    saved_buffers = {name: value.detach().cpu().clone() for name, value in model.named_buffers()}
    report = {'model': args.model, 'buffers': {name: list(value.shape) for name, value in saved_buffers.items()}, 'stages': []}

    def record(stage):
        changed = [name for name, value in model.named_buffers()
                   if not torch.equal(value.detach().cpu(), saved_buffers[name])]
        output = engine.generate([prompt], SamplingParams(temperature=0, max_tokens=48), use_tqdm=False)[0].outputs[0]
        item = {'stage': stage, 'changed_buffers': changed, 'text': output.text, 'token_ids': output.token_ids}
        report['stages'].append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')

    record('before_sleep')
    engine.sleep(level=1)
    engine.wake_up()
    record('after_sleep_1')
    engine.sleep(level=2)
    engine.wake_up()
    with torch.no_grad():
        for name, value in model.named_parameters():
            value.copy_(saved_parameters[name])
    record('after_sleep_2_parameters_restored')
    with torch.no_grad():
        for name, value in model.named_buffers():
            value.copy_(saved_buffers[name])
    record('after_sleep_2_parameters_and_buffers_restored')


if __name__ == '__main__':
    main()
