#!/usr/bin/env python3
"""One real E-VQA image on one idle GPU: no training, Ray, or remote APIs.

An optional public-model download uses the existing hf CLI, pinned revision,
one download worker, data-disk storage, and no implicit authentication.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from urllib.parse import urlsplit, urlunsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evograph_mm.kb.gldv2_subset import atomic_json, image_info
from scripts.download_gldv2_thumbnails import now

ROOT = Path('/home/data/dataset/wjz/EvoGraph-R1')
SUBSET = 'paper_64_16_seed0'
MODEL = Path('/home/data/dataset/wjz/models/Qwen2.5-VL-3B-Instruct')
REPO = 'Qwen/Qwen2.5-VL-3B-Instruct'
REVISION = '66285546d2b821cf421d4f5eb2576359d3770cd3'
LOGS = ROOT / 'logs'


def sanitized(message):
    def clean(match):
        p = urlsplit(match.group(0))
        return urlunsplit((p.scheme, p.hostname or '', p.path, '', ''))
    return re.sub(r'https?://[^\s\"\'<>]+', clean, str(message))


def idle_gpu(gpu):
    output = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'
    ], text=True)
    devices = {int(r[0]): (r[1].strip(), int(r[2])) for r in csv.reader(output.splitlines())}
    if gpu not in devices:
        raise RuntimeError('requested GPU does not exist')
    uuid, free_mib = devices[gpu]
    apps = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'
    ], text=True)
    if any(row and row[0].strip() == uuid for row in csv.reader(apps.splitlines())):
        raise RuntimeError('selected GPU now has a compute process; no process was stopped')
    if free_mib < 20000:
        raise RuntimeError('selected GPU has less than 20000 MiB free; no process was stopped')
    return {'physical_gpu': gpu, 'uuid': uuid, 'free_mib_before': free_mib}


def local_snapshot_valid(path):
    config = json.loads((path / 'config.json').read_text())
    if config.get('model_type') != 'qwen2_5_vl':
        raise RuntimeError('policy snapshot is not Qwen2.5-VL')
    index = json.loads((path / 'model.safetensors.index.json').read_text())
    shards = sorted(set(index['weight_map'].values()))
    if not shards or any(Path(name).name != name for name in shards):
        raise RuntimeError('unsafe or empty weight shard index')
    if any(not (path / name).is_file() or (path / name).stat().st_size == 0 for name in shards):
        raise RuntimeError('model weight shard is missing or empty')
    for name in ['tokenizer.json', 'tokenizer_config.json', 'preprocessor_config.json']:
        if not (path / name).is_file():
            raise RuntimeError('model processor/tokenizer file is missing: ' + name)
    return {'repo': REPO, 'revision': REVISION, 'path': str(path), 'shards': shards,
            'weight_bytes': sum((path / name).stat().st_size for name in shards)}


def run_hf(arguments):
    cli = Path(sys.executable).parent / 'hf'
    if not cli.is_file():
        raise RuntimeError('existing environment has no hf CLI; no installation attempted')
    env = dict(os.environ)
    for key in list(env):
        if any(word in key.upper() for word in ('TOKEN', 'API_KEY', 'SECRET', 'PASSWORD')):
            env.pop(key)
    env.update(HTTP_PROXY='http://127.0.0.1:7890', HTTPS_PROXY='http://127.0.0.1:7890',
               HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1',
               HF_HUB_DISABLE_XET='1', HF_HOME=str(MODEL.parent / '.hf_cache'))
    env.pop('HF_HUB_OFFLINE', None)
    env.pop('TRANSFORMERS_OFFLINE', None)
    with subprocess.Popen([str(cli), *arguments], env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        for line in process.stdout:
            print(sanitized(line.rstrip()), flush=True)
        if process.wait() != 0:
            raise RuntimeError('hf CLI failed; download cache retained for resumption')


def first_example():
    subset = ROOT / 'datasets_mm/E-VQA/subsets' / SUBSET
    manifest = json.loads((subset / 'manifest.json').read_text())
    if manifest['summary']['status'] != 'complete':
        raise RuntimeError('small subset has not passed complete validation')
    with (subset / 'qa_train.csv').open(newline='') as f:
        row = next(csv.DictReader(f))
    image_id = row['dataset_image_ids'].split(',')[0].strip()
    matches = [r for r in manifest['copied_images'] if r['image_id'] == image_id]
    if not matches:
        raise RuntimeError('QA image id is not in subset manifest')
    image_path = ROOT / matches[0]['subset_path']
    if not image_info(image_path):
        raise RuntimeError('first real image cannot be decoded')
    return row, image_id, image_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, choices=range(4), default=0)
    parser.add_argument('--download-model', action='store_true')
    parser.add_argument('--max-new-tokens', type=int, default=64)
    args = parser.parse_args(argv)
    if not 1 <= args.max_new_tokens <= 128:
        parser.error('smoke test permits at most 128 new tokens')
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / '.evqa_gpu_smoke.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = {'pid': os.getpid(), 'started_at': now(), 'status': 'running',
                  'training_started': False, 'api_calls': False, 'retrieval_tested': False,
                  'single_gpu': args.gpu, 'subset': SUBSET, 'inference_samples': 1}

        def checkpoint(phase):
            report.update(phase=phase, updated_at=now())
            atomic_json(LOGS / 'evqa_gpu_smoke_progress.json', report)

        try:
            checkpoint('preflight')
            if shutil.disk_usage(ROOT).free < 100_000_000_000:
                raise RuntimeError('free disk space below 100 GB')
            row, image_id, image_path = first_example()
            report['gpu_before_download'] = idle_gpu(args.gpu)
            if args.download_model:
                checkpoint('downloading_policy_model')
                run_hf(['download', REPO, '--revision', REVISION, '--local-dir', str(MODEL),
                        '--max-workers', '1', '--quiet'])
                checkpoint('verifying_policy_checksums')
                run_hf(['cache', 'verify', REPO, '--revision', REVISION,
                        '--local-dir', str(MODEL), '--fail-on-missing-files'])
                report['policy_checksums_verified'] = True
            report['model'] = local_snapshot_valid(MODEL)
            checkpoint('checking_gpu_before_inference')
            report['gpu_before_inference'] = idle_gpu(args.gpu)
            # No .env loading: it defaults embedding visibility to CPU. This test
            # controls exactly one physical GPU explicitly before importing torch.
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            checkpoint('loading_policy_on_single_gpu')
            import torch
            from PIL import Image, ImageOps
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            torch.set_num_threads(4)
            if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
                raise RuntimeError('expected one visible BF16-capable GPU')
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            processor = AutoProcessor.from_pretrained(
                str(MODEL), local_files_only=True, trust_remote_code=False,
                min_pixels=128*28*28, max_pixels=512*28*28)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(MODEL), local_files_only=True, trust_remote_code=False,
                dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': 'cuda:0'})
            model.eval()
            report['model_load_seconds'] = round(time.perf_counter()-started, 3)
            with Image.open(image_path) as original:
                image = ImageOps.exif_transpose(original).convert('RGB')
            messages = [{'role': 'user', 'content': [
                {'type': 'image'}, {'type': 'text', 'text': row['question']}]}]
            prompt = processor.apply_chat_template(messages, tokenize=False,
                                                    add_generation_prompt=True)
            inputs = processor(text=[prompt], images=[image], return_tensors='pt').to('cuda:0')
            report.update(image_id=image_id, image_path=str(image_path), question=row['question'],
                          input_tokens=int(inputs['input_ids'].shape[-1]),
                          image_grid_thw=inputs['image_grid_thw'].tolist(),
                          original_answer_not_in_prompt=True)
            checkpoint('real_image_inference')
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                           do_sample=False, use_cache=True)
            torch.cuda.synchronize()
            continuation = generated[:, inputs['input_ids'].shape[-1]:]
            answer = processor.batch_decode(continuation, skip_special_tokens=True)[0]
            if not answer.strip():
                raise RuntimeError('model generated an empty answer')
            report.update(status='success', output=answer, raw_reference_answer=row['answer'],
                          inference_seconds=round(time.perf_counter()-started, 3),
                          output_tokens=int(continuation.shape[-1]),
                          peak_gpu_allocated_mib=round(torch.cuda.max_memory_allocated()/1024**2, 2),
                          peak_gpu_reserved_mib=round(torch.cuda.max_memory_reserved()/1024**2, 2),
                          torch_version=torch.__version__,
                          gpu_name=torch.cuda.get_device_name(0),
                          meaningful_model_inference=True,
                          training_readiness='not_verified', finished_at=now())
            checkpoint('finished')
        except Exception as error:
            report.update(status='failed', error_type=type(error).__name__,
                          error=sanitized(error), finished_at=now())
            checkpoint(report.get('phase', 'failed'))
        atomic_json(LOGS / 'evqa_gpu_smoke_report.json', report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0 if report['status'] == 'success' else 1


if __name__ == '__main__':
    raise SystemExit(main())
