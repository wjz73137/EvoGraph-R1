#!/usr/bin/env python3
"""Run two real image -> strict KB -> local VLM -> score smoke examples."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evograph_mm.kb.gldv2_subset import atomic_json, image_info
from scripts.download_gldv2_thumbnails import now
from scripts.run_evqa_gpu_smoke import MODEL, idle_gpu, local_snapshot_valid, sanitized


DATA_ROOT = Path("/home/data/dataset/wjz/EvoGraph-R1")
DATA_FILE = DATA_ROOT / "datasets_mm/E-VQA/processed/paper_64_16_seed0/train.parquet"
GRAPH_ROOT = DATA_ROOT / "expr_mm/evqa_api_strict_max_graph_v1"
OUTPUT_ROOT = DATA_ROOT / "expr_mm/evqa_strict_agent_smoke_v1"
SEARCH_URL = "http://127.0.0.1:8004/search"
STATUS_URL = "http://127.0.0.1:8004/status"
SAMPLES = {
    "Charles William Jones House": {
        "question": "In what month was this building added to the national register?",
        "gold": ["December"],
    },
    "Pamban Bridge": {
        "question": "In what year was the bascule of this bridge damaged?",
        "gold": ["2018"],
    },
}


def normalize_answer(value: str) -> str:
    return " ".join(re.findall(r"\w+", value.casefold(), flags=re.UNICODE))


def extract_answer(value: str) -> str:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", value, flags=re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else value).strip()


def question_from_row(row) -> str:
    prompt = str(row["prompt"][0]["content"])
    if "Question:" not in prompt:
        raise ValueError("processed sample prompt lacks Question marker")
    return prompt.split("Question:", 1)[1].strip()


def select_samples(dataframe) -> list:
    selected = []
    for title, expected in SAMPLES.items():
        matches = [row for _, row in dataframe.iterrows()
                   if str(row["context"][0]) == title
                   and question_from_row(row) == expected["question"]]
        if len(matches) != 1:
            raise RuntimeError(f"expected exactly one processed sample for {title}")
        row = matches[0]
        stored_gold = [str(value) for value in row["extra_info"]["golden_answers"]]
        if stored_gold != expected["gold"]:
            raise RuntimeError(f"processed gold changed for {title}")
        if not image_info(Path(row["image_path"])):
            raise RuntimeError(f"sample image is invalid for {title}")
        selected.append(row)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=range(4), default=2)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 96:
        parser.error("smoke generation permits at most 96 new tokens")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    report_file = OUTPUT_ROOT / "report.json"
    owner = {
        "pipeline": "evqa-strict-agent-smoke-v1",
        "dataset": "E-VQA/paper_64_16_seed0",
        "samples": list(SAMPLES),
        "policy_model": str(MODEL),
        "search_url": SEARCH_URL,
        "strict_graph_report": str(GRAPH_ROOT / "report.json"),
        "physical_gpu": args.gpu,
        "max_new_tokens": args.max_new_tokens,
    }
    with (OUTPUT_ROOT / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if report_file.exists():
            existing = json.loads(report_file.read_text())
            if existing.get("status") == "complete" and existing.get("owner") == owner:
                print(json.dumps(existing, ensure_ascii=False))
                return 0
        report = {
            "owner": owner,
            "status": "running",
            "started_at": now(),
            "training_started": False,
            "remote_api_calls": False,
            "gold_answers_in_model_prompt": False,
            "samples": [],
        }
        atomic_json(report_file, report)
        try:
            import pandas as pd
            import requests
            graph_report = json.loads((GRAPH_ROOT / "report.json").read_text())
            if graph_report.get("index_status") != "complete":
                raise RuntimeError("strict Max graph indexes are incomplete")
            status = requests.get(STATUS_URL, timeout=30).json()
            if not status.get("ready") or Path(status["working_dir"]).resolve() != (
                    GRAPH_ROOT / "E-VQA").resolve():
                raise RuntimeError("port 8004 is not serving the strict Max graph")
            dataframe = pd.read_parquet(DATA_FILE)
            rows = select_samples(dataframe)
            report["gpu_preflight"] = idle_gpu(args.gpu)
            report["model_snapshot"] = local_snapshot_valid(MODEL)

            os.environ.update(
                CUDA_VISIBLE_DEVICES=str(args.gpu),
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
                MM_SEARCH_API_URL=SEARCH_URL,
                TEXT_SEARCH_API_URL=SEARCH_URL,
                MM_SEARCH_TIMEOUT="120",
            )
            from agent.tool.tools.mm.kb_search_tool import MMKBSearchTool
            search_tool = MMKBSearchTool()
            retrievals = []
            for row in rows:
                title = str(row["context"][0])
                question = question_from_row(row)
                image_response = json.loads(search_tool.execute({
                    "query": "<img>",
                    "image_id": str(row["image_id"]),
                    "image_path": str(row["image_path"]),
                    "context_query": question,
                }))
                candidates = image_response.get("results", [])
                if not candidates or candidates[0].get("entity") != title:
                    raise RuntimeError(f"image grounding did not rank {title!r} first")
                anchor = candidates[0]["entity"]
                text_response = json.loads(search_tool.execute({
                    "query": f"{anchor}. {question}",
                }))
                raw_results = text_response.get("results", [])
                evidence, seen = [], set()
                for item in raw_results:
                    value = str(item.get("<knowledge>", "")).strip().strip('"')
                    key = value.casefold()
                    if value and key not in seen:
                        evidence.append(value)
                        seen.add(key)
                    if len(evidence) == 6:
                        break
                if not evidence:
                    raise RuntimeError(f"text retrieval returned no evidence for {title}")
                retrievals.append({
                    "title": title,
                    "question": question,
                    "image_id": str(row["image_id"]),
                    "image_path": str(row["image_path"]),
                    "image_candidates": candidates,
                    "anchor": anchor,
                    "evidence": evidence,
                    "gold": [str(value) for value in row["extra_info"]["golden_answers"]],
                })

            import torch
            from PIL import Image, ImageOps
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            torch.set_num_threads(4)
            torch.set_num_interop_threads(2)
            if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
                raise RuntimeError("expected exactly one visible BF16-capable GPU")
            torch.cuda.reset_peak_memory_stats()
            load_started = time.perf_counter()
            processor = AutoProcessor.from_pretrained(
                str(MODEL), local_files_only=True, trust_remote_code=False,
                min_pixels=128 * 28 * 28, max_pixels=512 * 28 * 28,
            )
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(MODEL), local_files_only=True, trust_remote_code=False,
                dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": "cuda:0"},
            )
            model.eval()
            report["model_load_seconds"] = round(time.perf_counter() - load_started, 3)

            for item in retrievals:
                evidence_text = "\n".join(f"- {fact}" for fact in item["evidence"])
                instruction = (
                    "Answer the question using the image and the retrieved source-grounded facts. "
                    "Do not add outside knowledge. Return only a short answer inside "
                    "<answer>...</answer>.\n\n"
                    f"Image-grounded entity: {item['anchor']}\n"
                    f"Retrieved facts:\n{evidence_text}\n\n"
                    f"Question: {item['question']}"
                )
                messages = [{"role": "user", "content": [
                    {"type": "image"}, {"type": "text", "text": instruction},
                ]}]
                prompt = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                with Image.open(item["image_path"]) as original:
                    image = ImageOps.exif_transpose(original).convert("RGB")
                inputs = processor(text=[prompt], images=[image], return_tensors="pt").to("cuda:0")
                started = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs, max_new_tokens=args.max_new_tokens,
                        do_sample=False, use_cache=True,
                    )
                torch.cuda.synchronize()
                continuation = generated[:, inputs["input_ids"].shape[-1]:]
                raw_output = processor.batch_decode(continuation, skip_special_tokens=True)[0]
                answer = extract_answer(raw_output)
                normalized = normalize_answer(answer)
                gold_normalized = [normalize_answer(value) for value in item["gold"]]
                exact = normalized in gold_normalized
                contains = any(gold and (gold in normalized or normalized in gold)
                               for gold in gold_normalized)
                report["samples"].append({
                    "title": item["title"],
                    "question": item["question"],
                    "image_id": item["image_id"],
                    "image_anchor": item["anchor"],
                    "retrieved_evidence": item["evidence"],
                    "raw_model_output": raw_output,
                    "answer": answer,
                    "gold": item["gold"],
                    "exact_match": exact,
                    "answer_contains_gold": contains,
                    "input_tokens": int(inputs["input_ids"].shape[-1]),
                    "output_tokens": int(continuation.shape[-1]),
                    "inference_seconds": round(time.perf_counter() - started, 3),
                })
                atomic_json(report_file, report)
            correct = sum(sample["answer_contains_gold"] for sample in report["samples"])
            report.update(
                status="complete",
                finished_at=now(),
                sample_count=len(report["samples"]),
                correct_count=correct,
                smoke_accuracy=correct / len(report["samples"]),
                all_image_anchors_correct=True,
                all_retrievals_nonempty=True,
                gpu_name=torch.cuda.get_device_name(0),
                peak_gpu_allocated_mib=round(torch.cuda.max_memory_allocated(0) / 1024**2, 1),
                peak_gpu_reserved_mib=round(torch.cuda.max_memory_reserved(0) / 1024**2, 1),
            )
            atomic_json(report_file, report)
            print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
            return 0
        except Exception as exc:
            report.update(status="failed", finished_at=now(),
                          error_type=type(exc).__name__, error=sanitized(exc))
            atomic_json(report_file, report)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
