#!/usr/bin/env python3
"""Run eight image -> 63-doc strict KB -> local VLM -> score smoke samples."""
from __future__ import annotations

import argparse
import csv
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
SUBSET_ROOT = DATA_ROOT / "datasets_mm/E-VQA/subsets/paper_64_16_seed0"
GRAPH_ROOT = DATA_ROOT / "expr_mm/evqa_api_strict_max_graph_63_v1"
EXTRACTION_ROOT = DATA_ROOT / "expr_mm/evqa_api_strict_max_extraction_63_v1"
OUTPUT_ROOT = DATA_ROOT / "expr_mm/evqa_strict_agent_smoke_63_v2"
SEARCH_URL = "http://127.0.0.1:8005/search"
STATUS_URL = "http://127.0.0.1:8005/status"
ROW_INDEXES = [1, 3, 9, 17, 28, 39, 47, 61]


def normalize_answer(value: str) -> str:
    return " ".join(re.findall(r"\w+", value.casefold(), flags=re.UNICODE))


def extract_answer(value: str) -> str:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", value,
                      flags=re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else value).strip()


def selected_samples() -> list[dict]:
    manifest = json.loads((EXTRACTION_ROOT / "source_manifest.json").read_text())
    by_url = {item["wikipedia_url"]: item for item in manifest["documents"]}
    with (SUBSET_ROOT / "qa_train.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    samples = []
    for row_index in ROW_INDEXES:
        row = rows[row_index]
        url = row["wikipedia_url"].split("|")[0].strip()
        document = by_url[url]
        if not image_info(Path(document["image_path"])):
            raise RuntimeError(f"sample image is invalid: {document['image_id']}")
        if normalize_answer(row["answer"]) not in normalize_answer(document["contents"]):
            raise RuntimeError(f"gold answer is outside selected source passage: row {row_index}")
        samples.append({
            "row_index": row_index,
            "title": document["title"],
            "question": row["question"],
            "gold": [row["answer"].strip()],
            "image_id": document["image_id"],
            "image_path": document["image_path"],
        })
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=range(4), default=2)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 96:
        parser.error("smoke generation permits at most 96 new tokens")
    samples = selected_samples()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    report_file = OUTPUT_ROOT / "report.json"
    owner = {
        "pipeline": "evqa-strict-agent-smoke-63-v2",
        "dataset": "E-VQA/paper_64_16_seed0",
        "sample_rows": ROW_INDEXES,
        "sample_titles": [sample["title"] for sample in samples],
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
            "owner": owner, "status": "running", "started_at": now(),
            "training_started": False, "remote_api_calls": False,
            "gold_answers_in_model_prompt": False, "gold_answers_used_to_rank_evidence": False,
            "samples": [],
        }
        atomic_json(report_file, report)
        try:
            import requests
            graph_report = json.loads((GRAPH_ROOT / "report.json").read_text())
            if graph_report.get("index_status") != "complete":
                raise RuntimeError("63-document graph indexes are incomplete")
            status = requests.get(STATUS_URL, timeout=30).json()
            if not status.get("ready") or Path(status["working_dir"]).resolve() != (
                GRAPH_ROOT / "E-VQA"
            ).resolve():
                raise RuntimeError("port 8005 is not serving the 63-document strict graph")

            os.environ.update(
                MM_SEARCH_API_URL=SEARCH_URL, TEXT_SEARCH_API_URL=SEARCH_URL,
                MM_SEARCH_TIMEOUT="120",
            )
            from agent.tool.tools.mm.kb_search_tool import MMKBSearchTool
            search_tool = MMKBSearchTool()
            retrievals = []
            for sample in samples:
                image_response = json.loads(search_tool.execute({
                    "query": "<img>", "image_id": sample["image_id"],
                    "image_path": sample["image_path"],
                    "context_query": sample["question"], "image_top_k": 8,
                }))
                candidates = image_response.get("results", [])
                if not candidates or candidates[0].get("entity") != sample["title"]:
                    raise RuntimeError(f"image grounding did not rank {sample['title']!r} first")
                anchor = candidates[0]["entity"]
                text_response = json.loads(search_tool.execute({
                    "query": f"{anchor}. {sample['question']}",
                    "entity_top_k": 8, "hyperedge_top_k": 12,
                }))
                evidence, seen = [], set()
                for item in text_response.get("results", []):
                    value = str(item.get("<knowledge>", "")).strip().strip('"')
                    key = value.casefold()
                    if value and key not in seen:
                        evidence.append(value)
                        seen.add(key)
                    if len(evidence) == 10:
                        break
                if not evidence:
                    raise RuntimeError(f"text retrieval returned no evidence for {sample['title']}")
                rendered = normalize_answer("\n".join(evidence))
                gold_recalled = any(normalize_answer(gold) in rendered for gold in sample["gold"])
                retrievals.append({**sample, "image_candidates": candidates, "anchor": anchor,
                                   "evidence": evidence, "gold_recalled": gold_recalled})

            report["retrieval_gold_recall_count"] = sum(item["gold_recalled"] for item in retrievals)
            report["gpu_preflight"] = idle_gpu(args.gpu)
            report["model_snapshot"] = local_snapshot_valid(MODEL)
            atomic_json(report_file, report)

            os.environ.update(
                CUDA_VISIBLE_DEVICES=str(args.gpu), HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
            )
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
                    messages, tokenize=False, add_generation_prompt=True
                )
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
                answer_normalized = normalize_answer(answer)
                gold_normalized = [normalize_answer(value) for value in item["gold"]]
                exact = answer_normalized in gold_normalized
                contains = any(gold and (gold in answer_normalized or answer_normalized in gold)
                               for gold in gold_normalized)
                report["samples"].append({
                    "row_index": item["row_index"], "title": item["title"],
                    "question": item["question"], "image_id": item["image_id"],
                    "image_anchor": item["anchor"], "retrieved_evidence": item["evidence"],
                    "gold_recalled": item["gold_recalled"], "raw_model_output": raw_output,
                    "answer": answer, "gold": item["gold"], "exact_match": exact,
                    "answer_contains_gold": contains,
                    "input_tokens": int(inputs["input_ids"].shape[-1]),
                    "output_tokens": int(continuation.shape[-1]),
                    "inference_seconds": round(time.perf_counter() - started, 3),
                })
                atomic_json(report_file, report)
                print(json.dumps({"title": item["title"], "answer": answer,
                                  "correct": contains}, ensure_ascii=False), flush=True)
            correct = sum(sample["answer_contains_gold"] for sample in report["samples"])
            report.update(
                status="complete", finished_at=now(), sample_count=len(report["samples"]),
                correct_count=correct, smoke_accuracy=correct / len(report["samples"]),
                all_image_anchors_correct=True,
                all_retrievals_nonempty=True,
                all_gold_answers_recalled=all(item["gold_recalled"] for item in retrievals),
                gpu_name=torch.cuda.get_device_name(0),
                peak_gpu_allocated_mib=round(torch.cuda.max_memory_allocated(0) / 1024**2, 1),
                peak_gpu_reserved_mib=round(torch.cuda.max_memory_reserved(0) / 1024**2, 1),
            )
            atomic_json(report_file, report)
            print(json.dumps({
                "status": report["status"], "samples": report["sample_count"],
                "correct": report["correct_count"], "accuracy": report["smoke_accuracy"],
                "retrieval_gold_recall": report["retrieval_gold_recall_count"],
                "peak_gpu_allocated_mib": report["peak_gpu_allocated_mib"],
            }, ensure_ascii=False), flush=True)
            return 0
        except Exception as exc:
            report.update(status="failed", finished_at=now(),
                          error_type=type(exc).__name__, error=sanitized(exc))
            atomic_json(report_file, report)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
