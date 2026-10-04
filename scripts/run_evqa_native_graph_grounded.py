#!/usr/bin/env python3
"""Run the isolated four-document graph smoke test with grounded prompts."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphr1.prompt_grounded import (
    GROUNDED_PROMPT_VERSION,
    GROUNDED_REVIEW_PROMPT,
    activate_grounded_prompts,
)
from scripts import run_evqa_native_graph_smoke as smoke


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=smoke.MODEL,
    )
    parser.add_argument(
        "--output-tag",
        choices=("evqa_native_graph_grounded", "evqa_native_graph_grounded_7b",
                 "evqa_native_graph_grounded_7b_v2", "evqa_native_graph_grounded_7b_v3"),
        default="evqa_native_graph_grounded",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--review-pass", action="store_true")
    config, remaining = parser.parse_known_args(argv)
    models_root = smoke.ROOT / ".." / "models"
    model_path = config.model_path.resolve()
    if models_root.resolve() not in model_path.parents or not model_path.is_dir():
        parser.error("--model-path must be an existing directory under the data-disk models root")
    if not 256 <= config.max_new_tokens <= 4096:
        parser.error("--max-new-tokens must be between 256 and 4096")
    activate_grounded_prompts()
    smoke.MODEL = model_path
    smoke.MAX_NEW_TOKENS = config.max_new_tokens
    smoke.PROMPT_VERSION = GROUNDED_PROMPT_VERSION
    smoke.REVIEW_EXTRACTION = config.review_pass
    smoke.REVIEW_PROMPT = GROUNDED_REVIEW_PROMPT if config.review_pass else None
    smoke.OUTPUT_ROOT = smoke.ROOT / "expr_mm" / config.output_tag
    smoke.OUTPUT = smoke.OUTPUT_ROOT / "E-VQA"
    smoke.PREFIX = config.output_tag
    return smoke.main(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
