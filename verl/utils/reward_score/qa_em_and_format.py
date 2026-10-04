# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import json
import string
import random
from collections import Counter

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def cal_em(gold_batches, predictions):
    """Corpus mean exact match using the module's QA normalization."""
    scores = []
    for golds, prediction in zip(gold_batches, predictions):
        if isinstance(golds, str):
            golds = [golds]
        normalized_prediction = normalize_answer(str(prediction))
        scores.append(float(any(
            normalize_answer(str(gold)) == normalized_prediction for gold in golds
        )))
    return sum(scores) / len(scores) if scores else 0.0


def cal_f1(gold_batches, predictions):
    """Corpus mean token F1, taking the best score across valid answers."""
    scores = []
    for golds, prediction in zip(gold_batches, predictions):
        if isinstance(golds, str):
            golds = [golds]
        prediction_tokens = normalize_answer(str(prediction)).split()
        best = 0.0
        for gold in golds:
            gold_tokens = normalize_answer(str(gold)).split()
            common = Counter(prediction_tokens) & Counter(gold_tokens)
            overlap = sum(common.values())
            if not prediction_tokens and not gold_tokens:
                score = 1.0
            elif overlap == 0:
                score = 0.0
            else:
                precision = overlap / len(prediction_tokens)
                recall = overlap / len(gold_tokens)
                score = 2 * precision * recall / (precision + recall)
            best = max(best, score)
        scores.append(best)
    return sum(scores) / len(scores) if scores else 0.0


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0.0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1.0
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0.0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1.0
            break
    return score


def extract_solution(solution_str):
    """Extract the answer from the solution string."""
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, solution_str, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    return None

def compute_score_format(solution_str):
    """The scoring function for format reward.

    Args:
        solution_str: the solution text
    
    """
    if solution_str is None:
        return 0.0
    
    try:
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        if not assistant_blocks:
            return 0.0

        def parse_tool_call(block):
            match = re.fullmatch(
                r'\s*<think>(.*?)</think>\s*<tool_call>(.*?)</tool_call>\s*',
                block,
                re.DOTALL,
            )
            if match is None:
                return None
            try:
                payload = json.loads(match.group(2))
            except (TypeError, ValueError):
                return None
            if not isinstance(payload, dict) or payload.get("tool") != "kb_search":
                return None
            args = payload.get("args")
            if not isinstance(args, dict):
                return None
            query = args.get("query")
            return query.strip() if isinstance(query, str) else None

        # Strong retrieval protocol: visual grounding, then natural-language
        # graph lookup, then one final answer.  A visual-only shortcut no
        # longer receives the same full format reward as the complete chain.
        format_reward = 0.0
        first_query = parse_tool_call(assistant_blocks[0])
        visual_ok = first_query == "<img>"
        if visual_ok:
            format_reward += 0.25

        text_ok = False
        if visual_ok and len(assistant_blocks) >= 2:
            second_query = parse_tool_call(assistant_blocks[1])
            text_ok = bool(second_query and second_query != "<img>")
            if text_ok:
                format_reward += 0.25

        final_match = re.fullmatch(
            r'\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*',
            assistant_blocks[-1],
            re.DOTALL,
        )
        if final_match is not None and final_match.group(2).strip():
            format_reward += 0.5
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_format: {e}")
        return 0.0
    
    return format_reward


def compute_score_answer(solution_str, ground_truth):
    """The scoring function for exact match (EM) with format reward.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
    
    Returns:
        float: Total reward score (format reward + answer reward)
    """
    if solution_str is None:
        return 0.0
    
    try:
        # Extract answer from <answer> tags
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        if not assistant_blocks:
            return 0.0
        solution_str = assistant_blocks[-1]
        answer = extract_solution(solution_str)

        answer_reward = 0.0
        
        if answer is not None:
            # Safe ground_truth handling
            if isinstance(ground_truth, str):
                ground_truth_list = [ground_truth]
            elif hasattr(ground_truth, 'tolist'):
                ground_truth_list = ground_truth.tolist()
            elif isinstance(ground_truth, list):
                ground_truth_list = ground_truth
            else:
                ground_truth_list = [str(ground_truth)]
            
            answer_reward = cal_f1([ground_truth_list],[answer])
        
        # If no match found within <answer>, check entire solution for substring match
        # if answer_reward == 0.0:
        #     if subem_check(solution_str, ground_truth):
        #         answer_reward = 0.2
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_answer: {e}")
        return 0.0
    
    return answer_reward

def compute_score_format_answer(solution_str, ground_truth):
    """The scoring function for format reward.

    Args:
        solution_str: the solution text
    
    """
    if solution_str is None or ground_truth is None:
        return 0.0

    try:
        format_reward = compute_score_format(solution_str)
        answer_reward = compute_score_answer(solution_str, ground_truth)

        if format_reward == 1.0:
            return -1.0 + format_reward + answer_reward
        else:
            return -1.0 + format_reward
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_format_answer: {e}")
        return 0.0

def compute_score_em(solution_str, ground_truth):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
    
    """
    if solution_str is None or ground_truth is None:
        return 0.0
    
    try:
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        if not assistant_blocks:
            return 0.0
        solution_str = assistant_blocks[-1]
        answer = extract_solution(solution_str)
        if answer is None:
            return 0.0
        
        # Safe ground_truth handling
        if isinstance(ground_truth, str):
            ground_truth_list = [ground_truth]
        elif hasattr(ground_truth, 'tolist'):
            ground_truth_list = ground_truth.tolist()
        elif isinstance(ground_truth, list):
            ground_truth_list = ground_truth
        else:
            ground_truth_list = [str(ground_truth)]
        
        return float(cal_em([ground_truth_list],[answer]))
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_em: {e}")
        return 0.0
    
def compute_score_f1(solution_str, ground_truth):
    """The scoring function for exact match (F1).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
    
    """
    if solution_str is None or ground_truth is None:
        return 0.0
    
    try:
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        if not assistant_blocks:
            return 0.0
        solution_str = assistant_blocks[-1]
        answer = extract_solution(solution_str)
        if answer is None:
            return 0.0
        
        # Safe ground_truth handling
        if isinstance(ground_truth, str):
            ground_truth_list = [ground_truth]
        elif hasattr(ground_truth, 'tolist'):
            ground_truth_list = ground_truth.tolist()
        elif isinstance(ground_truth, list):
            ground_truth_list = ground_truth
        else:
            ground_truth_list = [str(ground_truth)]
        
        return float(cal_f1([ground_truth_list],[answer]))
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_f1: {e}")
        return 0.0
