"""
Tool generation manager for LLM agents
"""

import torch
import re
import json
import os
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from copy import deepcopy

import random

from .tensor_helper import TensorHelper, TensorConfig
from agent.tool.tool_env import ToolEnv, step, step_batch
from verl import DataProto
from verl.utils.tracking import Tracking
from verl.utils.multimodal import pad_non_tensors, select_non_tensors
from verl.utils.multimodal import process_image

IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"

@dataclass
class ToolGenerationConfig:
    """Configuration for tool-based generation"""
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_tool_response_length: int  # Renamed from max_obs_length
    num_gpus: int
    # use_parallel_tool_calls: bool = False
    use_batch_tool_calls: bool = False  # New option for batch execution
    tool_call_start: str = "<tool_call>"
    tool_call_end: str = "</tool_call>"
    tool_response_start: str = "<knowledge>"
    tool_response_end: str = "</knowledge>"
    # Knowledge pipeline tools use different tags to avoid confusion
    pipeline_response_start: str = "<pipeline>"
    pipeline_response_end: str = "</pipeline>"
    tool_custom_response_template: str = ""
    force_first_image_search: bool = True
    force_graph_edit_verification: bool = False
    response_guidance: bool = False
    max_turn_response_length: int | None = None


@dataclass
class TrajectoryState:
    """Per-trajectory source of truth for dynamic multimodal rollout state."""

    prompt_token_ids: List[int]
    multi_modal_data: Dict[str, Any] | None = None
    tool_return_images: List[Any] = field(default_factory=list)
    
class ToolGenerationManager:
    """Manager for handling LLM tool-based generation and interaction"""
    
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: ToolGenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        
        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_tool_response_length=config.max_tool_response_length,  # Renamed
            max_start_length=config.max_start_length,
        ))
        self.image_processor = self._load_image_processor()

    def _load_image_processor(self):
        model_path = getattr(self.tokenizer, "name_or_path", None)
        if not model_path:
            return None
        try:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            return getattr(processor, "image_processor", None)
        except Exception as exc:
            print(f"[warn] failed to load image processor for final mm alignment: {exc}", flush=True)
            return None

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _process_tool_call(self, responses_str) -> Tuple[List[str], List[bool]]:
        """
        Process a list of response strings to extract the first tool call
        while preserving the rest of the string content.
        
        Args:
            responses_str (List[str]): List of response strings potentially containing tool calls
            
        Returns:
            List[str]: Processed responses with only first tool call preserved
        """
        def process_single_response(resp):
            tool_pattern = r'<tool_call>(.*?)</tool_call>'
            match = re.search(tool_pattern, resp, re.DOTALL)

            if not match:
                repaired = self._repair_unclosed_tool_call(resp)
                if repaired is not None:
                    return repaired + self.tokenizer.eos_token, True
                # A dangling tool-call start often makes HF rollout spend a full
                # max_new_tokens pass on every retry. End the trajectory instead
                # of letting a malformed call dominate validation/training time.
                return resp + self.tokenizer.eos_token, False

            resp = resp.split(self.config.tool_call_end)[0] + self.config.tool_call_end

            return resp + self.tokenizer.eos_token, True
        
        # Process each response string (single pass)
        processed = [process_single_response(resp) for resp in responses_str]
        return [p[0] for p in processed], [p[1] for p in processed]

    def _repair_unclosed_tool_call(self, response: str) -> str | None:
        """Normalize a complete JSON tool call whose XML closing tag is malformed.

        The generated tool name and arguments are preserved verbatim at the JSON
        value level. A narrowly scoped fallback also repairs JSON punctuation
        after all required string values have already been closed; it never
        invents or completes argument text.
        """
        start = response.find(self.config.tool_call_start)
        if start < 0:
            return None
        payload_start = start + len(self.config.tool_call_start)
        payload = response[payload_start:].lstrip()
        try:
            value, _ = json.JSONDecoder().raw_decode(payload)
        except (json.JSONDecodeError, TypeError):
            value = self._recover_closed_string_tool_call(payload)
            if value is None:
                return None
        if not isinstance(value, dict):
            return None
        if not isinstance(value.get("tool"), str) or not isinstance(value.get("args"), dict):
            return None
        normalized = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        prefix = response[:start]
        return f"{prefix}{self.config.tool_call_start}{normalized}{self.config.tool_call_end}"

    @staticmethod
    def _recover_closed_string_tool_call(payload: str) -> Dict[str, Any] | None:
        """Recover missing braces/tags only when every required value is complete."""
        def string_field(name: str) -> str | None:
            match = re.search(
                rf'"{re.escape(name)}"\s*:\s*("(?:\\.|[^"\\])*")',
                payload,
                flags=re.DOTALL,
            )
            if not match:
                return None
            try:
                value = json.loads(match.group(1))
            except (TypeError, ValueError):
                return None
            return value if isinstance(value, str) and value.strip() else None

        tool = string_field("tool")
        required_fields = {
            "kb_search": ("query",),
            "websearch": ("query",),
            "insert": ("content",),
            "update": ("content", "new_content"),
            "delete": ("content",),
        }.get(tool)
        if required_fields is None:
            return None
        args = {name: string_field(name) for name in required_fields}
        if any(value is None for value in args.values()):
            return None
        return {"tool": tool, "args": args}

    def _forced_image_search_response(self) -> str:
        return (
            '<think>Identify the image entity before doing text lookup.</think>\n'
            '<tool_call>{"tool":"kb_search","args":{"query":"<img>"}}</tool_call>'
            + self.tokenizer.eos_token
        )

    def _build_forced_image_search_batch(self, active_mask: torch.Tensor) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        responses_str = [self._forced_image_search_response() for _ in range(int(active_mask.sum().item()))]
        responses_ids = self._batch_tokenize(responses_str)
        return responses_ids, responses_str, torch.ones(len(responses_str), dtype=torch.bool)

    def _should_force_first_image_search(self, step: int, rollings: DataProto) -> bool:
        return (
            self.config.force_first_image_search
            and step == 0
            and self._has_multimodal_data(rollings.non_tensor_batch)
        )

    def _graph_edit_verification_query(self, env: Any) -> str | None:
        """Return the fact that must be searched immediately after a successful edit."""
        history = list(getattr(env, "tool_history", []) or [])
        if not history:
            return None
        last_call = history[-1]
        if last_call.get("tool") not in {"insert", "update"}:
            return None
        if not self._tool_result_succeeded(last_call.get("result")):
            return None

        args = last_call.get("args") if isinstance(last_call.get("args"), dict) else {}
        value = args.get("new_content") if last_call.get("tool") == "update" else args.get("content")
        if isinstance(value, str):
            return value.strip() or None
        if isinstance(value, list):
            parts = [str(item).strip() for item in value if str(item).strip()]
            return " ".join(parts) or None
        return None

    def _forced_graph_edit_verification_response(self, query: str) -> str:
        payload = json.dumps(
            {"tool": "kb_search", "args": {"query": query}},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return (
            "<think>Verify that the successful graph edit is searchable.</think>\n"
            f"{self.config.tool_call_start}{payload}{self.config.tool_call_end}"
            + self.tokenizer.eos_token
        )

    def _apply_forced_graph_edit_verification(
        self,
        responses_ids: torch.Tensor,
        responses_str: List[str],
        new_active_masks: torch.Tensor,
        active_mask: torch.Tensor,
        envs: List[Any] | None,
    ) -> Tuple[torch.Tensor, List[str], torch.Tensor, int]:
        """Replace only trajectories that immediately require post-edit verification."""
        if not getattr(self.config, "force_graph_edit_verification", False) or not envs:
            return responses_ids, responses_str, new_active_masks, 0

        active_envs = [env for env, active in zip(envs, active_mask.tolist()) if active]
        forced_count = 0
        updated_responses = list(responses_str)
        updated_masks = new_active_masks.clone()
        for local_index, env in enumerate(active_envs):
            query = self._graph_edit_verification_query(env)
            if query is None:
                continue
            updated_responses[local_index] = self._forced_graph_edit_verification_response(query)
            updated_masks[local_index] = True
            forced_count += 1

        if not forced_count:
            return responses_ids, responses_str, new_active_masks, 0
        return (
            self._batch_tokenize(updated_responses),
            updated_responses,
            updated_masks,
            forced_count,
        )

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to extract tool calls."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        # Extract the first tool call from each response
        responses_str, active_masks = self._process_tool_call(responses_str)
        
        # Tokenize processed responses
        cleaned_token_ids = self._batch_tokenize(responses_str)
        
        return cleaned_token_ids, responses_str, torch.tensor(active_masks, dtype=torch.bool)
    
    def _process_tool_responses(self, tool_responses: List[str]) -> torch.Tensor:
        """Process tool responses to token ids"""

        encoded_rows = self.tokenizer(
            tool_responses,
            add_special_tokens=False,
            padding=False,
        )["input_ids"]
        marker_ids = self.tokenizer(
            "\n[Earlier tool results truncated; required next action follows.]\n",
            add_special_tokens=False,
        )["input_ids"]
        truncated_rows = []
        truncated_count = 0
        for token_ids in encoded_rows:
            if len(token_ids) > self.config.max_tool_response_length:
                truncated_count += 1
                token_ids = self._middle_truncate_token_ids(
                    token_ids,
                    self.config.max_tool_response_length,
                    marker_ids,
                )
            truncated_rows.append(token_ids)
        if truncated_count:
            print(
                "TOOL_RESPONSES_MIDDLE_TRUNCATED: "
                f"rows={truncated_count}/{len(encoded_rows)} "
                f"limit={self.config.max_tool_response_length}",
                flush=True,
            )
        return self.tokenizer.pad(
            {"input_ids": truncated_rows},
            padding=True,
            return_tensors="pt",
        )["input_ids"]

    @staticmethod
    def _middle_truncate_token_ids(
        token_ids: List[int],
        max_length: int,
        marker_ids: List[int],
    ) -> List[int]:
        """Preserve leading evidence and trailing state guidance under a token cap."""
        if len(token_ids) <= max_length:
            return list(token_ids)
        marker = list(marker_ids[:max_length])
        remaining = max_length - len(marker)
        if remaining <= 0:
            return marker
        head_length = (remaining * 3) // 5
        tail_length = remaining - head_length
        if tail_length == 0:
            return list(token_ids[:head_length]) + marker
        return list(token_ids[:head_length]) + marker + list(token_ids[-tail_length:])

    def _to_token_list(self, token_ids: Any) -> List[int]:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        elif hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        return [int(token_id) for token_id in token_ids]

    def _row_valid_token_list(self, row: torch.Tensor) -> List[int]:
        valid = row[row != self.tokenizer.pad_token_id]
        if valid.numel() == 0:
            return []
        return self._to_token_list(valid)

    def _images_from_item(self, item: Any) -> List[Any]:
        if isinstance(item, dict):
            images = item.get("image")
            if images is not None:
                if isinstance(images, (list, tuple, np.ndarray)):
                    return list(images)
                return [images]
        return []

    def _init_trajectory_states(self, gen_batch: DataProto) -> List[TrajectoryState]:
        batch_size = gen_batch.batch["input_ids"].shape[0]
        non_tensors = gen_batch.non_tensor_batch or {}
        raw_prompt_ids = non_tensors.get("raw_prompt_ids")
        multi_modal_data = non_tensors.get("multi_modal_data")

        states = []
        for row_idx in range(batch_size):
            if raw_prompt_ids is not None and row_idx < len(raw_prompt_ids):
                prompt_token_ids = self._to_token_list(raw_prompt_ids[row_idx])
            else:
                prompt_token_ids = self._row_valid_token_list(gen_batch.batch["input_ids"][row_idx])

            item = None
            if multi_modal_data is not None and row_idx < len(multi_modal_data):
                item = multi_modal_data[row_idx]
            item = deepcopy(item) if isinstance(item, dict) else None
            states.append(TrajectoryState(prompt_token_ids=prompt_token_ids, multi_modal_data=item))
        return states

    def _states_to_non_tensors(self, states: List[TrajectoryState]) -> Dict[str, Any]:
        non_tensors = {
            "raw_prompt_ids": np.array(
                [list(state.prompt_token_ids) for state in states],
                dtype=object,
            )
        }
        if any(state.multi_modal_data is not None for state in states):
            non_tensors["multi_modal_data"] = np.array(
                [
                    deepcopy(state.multi_modal_data) if state.multi_modal_data is not None else {}
                    for state in states
                ],
                dtype=object,
            )
        return non_tensors

    def _sync_rollings_from_states(self, rollings: DataProto, states: List[TrajectoryState]) -> DataProto:
        non_tensors = dict(rollings.non_tensor_batch or {})
        non_tensors.update(self._states_to_non_tensors(states))
        return DataProto.from_dict(
            {key: value for key, value in rollings.batch.items()},
            non_tensors=non_tensors,
        )

    def _prepare_tool_responses_multimodal(
        self,
        tool_responses: List[str],
        envs: List[Any] | None = None,
    ) -> Tuple[List[str], List[List[Any]]]:
        prepared_responses = []
        response_images = []
        for index, tool_response in enumerate(tool_responses):
            image_paths = self._extract_tool_response_image_paths(tool_response)
            images = self._load_tool_response_images(image_paths)
            tool_response = self._strip_tool_response_image_fields(tool_response)
            anchor_entity = self._first_result_entity(tool_response) if image_paths else ""
            if self.config.response_guidance and tool_response:
                tool_response = self._append_next_action_guidance(
                    tool_response,
                    image_response=bool(image_paths),
                    env=envs[index] if envs and index < len(envs) else None,
                    anchor_entity=anchor_entity,
                )
            if images:
                tool_response = self._append_image_placeholders(tool_response, len(images))
            prepared_responses.append(tool_response)
            response_images.append(images)
        return prepared_responses, response_images

    def _append_next_action_guidance(
        self,
        tool_response: str,
        *,
        image_response: bool,
        env: Any = None,
        anchor_entity: str = "",
    ) -> str:
        """Add the next GraphEdit state transition to a tool observation."""
        history = list(getattr(env, "tool_history", []) or [])
        last_call = history[-1] if history else {}
        last_tool = last_call.get("tool")
        last_args = last_call.get("args") if isinstance(last_call.get("args"), dict) else {}
        successful_edit_seen = any(
            call.get("tool") in {"insert", "update", "delete"}
            and self._tool_result_succeeded(call.get("result"))
            for call in history
        )

        if image_response:
            anchor = anchor_entity or "<top-ranked entity name>"
            guidance = (
                f"The required visual anchor is the first and top-ranked entity: {anchor}. "
                "Do not substitute a lower-ranked candidate. Your next response MUST be the "
                "required text lookup even if you think you already know the answer. Use exactly "
                "this JSON shape with a concrete query: "
                f'<tool_call>{{"tool":"kb_search","args":{{"query":"{anchor} <fact asked by the original question>"}}}}</tool_call>. '
                "Do not answer and do not call websearch before this text KB lookup."
            )
        elif "Invalid arguments for tool 'websearch'" in tool_response:
            guidance = (
                "Next required action: retry websearch now. Put the concrete search phrase "
                "inside args.query, not in log or another top-level field. Use exactly: "
                '<tool_call>{"tool":"websearch","args":{"query":"<entity name> <missing fact>"}}</tool_call>.'
            )
        elif last_tool == "websearch":
            guidance = (
                "Next required action: if the web evidence states the requested fact "
                "specifically, insert one short atomic fact into the graph. Use exactly: "
                '<tool_call>{"tool":"insert","args":{"content":"<entity> <relation> <verified value>."}}</tool_call>. '
                "Do not answer before the graph edit and its verification."
            )
        elif last_tool in {"insert", "update", "delete"}:
            if self._tool_result_succeeded(last_call.get("result")):
                guidance = (
                    "Next required action: verify the successful graph edit with a text "
                    "kb_search for the edited entity and fact. Use exactly: "
                    '<tool_call>{"tool":"kb_search","args":{"query":"<entity name> <edited fact>"}}</tool_call>. '
                    "Do not answer before verification."
                )
            else:
                if "pre-commit evidence gate" in tool_response:
                    guidance = (
                        "The pre-commit gate rejected this graph edit. Read its conflict type "
                        "and reason, keep the top-ranked visual anchor fixed, and use only prior "
                        "web evidence that explicitly refers to that exact entity and location. "
                        "Retry one short atomic edit; do not reuse a namesake fact and do not "
                        "answer before a successful edit and verification."
                    )
                else:
                    guidance = (
                        "The graph edit did not succeed. Correct the edit arguments using the "
                        "reported error, then retry one insert, update, or delete call."
                    )
        elif last_tool == "kb_search" and last_args.get("query") != "<img>":
            if successful_edit_seen:
                guidance = (
                    "The edited fact has now been queried for verification. If it appears in "
                    "the returned KB evidence, answer with exactly one <answer>...</answer> "
                    "block containing only the shortest answer span."
                )
            else:
                guidance = (
                "Check whether this KB result explicitly states the exact fact requested "
                "by the original question. Do not infer the answer from the entity name or "
                "merely related context. If the exact fact is present, answer with only the "
                "short answer span. If it is absent, the next required action is websearch "
                "using an entity-first query that includes a distinguishing location or other "
                "identity detail from the KB result. Use this exact shape: "
                '<tool_call>{"tool":"websearch","args":{"query":"<entity name> <location or identity detail> <missing fact>"}}</tool_call>.'
            )
        else:
            guidance = (
                "Follow the required GraphEdit sequence using only the listed tools and exact "
                "JSON argument shapes."
            )

        end_tag = self.config.tool_response_end
        if end_tag in tool_response:
            return tool_response.replace(end_tag, f"\n{guidance}\n{end_tag}", 1)
        return f"{tool_response}\n{guidance}"

    def _first_result_entity(self, tool_response: str) -> str:
        pattern = (
            re.escape(self.config.tool_response_start)
            + r"\s*(.*?)\s*"
            + re.escape(self.config.tool_response_end)
        )
        match = re.search(pattern, tool_response, flags=re.DOTALL)
        payload = match.group(1) if match else tool_response
        try:
            parsed = json.loads(payload)
        except (TypeError, ValueError):
            return ""
        results = parsed.get("results") if isinstance(parsed, dict) else None
        if not isinstance(results, list) or not results or not isinstance(results[0], dict):
            return ""
        entity = results[0].get("entity")
        return str(entity).strip() if entity is not None else ""

    def _strip_tool_response_image_fields(self, tool_response: str) -> str:
        """Hide local image locators from the model-visible tool response."""
        start_tag = self.config.tool_response_start
        end_tag = self.config.tool_response_end

        def sanitize_payload(payload: str) -> str:
            try:
                parsed = json.loads(payload)
            except Exception:
                return re.sub(
                    r',?\s*"(?:image_path|image_url)"\s*:\s*"[^"]*"',
                    "",
                    payload,
                )
            return json.dumps(self._remove_image_locator_fields(parsed), ensure_ascii=False)

        pattern = re.escape(start_tag) + r"\s*(.*?)\s*" + re.escape(end_tag)

        def replace_match(match: re.Match) -> str:
            return f"{start_tag}{sanitize_payload(match.group(1))}{end_tag}"

        updated, count = re.subn(pattern, replace_match, tool_response, flags=re.DOTALL)
        if count:
            return updated
        return sanitize_payload(tool_response)

    def _remove_image_locator_fields(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: self._remove_image_locator_fields(item)
                for key, item in value.items()
                if key not in {"image_path", "image_url"}
            }
        if isinstance(value, list):
            return [self._remove_image_locator_fields(item) for item in value]
        return value

    def _extract_tool_response_image_paths(self, tool_response: str) -> List[str]:
        limit = int(os.getenv("TOOL_RESPONSE_IMAGE_LIMIT", "3"))
        if limit <= 0:
            return []

        payloads = re.findall(
            re.escape(self.config.tool_response_start) + r"\s*(.*?)\s*" + re.escape(self.config.tool_response_end),
            tool_response,
            flags=re.DOTALL,
        ) or [tool_response]

        paths = []
        seen = set()
        for payload in payloads:
            parsed_items = []
            try:
                parsed_items.append(json.loads(payload))
            except Exception:
                parsed_items = []

            for parsed in parsed_items:
                for result in self._iter_json_results(parsed):
                    if not isinstance(result, dict):
                        continue
                    image_path = str(result.get("image_path") or "").strip()
                    if image_path and image_path not in seen:
                        seen.add(image_path)
                        paths.append(image_path)
                        if len(paths) >= limit:
                            return paths

            if not parsed_items:
                for match in re.finditer(r'"image_path"\s*:\s*"([^"]+)"', payload):
                    image_path = match.group(1).strip()
                    if image_path and image_path not in seen:
                        seen.add(image_path)
                        paths.append(image_path)
                        if len(paths) >= limit:
                            return paths
        return paths

    def _iter_json_results(self, parsed: Any):
        if isinstance(parsed, dict):
            results = parsed.get("results")
            if isinstance(results, list):
                yield from results
            else:
                yield parsed
        elif isinstance(parsed, list):
            for item in parsed:
                yield from self._iter_json_results(item)

    def _load_tool_response_images(self, image_paths: List[str]) -> List[Any]:
        images = []
        for image_path in image_paths:
            try:
                images.append(process_image(image_path))
            except Exception as exc:
                print(
                    f"[warn] failed to load tool-returned image: {image_path} ({exc})",
                    flush=True,
                )
        return images

    def _append_image_placeholders(self, tool_response: str, image_count: int) -> str:
        placeholders = "\nRetrieved images:\n" + "\n".join(
            f"image {index + 1}: {IMAGE_PLACEHOLDER}"
            for index in range(image_count)
        )
        end_tag = self.config.tool_response_end
        if end_tag in tool_response:
            return tool_response.replace(end_tag, placeholders + "\n" + end_tag, 1)
        return tool_response + placeholders
    
    def _execute_tool_calls(self, response_strs: List[str], 
                          envs: List[ToolEnv], 
                          active_mask: torch.Tensor) -> List[str]:
        """Execute tool calls sequentially and return tool responses."""
        # Convert torch tensor to list of booleans if needed
        active_list = active_mask.tolist() if isinstance(active_mask, torch.Tensor) else active_mask
        
        # Initialize result list with empty strings
        tool_responses = [""] * len(response_strs)
        # Process each environment sequentially
        for i, (resp, env, active) in enumerate(zip(response_strs, envs, active_list)):
            if not active:
                continue
                
            # Step the environment using the agent's response
            result = step(env, resp)
            tool_response = result[0]  # Extract observation from (observation, reward, done, info)
            
            # Determine tool type and use appropriate template
            tool_name = self._extract_tool_name(resp)
            if tool_name in ["insert", "update", "delete"]:
                # Use pipeline template for knowledge management tools
                template = self.config.tool_custom_response_template.replace(
                    self.config.tool_response_start, 
                    self.config.pipeline_response_start
                ).replace(
                    self.config.tool_response_end, 
                    self.config.pipeline_response_end
                )
                tool_responses[i] = template.format(tool_response=tool_response)
            else:
                # Use knowledge template for search tools
                tool_responses[i] = self.config.tool_custom_response_template.format(tool_response=tool_response)            
        return tool_responses
    
    def _extract_tool_name(self, response_str: str) -> str:
        """Extract tool name from response string"""
        import re
        import json
        
        # Try to extract tool call
        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
        tool_call_match = re.search(tool_call_pattern, response_str, re.DOTALL)
        
        if tool_call_match:
            try:
                tool_call_json = tool_call_match.group(1).strip()
                tool_call_data = json.loads(tool_call_json)
                
                # Handle both dictionary and list formats
                if isinstance(tool_call_data, dict):
                    return tool_call_data.get("tool", "unknown")
                elif isinstance(tool_call_data, list) and len(tool_call_data) > 0:
                    # If it's a list, take the first element and extract tool name
                    first_call = tool_call_data[0]
                    if isinstance(first_call, dict):
                        return first_call.get("tool", "unknown")
                    else:
                        return "unknown"
                else:
                    return "unknown"
            except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                pass
        
        # Try legacy query format
        query_pattern = r'<query>(.*?)</query>'
        query_match = re.search(query_pattern, response_str, re.DOTALL)
        if query_match:
            return "kb_search"  # Legacy format defaults to kb_search
            
        return "unknown"
    
    def _execute_tool_calls_batch(self, response_strs: List[str], 
                                 envs: List[ToolEnv], 
                                 active_mask: torch.Tensor) -> List[str]:
        """Execute tool calls in batch for tools that support batch operations."""
        # Convert torch tensor to list of booleans
        active_list = active_mask.tolist() if isinstance(active_mask, torch.Tensor) else active_mask
        
        # Filter active environments and responses
        active_envs = []
        active_responses = []
        active_indices = []
        
        for i, (env, resp, active) in enumerate(zip(envs, response_strs, active_list)):
            if active:
                active_envs.append(env)
                active_responses.append(resp)
                active_indices.append(i)
        
        # Initialize result list with empty strings
        tool_responses = [""] * len(response_strs)
        
        if not active_envs:
            return tool_responses
            
        # Use the independent step_batch function for active environments with step numbers
        # Generate step numbers for batch processing (every 10 steps)
        step_numbers = [i + 1 for i in range(len(active_envs))]  # Simple step numbering
        batch_results = step_batch(active_envs, active_responses, step_numbers)
        
        # Map results back to original indices
        for idx, result, resp in zip(active_indices, batch_results, active_responses):
            if result is None:
                tool_responses[idx] = ""
            else:
                tool_response = result[0]  # Extract observation from (observation, reward, done, info)
                
                # Determine tool type and use appropriate template
                tool_name = self._extract_tool_name(resp)
                if tool_name in ["insert", "update", "delete"]:
                    # Use pipeline template for knowledge management tools
                    template = self.config.tool_custom_response_template.replace(
                        self.config.tool_response_start, 
                        self.config.pipeline_response_start
                    ).replace(
                        self.config.tool_response_end, 
                        self.config.pipeline_response_end
                    )
                    tool_responses[idx] = template.format(tool_response=tool_response)
                else:
                    # Use knowledge template for search tools
                    tool_responses[idx] = self.config.tool_custom_response_template.format(tool_response=tool_response)
        return tool_responses
    
    def _update_rolling_state(self, rollings, cur_responses: torch.Tensor, 
                            tool_responses_ids: torch.Tensor,
                            tool_response_images: List[List[Any]] | None = None,
                            trajectory_states: List[TrajectoryState] | None = None) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            tool_responses_ids
        ])

        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, int(effective_len.item()))

        if self._has_multimodal_data(rollings.non_tensor_batch):
            new_input_ids = self._truncate_preserving_vision_tokens(new_input_ids, max_len)
            new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        else:
            new_input_ids = new_input_ids[:, -max_len:]
            new_attention_mask = new_attention_mask[:, -max_len:]

        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)
        if rollings.batch['position_ids'].dim() == 3:
            new_position_ids = new_position_ids.unsqueeze(1).expand(-1, 3, -1)

        non_tensors = dict(rollings.non_tensor_batch or {})
        if trajectory_states is not None:
            self._update_trajectory_states(
                trajectory_states,
                cur_responses,
                tool_responses_ids,
                tool_response_images=tool_response_images,
                max_len=max_len,
            )
            non_tensors.update(self._states_to_non_tensors(trajectory_states))
        else:
            if "raw_prompt_ids" in non_tensors:
                non_tensors["raw_prompt_ids"] = self._update_raw_prompt_ids(
                    non_tensors["raw_prompt_ids"],
                    cur_responses,
                    tool_responses_ids,
                    max_len=max_len,
                )
            if tool_response_images:
                non_tensors = self._append_tool_images_to_non_tensors(
                    non_tensors,
                    tool_response_images,
                )

        return DataProto.from_dict(
            {
                'input_ids': new_input_ids,
                'position_ids': new_position_ids,
                'attention_mask': new_attention_mask
            },
            non_tensors=non_tensors,
        )

    def _update_trajectory_states(
        self,
        states: List[TrajectoryState],
        cur_responses: torch.Tensor,
        tool_responses_ids: torch.Tensor,
        *,
        tool_response_images: List[List[Any]] | None,
        max_len: int,
    ) -> None:
        appended = 0
        for row_idx, state in enumerate(states):
            response_ids = cur_responses[row_idx]
            response_ids = response_ids[response_ids != self.tokenizer.pad_token_id]
            tool_ids = tool_responses_ids[row_idx]
            tool_ids = tool_ids[tool_ids != self.tokenizer.pad_token_id]

            combined = state.prompt_token_ids + self._to_token_list(response_ids) + self._to_token_list(tool_ids)
            state.prompt_token_ids = self._truncate_token_list_preserving_vision(combined, max_len)

            row_images = tool_response_images[row_idx] if tool_response_images and row_idx < len(tool_response_images) else []
            if row_images:
                item = deepcopy(state.multi_modal_data) if isinstance(state.multi_modal_data, dict) else {}
                images = self._images_from_item(item)
                images.extend(row_images)
                item["image"] = images
                state.multi_modal_data = item
                state.tool_return_images.extend(row_images)
                appended += len(row_images)

        if appended:
            print(f"ROLLOUT_TRAJECTORY_TOOL_IMAGES_APPENDED: images={appended}", flush=True)

    def _append_tool_images_to_non_tensors(
        self,
        non_tensors: Dict[str, Any],
        tool_response_images: List[List[Any]],
    ) -> Dict[str, Any]:
        multi_modal_data = non_tensors.get("multi_modal_data")
        if multi_modal_data is None:
            return non_tensors

        updated = []
        appended = 0
        for row_idx, item in enumerate(multi_modal_data):
            row_images = tool_response_images[row_idx] if row_idx < len(tool_response_images) else []
            if isinstance(item, dict):
                next_item = deepcopy(item)
            else:
                next_item = {}
            if row_images:
                images = list(next_item.get("image") or [])
                images.extend(row_images)
                next_item["image"] = images
                appended += len(row_images)
            updated.append(next_item)

        if appended:
            print(f"ROLLOUT_TOOL_IMAGES_APPENDED: images={appended}", flush=True)
        non_tensors["multi_modal_data"] = np.array(updated, dtype=object)
        return non_tensors

    def _has_multimodal_data(self, non_tensor_batch: Dict[str, Any]) -> bool:
        data = (non_tensor_batch or {}).get("multi_modal_data")
        if data is None:
            return False
        for item in data:
            if isinstance(item, dict) and item.get("image"):
                return True
        return False

    def _align_final_multimodal_data(
        self,
        input_ids: torch.Tensor,
        non_tensors: Dict[str, Any],
    ) -> Dict[str, Any]:
        multi_modal_data = (non_tensors or {}).get("multi_modal_data")
        if multi_modal_data is None:
            return non_tensors

        aligned = []
        trimmed = 0
        dropped = 0
        for row_idx, item in enumerate(multi_modal_data):
            if not isinstance(item, dict):
                aligned.append(item)
                continue

            row = input_ids[row_idx]
            valid = row[row != self.tokenizer.pad_token_id]
            token_ids = valid.detach().cpu().tolist()
            span_count = len(self._find_vision_spans(token_ids))
            images = item.get("image") or []
            if not isinstance(images, (list, tuple, np.ndarray)):
                images = [images]

            if span_count <= 0:
                next_item = dict(item)
                if images:
                    next_item.pop("image", None)
                    dropped += len(images)
                aligned.append(next_item)
                continue

            if len(images) > span_count:
                next_item = dict(item)
                next_item["image"] = list(images[:span_count])
                trimmed += len(images) - span_count
                aligned.append(next_item)
                continue

            aligned.append(item)

        if trimmed or dropped:
            print(
                f"FINAL_MM_ALIGN_IMAGES: trimmed={trimmed} dropped={dropped}",
                flush=True,
            )
        next_non_tensors = dict(non_tensors)
        next_non_tensors["multi_modal_data"] = np.array(aligned, dtype=object)
        return next_non_tensors

    def _image_token_counts(self, images: List[Any]) -> List[int]:
        if self.image_processor is None or not images:
            return [1 for _ in images]
        image_inputs = self.image_processor(images, return_tensors="pt")
        image_grid_thw = image_inputs.get("image_grid_thw")
        if image_grid_thw is None:
            return [1 for _ in images]
        merge_size = int(getattr(self.image_processor, "merge_size", 1))
        merge_length = merge_size ** 2
        return [max(1, int(grid.prod().item() // merge_length)) for grid in image_grid_thw]

    def _rewrite_vision_spans_to_counts(self, token_ids: List[int], token_counts: List[int]) -> Tuple[List[int], int, int]:
        vision_start_id, image_token_id, vision_end_id = self._vision_token_ids()
        if image_token_id is None:
            return token_ids, 0, 0

        output = []
        cursor = 0
        image_index = 0
        spans_seen = 0
        changed = 0

        def append_span(count: int) -> None:
            if vision_start_id is not None:
                output.append(vision_start_id)
            output.extend([image_token_id] * int(count))
            if vision_end_id is not None:
                output.append(vision_end_id)

        while cursor < len(token_ids):
            current = token_ids[cursor]
            if vision_start_id is not None and vision_end_id is not None and current == vision_start_id:
                try:
                    end = token_ids.index(vision_end_id, cursor + 1)
                except ValueError:
                    output.append(current)
                    cursor += 1
                    continue

                span = token_ids[cursor:end + 1]
                if image_token_id in span:
                    spans_seen += 1
                    if image_index < len(token_counts):
                        count = int(token_counts[image_index])
                        append_span(count)
                        if span.count(image_token_id) != count:
                            changed += 1
                        image_index += 1
                    else:
                        changed += 1
                    cursor = end + 1
                    continue

                output.extend(span)
                cursor = end + 1
                continue

            if current == image_token_id:
                start = cursor
                while cursor < len(token_ids) and token_ids[cursor] == image_token_id:
                    cursor += 1
                spans_seen += 1
                if image_index < len(token_counts):
                    count = int(token_counts[image_index])
                    append_span(count)
                    if cursor - start != count:
                        changed += 1
                    image_index += 1
                else:
                    changed += 1
                continue

            output.append(current)
            cursor += 1

        return output, spans_seen, changed

    def _pad_token_lists(self, token_lists: List[List[int]], width: int, *, pad_to_left: bool) -> torch.Tensor:
        output = torch.full(
            (len(token_lists), width),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        for row_idx, token_ids in enumerate(token_lists):
            token_ids = token_ids[-width:] if len(token_ids) > width else token_ids
            if not token_ids:
                continue
            value = torch.tensor(token_ids, dtype=torch.long)
            if pad_to_left:
                output[row_idx, -len(token_ids):] = value
            else:
                output[row_idx, :len(token_ids)] = value
        return output

    def _expand_final_vision_tokens(
        self,
        prompts: torch.Tensor,
        responses: torch.Tensor,
        non_tensors: Dict[str, Any] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any] | None]:
        if not non_tensors or "multi_modal_data" not in non_tensors:
            return prompts, responses, non_tensors

        multi_modal_data = list(non_tensors["multi_modal_data"])
        prompt_lists = []
        response_lists = []
        aligned_mm = []
        changed_rows = 0
        trimmed_images = 0
        dropped_spans = 0

        for row_idx in range(prompts.shape[0]):
            prompt_ids = self._row_valid_token_list(prompts[row_idx])
            response_ids = self._row_valid_token_list(responses[row_idx])
            item = multi_modal_data[row_idx] if row_idx < len(multi_modal_data) else {}
            item = deepcopy(item) if isinstance(item, dict) else {}
            images = self._images_from_item(item)

            prompt_span_count = len(self._find_vision_spans(prompt_ids))
            response_span_count = len(self._find_vision_spans(response_ids))
            total_span_count = prompt_span_count + response_span_count
            if len(images) > total_span_count:
                item["image"] = images[:total_span_count]
                images = images[:total_span_count]
                trimmed_images += len(self._images_from_item(multi_modal_data[row_idx])) - len(images)

            token_counts = self._image_token_counts(images)
            prompt_counts = token_counts[:prompt_span_count]
            response_counts = token_counts[prompt_span_count:prompt_span_count + response_span_count]

            new_prompt_ids, prompt_spans_seen, prompt_changed = self._rewrite_vision_spans_to_counts(
                prompt_ids,
                prompt_counts,
            )
            new_response_ids, response_spans_seen, response_changed = self._rewrite_vision_spans_to_counts(
                response_ids,
                response_counts,
            )

            represented_spans = min(prompt_spans_seen, len(prompt_counts)) + min(response_spans_seen, len(response_counts))
            if represented_spans < total_span_count:
                dropped_spans += total_span_count - represented_spans
            new_prompt_ids, kept_prompt_spans, truncated_prompt_spans = self._drop_trailing_vision_spans_to_fit(
                new_prompt_ids,
                prompts.shape[1],
            )
            new_response_ids, kept_response_spans, truncated_response_spans = self._drop_trailing_vision_spans_to_fit(
                new_response_ids,
                self.config.max_response_length,
            )
            dropped_spans += truncated_prompt_spans + truncated_response_spans

            # Images follow the same order as their placeholders: prompt images
            # first, then images returned by tools.  When an oversized final
            # sequence forces us to remove a complete trailing vision span, the
            # corresponding image must be removed as well.  Merely slicing the
            # image list to the total number of spans is incorrect when a prompt
            # image is retained while a response image is dropped.
            retained_images = (
                images[:kept_prompt_spans]
                + images[prompt_span_count:prompt_span_count + kept_response_spans]
            )
            if retained_images:
                item["image"] = retained_images
            else:
                item.pop("image", None)
            trimmed_images += max(0, len(images) - len(retained_images))

            if prompt_changed or response_changed or truncated_prompt_spans or truncated_response_spans:
                changed_rows += 1

            prompt_lists.append(self._truncate_token_list_preserving_vision(new_prompt_ids, prompts.shape[1]))
            response_lists.append(self._truncate_token_list_preserving_vision(new_response_ids, self.config.max_response_length))
            aligned_mm.append(item)

        # The dataset pads every prompt to max_prompt_length.  Keeping that
        # width after rebuilding the multimodal batch wastes activation memory
        # when remove-padding is disabled (the supported Qwen2.5-VL path in
        # this project).  Re-pad only to the longest real prompt in this batch.
        prompt_width = max(1, min(prompts.shape[1], max(len(ids) for ids in prompt_lists)))
        response_width = max(1, min(self.config.max_response_length, max(len(ids) for ids in response_lists)))
        new_prompts = self._pad_token_lists(prompt_lists, prompt_width, pad_to_left=True).to(prompts.device)
        new_responses = self._pad_token_lists(response_lists, response_width, pad_to_left=False).to(responses.device)

        if prompt_width < prompts.shape[1] or response_width < responses.shape[1]:
            print(
                f"FINAL_MM_COMPACT_WIDTHS: prompt={prompts.shape[1]}->{prompt_width} "
                f"response={responses.shape[1]}->{response_width}",
                flush=True,
            )

        if changed_rows or trimmed_images or dropped_spans:
            print(
                f"FINAL_MM_EXPAND_IMAGES: rows={changed_rows}/{prompts.shape[0]} "
                f"trimmed_images={trimmed_images} dropped_spans={dropped_spans}",
                flush=True,
            )

        next_non_tensors = dict(non_tensors)
        next_non_tensors["multi_modal_data"] = np.array(aligned_mm, dtype=object)
        return new_prompts, new_responses, next_non_tensors

    def _vision_token_ids(self) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        convert = getattr(self.tokenizer, "convert_tokens_to_ids", None)
        if not callable(convert):
            return None, None, None
        ids = []
        for token in ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>"):
            token_id = convert(token)
            if token_id is None or token_id == getattr(self.tokenizer, "unk_token_id", object()):
                ids.append(None)
            else:
                ids.append(int(token_id))
        return tuple(ids)

    def _find_vision_spans(self, token_ids: List[int]) -> List[Tuple[int, int]]:
        vision_start_id, image_token_id, vision_end_id = self._vision_token_ids()
        if vision_start_id is None or image_token_id is None or vision_end_id is None:
            return []

        spans = []
        cursor = 0
        while cursor < len(token_ids):
            try:
                start = token_ids.index(vision_start_id, cursor)
            except ValueError:
                break
            try:
                end = token_ids.index(vision_end_id, start + 1) + 1
            except ValueError:
                break
            if image_token_id in token_ids[start:end]:
                spans.append((start, end))
            cursor = end
        return spans

    def _drop_trailing_vision_spans_to_fit(
        self,
        token_ids: List[int],
        max_len: int,
    ) -> Tuple[List[int], int, int]:
        """Drop complete trailing vision spans until all retained spans fit.

        Text is deliberately left in place here; the normal left-truncation
        step below chooses the most recent text with the remaining budget.  The
        important invariant is that truncation can never bisect a Qwen-VL
        ``vision_start ... vision_end`` block.
        """
        max_len = max(0, int(max_len))
        spans = self._find_vision_spans(token_ids)
        retained = len(spans)
        span_token_count = sum(end - start for start, end in spans)
        if span_token_count <= max_len:
            return token_ids, retained, 0

        keep = [True] * len(token_ids)
        dropped = 0
        for start, end in reversed(spans):
            if span_token_count <= max_len:
                break
            for pos in range(start, end):
                keep[pos] = False
            span_token_count -= end - start
            retained -= 1
            dropped += 1

        return [token for token, should_keep in zip(token_ids, keep) if should_keep], retained, dropped

    def _truncate_token_list_preserving_vision(self, token_ids: List[int], max_len: int) -> List[int]:
        """Left-truncate token lists without cutting Qwen-VL vision spans."""
        max_len = int(max_len)
        if len(token_ids) <= max_len:
            return token_ids

        token_ids, _, _ = self._drop_trailing_vision_spans_to_fit(token_ids, max_len)
        if len(token_ids) <= max_len:
            return token_ids

        spans = self._find_vision_spans(token_ids)
        span_token_count = sum(end - start for start, end in spans)
        if not spans:
            return token_ids[-max_len:]

        keep = [False] * len(token_ids)
        for start, end in spans:
            for pos in range(start, end):
                keep[pos] = True

        tail_budget = max_len - span_token_count
        if tail_budget <= 0:
            return [
                token
                for pos, token in enumerate(token_ids)
                if keep[pos]
            ]
        for pos in range(len(token_ids) - 1, -1, -1):
            if keep[pos]:
                continue
            keep[pos] = True
            tail_budget -= 1
            if tail_budget == 0:
                break

        kept = [token for token, should_keep in zip(token_ids, keep) if should_keep]
        return kept[-max_len:]

    def _update_raw_prompt_ids(
        self,
        raw_prompt_ids,
        cur_responses: torch.Tensor,
        tool_responses_ids: torch.Tensor,
        *,
        max_len: int,
    ):
        updated = []
        for row_idx, raw_ids in enumerate(raw_prompt_ids):
            if isinstance(raw_ids, torch.Tensor):
                ids = raw_ids.detach().cpu().tolist()
            elif hasattr(raw_ids, "tolist"):
                ids = raw_ids.tolist()
            else:
                ids = list(raw_ids)

            response_ids = cur_responses[row_idx]
            response_ids = response_ids[response_ids != self.tokenizer.pad_token_id].detach().cpu().tolist()
            tool_ids = tool_responses_ids[row_idx]
            tool_ids = tool_ids[tool_ids != self.tokenizer.pad_token_id].detach().cpu().tolist()

            combined = [int(token) for token in ids + response_ids + tool_ids]
            updated.append(self._truncate_token_list_preserving_vision(combined, max_len))
        return updated

    def _truncate_preserving_vision_tokens(self, input_ids: torch.Tensor, max_len: int) -> torch.Tensor:
        """Left-truncate prompts without cutting Qwen-VL vision spans."""
        max_len = int(max_len)
        if max_len >= input_ids.shape[1]:
            return input_ids

        output = torch.full(
            (input_ids.shape[0], max_len),
            self.tokenizer.pad_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )

        for row_idx, row in enumerate(input_ids):
            valid = row[row != self.tokenizer.pad_token_id]
            if valid.numel() == 0:
                continue
            if valid.numel() <= max_len:
                output[row_idx, -valid.numel():] = valid
                continue

            token_ids = valid.detach().cpu().tolist()
            kept_ids = self._truncate_token_list_preserving_vision(token_ids, max_len)
            if kept_ids:
                kept = torch.tensor(kept_ids, dtype=valid.dtype, device=valid.device)
                output[row_idx, -kept.numel():] = kept

        return output

    def _env_enabled(self, name: str, default: str = "false") -> bool:
        return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}

    def _drop_vision_spans_for_generation(self, active_batch: DataProto, step: int) -> DataProto:
        """Remove image spans before vLLM tool-turn generation.

        vLLM 0.7.x can misalign Qwen2.5-VL image features and placeholder
        tokens after the tool loop expands/repeats multimodal prompts. The tool
        response already carries textual evidence, so later turns can generate
        from text-only context while the original trajectory remains unchanged.
        """
        if step == 0 or not self._env_enabled("DROP_MM_IN_VLLM_TOOL_TURNS"):
            return active_batch
        if not self._has_multimodal_data(active_batch.non_tensor_batch):
            return active_batch

        input_ids = active_batch.batch["input_ids"]
        output = torch.full_like(input_ids, self.tokenizer.pad_token_id)
        dropped = 0

        for row_idx, row in enumerate(input_ids):
            valid = row[row != self.tokenizer.pad_token_id]
            if valid.numel() == 0:
                continue

            token_ids = valid.detach().cpu().tolist()
            spans = self._find_vision_spans(token_ids)
            if spans:
                keep = torch.ones(valid.shape[0], dtype=torch.bool, device=valid.device)
                for start, end in spans:
                    keep[start:end] = False
                valid = valid[keep]
                dropped += len(spans)

            if valid.numel() > input_ids.shape[1]:
                valid = valid[-input_ids.shape[1]:]
            if valid.numel() > 0:
                output[row_idx, -valid.numel():] = valid

        if dropped == 0:
            return active_batch

        attention_mask = self.tensor_fn.create_attention_mask(output)
        position_ids = self.tensor_fn.create_position_ids(attention_mask)
        if active_batch.batch["position_ids"].dim() == 3:
            position_ids = position_ids.unsqueeze(1).expand(-1, 3, -1)

        non_tensors = dict(active_batch.non_tensor_batch or {})
        non_tensors.pop("multi_modal_data", None)
        non_tensors.pop("multi_modal_inputs", None)
        non_tensors.pop("raw_prompt_ids", None)

        print(
            f"ROLLOUT_DROP_MM_FOR_VLLM: step={step} rows={input_ids.shape[0]} spans={dropped}",
            flush=True,
        )
        return DataProto.from_dict(
            {
                "input_ids": output,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
            },
            non_tensors=non_tensors,
        )

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          tool_responses_ids: torch.Tensor) -> Dict:
        """Update right side state."""
        responses = self.tensor_fn.concatenate_with_padding([
            right_side['responses'],
            cur_responses,
            tool_responses_ids
        ], pad_to_left=False)
        
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len]}


    def _flush_deferred_env_tools(self, envs: List[Any] = None) -> None:
        seen = set()
        for env in envs or []:
            if env is not None and id(env) not in seen:
                seen.add(id(env))
                flush_deferred_tools = getattr(env, "flush_deferred_tools", None)
                if callable(flush_deferred_tools):
                    flush_deferred_tools()
        from agent.tool.tools.graphr1_base_tool import GraphR1BaseTool
        GraphR1BaseTool.flush_deferred_io()

    @staticmethod
    def _tool_result_succeeded(result: Any) -> bool:
        if isinstance(result, dict):
            return bool(result.get("success", False))
        if not isinstance(result, str):
            return False
        try:
            payload = json.loads(result)
        except (TypeError, ValueError):
            return False
        return isinstance(payload, dict) and bool(payload.get("success", False))

    @staticmethod
    def _kb_search_returned_evidence(result: Any) -> bool:
        if not isinstance(result, str) or not result.strip():
            return False
        try:
            payload = json.loads(result)
        except (TypeError, ValueError):
            return False
        if not isinstance(payload, dict):
            return False
        results = payload.get("results")
        return isinstance(results, list) and bool(results)

    @classmethod
    def _trajectory_tool_metrics(cls, env: Any) -> Dict[str, int]:
        history = list(getattr(env, "tool_history", []) or [])
        successful_edit_indices = [
            index
            for index, call in enumerate(history)
            if call.get("tool") in {"insert", "update", "delete"}
            and cls._tool_result_succeeded(call.get("result"))
        ]
        verified_edits = sum(
            any(
                later.get("tool") == "kb_search"
                and cls._kb_search_returned_evidence(later.get("result"))
                for later in history[index + 1 :]
            )
            for index in successful_edit_indices
        )
        return {
            "duplicate_search_count": int(getattr(env, "duplicate_search_count", 0)),
            "successful_graph_edit_count": len(successful_edit_indices),
            "verified_graph_edit_count": int(verified_edits),
            "websearch_count": sum(call.get("tool") == "websearch" for call in history),
        }


    def _select_active_rollings(self, rollings: DataProto, active_mask: torch.Tensor) -> DataProto:
        return DataProto.from_dict(
            {k: v[active_mask] for k, v in rollings.batch.items()},
            non_tensors=select_non_tensors(rollings.non_tensor_batch, active_mask),
        )


    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        max_turn_response_length = getattr(self.config, "max_turn_response_length", None)
        if max_turn_response_length is not None:
            active_batch.meta_info["response_length"] = int(max_turn_response_length)

        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)
            
        padded_active_batch = DataProto.from_dict(
            padded_batch,
            non_tensors=pad_non_tensors(active_batch.non_tensor_batch, padding_size),
        )
        
        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        
        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
        if getattr(padded_output, 'non_tensor_batch', None):
            padded_output.non_tensor_batch = {
                k: v[:-padding_size] for k, v in padded_output.non_tensor_batch.items()
            }
            
        padded_output.batch = trimmed_batch
        return padded_output
    
    def run_llm_loop(self, gen_batch, envs: List[Any] = None,
                    initial_input_ids: torch.Tensor = None) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        
        initial_position_ids = gen_batch.batch.get('position_ids')
        if initial_position_ids is not None and initial_position_ids.dim() == 3:
            initial_position_ids = initial_position_ids[..., -self.config.max_start_length:]
        elif initial_position_ids is not None:
            initial_position_ids = initial_position_ids[:, -self.config.max_start_length:]

        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        if initial_position_ids is not None:
            original_left_side['position_ids'] = initial_position_ids
        original_right_side = {'responses': initial_input_ids[:, []]}
        
        batch_size = gen_batch.batch['input_ids'].shape[0]
        
        active_mask = torch.ones(batch_size, dtype=torch.bool)
        turns = torch.zeros(batch_size, dtype=torch.int32)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch
        trajectory_states = self._init_trajectory_states(gen_batch)
        rollings = self._sync_rollings_from_states(rollings, trajectory_states)

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            print(f"ROLLOUT_TURN_START: step={step} active={active_mask.sum().item()}", flush=True)
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            rollings_active = self._select_active_rollings(rollings, active_mask)
            if self._should_force_first_image_search(step, rollings):
                responses_ids, responses_str, new_active_masks = self._build_forced_image_search_batch(active_mask)
                meta_info = dict(getattr(rollings_active, "meta_info", {}) or {})
                print(f"ROLLOUT_TURN_FORCED_IMAGE_SEARCH: step={step} active={active_mask.sum().item()}", flush=True)
            else:
                rollings_active = self._drop_vision_spans_for_generation(rollings_active, step)
                gen_output = self._generate_with_gpu_padding(rollings_active)
                print(f"ROLLOUT_TURN_GENERATED: step={step} active={active_mask.sum().item()}", flush=True)

                meta_info = gen_output.meta_info
                responses_ids, responses_str, new_active_masks = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str, new_active_masks, forced_verifications = (
                self._apply_forced_graph_edit_verification(
                    responses_ids,
                    responses_str,
                    new_active_masks,
                    active_mask,
                    envs,
                )
            )
            if forced_verifications:
                print(
                    f"ROLLOUT_TURN_FORCED_GRAPH_EDIT_VERIFICATION: "
                    f"step={step} forced={forced_verifications} "
                    f"active={active_mask.sum().item()}",
                    flush=True,
                )
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            active_mask[active_mask.clone()] = new_active_masks

            turns[active_mask] += 1

            if self.config.use_batch_tool_calls:
                # Use batch execution for tool calls
                tool_responses = self._execute_tool_calls_batch(responses_str, envs, active_mask)
            else:
                # Use sequential execution for tool calls
                tool_responses = self._execute_tool_calls(responses_str, envs, active_mask)

            active_num_list.append(active_mask.sum().item())
            tool_responses, tool_response_images = self._prepare_tool_responses_multimodal(
                tool_responses,
                envs=envs,
            )
            tool_responses_ids = self._process_tool_responses(tool_responses)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                tool_responses_ids,
                tool_response_images=tool_response_images,
                trajectory_states=trajectory_states,
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                tool_responses_ids
            )
        self._flush_deferred_env_tools(envs)
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        original_right_side['turns'] = turns

        tool_metrics = [self._trajectory_tool_metrics(env) for env in (envs or [])]
        if len(tool_metrics) == batch_size:
            for metric_name in (
                "duplicate_search_count",
                "successful_graph_edit_count",
                "verified_graph_edit_count",
                "websearch_count",
            ):
                original_right_side[metric_name] = torch.tensor(
                    [metrics[metric_name] for metrics in tool_metrics],
                    dtype=torch.int32,
                )

        # Save trajectory and return final output
        output_non_tensors = {}
        if not self.is_validation:
            output_non_tensors.update(self._states_to_non_tensors(trajectory_states))
            output_non_tensors.pop("raw_prompt_ids", None)
        return self._compose_final_output(
            original_left_side,
            original_right_side,
            meta_info,
            non_tensors=output_non_tensors,
        )


    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict,
                            non_tensors: Dict = None) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        prompts = left_side['input_ids']
        responses = right_side['responses']
        prompts, responses, non_tensors = self._expand_final_vision_tokens(
            prompts,
            responses,
            non_tensors,
        )
        final_output['prompts'] = prompts
        final_output['responses'] = responses

        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            prompts,
            responses
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(prompts),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        
        position_ids = self.tensor_fn.create_position_ids(final_output['attention_mask'])
        left_position_ids = left_side.get('position_ids')
        if left_position_ids is not None and left_position_ids.dim() == 3:
            prompt_attention_mask = self.tensor_fn.create_attention_mask(prompts)
            left_position_ids = self.tensor_fn.create_position_ids(prompt_attention_mask)
            left_position_ids = left_position_ids.unsqueeze(1).expand(-1, 3, -1)
            prompt_len = prompts.shape[1]
            response_position_ids = position_ids[:, prompt_len:].unsqueeze(1).expand(-1, 3, -1)
            position_ids = torch.cat([left_position_ids, response_position_ids], dim=-1)
        final_output['position_ids'] = position_ids

        if non_tensors and "multi_modal_data" in non_tensors:
            non_tensors = self._align_final_multimodal_data(
                final_output["input_ids"],
                non_tensors,
            )
        
        final_output = DataProto.from_dict(final_output, non_tensors=non_tensors)
        final_output.meta_info.update(meta_info)

        return final_output
