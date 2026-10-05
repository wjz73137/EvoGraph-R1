from types import SimpleNamespace
import json

import torch

from agent.llm_agent.generation import ToolGenerationManager
from agent.llm_agent.tensor_helper import TensorConfig, TensorHelper


def manager():
    instance = ToolGenerationManager.__new__(ToolGenerationManager)
    instance.tokenizer = SimpleNamespace(eos_token="<eos>")
    instance.config = SimpleNamespace(
        tool_call_start="<tool_call>",
        tool_call_end="</tool_call>",
        tool_response_start="<knowledge>",
        tool_response_end="</knowledge>",
    )
    return instance


def response_manager(response_limit=8):
    instance = manager()
    instance.tokenizer = SimpleNamespace(pad_token_id=0)
    instance.config.max_prompt_length = 3
    instance.config.max_response_length = response_limit
    instance.tensor_fn = TensorHelper(TensorConfig(0, 3, 4, 3))
    instance._vision_token_ids = lambda: (90, 91, 92)
    return instance


def test_trajectory_uses_response_budget_not_prompt_budget():
    instance = response_manager()
    result = instance._update_right_side(
        {'responses': torch.tensor([[1, 2, 3, 0]])},
        torch.tensor([[4, 5]]), torch.tensor([[6, 0]]),
    )
    assert result['responses'].tolist() == [[1, 2, 3, 4, 5, 6]]


def test_overflow_retains_final_answer_and_complete_vision_span():
    instance = response_manager(response_limit=6)
    result = instance._update_right_side(
        {'responses': torch.tensor([[90, 91, 92, 1, 2, 3]])},
        torch.tensor([[4, 5]]), torch.tensor([[6]]),
    )
    assert result['responses'].tolist() == [[90, 91, 92, 4, 5, 6]]


def test_response_truncation_preserves_right_padding_for_short_rows():
    instance = response_manager(response_limit=4)
    result = instance._update_right_side(
        {'responses': torch.tensor([[1, 2, 3], [1, 0, 0]])},
        torch.tensor([[4, 5], [2, 0]]), torch.tensor([[6], [0]]),
    )
    assert result['responses'].tolist() == [[3, 4, 5, 6], [1, 2, 0, 0]]


def test_repairs_complete_json_with_wrong_closing_tag():
    value = (
        '<think>lookup</think>\n<tool_call>{"tool":"kb_search",'
        '"args":{"query":"castle date"}}</knowledge>'
    )
    responses, active = manager()._process_tool_call([value])
    assert active == [True]
    assert responses[0].endswith(
        '<tool_call>{"tool":"kb_search","args":{"query":"castle date"}}</tool_call><eos>'
    )


def test_does_not_repair_incomplete_json():
    value = '<think>lookup</think>\n<tool_call>{"tool":"kb_search","args":{"query":"castle'
    responses, active = manager()._process_tool_call([value])
    assert active == [False]
    assert responses[0] == value + "<eos>"


def test_repairs_missing_json_braces_after_complete_query_string():
    value = (
        '<think>search</think>\n<tool_call>{"tool":"websearch",'
        '"args":{"query":"Hotel Bohema Bydgoszcz star rating">'
    )
    responses, active = manager()._process_tool_call([value])

    assert active == [True]
    assert responses[0].endswith(
        '<tool_call>{"tool":"websearch","args":'
        '{"query":"Hotel Bohema Bydgoszcz star rating"}}</tool_call><eos>'
    )


def test_does_not_invent_an_unfinished_query_value():
    value = '<think>search</think>\n<tool_call>{"tool":"websearch","args":{"query":"Hotel'
    responses, active = manager()._process_tool_call([value])

    assert active == [False]
    assert responses[0] == value + "<eos>"


def test_appends_phase_specific_guidance_inside_knowledge_block():
    instance = manager()
    image = instance._append_next_action_guidance(
        '<knowledge>{"results":[]}</knowledge>',
        image_response=True,
        anchor_entity="Hotel Bohema",
    )
    text = instance._append_next_action_guidance(
        '<knowledge>{"results":[]}</knowledge>',
        image_response=False,
        env=SimpleNamespace(
            tool_history=[
                {
                    "tool": "kb_search",
                    "args": {"query": "Hotel Bohema stars"},
                    "result": '{"results": []}',
                }
            ]
        ),
    )
    assert '"tool":"kb_search"' in image
    assert "Hotel Bohema" in image
    assert "Do not substitute" in image
    assert "Do not answer" in image
    assert "exact fact" in text
    assert 'tool websearch' in text
    assert image.endswith("</knowledge>")
    assert text.endswith("</knowledge>")


def test_guidance_requires_insert_after_websearch_and_verification_after_edit():
    instance = manager()
    after_web = instance._append_next_action_guidance(
        "<knowledge>reliable evidence</knowledge>",
        image_response=False,
        env=SimpleNamespace(
            tool_history=[{"tool": "websearch", "args": {"query": "entity fact"}, "result": "evidence"}]
        ),
    )
    after_edit = instance._append_next_action_guidance(
        '<pipeline>{"success":true}</pipeline>',
        image_response=False,
        env=SimpleNamespace(
            tool_history=[{"tool": "insert", "args": {"content": "fact"}, "result": '{"success":true}'}]
        ),
    )
    assert 'tool insert' in after_web
    assert "Do not answer before" in after_web
    assert 'tool kb_search' in after_edit
    assert "verification" in after_edit


def test_visual_guidance_uses_real_question_without_copyable_placeholders():
    text = manager()._append_next_action_guidance(
        '<knowledge>{"results": []}</knowledge>', image_response=True,
        anchor_entity='Hotel Bohema',
        env=SimpleNamespace(tool_context={'question': 'How many stars does this hotel have?'}, tool_history=[]),
    )
    assert 'Hotel Bohema How many stars does this hotel have?' in text
    assert '<fact asked' not in text


def test_gate_rejection_requests_new_evidence_not_repeated_bad_edit():
    env = SimpleNamespace(tool_context={'question': 'How many stars?'}, tool_history=[
        {'tool': 'kb_search', 'args': {'query': '<img>'}, 'result': '{"results": [{"entity": "Hotel Bohema"}]}'},
        {'tool': 'kb_search', 'args': {'query': 'Hotel Bohema stars'},
         'result': '{"results": [{"knowledge": "Hotel Bohema is in Bydgoszcz, Poland."}]}'},
        {'tool': 'insert', 'args': {'content': 'Hotel Bohema has three stars.'}, 'result': '{"success": false}'},
    ])
    text = manager()._append_next_action_guidance(
        '<pipeline>Graph edit rejected by pre-commit evidence gate</pipeline>',
        image_response=False, env=env,
    )
    assert 'Do not repeat the rejected edit' in text
    assert 'Bydgoszcz, Poland' in text
    assert '"tool":"websearch"' in text


def test_successful_insert_builds_concrete_forced_verification_call():
    instance = manager()
    instance.config.force_graph_edit_verification = True
    env = SimpleNamespace(
        tool_history=[
            {
                "tool": "insert",
                "args": {"content": "Hotel Bohema is a five-star hotel."},
                "result": '{"success":true}',
            }
        ]
    )

    query = instance._graph_edit_verification_query(env)
    response = instance._forced_graph_edit_verification_response(query)
    payload = response.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0]

    assert query == "Hotel Bohema is a five-star hotel."
    assert json.loads(payload) == {
        "tool": "kb_search",
        "args": {"query": "Hotel Bohema is a five-star hotel."},
    }


def test_forced_verification_only_applies_immediately_after_successful_edit():
    instance = manager()
    instance.config.force_graph_edit_verification = True
    failed_edit = SimpleNamespace(
        tool_history=[
            {"tool": "insert", "args": {"content": "fact"}, "result": '{"success":false}'}
        ]
    )
    already_verified = SimpleNamespace(
        tool_history=[
            {"tool": "insert", "args": {"content": "fact"}, "result": '{"success":true}'},
            {"tool": "kb_search", "args": {"query": "fact"}, "result": '{"results":[]}'},
        ]
    )

    assert instance._graph_edit_verification_query(failed_edit) is None
    assert instance._graph_edit_verification_query(already_verified) is None


def test_forced_verification_replaces_only_the_matching_active_trajectory():
    instance = manager()
    instance.config.force_graph_edit_verification = True
    instance._batch_tokenize = lambda rows: rows
    edit_env = SimpleNamespace(
        tool_history=[
            {
                "tool": "insert",
                "args": {"content": "Hotel Bohema is a five-star hotel."},
                "result": '{"success":true}',
            }
        ]
    )
    ordinary_env = SimpleNamespace(tool_history=[])
    inactive_env = SimpleNamespace(tool_history=[])

    ids, responses, masks, count = instance._apply_forced_graph_edit_verification(
        responses_ids=["ordinary generated", "edit generated"],
        responses_str=["ordinary generated", "edit generated"],
        new_active_masks=torch.tensor([False, False]),
        active_mask=torch.tensor([True, False, True]),
        envs=[ordinary_env, inactive_env, edit_env],
    )

    assert count == 1
    assert responses[0] == "ordinary generated"
    assert "Hotel Bohema is a five-star hotel." in responses[1]
    assert masks.tolist() == [False, True]
    assert ids == responses


def test_middle_truncation_preserves_evidence_and_state_guidance():
    token_ids = list(range(100))
    truncated = ToolGenerationManager._middle_truncate_token_ids(
        token_ids,
        max_length=30,
        marker_ids=[1000, 1001],
    )

    assert len(truncated) == 30
    assert truncated[:3] == [0, 1, 2]
    assert [1000, 1001] == truncated[16:18]
    assert truncated[-3:] == [97, 98, 99]


def test_extracts_first_visual_entity_as_required_anchor():
    instance = manager()
    response = (
        '<knowledge>{"results":[{"entity":"Hotel Bohema"},'
        '{"entity":"Mandarin Oriental Hyde Park"}]}</knowledge>'
    )
    assert instance._first_result_entity(response) == "Hotel Bohema"
