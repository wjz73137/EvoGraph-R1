from verl.utils.reward_score.qa_em_and_format import (
    compute_score_format,
    compute_score_format_answer,
)


def _assistant(body):
    return f"<|im_start|>assistant\n{body}<|im_end|>"


def test_complete_visual_text_answer_protocol_gets_full_reward():
    solution = "".join(
        [
            _assistant(
                '<think>ground image</think>\n'
                '<tool_call>{"tool":"kb_search","args":{"query":"<img>"}}</tool_call>'
            ),
            _assistant(
                '<think>lookup fact</think>\n'
                '<tool_call>{"tool":"kb_search","args":{"query":"Mitla town type"}}</tool_call>'
            ),
            _assistant('<think>use evidence</think>\n<answer>modern</answer>'),
        ]
    )

    assert compute_score_format(solution) == 1.0
    assert compute_score_format_answer(solution, ["modern"]) == 1.0


def test_visual_only_shortcut_is_not_full_format_reward():
    solution = "".join(
        [
            _assistant(
                '<think>ground image</think>\n'
                '<tool_call>{"tool":"kb_search","args":{"query":"<img>"}}</tool_call>'
            ),
            _assistant('<think>guess</think>\n<answer>modern</answer>'),
        ]
    )

    assert compute_score_format(solution) == 0.75
    assert compute_score_format_answer(solution, ["modern"]) == -0.25


def test_wrong_or_repeated_tool_query_cannot_complete_protocol():
    solution = "".join(
        [
            _assistant(
                '<think>ground image</think>\n'
                '<tool_call>{"tool":"kb_search","args":{"query":"<img>"}}</tool_call>'
            ),
            _assistant(
                '<think>repeat image</think>\n'
                '<tool_call>{"tool":"kb_search","args":{"query":"<img>"}}</tool_call>'
            ),
            _assistant('<think>guess</think>\n<answer>modern</answer>'),
        ]
    )

    assert compute_score_format(solution) == 0.75
