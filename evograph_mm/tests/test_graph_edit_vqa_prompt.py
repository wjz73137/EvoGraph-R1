from evograph_mm.kb.vqa_prompt import build_graph_edit_user_prompt


def test_graph_edit_prompt_removes_static_search_only_constraint():
    prompt = build_graph_edit_user_prompt(question="How tall is this lighthouse?")

    assert "call websearch once" in prompt
    assert "insert a missing atomic fact" in prompt
    assert "call kb_search again" in prompt
    assert "Never repeat an identical tool call" in prompt
    assert "Use only kb_search" not in prompt
    assert "How tall is this lighthouse?" in prompt
