import json
from types import SimpleNamespace

from agent.llm_agent.generation import ToolGenerationManager
from agent.tool.tool_base import Tool
from agent.tool.tool_env import ToolEnv, _inject_tool_context, _normalize_tool_call, step
from agent.tool.tools.mm import MMGraphR1InsertTool
from verl.trainer.main_ppo import compute_graph_edit_shaping


class RecordingTool(Tool):
    def __init__(self, name):
        super().__init__(
            name=name,
            description=name,
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
        self.calls = []

    def execute(self, args):
        self.calls.append(dict(args))
        return json.dumps({"results": [args["query"]]})


class EditTool(Tool):
    def __init__(self):
        super().__init__(
            name="insert",
            description="insert",
            parameters={
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": ["content"],
            },
        )

    def execute(self, args):
        return json.dumps({"success": True, "content": args["content"]})


def call(tool, args):
    return (
        "<think>act</think>\n"
        f"<tool_call>{json.dumps({'tool': tool, 'args': args})}</tool_call>"
    )


def test_search_query_alias_is_normalized_but_log_text_is_not_inferred():
    search = RecordingTool("websearch")
    env = ToolEnv([search], max_turns=2)

    alias_result = step(env, call("websearch", {"search_query": "entity height"}))
    assert alias_result[3]["action_is_effective"] is True
    assert search.calls == [{"search_query": "entity height", "query": "entity height"}]

    malformed = (
        '<think>search</think><tool_call>{"tool":"websearch","args":{},'
        '"log":"query is entity location"}</tool_call>'
    )
    invalid_result = step(ToolEnv([RecordingTool("websearch")], max_turns=1), malformed)
    assert invalid_result[3]["action_is_effective"] is False
    assert "Missing required parameter: query" in invalid_result[0]


def test_update_with_only_new_fact_is_normalized_to_insert():
    normalized = _normalize_tool_call(
        "update",
        {"content": "<Hotel Bohema> is a <five-star hotel>."},
    )

    assert normalized == {
        "tool": "insert",
        "args": {"content": "Hotel Bohema is a five-star hotel."},
    }


def test_private_trajectory_route_reaches_search_and_edit_tools():
    env = SimpleNamespace(
        tool_context={"mm_search_api_url": "http://127.0.0.1:8012/search"}
    )
    search_args = _inject_tool_context(env, "kb_search", {"query": "entity fact"})
    edit_args = _inject_tool_context(env, "insert", {"content": "Entity has fact."})

    assert search_args["__mm_api_url"] == "http://127.0.0.1:8012/search"
    assert edit_args["__mm_api_url"] == "http://127.0.0.1:8012/search"
    assert MMGraphR1InsertTool()._endpoint_url(
        "/insert", api_base_url=edit_args["__mm_api_url"]
    ) == "http://127.0.0.1:8012/insert"


def test_duplicate_search_is_blocked_until_graph_edit_changes_state():
    search = RecordingTool("websearch")
    env = ToolEnv([search, EditTool()], max_turns=6)

    first = step(env, call("websearch", {"query": "entity height"}))
    duplicate = step(env, call("websearch", {"query": "entity height"}))

    assert len(search.calls) == 1
    assert first[3]["action_is_effective"] is True
    assert duplicate[3]["duplicate_search"] is True
    assert duplicate[1] == -0.1
    assert env.duplicate_search_count == 1

    edit = step(env, call("insert", {"content": "Entity is 75 feet tall."}))
    after_edit = step(env, call("websearch", {"query": "entity height"}))

    assert edit[3]["action_is_effective"] is True
    assert after_edit[3]["action_is_effective"] is True
    assert len(search.calls) == 2


def test_trajectory_metrics_require_search_after_successful_edit():
    env = SimpleNamespace(
        duplicate_search_count=2,
        tool_history=[
            {"tool": "websearch", "result": "evidence"},
            {"tool": "insert", "result": '{"success": true}'},
            {"tool": "kb_search", "result": '{"results": ["fact"]}'},
        ],
    )
    metrics = ToolGenerationManager._trajectory_tool_metrics(env)
    assert metrics == {
        "duplicate_search_count": 2,
        "successful_graph_edit_count": 1,
        "verified_graph_edit_count": 1,
        "websearch_count": 1,
    }


def test_empty_search_does_not_count_as_verified_edit():
    env = SimpleNamespace(
        duplicate_search_count=0,
        tool_history=[
            {"tool": "insert", "result": '{"success": true}'},
            {"tool": "kb_search", "result": '{"results": []}'},
        ],
    )
    assert ToolGenerationManager._trajectory_tool_metrics(env)[
        "verified_graph_edit_count"
    ] == 0


def test_graph_edit_shaping_rewards_only_correct_verified_edit():
    assert compute_graph_edit_shaping(
        "E-VQA/paper_grpo_graph_edit_v1",
        answer_em_score=1.0,
        duplicate_search_count=0,
        successful_graph_edit_count=1,
        verified_graph_edit_count=1,
    ) == 0.15
    assert compute_graph_edit_shaping(
        "E-VQA/paper_grpo_graph_edit_v1",
        answer_em_score=0.0,
        duplicate_search_count=2,
        successful_graph_edit_count=1,
        verified_graph_edit_count=0,
    ) == -0.25
    assert compute_graph_edit_shaping(
        "E-VQA/paper_grpo_graph_covered_full_v1",
        answer_em_score=1.0,
        duplicate_search_count=4,
        successful_graph_edit_count=1,
        verified_graph_edit_count=1,
    ) == 0.0
