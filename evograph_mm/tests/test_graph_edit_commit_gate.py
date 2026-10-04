import json
from types import SimpleNamespace

from agent.tool import tool_env
from agent.tool.tools.mm import commit_gate
from agent.tool.tools.mm.commit_gate import GraphEditGateDecision
from agent.tool.tools.mm.edit_tools import MMGraphR1InsertTool


def allow(reason="supported"):
    return GraphEditGateDecision(
        allowed=True,
        decision="PERMANENT",
        validity="support",
        conflict_type="none",
        confidence=0.98,
        reason=reason,
        model="judge",
    )


def reject(reason="wrong namesake"):
    return GraphEditGateDecision(
        allowed=False,
        decision="REJECT",
        validity="refute",
        conflict_type="entity_identity_mismatch",
        confidence=0.99,
        reason=reason,
        model="judge",
    )


def test_edit_context_carries_question_and_prior_evidence_without_exposing_it():
    env = SimpleNamespace(
        tool_context={"question": "How many stars?", "mm_search_api_url": "http://x/search"},
        tool_history=[
            {"tool": "kb_search", "args": {"query": "<img>"}, "result": '{"entity":"Hotel"}'},
            {"tool": "websearch", "args": {"query": "Hotel stars"}, "result": "five stars"},
        ],
    )

    enriched = tool_env._inject_tool_context(env, "insert", {"content": "Hotel is five-star."})

    assert enriched["__question"] == "How many stars?"
    assert enriched["__mm_api_url"] == "http://x/search"
    assert enriched["__trajectory_history"] == env.tool_history
    assert tool_env._public_tool_args(enriched) == {"content": "Hotel is five-star."}


def test_graph_edit_websearch_query_is_grounded_in_anchor_and_kb_identity():
    env = SimpleNamespace(
        tool_context={
            "question": "How many stars does this hotel have?",
            "data_source": "paper_grpo_graph_edit_controlled_missing",
            "dataset": "E-VQA",
        },
        tool_history=[
            {
                "tool": "kb_search",
                "args": {"query": "<img>"},
                "result": json.dumps({"results": [{"entity": "Hotel Bohema"}]}),
            },
            {
                "tool": "kb_search",
                "args": {"query": "Hotel Bohema stars"},
                "result": json.dumps(
                    {
                        "results": [
                            {"<knowledge>": '"The Hotel Bohema is located in downtown Bydgoszcz, Poland."'},
                            {"<knowledge>": '"The Hotel Bohema is located at Konarskiego Street No. 9."'},
                        ]
                    }
                ),
            },
        ],
    )

    enriched = tool_env._inject_tool_context(
        env,
        "websearch",
        {"query": "Hotel Bohema <star rating>"},
    )

    assert enriched["query"].startswith("Hotel Bohema")
    assert "Bydgoszcz" in enriched["query"]
    assert "Konarskiego Street No. 9" in enriched["query"]
    assert "How many stars" in enriched["query"]
    assert "<" not in enriched["query"]
    assert enriched["__dataset"] == "E-VQA"


def test_commit_gate_requires_all_permanent_support_conditions(monkeypatch):
    monkeypatch.setenv("GRAPH_EDIT_GATE_MIN_CONFIDENCE", "0.85")
    accepted = commit_gate._normalize_decision(
        {
            "decision": "PERMANENT",
            "validity": "support",
            "conflict_type": "none",
            "confidence": 0.94,
            "reason": "same entity and explicit evidence",
        },
        model="judge",
    )
    low_confidence = commit_gate._normalize_decision(
        {
            "decision": "PERMANENT",
            "validity": "support",
            "conflict_type": "none",
            "confidence": 0.4,
            "reason": "uncertain",
        },
        model="judge",
    )
    mismatch = commit_gate._normalize_decision(
        {
            "decision": "PERMANENT",
            "validity": "support",
            "conflict_type": "entity_identity_mismatch",
            "confidence": 0.99,
            "reason": "different city",
        },
        model="judge",
    )

    assert accepted.allowed is True
    assert low_confidence.allowed is False
    assert low_confidence.decision == "TENTATIVE"
    assert mismatch.allowed is False


def test_single_insert_rejection_never_posts(monkeypatch):
    tool = MMGraphR1InsertTool()
    monkeypatch.setattr(
        "agent.tool.tools.mm.edit_tools.validate_graph_edit_commit",
        lambda operation, args: reject(),
    )
    monkeypatch.setattr(tool, "_post_json", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))

    result = json.loads(tool.execute({"content": "wrong fact"}))

    assert result["success"] is False
    assert result["precommit_gate"]["conflict_type"] == "entity_identity_mismatch"


def test_batch_gate_filters_rejected_item_and_preserves_positions(monkeypatch):
    tool = MMGraphR1InsertTool()
    monkeypatch.setattr(
        "agent.tool.tools.mm.edit_tools.validate_graph_edit_commit",
        lambda operation, args: reject() if args["content"] == "wrong" else allow(),
    )
    posted = []

    def fake_post(path, payload, **kwargs):
        posted.append((path, payload, kwargs))
        return {"results": [{"success": True, "message": "inserted"}]}

    monkeypatch.setattr(tool, "_post_json", fake_post)
    results = [
        json.loads(item)
        for item in tool.batch_execute(
            [
                {"content": "wrong", "__mm_api_url": "http://x/search"},
                {"content": "right", "__mm_api_url": "http://x/search"},
            ]
        )
    ]

    assert results[0]["success"] is False
    assert results[0]["precommit_gate"]["decision"] == "REJECT"
    assert results[1]["success"] is True
    assert posted[0][1] == {"items": [{"content": "right"}]}
    assert posted[0][2]["api_base_url"] == "http://x/search"


def test_deferred_rejection_is_not_queued(monkeypatch):
    tool = MMGraphR1InsertTool()
    monkeypatch.setattr(
        "agent.tool.tools.mm.edit_tools.validate_graph_edit_commit",
        lambda operation, args: reject("not grounded"),
    )

    result = json.loads(tool.time_batch_submit({"content": "wrong"}, 1, 2))

    assert result["success"] is False
    assert tool.get_time_batch_stats()["pending"] == 0


def test_evidence_packet_excludes_hidden_transport_fields():
    packet = commit_gate._build_evidence_packet(
        "insert",
        {
            "content": "Hotel Bohema is a five-star hotel.",
            "__question": "How many stars?",
            "__mm_api_url": "http://secret/search",
            "__trajectory_history": [
                {
                    "tool": "websearch",
                    "args": {"query": "Hotel Bohema Bydgoszcz stars", "__dataset": "E-VQA"},
                    "result": "Hotel Bohema in Bydgoszcz is five-star.",
                }
            ],
        },
    )

    assert packet["candidate_edit"] == {"content": "Hotel Bohema is a five-star hotel."}
    assert packet["prior_tool_evidence_in_chronological_order"][0]["args"] == {
        "query": "Hotel Bohema Bydgoszcz stars"
    }
    assert "secret" not in json.dumps(packet)
