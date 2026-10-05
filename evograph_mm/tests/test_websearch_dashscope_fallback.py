from types import SimpleNamespace

import openai

from agent.tool.tools.websearch_tool import WebSearchTool


def test_uses_dashscope_search_when_jina_key_is_absent(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="Evidence: 75 feet")
                    )
                ]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setenv("WEBSEARCH_CACHE_ENABLED", "false")
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("WEBSEARCH_MODEL", "test-search-model")
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)

    result = WebSearchTool()._jina_search("Ocracoke Light height")

    assert result == "Evidence: 75 feet"
    assert captured["client"]["base_url"] == "https://example.test/v1"
    assert captured["model"] == "test-search-model"
    assert captured["extra_body"]["enable_search"] is True
    assert captured["extra_body"]["search_options"]["forced_search"] is True

    FakeCompletions.create = lambda self, **kwargs: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content='{"command":"search","query":"Entity fact"}'
        ))]
    )
    result = WebSearchTool()._dashscope_search('Entity fact')
    assert 'not retrieved evidence' in result


def test_cached_search_commands_are_not_factual_evidence(monkeypatch):
    monkeypatch.setenv('WEBSEARCH_WIKIPEDIA_AUGMENT', 'false')
    tool = WebSearchTool()
    tool.cache_enabled = True
    monkeypatch.setattr(tool, '_search_with_cache', lambda *args: '{"command":"search","query":"Entity fact"}')
    assert 'not retrieved evidence' in tool.execute({'query': 'Entity fact'})


def test_optionally_augments_websearch_with_wikipedia(monkeypatch):
    tool = WebSearchTool()
    monkeypatch.setenv("WEBSEARCH_WIKIPEDIA_AUGMENT", "true")
    monkeypatch.setattr(
        tool,
        "_wikipedia_search",
        lambda query: f"Wikipedia evidence [Entity]: evidence for {query}",
    )

    result = tool._augment_with_wikipedia("Entity location", "Primary evidence")

    assert result == (
        "Wikipedia evidence [Entity]: evidence for Entity location\n\n"
        "Primary evidence"
    )
