import json

import pytest

from scripts import retire_local_extraction_graphs as retire
from scripts.run_evqa_api_graph import (
    COMPATIBLE_PREVIOUS_RUNNER, checked_api_output, compatible_owner_migration,
    index_artifacts_present, load_api_config,
)


def make_targets(root, model='local/Qwen2.5-VL-7B-Instruct'):
    for name in retire.NAMES:
        target = root / 'expr_mm' / name / 'E-VQA'
        target.mkdir(parents=True)
        (target / 'owner.json').write_text(json.dumps({
            'graph_llm': model, 'documents': [{'contents': 'Real source.'}]}))


def test_retirement_allowlist_only_accepts_local_graphs(tmp_path, monkeypatch):
    monkeypatch.setattr(retire, 'ROOT', tmp_path)
    make_targets(tmp_path)
    targets, owners = retire.validate_targets()
    assert len(targets) == len(owners) == 5
    owner = targets[0] / 'E-VQA/owner.json'
    owner.write_text(json.dumps({'graph_llm': 'api/qwen', 'documents': []}))
    with pytest.raises(RuntimeError, match='not an identified'):
        retire.validate_targets()


def test_retirement_refuses_symlinks_and_source_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(retire, 'ROOT', tmp_path)
    make_targets(tmp_path)
    owner = tmp_path / 'expr_mm' / retire.NAMES[0] / 'E-VQA/owner.json'
    owner.write_text(json.dumps({'graph_llm': 'local/Qwen2.5-VL-3B-Instruct',
                                'documents': [{'contents': 'Different source.'}]}))
    with pytest.raises(RuntimeError, match='source mismatch'):
        retire.validate_targets()
    last = tmp_path / 'expr_mm' / retire.NAMES[-1]
    moved = tmp_path / 'preserved'
    last.rename(moved)
    last.symlink_to(moved, target_is_directory=True)
    with pytest.raises(RuntimeError, match='invalid exact'):
        retire.validate_targets()


@pytest.mark.parametrize('record', [
    {'output': '', 'finish_reason': 'stop'},
    {'output': 'Partial output', 'finish_reason': 'length'},
])
def test_empty_or_truncated_api_outputs_are_never_accepted(record):
    with pytest.raises(RuntimeError, match='empty or truncated'):
        checked_api_output(record)


def test_api_config_comes_from_project_env_not_local_model(monkeypatch):
    import dotenv
    import os
    monkeypatch.setattr(dotenv, 'dotenv_values', lambda path: {
        'OPENAI_API_KEY': 'test-key-not-a-real-credential',
        'OPENAI_BASE_URL': 'https://example.invalid/v1',
        'GRAPH_LLM_MODEL': 'api-test-model', 'OPENAI_MODEL': 'other-model'})
    for name in ('OPENAI_API_KEY', 'OPENAI_BASE_URL', 'GRAPH_LLM_MODEL', 'OPENAI_MODEL'):
        monkeypatch.setenv(name, 'old-value')
    assert load_api_config()[2] == 'api-test-model'
    assert os.environ['GRAPH_LLM_MODEL'] == 'api-test-model'
    assert checked_api_output({'output': 'Original API text.', 'finish_reason': 'stop'}) == 'Original API text.'


def test_missing_api_credentials_stop_before_any_request(monkeypatch):
    import dotenv
    monkeypatch.setattr(dotenv, 'dotenv_values', lambda path: {'GRAPH_LLM_MODEL': 'api-model'})
    with pytest.raises(RuntimeError, match='lacks API'):
        load_api_config()


def test_manifest_fix_migration_cannot_change_source_model_or_prompt():
    previous = {'runner_sha256': COMPATIBLE_PREVIOUS_RUNNER, 'source_sha256': 'source',
                'graph_llm': 'api/model', 'original_prompt_sha256': 'prompt'}
    current = {**previous, 'runner_sha256': 'new-code'}
    report = {'extraction_complete': True}
    assert compatible_owner_migration(previous, current, report)
    assert not compatible_owner_migration(previous, current, {})
    for key in ('source_sha256', 'graph_llm', 'original_prompt_sha256'):
        assert not compatible_owner_migration(previous, {**current, key: 'changed'}, report)


def test_index_reuse_requires_all_recorded_artifacts(tmp_path):
    artifact = tmp_path / 'index.faiss'
    assert not index_artifacts_present({'index_path': str(artifact)})
    artifact.write_bytes(b'fixture')
    assert index_artifacts_present({'index_path': str(artifact), 'vector_count': 4})
    assert not index_artifacts_present({'index_path': str(artifact), 'ids_path': str(tmp_path / 'missing.json')})
