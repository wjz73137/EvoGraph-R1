import json
import sys

import pytest

from scripts import prepare_evqa_graphedit_resume as resume


def fixture_run(tmp_path):
    source = tmp_path / 'failed'
    output = tmp_path / 'resume'
    actor = source / 'checkpoints/global_step_328/actor'
    actor.mkdir(parents=True)
    for rank in (0, 1):
        for kind in ('model', 'optim', 'extra_state'):
            (actor / f'{kind}_world_size_2_rank_{rank}.pt').write_bytes(b'checkpoint')
    (actor.parent / 'data.pt').write_bytes(b'loader state')
    results = source / 'expr_results/experiment'
    results.mkdir(parents=True)
    rows = [{'global_steps': step} for step in range(327, 331)]
    metrics = results / 'evals_training.jsonl'
    metrics.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    for row in rows:
        (results / f'train_trajectories_step{row["global_steps"]}.json').write_text('[]')
    graph = source / 'isolated_graphs/train_0'
    graph.mkdir(parents=True)
    for name in ('kv_store_entities.json', 'kv_store_hyperedges.json', 'graph_chunk_entity_relation.graphml'):
        (graph / name).write_text('{}')
    (graph / 'hyperedge_recent_mutations.json').write_text(json.dumps({'mutations': [
        {'timestamp': '2020-01-01T00:00:00'}]}))
    return source, output, metrics, graph


def invoke(monkeypatch, source, output):
    monkeypatch.setattr(sys, 'argv', ['prepare', '--source', str(source), '--output', str(output),
                                   '--experiment', 'experiment', '--step', '328'])
    monkeypatch.setattr(resume, 'copy_graphs', lambda *args: None)
    resume.main()


def test_resume_retains_only_saved_updates_without_changing_source(tmp_path, monkeypatch):
    source, output, metrics, _ = fixture_run(tmp_path)
    original = metrics.read_bytes()
    invoke(monkeypatch, source, output)
    assert metrics.read_bytes() == original
    retained = (output / 'expr_results/experiment/evals_training.jsonl').read_text().splitlines()
    assert [json.loads(line)['global_steps'] for line in retained] == [327, 328]
    assert not (output / 'expr_results/experiment/train_trajectories_step329.json').exists()
    report = json.loads((output / 'resume_manifest.json').read_text())
    assert report['discarded_optimizer_updates_to_replay'] == 2
    assert report['reset_dataloader_on_resume'] is False


def test_resume_refuses_post_checkpoint_graph_mutation(tmp_path, monkeypatch):
    source, output, _, graph = fixture_run(tmp_path)
    (graph / 'hyperedge_recent_mutations.json').write_text(json.dumps({'mutations': [
        {'timestamp': '2099-01-01T00:00:00+00:00'}]}))
    with pytest.raises(RuntimeError, match='post-checkpoint mutation'):
        invoke(monkeypatch, source, output)
    assert not output.exists()


def test_resume_refuses_unrewound_successful_edits(tmp_path, monkeypatch):
    source, output, metrics, _ = fixture_run(tmp_path)
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    rows[-1]['trajectory/successful_graph_edit_count/mean'] = 0.25
    metrics.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(RuntimeError, match='successful graph edits after checkpoint'):
        invoke(monkeypatch, source, output)
    assert not output.exists()
