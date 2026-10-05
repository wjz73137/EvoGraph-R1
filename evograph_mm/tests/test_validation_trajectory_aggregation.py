import pytest

from verl.trainer.ppo.ray_trainer import _group_validation_trajectory_metrics


def test_multiple_validation_batches_are_grouped_from_all_accumulated_rows():
    # Two two-row batches: reading only the final batch loses rows / indexes past it.
    rows = [{'successful_graph_edit_count': value, 'websearch_count': value + 1}
            for value in [1, 2, 4, 8]]
    grouped = _group_validation_trajectory_metrics(rows, ['a', 'b', 'a', 'b'])
    assert grouped['successful_graph_edit_count'] == {'a': [1, 4], 'b': [2, 8]}
    assert grouped['websearch_count'] == {'a': [2, 5], 'b': [3, 9]}
    assert grouped['duplicate_search_count'] == {'a': [0, 0], 'b': [0, 0]}


def test_validation_metric_alignment_mismatch_fails_explicitly():
    with pytest.raises(ValueError, match='count mismatch'):
        _group_validation_trajectory_metrics([{}], ['a', 'b'])
