from contextlib import nullcontext
import weakref

import torch

from verl.utils.checkpoint import fsdp_checkpoint_manager as checkpoint


def test_native_mmap_cpu_loading_preserves_tensors(tmp_path, monkeypatch):
    path = tmp_path / 'state.pt'
    expected = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    torch.save({'weight': expected}, path)
    monkeypatch.setenv('EVOGRAPH_CHECKPOINT_MMAP_LOAD', 'true')
    actual = checkpoint._load_checkpoint_state(path)
    assert actual['weight'].device.type == 'cpu'
    assert torch.equal(actual['weight'], expected)


def test_model_temporary_state_is_released_before_optimizer_load(monkeypatch):
    manager = object.__new__(checkpoint.FSDPCheckpointManager)
    manager.rank, manager.world_size = 0, 2
    manager.model = type('Model', (), {'load_state_dict': lambda self, state: None})()
    manager.optimizer = type('Optim', (), {'load_state_dict': lambda self, state: None})()
    scheduler_states = []
    manager.lr_scheduler = type('LR', (), {'load_state_dict': lambda self, state: scheduler_states.append(state)})()
    restored_rng = []
    manager.load_rng_state = restored_rng.append
    monkeypatch.setattr(checkpoint.FSDP, 'state_dict_type', lambda *args: nullcontext())
    monkeypatch.setattr(checkpoint, 'copy_to_local', lambda path: path)
    tensor_refs, order = [], []

    def load(path):
        if 'model_world' in path:
            order.append('model')
            tensor = torch.ones(2)
            tensor_refs.append(weakref.ref(tensor))
            return {'weight': tensor}
        if 'optim_world' in path:
            assert tensor_refs[0]() is None
            order.append('optimizer')
            return {'state': {}}
        order.append('extra')
        return {'lr_scheduler': {'last_epoch': 650}, 'rng': {'test': True}}

    monkeypatch.setattr(checkpoint, '_load_checkpoint_state', load)
    manager.load_checkpoint('/synthetic/checkpoint')
    assert order == ['model', 'optimizer', 'extra']
    assert scheduler_states == [{'last_epoch': 650}]
    assert restored_rng == [{'test': True}]
