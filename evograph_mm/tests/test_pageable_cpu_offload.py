from types import SimpleNamespace

import pytest
import torch

from verl.utils import fsdp_utils


def test_original_async_policy_is_default(monkeypatch):
    monkeypatch.delenv('EVOGRAPH_FSDP_CPU_OFFLOAD_NON_BLOCKING', raising=False)
    assert fsdp_utils._cpu_offload_non_blocking() is True


def test_invalid_transfer_policy_is_rejected(monkeypatch):
    monkeypatch.setenv('EVOGRAPH_FSDP_CPU_OFFLOAD_NON_BLOCKING', 'maybe')
    with pytest.raises(ValueError, match='true or false'):
        fsdp_utils._cpu_offload_non_blocking()


def test_optimizer_uses_blocking_native_copy_and_preserves_values(monkeypatch):
    monkeypatch.setenv('EVOGRAPH_FSDP_CPU_OFFLOAD_NON_BLOCKING', 'false')
    parameter = torch.nn.Parameter(torch.ones(3))
    optimizer = torch.optim.AdamW([parameter])
    optimizer.state[parameter] = {'exp_avg': torch.arange(3.), 'step': 9}
    original_to = torch.Tensor.to
    calls = []

    def record_to(self, *args, **kwargs):
        calls.append((args, kwargs))
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'to', record_to)
    fsdp_utils.offload_fsdp_optimizer(optimizer)
    assert calls == [(('cpu',), {'non_blocking': False})]
    assert torch.equal(optimizer.state[parameter]['exp_avg'], torch.arange(3.))
    assert optimizer.state[parameter]['step'] == 9


def test_manual_flat_parameter_uses_blocking_copy_but_native_offload_is_skipped(monkeypatch):
    monkeypatch.setenv('EVOGRAPH_FSDP_CPU_OFFLOAD_NON_BLOCKING', 'false')
    flat = torch.nn.Parameter(torch.arange(4.))
    flat._local_shard = flat.data
    calls = []
    handle = SimpleNamespace(flat_param=flat, _offload_params=False,
                             flat_param_to=lambda *args, **kwargs: calls.append((args, kwargs)))
    native = SimpleNamespace(_offload_params=True)
    model = SimpleNamespace(_is_root=True, _all_handles=[handle, native])
    monkeypatch.setattr(fsdp_utils, 'FSDP', SimpleNamespace)
    monkeypatch.setattr(fsdp_utils, '_lazy_init', lambda *args: None)
    fsdp_utils.offload_fsdp_model_to_cpu(model, empty_cache=False)
    assert calls == [((torch.device('cpu'),), {'non_blocking': False})]
    assert torch.equal(flat._local_shard, torch.arange(4.))
