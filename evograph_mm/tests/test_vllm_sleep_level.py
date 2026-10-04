from unittest.mock import Mock, patch

import pytest

from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager


def _manager(sleep_level):
    manager = FSDPVLLMShardingManager.__new__(FSDPVLLMShardingManager)
    manager.module = Mock()
    manager.inference_engine = Mock()
    manager.device_mesh = None
    manager.sleep_level = sleep_level
    return manager


def test_rejects_invalid_vllm_sleep_level():
    with pytest.raises(ValueError, match="sleep_level must be 1 or 2"):
        FSDPVLLMShardingManager(
            module=Mock(),
            inference_engine=Mock(),
            model_config=Mock(),
            sleep_level=3,
        )


@patch("verl.workers.sharding_manager.fsdp_vllm.torch.cuda.empty_cache")
@patch("verl.workers.sharding_manager.fsdp_vllm.log_gpu_memory_usage")
def test_exit_uses_configured_deep_sleep(_log_memory, _empty_cache):
    manager = _manager(sleep_level=2)

    manager.__exit__(None, None, None)

    manager.inference_engine.sleep.assert_called_once_with(level=2)
    manager.module.train.assert_called_once_with()
