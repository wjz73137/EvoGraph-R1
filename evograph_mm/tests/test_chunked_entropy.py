from __future__ import annotations

import torch

from verl.utils.torch_functional import (
    entropy_from_logits,
    entropy_from_logits_chunked,
)


def test_chunked_entropy_matches_values_and_gradients() -> None:
    baseline_logits = torch.randn(2, 11, 37, dtype=torch.float64, requires_grad=True)
    chunked_logits = baseline_logits.detach().clone().requires_grad_(True)

    baseline = entropy_from_logits(baseline_logits)
    chunked = entropy_from_logits_chunked(chunked_logits, chunk_size=4)
    assert torch.allclose(chunked, baseline, atol=1e-12, rtol=1e-12)

    baseline.sum().backward()
    chunked.sum().backward()
    assert torch.allclose(
        chunked_logits.grad,
        baseline_logits.grad,
        atol=1e-12,
        rtol=1e-12,
    )
