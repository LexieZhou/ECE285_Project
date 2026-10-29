"""Ancestral sampling in generation order."""

from __future__ import annotations

import torch

from tabstruct.models.masked_transformer import MaskedTabularTransformer


@torch.no_grad()
def ancestral_sample(
    model: MaskedTabularTransformer,
    attn_mask: torch.Tensor,
    n: int,
    generator: torch.Generator,
    batch_size: int = 4096,
) -> torch.Tensor:
    """Sample n rows [n, d] in position order, one column at a time.

    Columns not yet generated hold placeholder zeros; the mask guarantees position j
    only reads its parents (all < j), so the placeholders never affect a prediction.
    `generator` must live on the model's device.
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    d = len(model.cardinalities)
    out = torch.empty(n, d, dtype=torch.long, device=device)
    for start in range(0, n, batch_size):
        x = torch.zeros(min(batch_size, n - start), d, dtype=torch.long, device=device)
        for j in range(d):
            probs = torch.softmax(model(x, attn_mask)[j], dim=-1)
            x[:, j] = torch.multinomial(probs, 1, generator=generator).squeeze(1)
        out[start : start + len(x)] = x
    model.train(was_training)
    return out
