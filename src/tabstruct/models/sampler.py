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
) -> torch.Tensor:
    """Sample n rows [n, d] in position order, one column at a time."""
    raise NotImplementedError
