"""Capacity-controlled masked Transformer (CaTs-style, two token streams).

All variants share this architecture and parameter count; only the mask differs.
"""

from __future__ import annotations

import torch
from torch import nn


class MaskedTabularTransformer(nn.Module):
    def __init__(
        self,
        cardinalities: list[int],
        dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        """`cardinalities` are in position (generation) order."""
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> list[torch.Tensor]:
        """x: [B, d] integer codes in position order; attn_mask from `build_attention_mask`.

        Returns one logits tensor [B, cardinalities[j]] per position j, read from query
        token j.
        """
        raise NotImplementedError

    def log_prob(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """Per-column log p(x_j | x_C(j)), shape [B, d]."""
        raise NotImplementedError
