"""Capacity-controlled masked Transformer (CaTs-style, two token streams).

All variants share this architecture and parameter count; only the mask differs.

Content token k = value embedding of x_k + content column embedding k. Query token j =
query column embedding j (no value). With a mask from `build_attention_mask`, content
tokens only attend to themselves, so after any number of layers query token j has seen
exactly x_{C(j)}.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
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
        self.cardinalities = list(cardinalities)
        d = len(self.cardinalities)
        # One shared value table; column k's codes live at offsets[k] .. offsets[k] + card_k - 1.
        offsets = torch.tensor([0, *self.cardinalities[:-1]]).cumsum(0)
        self.register_buffer("offsets", offsets, persistent=False)
        self.value_embed = nn.Embedding(sum(self.cardinalities), dim)
        self.content_col_embed = nn.Embedding(d, dim)
        self.query_col_embed = nn.Embedding(d, dim)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                dim, n_heads, dim_feedforward=4 * dim, dropout=dropout,
                batch_first=True, norm_first=True,
            )
            for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(dim)
        self.heads = nn.ModuleList(nn.Linear(dim, card) for card in self.cardinalities)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> list[torch.Tensor]:
        """x: [B, d] integer codes in position order; attn_mask from `build_attention_mask`.

        Returns one logits tensor [B, cardinalities[j]] per position j, read from query
        token j.
        """
        batch, d = x.shape
        cols = torch.arange(d, device=x.device)
        content = self.value_embed(x + self.offsets) + self.content_col_embed(cols)
        query = self.query_col_embed(cols).expand(batch, d, -1)
        h = torch.cat([content, query], dim=1)

        blocked = ~attn_mask.to(x.device)  # PyTorch convention: True = may not attend
        for layer in self.layers:
            h = layer(h, src_mask=blocked)
        h = self.norm(h[:, d:])
        return [head(h[:, j]) for j, head in enumerate(self.heads)]

    def log_prob(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """Per-column log p(x_j | x_C(j)), shape [B, d]."""
        logits = self.forward(x, attn_mask)
        return torch.stack(
            [F.log_softmax(lj, dim=-1).gather(1, x[:, j : j + 1]).squeeze(1)
             for j, lj in enumerate(logits)],
            dim=1,
        )
