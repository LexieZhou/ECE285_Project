"""Attention masks that restrict each column's prediction to its conditioning set.

Token layout for d columns: positions 0..d-1 are content tokens (value + column
embedding), positions d..2d-1 are query tokens (column embedding only).
"""

from __future__ import annotations

import torch


def build_attention_mask(parents: list[list[int]]) -> torch.Tensor:
    """Boolean [2d, 2d] mask, True = the row token may attend to the column token.

    Content token k attends only to itself. Query token d+j attends to content tokens in
    `parents[j]` and to itself. Note: PyTorch attention modules use True = *blocked*, so
    invert before passing this to them.
    """
    raise NotImplementedError
