"""Support-Conditioned Structural Attention (SCSA).

p(x_j | x_<j) = sum_r pi_{j,r} p_r(x_j | x_{C_r(j)}), r in {BN parents, full prefix,
marginal}. Components share one backbone and differ only in masks; the gate sees
u_j = [log(1 + s_{j,BN}), log(1 + s_{j,full})]. Train with leave-one-out support.
"""

from __future__ import annotations

import torch
from torch import nn

from tabstruct.models.masked_transformer import MaskedTabularTransformer


class SCSA(nn.Module):
    def __init__(
        self,
        backbone: MaskedTabularTransformer,
        bn_parents: list[list[int]],
        gate_hidden: int = 16,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(
        self, x: torch.Tensor, support_bn: torch.Tensor, support_full: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (mixture log-prob [B, d], per-component log-probs [B, d, 3],
        gate weights pi [B, d, 3])."""
        raise NotImplementedError
