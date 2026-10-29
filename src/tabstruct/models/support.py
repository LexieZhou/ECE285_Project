"""Support counts s_{j,r}(x): training rows matching x on conditioning set C_r(j)."""

from __future__ import annotations

import numpy as np


class SupportTable:
    def __init__(self, conditioning_sets: list[list[int]]) -> None:
        """`conditioning_sets[j]` lists positions; an empty set has support n."""
        self.conditioning_sets = conditioning_sets

    def fit(self, train: np.ndarray) -> SupportTable:
        """Build per-position hash tables (value tuple -> count) from train [n, d]."""
        raise NotImplementedError

    def counts(self, x: np.ndarray, leave_one_out: bool = False) -> np.ndarray:
        """Support of each row of x [B, d] for each position, shape [B, d].

        Use `leave_one_out=True` for rows that are themselves in the training set.
        """
        raise NotImplementedError
