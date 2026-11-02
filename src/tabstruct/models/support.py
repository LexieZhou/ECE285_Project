"""Support counts s_{j,r}(x): training rows matching x on conditioning set C_r(j)."""

from __future__ import annotations

import numpy as np


class SupportTable:
    def __init__(self, conditioning_sets: list[list[int]]) -> None:
        """`conditioning_sets[j]` lists positions; an empty set has support n."""
        self.conditioning_sets = conditioning_sets

    def fit(self, train: np.ndarray) -> SupportTable:
        """Build per-position hash tables (value tuple -> count) from train [n, d]."""
        self.n = len(train)
        self.tables = []
        for cond in self.conditioning_sets:
            keys, counts = np.unique(train[:, cond], axis=0, return_counts=True)
            self.tables.append(dict(zip(map(tuple, keys.tolist()), counts.tolist())))
        return self

    def counts(self, x: np.ndarray, leave_one_out: bool = False) -> np.ndarray:
        """Support of each row of x [B, d] for each position, shape [B, d].

        Use `leave_one_out=True` for rows that are themselves in the training set.
        """
        out = np.empty((len(x), len(self.conditioning_sets)), dtype=np.int64)
        for j, (cond, table) in enumerate(zip(self.conditioning_sets, self.tables)):
            if not cond:
                out[:, j] = self.n
            else:
                out[:, j] = [table.get(key, 0) for key in map(tuple, x[:, cond].tolist())]
        return out - 1 if leave_one_out else out
