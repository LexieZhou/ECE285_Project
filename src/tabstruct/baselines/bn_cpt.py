"""BN with smoothed CPTs (Laplace / BDeu, as in pgmpy `BayesianEstimator`).

The CPTs are counted directly with numpy; `tests/test_bn_cpt.py` checks them against
pgmpy's `BayesianEstimator`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tabstruct.structure.graphs import Graph


class BNCPTGenerator:
    def __init__(self, graph: Graph, prior: str = "BDeu", equivalent_sample_size: float = 1.0):
        """`prior` is "BDeu" (pseudo-count ess / (r * q) per cell, r states and q parent
        configurations) or "K2" (pseudo-count 1 per cell, i.e. Laplace)."""
        if prior not in ("BDeu", "K2"):
            raise ValueError(f"unknown prior: {prior}")
        self.graph = graph
        self.prior = prior
        self.equivalent_sample_size = equivalent_sample_size

    def fit(self, train: pd.DataFrame, cardinalities: list[int]) -> BNCPTGenerator:
        """`train` and `cardinalities` are in original column order."""
        self.columns = list(train.columns)
        self.cards = [cardinalities[c] for c in self.graph.order]  # position order
        x = train.to_numpy()[:, self.graph.order]
        self.cpts = []  # position j -> [q_j, r_j] conditional probabilities
        for j, parents in enumerate(self.graph.parents):
            r, q = self.cards[j], self._n_configs(parents)
            counts = np.zeros((q, r))
            np.add.at(counts, (self._config(x, parents), x[:, j]), 1)
            alpha = self.equivalent_sample_size / (r * q) if self.prior == "BDeu" else 1.0
            counts += alpha
            self.cpts.append(counts / counts.sum(axis=1, keepdims=True))
        return self

    def n_params(self) -> int:
        """Free CPT parameters: sum over positions of q_j * (r_j - 1)."""
        return sum(cpt.shape[0] * (cpt.shape[1] - 1) for cpt in self.cpts)

    def log_prob(self, df: pd.DataFrame) -> np.ndarray:
        """Per-column log p(x_j | x_pa(j)), shape [N, d] in position order."""
        x = df[self.columns].to_numpy()[:, self.graph.order]
        return np.stack(
            [np.log(cpt[self._config(x, pa), x[:, j]])
             for j, (cpt, pa) in enumerate(zip(self.cpts, self.graph.parents))],
            axis=1,
        )

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        """Ancestral sampling; returns codes in original column order."""
        rng = np.random.default_rng(seed)
        x = np.zeros((n, len(self.cards)), dtype=np.int64)
        for j, (cpt, pa) in enumerate(zip(self.cpts, self.graph.parents)):
            cdf = cpt[self._config(x, pa)].cumsum(axis=1)
            u = rng.random((n, 1)) * cdf[:, -1:]  # scale by the last value: float round-off
            x[:, j] = (u > cdf).sum(axis=1)
        out = np.empty_like(x)
        out[:, self.graph.order] = x
        return pd.DataFrame(out, columns=self.columns)

    def _n_configs(self, parents: list[int]) -> int:
        return int(np.prod([self.cards[p] for p in parents]))  # 1 for no parents

    def _config(self, x: np.ndarray, parents: list[int]) -> np.ndarray:
        """Mixed-radix index of each row's parent configuration (last parent fastest)."""
        index = np.zeros(len(x), dtype=np.int64)
        for p in parents:
            index = index * self.cards[p] + x[:, p]
        return index
