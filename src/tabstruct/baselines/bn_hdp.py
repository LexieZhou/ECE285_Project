"""BN with hierarchical (HDP-style) CPT smoothing, after Petitjean et al. (2018).

Substitute for the reference Java code (documented in PLAN.md, Phase 1). As in the
paper, the CPT of each column is a tree over its parents: level 0 is the marginal,
level l conditions on the first l parents, and the leaves are full parent
configurations. Each node's distribution is smoothed toward its parent node's, with one
concentration per (column, level) shared by all nodes of that level:

    p_l(x | u_1..u_l) = (n(x, u_1..u_l) + a_l * p_{l-1}(x | u_1..u_{l-1})) / (n(u_1..u_l) + a_l)

with p_{-1} uniform. Differences from the paper's HDP: nodes pass their raw counts up
(no Chinese-restaurant table counts), and the concentrations are point estimates that
maximize the leave-one-out log-likelihood of the training rows instead of being
Gibbs-sampled. Both keep the fit deterministic. LOO puts no mass on unseen values of a
CPT row without singletons and drives its concentration to ~0, which the paper's
prior on the concentrations prevents; `min_concentration` plays that role here (selected
on validation NLL in `train.fit_bn_hdp`). Parents enter the tree in decreasing mutual
information with the child (on the training split), so the coarse-grained levels keep
the most informative parents.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from tabstruct.baselines.bn_cpt import BNCPTGenerator
from tabstruct.structure.graphs import Graph


class BNHDPGenerator(BNCPTGenerator):
    def __init__(
        self,
        graph: Graph,
        concentration: float | None = None,
        min_concentration: float = 1e-2,
        max_concentration: float = 1e4,
    ):
        """`concentration` fixes a_l for every level; None fits them by leave-one-out
        within [min_concentration, max_concentration]. `log_prob` and `sample` come from
        `BNCPTGenerator`."""
        self.graph = graph
        self.concentration = concentration
        self.bounds = (min_concentration, max_concentration)

    def fit(self, train: pd.DataFrame, cardinalities: list[int]) -> BNHDPGenerator:
        """`train` and `cardinalities` are in original column order."""
        self.columns = list(train.columns)
        self.cards = [cardinalities[c] for c in self.graph.order]  # position order
        x = train.to_numpy()[:, self.graph.order]
        self.cpts = []           # position j -> [q_j, r_j], parents in graph order
        self.parent_orders = []  # position j -> parents in tree order
        self.concentrations = []  # position j -> a_0..a_k
        for j, parents in enumerate(self.graph.parents):
            r = self.cards[j]
            tree = sorted(parents, key=lambda p: -self._mutual_information(x[:, p], x[:, j]))
            configs = [self._config(x, tree[:l]) for l in range(len(tree) + 1)]
            counts = []
            for l, config in enumerate(configs):
                n = np.zeros((self._n_configs(tree[:l]), r))
                np.add.at(n, (config, x[:, j]), 1)
                counts.append(n)
            if self.concentration is None:
                alpha = self._fit_concentrations(counts, configs, x[:, j])
            else:
                alpha = np.full(len(counts), self.concentration)
            self.cpts.append(self._to_graph_order(self._leaf_cpt(counts, tree, alpha), tree, parents))
            self.parent_orders.append(tree)
            self.concentrations.append(alpha.tolist())
        return self

    def _leaf_cpt(self, counts: list[np.ndarray], tree: list[int], alpha: np.ndarray) -> np.ndarray:
        """Smooth down the tree; returns the leaf CPT [q, r] indexed in tree order."""
        p = np.full((1, counts[0].shape[1]), 1.0 / counts[0].shape[1])
        for l, (n, a) in enumerate(zip(counts, alpha)):
            # Level-l config c has level-(l-1) config c // card(tree[l-1]) (last parent fastest).
            base = p if l == 0 else np.repeat(p, self.cards[tree[l - 1]], axis=0)
            p = (n + a * base) / (n.sum(1, keepdims=True) + a)
        return p

    def _to_graph_order(self, cpt: np.ndarray, tree: list[int], parents: list[int]) -> np.ndarray:
        """Re-index the configuration axis from tree order to `parents` order, so the
        inherited `_config` / `log_prob` / `sample` apply unchanged."""
        if not tree:
            return cpt
        r = cpt.shape[1]
        grid = cpt.reshape([self.cards[p] for p in tree] + [r])
        axes = [tree.index(p) for p in parents] + [len(tree)]
        return grid.transpose(axes).reshape(-1, r)

    def _fit_concentrations(
        self, counts: list[np.ndarray], configs: list[np.ndarray], x_j: np.ndarray
    ) -> np.ndarray:
        """Maximize sum_i log p^{-i}(x_i), where every count is taken without row i."""
        cell = [n[c, x_j] - 1 for n, c in zip(counts, configs)]          # [L][N]
        total = [n.sum(1)[c] - 1 for n, c in zip(counts, configs)]       # [L][N]
        r = counts[0].shape[1]

        def loss(log_alpha: np.ndarray) -> float:
            p = np.full(len(x_j), 1.0 / r)
            for a, n_cell, n_total in zip(np.exp(log_alpha), cell, total):
                p = (n_cell + a * p) / (n_total + a)
            return -np.log(p).sum()

        log_bounds = tuple(np.log(self.bounds))
        result = minimize(
            loss, np.zeros(len(counts)), method="L-BFGS-B", bounds=[log_bounds] * len(counts)
        )
        return np.exp(result.x)

    def _mutual_information(self, a: np.ndarray, b: np.ndarray) -> float:
        joint = np.zeros((a.max() + 1, b.max() + 1))
        np.add.at(joint, (a, b), 1)
        joint /= joint.sum()
        outer = joint.sum(1, keepdims=True) * joint.sum(0, keepdims=True)
        nz = joint > 0
        return float((joint[nz] * np.log(joint[nz] / outer[nz])).sum())
