"""Degree-matched random DAGs: the sparsity-only control for RQ2."""

from __future__ import annotations

import numpy as np

from tabstruct.structure.graphs import Graph


def degree_matched_random_dag(reference: Graph, rng: np.random.Generator) -> Graph:
    """Same order and per-position in-degree as `reference`; for position j with in-degree
    k_j, sample k_j distinct parents uniformly from positions < j."""
    raise NotImplementedError
