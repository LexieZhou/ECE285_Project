"""Bayesian-network structure learning on the training split (pgmpy)."""

from __future__ import annotations

import pandas as pd

from tabstruct.structure.graphs import Graph


def learn_bn(train: pd.DataFrame, max_indegree: int = 3) -> Graph:
    """Hill-climbing with BIC (`HillClimbSearch` + `BicScore`) on integer-coded `train`.

    Returns a Graph whose order is a topological order of the learned DAG.
    """
    raise NotImplementedError


def graph_from_dag(edges: list[tuple[int, int]], d: int, name: str) -> Graph:
    """Convert a DAG over original column indices into position space via a topological
    order (used for both learned and true DAGs)."""
    raise NotImplementedError
