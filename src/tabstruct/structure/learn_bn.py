"""Bayesian-network structure learning on the training split (pgmpy)."""

from __future__ import annotations

import networkx as nx
import pandas as pd

from tabstruct.structure.graphs import Graph


def learn_bn(train: pd.DataFrame, max_indegree: int = 3) -> Graph:
    """Hill-climbing with BIC (`HillClimbSearch` + `BicScore`) on integer-coded `train`.

    Returns a Graph whose order is a topological order of the learned DAG.
    """
    from pgmpy.causal_discovery import HillClimbSearch

    # Positional column names, so the learned edges are original column indices.
    data = train.set_axis(range(train.shape[1]), axis=1)
    search = HillClimbSearch(
        scoring_method="bic-d",  # explicit: integer codes must not be read as continuous
        max_indegree=max_indegree,
        return_type="dag",
        show_progress=False,
    )
    dag = search.fit(data).causal_graph_
    return graph_from_dag(list(dag.edges()), train.shape[1], name="bn")


def graph_from_dag(edges: list[tuple[int, int]], d: int, name: str) -> Graph:
    """Convert a DAG over original column indices into position space via a topological
    order (used for both learned and true DAGs)."""
    dag = nx.DiGraph()
    dag.add_nodes_from(range(d))
    dag.add_edges_from(edges)
    if dag.number_of_nodes() != d:
        raise ValueError(f"edges reference columns outside 0..{d - 1}: {edges}")
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError(f"{name}: edges contain a cycle")
    # Ties broken by original column index, so the order is deterministic.
    order = list(nx.lexicographical_topological_sort(dag))
    position = {col: j for j, col in enumerate(order)}
    parents = [sorted(position[p] for p in dag.predecessors(col)) for col in order]
    return Graph(order=order, parents=parents, name=name)
