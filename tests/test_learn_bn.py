import numpy as np
import pandas as pd
import pytest

from tabstruct.structure.learn_bn import graph_from_dag, learn_bn


def test_graph_from_dag_positions():
    # Columns 0..3 with edges 2 -> 0, 2 -> 3, 0 -> 3; column 1 isolated.
    g = graph_from_dag([(2, 0), (2, 3), (0, 3)], d=4, name="true_dag")
    position = {col: j for j, col in enumerate(g.order)}
    assert g.order == [1, 2, 0, 3]  # topological, ties broken by column index
    assert g.parents[position[3]] == sorted([position[0], position[2]])
    assert g.parents[position[0]] == [position[2]]
    assert g.parents[position[1]] == []


def test_graph_from_dag_rejects_cycles_and_bad_columns():
    with pytest.raises(ValueError):
        graph_from_dag([(0, 1), (1, 0)], d=2, name="cycle")
    with pytest.raises(ValueError):
        graph_from_dag([(0, 5)], d=2, name="out_of_range")


def test_learn_bn_recovers_chain():
    # x0 -> x1 -> x2 with strong dependence, x3 independent of everything.
    rng = np.random.default_rng(0)
    n = 2000
    x0 = rng.integers(0, 3, n)
    x1 = np.where(rng.random(n) < 0.9, x0, rng.integers(0, 3, n))
    x2 = np.where(rng.random(n) < 0.9, x1, rng.integers(0, 3, n))
    x3 = rng.integers(0, 2, n)
    train = pd.DataFrame({"a": x0, "b": x1, "c": x2, "d": x3})

    g = learn_bn(train, max_indegree=3)
    skeleton = {
        frozenset((g.order[j], g.order[p])) for j, pa in enumerate(g.parents) for p in pa
    }
    assert skeleton == {frozenset((0, 1)), frozenset((1, 2))}
    assert max(g.in_degrees()) <= 3


def test_learn_bn_respects_max_indegree():
    # y depends on four parents; with max_indegree=2 it may keep at most two of them.
    rng = np.random.default_rng(1)
    n = 3000
    xs = rng.integers(0, 2, (n, 4))
    y = np.where(rng.random(n) < 0.95, xs.sum(axis=1) % 2, rng.integers(0, 2, n))
    train = pd.DataFrame(np.column_stack([xs, y]), columns=list("abcdy"))
    assert max(learn_bn(train, max_indegree=2).in_degrees()) <= 2
