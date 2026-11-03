import numpy as np
import pandas as pd

from tabstruct.baselines.bn_cpt import BNCPTGenerator
from tabstruct.baselines.bn_hdp import BNHDPGenerator
from tabstruct.structure.graphs import Graph
from tabstruct.structure.learn_bn import graph_from_dag

COLUMNS = ["a", "b", "c", "d"]
CARDS = [3, 2, 4, 3]
EDGES = [(0, 2), (1, 2), (2, 3), (0, 3)]  # original column indices; c has two parents


def random_table(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({col: rng.integers(0, card, n) for col, card in zip(COLUMNS, CARDS)})


def all_cells() -> pd.DataFrame:
    grid = np.array(np.meshgrid(*[range(c) for c in CARDS], indexing="ij")).reshape(4, -1).T
    return pd.DataFrame(grid, columns=COLUMNS)


def test_log_prob_is_a_normalized_joint():
    graph = graph_from_dag(EDGES, d=4, name="bn")
    gen = BNHDPGenerator(graph).fit(random_table(100, 0), CARDS)
    log_p = gen.log_prob(all_cells())
    assert log_p.shape == (len(all_cells()), 4)
    assert np.isclose(np.exp(log_p.sum(1)).sum(), 1.0)
    for alpha, parents in zip(gen.concentrations, graph.parents):
        assert len(alpha) == len(parents) + 1  # one concentration per tree level


def test_without_parents_matches_bdeu():
    # One level with base 1/r: (n_x + a / r) / (N + a) is BDeu with ess = a and q = 1.
    graph = Graph.marginal([0, 1, 2, 3])
    train, held_out = random_table(60, 0), random_table(200, 1)
    hdp = BNHDPGenerator(graph, concentration=3.0).fit(train, CARDS)
    cpt = BNCPTGenerator(graph, "BDeu", 3.0).fit(train, CARDS)
    np.testing.assert_allclose(hdp.log_prob(held_out), cpt.log_prob(held_out), rtol=1e-12)


def test_independent_of_parent_list_order():
    # The tree order comes from mutual information, not from how parents are listed.
    graph = graph_from_dag(EDGES, d=4, name="bn")
    swapped = Graph(graph.order, [list(reversed(pa)) for pa in graph.parents], name="bn")
    train = random_table(150, 0)
    a = BNHDPGenerator(graph).fit(train, CARDS).log_prob(all_cells())
    b = BNHDPGenerator(swapped).fit(train, CARDS).log_prob(all_cells())
    np.testing.assert_allclose(a, b, rtol=1e-10)


def test_unseen_parent_configuration_backs_off_to_coarser_level():
    # c copies a; b is noise. Hold out every row with (a, b) = (2, 1): flat-prior CPTs know
    # nothing about that configuration, the hierarchy still predicts c = a = 2.
    rng = np.random.default_rng(0)
    n = 2000
    a, b = rng.integers(0, 3, n), rng.integers(0, 2, n)
    train = pd.DataFrame({
        "a": a, "b": b, "c": np.where(rng.random(n) < 0.9, a, 3), "d": rng.integers(0, 3, n),
    })
    train = train[~((train["a"] == 2) & (train["b"] == 1))]
    graph = graph_from_dag([(0, 2), (1, 2)], d=4, name="bn")
    hdp = BNHDPGenerator(graph).fit(train, CARDS)
    cpt = BNCPTGenerator(graph, "BDeu", 1.0).fit(train, CARDS)
    query = pd.DataFrame({"a": [2], "b": [1], "c": [2], "d": [0]})
    j = graph.order.index(2)
    assert np.exp(hdp.log_prob(query)[0, j]) > 0.8
    assert np.isclose(np.exp(cpt.log_prob(query)[0, j]), 0.25)
    # a is far more informative about c than b, so it enters the tree first.
    assert hdp.parent_orders[j] == [graph.order.index(0), graph.order.index(1)]


def test_samples_follow_the_cpts():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 3, 5000)
    train = pd.DataFrame({
        "a": a, "b": rng.integers(0, 2, 5000), "c": np.where(rng.random(5000) < 0.9, a, 3),
        "d": rng.integers(0, 3, 5000),
    })
    gen = BNHDPGenerator(graph_from_dag([(0, 2)], d=4, name="bn")).fit(train, CARDS)
    samples = gen.sample(20_000, seed=0)
    assert list(samples.columns) == COLUMNS
    assert abs((samples["a"] == samples["c"]).mean() - (train["a"] == train["c"]).mean()) < 0.02
