import numpy as np
import pandas as pd
import pytest

from tabstruct.baselines.bn_cpt import BNCPTGenerator
from tabstruct.data.bnlearn import true_log_likelihood
from tabstruct.structure.learn_bn import graph_from_dag

COLUMNS = ["a", "b", "c", "d"]
CARDS = [3, 2, 4, 3]
EDGES = [(0, 2), (1, 2), (2, 3), (0, 3)]  # original column indices; c has two parents


def random_table(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({col: rng.integers(0, card, n) for col, card in zip(COLUMNS, CARDS)})


@pytest.mark.parametrize("prior,ess", [("BDeu", 1.0), ("BDeu", 10.0), ("K2", 1.0)])
def test_matches_pgmpy_bayesian_estimator(prior, ess):
    from pgmpy.estimators import BayesianEstimator
    from pgmpy.models import DiscreteBayesianNetwork

    train, held_out = random_table(60, 0), random_table(200, 1)  # small n: priors matter
    graph = graph_from_dag(EDGES, d=4, name="bn")
    ours = BNCPTGenerator(graph, prior, ess).fit(train, CARDS)

    model = DiscreteBayesianNetwork([(COLUMNS[u], COLUMNS[v]) for u, v in EDGES])
    # Declare every state, including ones absent from the small training set.
    states = {col: list(range(card)) for col, card in zip(COLUMNS, CARDS)}
    estimator = BayesianEstimator(model, train, state_names=states)
    cpds = estimator.get_parameters(prior_type=prior, equivalent_sample_size=ess)
    model.add_cpds(*cpds)
    np.testing.assert_allclose(
        ours.log_prob(held_out).sum(1), true_log_likelihood(model, held_out), rtol=1e-10
    )


def test_log_prob_is_position_ordered_and_normalized():
    graph = graph_from_dag(EDGES, d=4, name="bn")
    gen = BNCPTGenerator(graph).fit(random_table(100, 0), CARDS)
    cells = pd.DataFrame(
        np.array(np.meshgrid(*[range(c) for c in CARDS], indexing="ij")).reshape(4, -1).T,
        columns=COLUMNS,
    )
    log_p = gen.log_prob(cells)
    assert log_p.shape == (len(cells), 4)
    assert np.isclose(np.exp(log_p.sum(1)).sum(), 1.0)  # a proper joint over all cells
    assert gen.n_params() == sum(
        int(np.prod([CARDS[p] for p in [u for u, v in EDGES if v == col]])) * (CARDS[col] - 1)
        for col in range(4)
    )


def test_samples_follow_the_cpts():
    # Strong chain a -> c so that sampling in the wrong order would show.
    rng = np.random.default_rng(0)
    a = rng.integers(0, 3, 5000)
    train = pd.DataFrame({
        "a": a, "b": rng.integers(0, 2, 5000), "c": np.where(rng.random(5000) < 0.9, a, 3),
        "d": rng.integers(0, 3, 5000),
    })
    graph = graph_from_dag([(0, 2)], d=4, name="bn")
    gen = BNCPTGenerator(graph).fit(train, CARDS)
    samples = gen.sample(20_000, seed=0)
    assert list(samples.columns) == COLUMNS
    for col, card in zip(COLUMNS, CARDS):
        assert samples[col].between(0, card - 1).all()
    assert abs((samples["a"] == samples["c"]).mean() - (train["a"] == train["c"]).mean()) < 0.02
    assert samples.equals(gen.sample(20_000, seed=0))
