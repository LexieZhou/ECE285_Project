import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

from tabstruct.eval.nll import bucket_index
from tabstruct.eval.privacy import vulnerability_scores
from tabstruct.eval.support_analysis import (
    bn_support,
    bucket_comparisons,
    loss_mi_auc,
    to_original_order,
)
from tabstruct.structure.learn_bn import graph_from_dag


def test_bucket_edges():
    np.testing.assert_array_equal(bucket_index(np.array([0, 1, 4, 5, 19, 20, 500])),
                                  [0, 1, 1, 2, 2, 3, 3])


def test_vulnerability_matches_brute_force_cosine():
    rng = np.random.default_rng(0)
    train = pd.DataFrame(rng.integers(0, [3, 2, 4, 5], size=(300, 4)))
    onehot = pd.get_dummies(train.astype(str)).to_numpy(float)
    dist = cosine_distances(onehot)
    np.fill_diagonal(dist, np.inf)
    expected = np.sort(dist, axis=1)[:, :5].mean(1)
    np.testing.assert_allclose(vulnerability_scores(train, k=5, block=64), expected, atol=1e-6)


def test_bn_support_is_loo_for_train_and_in_original_order():
    rng = np.random.default_rng(0)
    train, test = rng.integers(0, 3, size=(200, 4)), rng.integers(0, 3, size=(50, 4))
    # Original columns: 3 -> 0 -> 1, 2 isolated, so positions differ from columns.
    bn = graph_from_dag([(3, 0), (0, 1), (3, 1)], d=4, name="bn")
    s_train, s_test = bn_support(train, test, bn)
    parents = {0: [3], 1: [0, 3], 2: [], 3: []}  # original column indices
    for col, pa in parents.items():
        brute_test = [(train[:, pa] == row[pa]).all(1).sum() for row in test]
        brute_train = [(train[:, pa] == row[pa]).all(1).sum() - 1 for row in train]
        np.testing.assert_array_equal(s_test[:, col], brute_test)
        np.testing.assert_array_equal(s_train[:, col], brute_train)


def test_to_original_order_inverts_position_order():
    order = [2, 0, 3, 1]
    x = np.arange(8).reshape(2, 4)  # original order
    np.testing.assert_array_equal(to_original_order(x[:, order], order), x)


def test_loss_mi_auc():
    assert loss_mi_auc(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 1.0  # members lower NLL
    assert loss_mi_auc(np.array([1.0, 3.0]), np.array([1.0, 3.0])) == 0.5
    assert np.isnan(loss_mi_auc(np.array([]), np.array([1.0])))


def test_bucket_comparisons_pair_by_seed_and_average_random_dags():
    rows = []
    for seed in range(3):
        for bucket, gain in [("zero", -0.4), ("high", 0.2)]:  # bn wins at zero, loses at high
            nll = {"full_prefix": 1.0 + seed, "bn": 1.0 + seed + gain,
                   "random_dag_0": 2.0 + seed, "random_dag_1": 4.0 + seed}
            rows += [{"dataset": "child", "n": "500", "seed": seed, "variant": v,
                      "bucket": bucket, "nll_test": x} for v, x in nll.items()]
    comp = bucket_comparisons(pd.DataFrame(rows))
    rq1 = comp[comp["comparison"].str.startswith("RQ1")].set_index("bucket")
    assert list(rq1.index) == ["zero", "high"]  # bucket order, not alphabetical
    assert np.isclose(rq1.loc["zero", "mean_diff"], -0.4) and rq1.loc["zero", "a_better"] == 3
    assert np.isclose(rq1.loc["high", "mean_diff"], 0.2) and rq1.loc["high", "a_better"] == 0
    rq2 = comp[comp["comparison"].str.startswith("RQ2")].set_index("bucket")
    assert np.isclose(rq2.loc["zero", "mean_diff"], -0.4 - 2.0)  # bn - mean(2, 4) offsets
