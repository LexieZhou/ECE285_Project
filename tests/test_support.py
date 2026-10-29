"""Invariant 5 (CLAUDE.md): training rows use leave-one-out support."""

import numpy as np

from tabstruct.models.support import SupportTable

SETS = [[], [0], [0, 1], [1, 2]]


def brute_force(train: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.array([[(train[:, c] == row[c]).all(axis=1).sum() for c in SETS] for row in x])


def test_counts_match_brute_force():
    rng = np.random.default_rng(0)
    train = rng.integers(0, 3, size=(50, 4))
    held_out = rng.integers(0, 3, size=(20, 4))
    table = SupportTable(SETS).fit(train)

    np.testing.assert_array_equal(table.counts(held_out), brute_force(train, held_out))
    np.testing.assert_array_equal(
        table.counts(train, leave_one_out=True), brute_force(train, train) - 1
    )
