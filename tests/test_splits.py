"""Invariant 4 (CLAUDE.md): fixed held-out sets, disjoint splits, no leakage into train.

Downloads Adult (UCI) and the CHILD network on first run; cached afterwards.
"""

from pathlib import Path

import pytest

from tabstruct.data.splits import SPLIT_SEED, make_splits
from tabstruct.utils import load_config

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "configs/default.yaml", {"paths": {"data_dir": str(ROOT / "data")}})


def index_sets(splits):
    return [set(df.index) for df in (splits.train, splits.val, splits.test, splits.reference)]


def assert_codes_in_range(splits):
    for df in (splits.train, splits.val, splits.test, splits.reference):
        assert list(df.columns) == splits.columns
        for col, card in zip(df.columns, splits.cardinalities):
            assert df[col].between(0, card - 1).all(), col


def test_adult_sizes_disjoint_and_fixed_held_out():
    s0 = make_splits("adult", 500, seed=0, cfg=CFG)
    s1 = make_splits("adult", 500, seed=1, cfg=CFG)
    assert (len(s0.train), len(s0.val), len(s0.test), len(s0.reference)) == (500, 200, 10000, 5000)
    assert len(s0.columns) == 14

    sets = index_sets(s0)
    for i in range(4):
        for j in range(i + 1, 4):
            assert not sets[i] & sets[j]

    # Same held-out rows for every seed; their codes differ because bins are fit per train split.
    assert s0.test.index.equals(s1.test.index) and s0.reference.index.equals(s1.reference.index)
    assert set(s0.train.index) != set(s1.train.index)
    assert_codes_in_range(s0)


def test_adult_small_n_bins_and_nested_subsets():
    small = make_splits("adult", 500, seed=0, cfg=CFG)
    large = make_splits("adult", 5000, seed=0, cfg=CFG)
    numerical = [c for c in small.columns if c in ("age", "hours-per-week")]
    for col in numerical:
        assert small.cardinalities[small.columns.index(col)] <= CFG["data"]["n_bins_small"]
    assert set(small.train.index) <= set(large.train.index)


def test_adult_full():
    s = make_splits("adult", None, seed=0, cfg=CFG)
    assert len(s.val) == pytest.approx(0.25 * len(s.train), rel=0.01)
    sets = index_sets(s)
    assert sum(len(x) for x in sets) == len(set().union(*sets)) == 48842


def test_child_fixed_held_out_and_schema():
    s0 = make_splits("child", 500, seed=0, cfg=CFG)
    s1 = make_splits("child", 500, seed=1, cfg=CFG)
    assert (len(s0.train), len(s0.val), len(s0.test), len(s0.reference)) == (500, 200, 10000, 5000)
    assert s0.test.equals(s1.test) and not s0.train.equals(s1.train)
    assert not s0.train.reset_index(drop=True).equals(s0.test.iloc[:500].reset_index(drop=True))
    assert s0.true_model is not None and len(s0.columns) == 20
    assert_codes_in_range(s0)


def test_reserved_seed_and_bad_inputs():
    with pytest.raises(ValueError):
        make_splits("adult", 500, seed=SPLIT_SEED, cfg=CFG)
    with pytest.raises(ValueError):
        make_splits("child", None, seed=0, cfg=CFG)
    with pytest.raises(ValueError):
        make_splits("nope", 500, seed=0, cfg=CFG)
