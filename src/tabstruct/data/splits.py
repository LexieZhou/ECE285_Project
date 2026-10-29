"""Fixed test/reference sets and per-seed train/val subsets for every dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from tabstruct.data import BNLEARN_NETWORKS, adult, bnlearn
from tabstruct.data.discretize import fit_discretizer

# Seed of the fixed test/reference sets. Run seeds must differ from it, otherwise a
# bnlearn training sample would replay the test sample.
SPLIT_SEED = 2**31 - 1


@dataclass
class DataSplits:
    """Integer-coded splits of one (dataset, n, seed). Columns are in original order."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame  # fixed across seeds; never used for training or tuning
    reference: pd.DataFrame  # fixed across seeds; DOMIAS reference data only
    columns: list[str]
    cardinalities: list[int]
    true_model: Any | None = None  # ground-truth pgmpy model for bnlearn datasets


def make_splits(dataset: str, n: int | None, seed: int, cfg: dict[str, Any]) -> DataSplits:
    """Build splits for `dataset` with `n` training rows (`None` = full pool).

    Test (cfg.data.test_size) and reference (cfg.data.reference_size) sets are fixed;
    train and a disjoint val set of size max(val_fraction * n, min_val_size) are drawn
    from the remaining pool with `seed`. The discretizer is fit on train only.

    Adult uses the official `adult.test` (16,281 rows) as its test set, ignoring
    cfg.data.test_size, so results are comparable with the literature; the reference
    set and the pool come from the official `adult.data`.

    For a fixed seed, smaller training sets are prefixes of larger ones, so NLL-vs-n
    curves compare nested subsets.
    """
    if seed == SPLIT_SEED:
        raise ValueError(f"seed {SPLIT_SEED} is reserved for the fixed test/reference sets")
    data_cfg = cfg["data"]
    n_test, n_fixed = data_cfg["test_size"], data_cfg["test_size"] + data_cfg["reference_size"]

    if dataset == "adult":
        raw = adult.load_adult(cfg["paths"]["data_dir"])
        official_train, test = raw.iloc[: adult.N_OFFICIAL_TRAIN], raw.iloc[adult.N_OFFICIAL_TRAIN :]
        n_ref = data_cfg["reference_size"]
        perm = np.random.default_rng(SPLIT_SEED).permutation(len(official_train))
        reference, pool = official_train.iloc[perm[:n_ref]], official_train.iloc[perm[n_ref:]]
        if n is None:
            fraction = data_cfg["val_fraction"]
            val_size = round(len(pool) * fraction / (1 + fraction))
            n = len(pool) - val_size
        else:
            val_size = _val_size(n, data_cfg)
        if n + val_size > len(pool):
            raise ValueError(f"adult: n={n} + val={val_size} exceeds the pool of {len(pool)} rows")
        idx = np.random.default_rng(seed).permutation(len(pool))
        train, val = pool.iloc[idx[:n]], pool.iloc[idx[n : n + val_size]]
        numerical = adult.NUMERICAL
        # Vocabularies are schema knowledge (the set of possible values), not fitted statistics.
        categories = {col: sorted(raw[col].unique()) for col in adult.CATEGORICAL}
        true_model = None

    elif dataset in BNLEARN_NETWORKS:
        if n is None:
            raise ValueError(f"{dataset}: bnlearn datasets need an explicit n")
        true_model = bnlearn.load_network(dataset)
        fixed = bnlearn.sample_network(true_model, n_fixed, seed=SPLIT_SEED)
        test, reference = fixed.iloc[:n_test], fixed.iloc[n_test:]
        rows = bnlearn.sample_network(true_model, n + _val_size(n, data_cfg), seed=seed)
        train, val = rows.iloc[:n], rows.iloc[n:]
        numerical = []
        categories = bnlearn.state_names(true_model)

    else:
        raise ValueError(f"unknown dataset: {dataset}")

    n_bins = data_cfg["n_bins_small"] if n <= data_cfg["small_n_threshold"] else data_cfg["n_bins"]
    disc = fit_discretizer(train, numerical, categories, n_bins)
    return DataSplits(
        train=disc.transform(train),
        val=disc.transform(val),
        test=disc.transform(test),
        reference=disc.transform(reference),
        columns=disc.columns,
        cardinalities=disc.cardinalities,
        true_model=true_model,
    )


def _val_size(n: int, data_cfg: dict[str, Any]) -> int:
    return max(round(data_cfg["val_fraction"] * n), data_cfg["min_val_size"])
