"""Stage 2 (RQ3): held-out NLL and membership-inference signals by BN-parent support.

For one (dataset, n, seed) of a Stage 1 grid, every cell (row, column) gets the support
s_{j,1}: the number of training rows that share its values on the column's learned-BN
parents (the `bn` run's graph). Training rows use leave-one-out counts (CLAUDE.md
invariant 5). Buckets are the fixed SUPPORT_BUCKETS and are the same for every variant,
so variants are compared on the same cells. A row's bucket uses min_j s_{j,1}.
`analyze_group(support="full_prefix")` buckets on full-prefix support s_{j,2} instead.

Per variant and bucket this gives the mean per-cell test and train NLL, and a
loss-based MI AUC: training rows (members) against test rows (non-members), scored by
-NLL of the whole row. Per (dataset, n, seed) it also gives the Spearman correlation
between a training row's min_j s_{j,1} and its vulnerability score.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.metrics import roc_auc_score

from tabstruct.data.splits import make_splits
from tabstruct.eval.compare import COMPARISONS, mean_ci, n_sort_key
from tabstruct.eval.nll import SUPPORT_BUCKETS, bucket_index
from tabstruct.eval.privacy import vulnerability_scores
from tabstruct.models.support import SupportTable
from tabstruct.structure.graphs import Graph

BUCKETS = list(SUPPORT_BUCKETS)


@dataclass
class RunGroup:
    """Every finished run of one (dataset, n, seed); `runs` maps variant -> run dir."""

    dataset: str
    n: str
    seed: int
    runs: dict[str, Path]


def find_groups(results_dir: str | Path, stage: str) -> list[RunGroup]:
    """Groups that have a finished `bn` run (it defines the support)."""
    groups = []
    for bn_dir in sorted(Path(results_dir, stage).glob("*/n*/bn/seed*")):
        if not (bn_dir / "metrics.json").exists():
            continue
        seed_tag, n_dir = bn_dir.name, bn_dir.parent.parent
        runs = {
            run.parent.name: run
            for run in sorted(n_dir.glob(f"*/{seed_tag}"))
            if (run / "metrics.json").exists() and (run / "per_column_nll.npy").exists()
        }
        groups.append(RunGroup(n_dir.parent.name, n_dir.name.removeprefix("n"),
                               int(seed_tag.removeprefix("seed")), runs))
    return sorted(groups, key=lambda g: (g.dataset, n_sort_key(g.n), g.seed))


def to_original_order(per_position: np.ndarray, order: list[int]) -> np.ndarray:
    """[N, d] in generation-position order -> [N, d] in original column order."""
    out = np.empty_like(per_position)
    out[:, order] = per_position
    return out


def bn_support(train: np.ndarray, test: np.ndarray, bn: Graph) -> tuple[np.ndarray, np.ndarray]:
    """Support under `bn`'s parent sets for training rows (leave-one-out) and test rows,
    both [N, d] in original column order; `train` and `test` are in original column
    order too."""
    table = SupportTable(bn.parents).fit(train[:, bn.order])
    s_train = table.counts(train[:, bn.order], leave_one_out=True)
    s_test = table.counts(test[:, bn.order])
    return to_original_order(s_train, bn.order), to_original_order(s_test, bn.order)


def loss_mi_auc(member_nll: np.ndarray, nonmember_nll: np.ndarray) -> float:
    """AUC of telling members from non-members by -NLL (0.5 = no signal)."""
    if not len(member_nll) or not len(nonmember_nll):
        return np.nan
    labels = np.r_[np.ones(len(member_nll)), np.zeros(len(nonmember_nll))]
    return float(roc_auc_score(labels, -np.r_[member_nll, nonmember_nll]))


def analyze_group(group: RunGroup, support: str = "bn", k: int = 5) -> dict[str, pd.DataFrame]:
    """Tables `cells` (NLL by bucket), `mi` (loss-based MI by row bucket) and
    `vulnerability` (support vs vulnerability) for one (dataset, n, seed).

    `support` picks the conditioning sets that define buckets: "bn" (s_{j,1}, the Stage 2
    protocol) or "full_prefix" (s_{j,2}, every earlier column in the BN's order; a
    secondary view, since the learned BN at small n is so sparse that almost every cell
    has high s_{j,1})."""
    bn_dir = group.runs["bn"]
    cfg = yaml.safe_load((bn_dir / "config.yaml").read_text())
    splits = make_splits(group.dataset, None if group.n == "full" else int(group.n), group.seed, cfg)
    train, test = splits.train.to_numpy(), splits.test.to_numpy()
    bn = Graph.from_dict(json.loads((bn_dir / "graph.json").read_text()))
    if support == "full_prefix":
        bn = Graph.full_prefix(bn.order)
    elif support != "bn":
        raise ValueError(f"support must be bn or full_prefix, got {support!r}")
    s_train, s_test = bn_support(train, test, bn)
    cell_train, cell_test = bucket_index(s_train), bucket_index(s_test)
    row_train, row_test = bucket_index(s_train.min(1)), bucket_index(s_test.min(1))
    key = {"dataset": group.dataset, "n": group.n, "seed": group.seed}

    cells, mi = [], []
    for variant, run in group.runs.items():
        graph = Graph.from_dict(json.loads((run / "graph.json").read_text()))
        nll_test = to_original_order(np.load(run / "per_column_nll.npy"), graph.order)
        if nll_test.shape != test.shape:
            raise ValueError(f"{run}: per_column_nll.npy {nll_test.shape}, test is {test.shape}")
        train_file = run / "train_per_column_nll.npy"
        nll_train = to_original_order(np.load(train_file), graph.order) if train_file.exists() else None
        if nll_train is not None and nll_train.shape != train.shape:
            raise ValueError(f"{run}: train_per_column_nll.npy {nll_train.shape}, train is {train.shape}")
        for i, bucket in enumerate(BUCKETS):
            test_cells = nll_test[cell_test == i]
            row = {**key, "variant": variant, "bucket": bucket,
                   "nll_test": test_cells.mean() if len(test_cells) else np.nan,
                   "count_test": len(test_cells)}
            if nll_train is not None:
                train_cells = nll_train[cell_train == i]
                row["nll_train"] = train_cells.mean() if len(train_cells) else np.nan
                row["count_train"] = len(train_cells)
                mi.append({**key, "variant": variant, "bucket": bucket,
                           "auc": loss_mi_auc(nll_train[row_train == i].sum(1),
                                              nll_test[row_test == i].sum(1)),
                           "members": int((row_train == i).sum()),
                           "nonmembers": int((row_test == i).sum())})
            cells.append(row)

    vuln = vulnerability_scores(splits.train, k=k)
    rho, p = stats.spearmanr(s_train.min(1), vuln)
    vulnerability = {**key, "spearman_rho": rho, "spearman_p": p, "rows": len(train)}
    for i, bucket in enumerate(BUCKETS):
        in_bucket = row_train == i
        vulnerability[f"vulnerability_{bucket}"] = vuln[in_bucket].mean() if in_bucket.any() else np.nan
        vulnerability[f"rows_{bucket}"] = int(in_bucket.sum())
    return {"cells": pd.DataFrame(cells), "mi": pd.DataFrame(mi),
            "vulnerability": pd.DataFrame([vulnerability])}


def average_random_dags(df: pd.DataFrame) -> pd.DataFrame:
    """random_dag_<k> -> one `random_dag` row per (dataset, n, seed, bucket): the mean."""
    df = df.assign(variant=df["variant"].str.replace(r"^random_dag_\d+$", "random_dag", regex=True))
    keys = ["dataset", "n", "seed", "variant", "bucket"]
    return df.groupby(keys, as_index=False, sort=False).mean(numeric_only=True)


def bucket_comparisons(cells: pd.DataFrame, value: str = "nll_test") -> pd.DataFrame:
    """COMPARISONS (a - b) per (dataset, n, bucket), paired by seed: mean, 95% CI, and
    seeds where a is lower."""
    wide = average_random_dags(cells).pivot_table(
        index=["dataset", "n", "bucket", "seed"], columns="variant", values=value
    )
    rows = []
    for (dataset, n, bucket), group in wide.groupby(level=["dataset", "n", "bucket"]):
        for name, a, b in COMPARISONS:
            if a not in group or b not in group:
                continue
            diff = (group[a] - group[b]).dropna().to_numpy()
            if not len(diff):
                continue
            mean, ci = mean_ci(diff)
            rows.append({"comparison": name, "dataset": dataset, "n": n, "bucket": bucket,
                         "mean_diff": mean, "ci95": ci, "n_seeds": len(diff),
                         "a_better": int((diff < 0).sum())})
    return _sorted(pd.DataFrame(rows), ["comparison"])


def over_seeds(df: pd.DataFrame, value: str, by: list[str]) -> pd.DataFrame:
    """Mean and 95% CI of `value` over seeds for every combination of `by`."""
    rows = []
    for keys, group in df.groupby(by, sort=False):
        mean, ci = mean_ci(group[value].to_numpy())
        rows.append({**dict(zip(by, keys)), "mean": mean, "ci95": ci,
                     "n_seeds": int(group[value].notna().sum())})
    return _sorted(pd.DataFrame(rows), [c for c in by if c not in ("dataset", "n", "bucket")])


def _sorted(df: pd.DataFrame, first: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    order = [name for name, _, _ in COMPARISONS]

    def key(col: pd.Series) -> pd.Series:
        if col.name == "n":
            return col.map(n_sort_key)
        if col.name == "bucket":
            return col.map(BUCKETS.index)
        if col.name == "comparison":
            return col.map(order.index)
        return col

    cols = ["dataset"] + first + ["n"] + (["bucket"] if "bucket" in df else [])
    return df.sort_values(cols, key=key).reset_index(drop=True)
