"""Stage 1 (RQ1-2) summaries over seeds: per-variant NLL and paired variant differences.

Input is `results/summary.csv` (one row per run, from scripts/aggregate.py). The five
`random_dag_<k>` variants are averaged within each seed into one `random_dag` value
before any comparison, so a seed counts once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# (name, a, b): rows report a - b, so a negative difference means a has lower NLL.
COMPARISONS = [
    ("RQ1: bn - full_prefix", "bn", "full_prefix"),
    ("RQ2: bn - random_dag", "bn", "random_dag"),
    ("sparsity: random_dag - full_prefix", "random_dag", "full_prefix"),
    ("bn - true_dag", "bn", "true_dag"),
    ("order: true_dag - full_prefix_trueorder", "true_dag", "full_prefix_trueorder"),
    ("neural vs counts: bn - bn_cpt", "bn", "bn_cpt"),
    ("bn_hdp - bn_cpt", "bn_hdp", "bn_cpt"),
]


def n_sort_key(n: str) -> float:
    return np.inf if n == "full" else float(n)


def per_seed(summary: pd.DataFrame, metric: str = "nll_test") -> pd.DataFrame:
    """Rows (dataset, n, seed), one column per variant; random_dag_<k> averaged."""
    s = summary.assign(
        n=summary["n"].astype(str),
        variant=summary["variant"].str.replace(r"^random_dag_\d+$", "random_dag", regex=True),
    )
    return s.pivot_table(index=["dataset", "n", "seed"], columns="variant", values=metric, aggfunc="mean")


def mean_ci(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Mean and half-width of the t confidence interval (nan for fewer than 2 values)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return (float(values.mean()) if len(values) else np.nan), np.nan
    sem = values.std(ddof=1) / np.sqrt(len(values))
    return float(values.mean()), float(stats.t.ppf(0.5 + level / 2, len(values) - 1) * sem)


def variant_table(summary: pd.DataFrame, metric: str = "nll_test") -> pd.DataFrame:
    """Mean and 95% CI of `metric` over seeds, per (dataset, n, variant)."""
    wide = per_seed(summary, metric)
    rows = []
    for (dataset, n), group in wide.groupby(level=["dataset", "n"]):
        for variant in group.columns:
            values = group[variant].dropna().to_numpy()
            if len(values):
                mean, ci = mean_ci(values)
                rows.append({"dataset": dataset, "n": n, "variant": variant,
                             "mean": mean, "ci95": ci, "n_seeds": len(values)})
    return _sorted(pd.DataFrame(rows), ["variant"])


def paired_comparisons(summary: pd.DataFrame, metric: str = "nll_test") -> pd.DataFrame:
    """Every comparison in COMPARISONS, per (dataset, n), paired by seed.

    `a_better` counts seeds where a has lower `metric`. The Wilcoxon signed-rank p-value
    is two-sided; with 5 seeds it cannot go below 0.0625, so read it alongside the CI
    and `a_better`.
    """
    wide = per_seed(summary, metric)
    rows = []
    for (dataset, n), group in wide.groupby(level=["dataset", "n"]):
        for name, a, b in COMPARISONS:
            if a not in group or b not in group:
                continue
            diff = (group[a] - group[b]).dropna().to_numpy()
            if not len(diff):
                continue
            mean, ci = mean_ci(diff)
            p = stats.wilcoxon(diff).pvalue if len(diff) > 1 and np.any(diff != 0) else np.nan
            rows.append({"comparison": name, "dataset": dataset, "n": n, "mean_diff": mean,
                         "ci95": ci, "n_seeds": len(diff), "a_better": int((diff < 0).sum()),
                         "wilcoxon_p": p})
    return _sorted(pd.DataFrame(rows), ["comparison"])


VARIANT_ORDER = ["marginal", "full_prefix", "bn", "random_dag", "true_dag",
                 "full_prefix_trueorder", "bn_cpt", "bn_hdp"]


def _sort_key(col: pd.Series) -> pd.Series:
    """Tables follow COMPARISONS / VARIANT_ORDER and increasing n, not alphabetical order."""
    if col.name == "n":
        return col.map(n_sort_key)
    if col.name == "comparison":
        return col.map([name for name, _, _ in COMPARISONS].index)
    if col.name == "variant":
        return col.map(lambda v: VARIANT_ORDER.index(v) if v in VARIANT_ORDER else len(VARIANT_ORDER))
    return col


def _sorted(df: pd.DataFrame, first: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_values(["dataset"] + first + ["n"], key=_sort_key).reset_index(drop=True)
