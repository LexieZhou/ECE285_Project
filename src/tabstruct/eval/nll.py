"""Held-out NLL summaries and per-support-bucket breakdowns."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Lower edges of the fixed support buckets: zero = 0, low = 1-4, medium = 5-19, high >= 20.
SUPPORT_BUCKETS = {"zero": 0, "low": 1, "medium": 5, "high": 20}


def summarize_nll(per_column_nll: np.ndarray, columns: list[str]) -> dict[str, object]:
    """per_column_nll [N, d] in position order, `columns` the names in that order ->
    {'nll': nats/row, 'per_column': {name: mean NLL}}.

    Per-column values are keyed by name because positions differ across seeds.
    """
    return {
        "nll": float(per_column_nll.sum(axis=1).mean()),
        "per_column": dict(zip(columns, per_column_nll.mean(axis=0).tolist())),
    }


def bucket_nll(per_column_nll: np.ndarray, support: np.ndarray) -> pd.DataFrame:
    """Mean NLL and cell count per support bucket; both inputs [N, d]."""
    edges = list(SUPPORT_BUCKETS.values())
    bucket = np.digitize(support.ravel(), edges[1:])  # 0 = zero, ..., 3 = high
    nll = per_column_nll.ravel()
    rows = []
    for i, name in enumerate(SUPPORT_BUCKETS):
        cells = nll[bucket == i]
        rows.append({"bucket": name, "nll": cells.mean() if len(cells) else np.nan, "count": len(cells)})
    return pd.DataFrame(rows)
