"""Held-out NLL summaries and per-support-bucket breakdowns."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Lower edges of the fixed support buckets: zero = 0, low = 1-4, medium = 5-19, high >= 20.
SUPPORT_BUCKETS = {"zero": 0, "low": 1, "medium": 5, "high": 20}


def summarize_nll(per_column_nll: np.ndarray) -> dict[str, float]:
    """per_column_nll [N, d] -> {'nll': nats/row, 'nll_col_<j>': mean per column}."""
    raise NotImplementedError


def bucket_nll(per_column_nll: np.ndarray, support: np.ndarray) -> pd.DataFrame:
    """Mean NLL and cell count per support bucket; both inputs [N, d]."""
    raise NotImplementedError
