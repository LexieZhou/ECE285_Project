"""TVAE baseline (SDV `TVAESynthesizer`) on the discretized table, default hyperparameters.

Requires the `baselines` extra: pip install -e ".[baselines]".
"""

from __future__ import annotations

import pandas as pd


class TVAEGenerator:
    def __init__(self, epochs: int = 300):
        self.epochs = epochs

    def fit(self, train: pd.DataFrame, cardinalities: list[int]) -> TVAEGenerator:
        raise NotImplementedError

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        raise NotImplementedError
