"""SMOTE-style unsupervised generator: interpolate a row with one of its k nearest
neighbours (one-hot space); each column takes either endpoint's value."""

from __future__ import annotations

import pandas as pd


class SMOTEGenerator:
    def __init__(self, k: int = 5):
        self.k = k

    def fit(self, train: pd.DataFrame, cardinalities: list[int]) -> SMOTEGenerator:
        raise NotImplementedError

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        raise NotImplementedError
