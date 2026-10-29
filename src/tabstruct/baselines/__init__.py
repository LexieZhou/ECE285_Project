"""Baseline generators. All fit and sample in the same discretized integer space."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class Generator(Protocol):
    def fit(self, train: pd.DataFrame, cardinalities: list[int]) -> Generator: ...

    def sample(self, n: int, seed: int) -> pd.DataFrame: ...


class LikelihoodGenerator(Generator, Protocol):
    def log_prob(self, df: pd.DataFrame) -> np.ndarray:
        """Per-column log-likelihood, shape [N, d]."""
        ...
