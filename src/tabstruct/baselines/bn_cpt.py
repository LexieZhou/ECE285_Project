"""BN with smoothed CPTs (Laplace / BDeu via pgmpy `BayesianEstimator`)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tabstruct.structure.graphs import Graph


class BNCPTGenerator:
    def __init__(self, graph: Graph, prior: str = "BDeu", equivalent_sample_size: float = 1.0):
        self.graph = graph
        self.prior = prior
        self.equivalent_sample_size = equivalent_sample_size

    def fit(self, train: pd.DataFrame, cardinalities: list[int]) -> BNCPTGenerator:
        raise NotImplementedError

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        raise NotImplementedError

    def log_prob(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError
