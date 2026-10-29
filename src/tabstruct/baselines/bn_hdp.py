"""BN with hierarchical (HDP-style) CPT smoothing (Petitjean et al., 2018).

If wrapping the reference Java code is too costly, implement hierarchical back-off of
each CPT row toward parent-subset/marginal estimates and document the substitution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tabstruct.structure.graphs import Graph


class BNHDPGenerator:
    def __init__(self, graph: Graph):
        self.graph = graph

    def fit(self, train: pd.DataFrame, cardinalities: list[int]) -> BNHDPGenerator:
        raise NotImplementedError

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        raise NotImplementedError

    def log_prob(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError
