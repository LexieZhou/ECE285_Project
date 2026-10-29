"""Discretization of raw tables into integer codes. Fit on the training split only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Discretizer:
    columns: list[str]
    bin_edges: dict[str, np.ndarray]  # numerical column -> interior quantile edges
    categories: dict[str, list]  # categorical column -> full vocabulary from the schema

    @property
    def cardinalities(self) -> list[int]:
        return [
            len(self.bin_edges[c]) + 1 if c in self.bin_edges else len(self.categories[c])
            for c in self.columns
        ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Integer-code `df` column by column.

        Numerical values outside the training range fall into the first/last bin (outer
        edges are ±inf), so held-out rows never go out of range.
        """
        codes = {}
        for col in self.columns:
            if col in self.bin_edges:
                codes[col] = np.searchsorted(self.bin_edges[col], df[col].to_numpy(), side="right")
            else:
                index = {value: i for i, value in enumerate(self.categories[col])}
                unknown = set(df[col].unique()) - index.keys()
                if unknown:
                    raise ValueError(f"{col}: values outside the schema vocabulary: {unknown}")
                codes[col] = df[col].map(index).to_numpy()
        return pd.DataFrame(codes, index=df.index).astype(np.int64)


def fit_discretizer(
    train: pd.DataFrame, numerical: list[str], categories: dict[str, list], n_bins: int
) -> Discretizer:
    """Quantile-bin `numerical` columns on `train` (duplicate edges merged); other
    columns use the vocabularies in `categories`. Column order follows `train`."""
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    bin_edges = {col: np.unique(np.quantile(train[col].to_numpy(), quantiles)) for col in numerical}
    return Discretizer(
        columns=list(train.columns),
        bin_edges=bin_edges,
        categories={col: list(categories[col]) for col in train.columns if col not in bin_edges},
    )
