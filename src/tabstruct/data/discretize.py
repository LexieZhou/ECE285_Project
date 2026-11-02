"""Discretization of raw tables into integer codes.

Adult bin edges are fit once on a fixed binning set (see `splits.make_splits`), never on
test or reference rows.
"""

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


def quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Interior edges for `n_bins` quantile bins (duplicate edges merged).

    Bin k holds e_{k-1} <= v < e_k. If more than 1/n_bins of the values equal the minimum
    (e.g. capital-gain, ~92% zeros), plain quantiles put every edge on that minimum and
    the column becomes constant. Instead the minimum gets bin 0 to itself and the values
    above it are quantile-binned into n_bins - 1 bins.
    """
    values = values.astype(float)
    lo = values.min()
    if np.mean(values == lo) <= 1 / n_bins:
        return np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1]))
    rest = values[values > lo]
    inner = np.quantile(rest, np.linspace(0, 1, n_bins)[1:-1]) if len(rest) else []
    return np.unique(np.concatenate([[np.nextafter(lo, np.inf)], inner]))


def fit_discretizer(
    data: pd.DataFrame, numerical: list[str], categories: dict[str, list], n_bins: int
) -> Discretizer:
    """Quantile-bin `numerical` columns on `data` (see `quantile_edges`); other columns
    use the vocabularies in `categories`. Column order follows `data`."""
    bin_edges = {col: quantile_edges(data[col].to_numpy(), n_bins) for col in numerical}
    return Discretizer(
        columns=list(data.columns),
        bin_edges=bin_edges,
        categories={col: list(categories[col]) for col in data.columns if col not in bin_edges},
    )
