"""UCI Adult loader (ucimlrepo id=2).

Returns the raw table only: drops `fnlwgt`, keeps `?` as its own category. Binning of
numerical columns lives in `discretize.py` and is fit on the training split.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET = "income"
NUMERICAL = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    "income",
]


def load_adult(cache_dir: str | Path) -> pd.DataFrame:
    """Return the full Adult table (~48k rows), caching a CSV under `cache_dir`.

    Normalizes income labels ('>50K.' -> '>50K') and maps missing values (NaN or '?') to '?'.
    """
    path = Path(cache_dir) / "adult.csv"
    if path.exists():
        return pd.read_csv(path, dtype={c: str for c in CATEGORICAL}, keep_default_na=False)

    from ucimlrepo import fetch_ucirepo

    raw = fetch_ucirepo(id=2).data
    df = pd.concat([raw.features, raw.targets], axis=1).drop(columns="fnlwgt")
    for col in CATEGORICAL:
        df[col] = df[col].fillna("?").astype(str).str.strip()
    df[TARGET] = df[TARGET].str.rstrip(".")
    df = df[NUMERICAL + CATEGORICAL].reset_index(drop=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return load_adult(cache_dir)  # same dtypes on first and cached loads
