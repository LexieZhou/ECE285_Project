"""UCI Adult loader (ucimlrepo id=2).

Returns the raw table only: drops `fnlwgt`, keeps `?` as its own category. Binning of
numerical columns lives in `discretize.py` and is fit on the training split.

Rows keep the ucimlrepo order: the first `N_OFFICIAL_TRAIN` rows are `adult.data`, the
rest are the official `adult.test` (16,281 rows).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET = "income"
N_OFFICIAL_TRAIN = 32561
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
    # Official test labels end in '.'; check they are exactly the tail before stripping it.
    is_test = df[TARGET].str.endswith(".").to_numpy()
    if is_test[:N_OFFICIAL_TRAIN].any() or not is_test[N_OFFICIAL_TRAIN:].all():
        raise ValueError("adult: official test rows are not the tail of the ucimlrepo table")
    df[TARGET] = df[TARGET].str.rstrip(".")
    df = df[NUMERICAL + CATEGORICAL].reset_index(drop=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return load_adult(cache_dir)  # same dtypes on first and cached loads
