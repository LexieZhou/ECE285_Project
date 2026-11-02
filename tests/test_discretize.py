import numpy as np
import pandas as pd

from tabstruct.data.discretize import fit_discretizer


def test_codes_in_range_and_extremes_clipped():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"age": rng.normal(40, 10, 200), "sex": rng.choice(["F", "M"], 200)})
    held_out = pd.DataFrame({"age": [-1000.0, 40.0, 1000.0], "sex": ["F", "M", "F"]})

    disc = fit_discretizer(train, numerical=["age"], categories={"sex": ["F", "M"]}, n_bins=5)
    codes = disc.transform(held_out)

    assert list(codes.columns) == ["age", "sex"]
    assert codes["age"].iloc[0] == 0
    assert codes["age"].iloc[2] == disc.cardinalities[0] - 1
    for col, card in zip(codes.columns, disc.cardinalities):
        assert codes[col].between(0, card - 1).all()


def test_spike_at_minimum_gets_its_own_bin():
    # Zero-inflated like capital-gain: plain quantiles would make this column constant.
    rng = np.random.default_rng(0)
    gain = np.where(rng.random(1000) < 0.9, 0, rng.integers(100, 20000, 1000))
    train = pd.DataFrame({"gain": gain})
    disc = fit_discretizer(train, numerical=["gain"], categories={}, n_bins=5)
    codes = disc.transform(pd.DataFrame({"gain": [0, 1, 150, 19000]}))["gain"]

    assert disc.cardinalities == [5]
    assert codes.iloc[0] == 0 and (codes.iloc[1:] > 0).all()  # zero vs any positive value
    train_codes = disc.transform(train)["gain"]
    assert (train_codes == 0).sum() == (gain == 0).sum()
    assert train_codes[gain > 0].nunique() == 4  # positives spread over the other bins


def test_spike_without_other_values_still_separates_them():
    train = pd.DataFrame({"gain": np.zeros(50)})
    disc = fit_discretizer(train, numerical=["gain"], categories={}, n_bins=5)
    assert disc.transform(pd.DataFrame({"gain": [0, 7]}))["gain"].tolist() == [0, 1]
