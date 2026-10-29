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
