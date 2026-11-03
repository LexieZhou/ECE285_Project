import numpy as np
import pandas as pd
from scipy import stats

from tabstruct.eval.compare import mean_ci, paired_comparisons, per_seed, variant_table

# NLL per variant; seed s adds s * 0.1 to every variant so that pairing matters.
BASE = {"full_prefix": 13.0, "bn": 12.5, "bn_cpt": 12.6}
RANDOM = [12.8, 12.9, 13.0, 13.1, 13.2]  # random_dag_0..4, mean 13.0


def fake_summary(seeds=range(5)) -> pd.DataFrame:
    rows = []
    for n in [500, "full"]:
        for seed in seeds:
            nll = {**BASE, **{f"random_dag_{k}": v for k, v in enumerate(RANDOM)}}
            for variant, value in nll.items():
                rows.append({"stage": "stage1", "dataset": "adult", "n": n, "variant": variant,
                             "seed": seed, "nll_test": value + 0.1 * seed})
    return pd.DataFrame(rows)


def test_random_dags_are_averaged_per_seed():
    wide = per_seed(fake_summary())
    assert "random_dag" in wide and "random_dag_0" not in wide
    np.testing.assert_allclose(wide.loc[("adult", "500", 0), "random_dag"], 13.0)


def test_paired_differences_cancel_seed_effects():
    comp = paired_comparisons(fake_summary())
    rq1 = comp[(comp["comparison"].str.startswith("RQ1")) & (comp["n"] == "500")].iloc[0]
    assert np.isclose(rq1["mean_diff"], -0.5) and np.isclose(rq1["ci95"], 0.0)  # constant diff
    assert rq1["a_better"] == 5 and rq1["n_seeds"] == 5
    # Comparisons whose variants did not run (no bnlearn variants here) are left out.
    assert not comp["comparison"].str.contains("true_dag").any()
    assert list(comp["n"].unique()) == ["500", "full"]  # numeric order, full last


def test_variant_table_ci_matches_t_interval():
    table = variant_table(fake_summary())
    row = table[(table["variant"] == "bn") & (table["n"] == "500")].iloc[0]
    values = 12.5 + 0.1 * np.arange(5)
    half = stats.t.ppf(0.975, 4) * values.std(ddof=1) / np.sqrt(5)
    assert np.isclose(row["mean"], values.mean()) and np.isclose(row["ci95"], half)
    assert table["variant"].iloc[0] == "full_prefix"  # VARIANT_ORDER, not alphabetical


def test_mean_ci_single_value():
    mean, ci = mean_ci(np.array([1.0]))
    assert mean == 1.0 and np.isnan(ci)
