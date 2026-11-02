import numpy as np

from tabstruct.eval.nll import bucket_nll, summarize_nll


def test_summarize_nll_is_nats_per_row():
    nll = np.array([[1.0, 2.0], [3.0, 4.0]])
    summary = summarize_nll(nll, ["b", "a"])
    assert summary["nll"] == 5.0  # sum over columns, mean over rows
    assert summary["per_column"] == {"b": 2.0, "a": 3.0}


def test_bucket_nll_edges():
    support = np.array([[0, 1, 4, 5, 19, 20, 100]])
    nll = np.arange(7, dtype=float)[None]
    table = bucket_nll(nll, support).set_index("bucket")
    assert table["count"].to_dict() == {"zero": 1, "low": 2, "medium": 2, "high": 2}
    assert table["nll"].to_dict() == {"zero": 0.0, "low": 1.5, "medium": 3.5, "high": 5.5}


def test_bucket_nll_empty_bucket_is_nan():
    table = bucket_nll(np.ones((2, 1)), np.full((2, 1), 30)).set_index("bucket")
    assert table.loc["zero", "count"] == 0 and np.isnan(table.loc["zero", "nll"])
