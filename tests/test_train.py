"""Graph construction per variant (invariant 3) and an end-to-end training run."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tabstruct.data.splits import DataSplits
from tabstruct.train import build_graph, fit_and_evaluate, hparam_candidates
from tabstruct.utils import deep_update, load_config, resolve_device

ROOT = Path(__file__).resolve().parents[1]
CFG = deep_update(
    load_config(ROOT / "configs/default.yaml"),
    {
        "model": {"dim": 16, "n_heads": 2, "n_layers": 1},
        "train": {"device": "cpu", "max_epochs": 40, "patience": 5, "hparam_grid": None},
    },
)


def chain_table(n: int, seed: int) -> pd.DataFrame:
    """x0 -> x1 -> x2 with strong dependence; x3 independent."""
    rng = np.random.default_rng(seed)
    x0 = rng.integers(0, 3, n)
    x1 = np.where(rng.random(n) < 0.9, x0, rng.integers(0, 3, n))
    x2 = np.where(rng.random(n) < 0.9, x1, rng.integers(0, 3, n))
    x3 = rng.integers(0, 2, n)
    return pd.DataFrame({"a": x0, "b": x1, "c": x2, "d": x3})


class FakeTrueModel:
    def edges(self):
        return [("a", "b"), ("b", "c")]


def chain_splits(true_model=None) -> DataSplits:
    return DataSplits(
        train=chain_table(1000, 1),
        val=chain_table(300, 2),
        test=chain_table(500, 3),
        reference=chain_table(10, 4),
        columns=list("abcd"),
        cardinalities=[3, 3, 3, 2],
        true_model=true_model,
    )


def test_variants_share_the_bn_order():
    splits = chain_splits()
    graphs = {
        v: build_graph(v, splits, CFG, np.random.default_rng(0))
        for v in ["marginal", "full_prefix", "bn", *(f"random_dag_{k}" for k in range(5))]
    }
    bn = graphs["bn"]
    assert all(g.order == bn.order for g in graphs.values())
    assert graphs["full_prefix"].parents == [list(range(j)) for j in range(4)]
    assert graphs["marginal"].parents == [[]] * 4
    for k in range(5):
        assert graphs[f"random_dag_{k}"].in_degrees() == bn.in_degrees()


def test_random_dag_depends_only_on_seed_and_k():
    splits = chain_splits()
    a = build_graph("random_dag_3", splits, CFG, np.random.default_rng(7))
    b = build_graph("random_dag_3", splits, CFG, np.random.default_rng(7))
    assert a == b
    with pytest.raises(ValueError):
        build_graph("random_dag_5", splits, CFG, np.random.default_rng(7))


def test_true_dag_variants_use_the_true_order():
    splits = chain_splits(FakeTrueModel())
    true = build_graph("true_dag", splits, CFG, np.random.default_rng(0))
    control = build_graph("full_prefix_trueorder", splits, CFG, np.random.default_rng(0))
    position = {col: j for j, col in enumerate(true.order)}
    assert true.parents[position[1]] == [position[0]]
    assert true.parents[position[2]] == [position[1]]
    assert true.parents[position[3]] == []
    assert control.order == true.order
    assert control.parents == [list(range(j)) for j in range(4)]
    with pytest.raises(ValueError):
        build_graph("true_dag", chain_splits(), CFG, np.random.default_rng(0))


def test_fit_and_evaluate_end_to_end(tmp_path):
    splits = chain_splits()
    results = {}
    for variant in ["marginal", "bn"]:
        out = tmp_path / variant
        out.mkdir()
        graph = build_graph(variant, splits, CFG, np.random.default_rng(0))
        results[variant] = fit_and_evaluate(variant, splits, graph, CFG, out, seed=0)

        nll = np.load(out / "per_column_nll.npy")
        assert nll.shape == (len(splits.test), 4)
        assert np.isclose(nll.sum(1).mean(), results[variant]["nll_test"])
        train_nll = np.load(out / "train_per_column_nll.npy")
        assert train_nll.shape == (len(splits.train), 4)
        assert np.isclose(train_nll.sum(1).mean(), results[variant]["nll_train"])
        samples = pd.read_csv(out / "samples.csv")
        assert list(samples.columns) == splits.columns and len(samples) == len(splits.train)
        for col, card in zip(splits.columns, splits.cardinalities):
            assert samples[col].between(0, card - 1).all()

    # Invariant 2: variants differ only in masks, so parameter counts match.
    assert results["bn"]["n_params"] == results["marginal"]["n_params"]
    # The chain carries ~1.5 nats/row of dependence; BN conditioning must capture it.
    assert results["bn"]["nll_test"] < results["marginal"]["nll_test"] - 0.5
    # Samples follow the chain: x1 copies x0 far more often than chance (1/3).
    samples = pd.read_csv(tmp_path / "bn" / "samples.csv")
    assert (samples["a"] == samples["b"]).mean() > 0.8


def test_unimplemented_variants_raise():
    splits = chain_splits()
    graph = build_graph("scsa", splits, CFG, np.random.default_rng(0))
    with pytest.raises(NotImplementedError):
        fit_and_evaluate("scsa", splits, graph, CFG, Path("unused"), seed=0)


def test_bn_hdp_end_to_end(tmp_path):
    splits = chain_splits()
    graph = build_graph("bn_hdp", splits, CFG, np.random.default_rng(0))
    metrics = fit_and_evaluate("bn_hdp", splits, graph, CFG, tmp_path, seed=0)

    nll = np.load(tmp_path / "per_column_nll.npy")
    assert nll.shape == (len(splits.test), 4)
    assert np.isclose(nll.sum(1).mean(), metrics["nll_test"])
    assert set(metrics["hparams"]["concentrations"]) == set(splits.columns)
    assert metrics["nll_val"] == min(h["nll_val"] for h in metrics["hparam_search"])
    assert metrics["nll_test"] < 2.6  # same bound as bn_cpt: true entropy ~2.4 nats/row


def test_bn_cpt_end_to_end(tmp_path):
    splits = chain_splits()
    graph = build_graph("bn_cpt", splits, CFG, np.random.default_rng(0))
    metrics = fit_and_evaluate("bn_cpt", splits, graph, CFG, tmp_path, seed=0)

    nll = np.load(tmp_path / "per_column_nll.npy")
    assert nll.shape == (len(splits.test), 4)
    assert np.isclose(nll.sum(1).mean(), metrics["nll_test"])
    train_nll = np.load(tmp_path / "train_per_column_nll.npy")
    assert np.isclose(train_nll.sum(1).mean(), metrics["nll_train"])
    assert len(metrics["hparam_search"]) == len(CFG["baselines"]["bn_cpt"]["smoothing_grid"])
    assert metrics["nll_val"] == min(h["nll_val"] for h in metrics["hparam_search"])
    # True entropy of the chain is ~2.4 nats/row (marginal ~4.0); counts on 1k rows get close.
    assert metrics["nll_test"] < 2.6
    samples = pd.read_csv(tmp_path / "samples.csv")
    assert list(samples.columns) == splits.columns and (samples["a"] == samples["b"]).mean() > 0.8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fit_and_evaluate_on_cuda(tmp_path):
    splits = chain_splits()
    graph = build_graph("bn", splits, CFG, np.random.default_rng(0))
    cfg = deep_update(CFG, {"train": {"device": "cuda"}})
    gpu = fit_and_evaluate("bn", splits, graph, cfg, tmp_path, seed=0)
    assert gpu["device"] == "cuda"
    assert gpu["nll_test"] < 3.8  # marginal is ~4.0 nats/row on this chain


def test_resolve_device():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type == ("cuda" if torch.cuda.is_available() else "cpu")
    with pytest.raises(ValueError):
        resolve_device("mps")


def test_hparam_candidates_reuse_the_hparam_seed_choice(tmp_path):
    grid = {"lr": [1e-3, 3e-4], "weight_decay": [0.0, 1e-2], "dropout": [0.1, 0.3]}
    cfg = deep_update(CFG, {"train": {"hparam_grid": grid, "hparam_seed": 0}})
    seed0, seed1 = tmp_path / "seed0", tmp_path / "seed1"

    assert hparam_candidates(CFG, 1, seed1) == ([{"lr": 1e-3, "weight_decay": 0.0, "dropout": 0.1}], "default")
    candidates, source = hparam_candidates(cfg, 0, seed0)
    assert len(candidates) == 8 and source == "search"
    with pytest.raises(FileNotFoundError):  # seed 0 has not run yet
        hparam_candidates(cfg, 1, seed1)

    seed0.mkdir()
    chosen = {"lr": 3e-4, "weight_decay": 1e-2, "dropout": 0.3}
    (seed0 / "metrics.json").write_text(json.dumps({"hparams": chosen, "hparams_from": "default"}))
    with pytest.raises(ValueError):  # seed 0 ran without the grid
        hparam_candidates(cfg, 1, seed1)
    (seed0 / "metrics.json").write_text(json.dumps({"hparams": chosen, "hparams_from": "search"}))
    assert hparam_candidates(cfg, 1, seed1) == ([chosen], "seed0")
