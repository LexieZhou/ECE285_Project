"""Single-run entry point: one (dataset, n, variant, seed) -> one results directory.

    python -m tabstruct.train --config configs/default.yaml \
        --dataset adult --n 500 --variant bn --seed 0

Writes config.yaml, graph.json, metrics.json, per_column_nll.npy and samples.csv to
results/<stage>/<dataset>/n<n>/<variant>/seed<s>/.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import itertools
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from tabstruct.data import DATASETS, bnlearn
from tabstruct.data.splits import DataSplits, make_splits
from tabstruct.eval.nll import summarize_nll
from tabstruct.models.masked_transformer import MaskedTabularTransformer
from tabstruct.models.masks import build_attention_mask
from tabstruct.models.sampler import ancestral_sample
from tabstruct.structure.graphs import Graph
from tabstruct.structure.learn_bn import graph_from_dag, learn_bn
from tabstruct.structure.random_dag import degree_matched_random_dag
from tabstruct.utils import (
    load_config,
    resolve_device,
    run_dir,
    save_json,
    save_yaml,
    set_seed,
)

NEURAL_VARIANTS = ("marginal", "full_prefix", "bn", "true_dag", "full_prefix_trueorder")
RANDOM_DAG = re.compile(r"random_dag_(\d+)")
# Variants that are not neural graph variants but still use the learned BN's graph/order.
BN_GRAPH_VARIANTS = ("bn_cpt", "bn_hdp", "smote", "tvae")


def parse_n(value: str) -> int | None:
    """`--n full` means the whole training pool."""
    return None if value == "full" else int(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--n", required=True, type=parse_n, help="training rows, or 'full'")
    parser.add_argument("--variant", required=True, help="see CLAUDE.md for variant names")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--stage", default="stage1")
    return parser.parse_args(argv)


def build_graph(
    variant: str, splits: DataSplits, cfg: dict[str, Any], rng: np.random.Generator
) -> Graph:
    """Map a variant name to its conditioning graph.

    `bn` and `random_dag_<k>` share the learned BN's topological order; `true_dag` and
    `full_prefix_trueorder` use the true DAG's order (bnlearn only). `rng` must be fresh
    from `set_seed`: random DAG k comes from its k-th spawned stream, so it does not
    depend on which other variants ran.
    """
    if variant in ("true_dag", "full_prefix_trueorder"):
        if splits.true_model is None:
            raise ValueError(f"{variant} needs a dataset with a known DAG")
        index = {col: i for i, col in enumerate(splits.columns)}
        edges = [(index[u], index[v]) for u, v in splits.true_model.edges()]
        true = graph_from_dag(edges, len(splits.columns), name="true_dag")
        if variant == "true_dag":
            return true
        return dataclasses.replace(Graph.full_prefix(true.order), name=variant)

    bn = learn_bn(splits.train, max_indegree=cfg["structure"]["max_indegree"])
    if variant == "marginal":
        return Graph.marginal(bn.order)
    if variant == "full_prefix":
        return Graph.full_prefix(bn.order)
    if match := RANDOM_DAG.fullmatch(variant):
        k, n_dags = int(match.group(1)), cfg["structure"]["n_random_dags"]
        if k >= n_dags:
            raise ValueError(f"{variant}: structure.n_random_dags is {n_dags}")
        return degree_matched_random_dag(bn, rng.spawn(n_dags)[k], name=variant)
    if variant == "bn" or variant in BN_GRAPH_VARIANTS or variant.startswith("scsa"):
        return bn
    raise ValueError(f"unknown variant: {variant}")


def hparam_configs(cfg: dict[str, Any]) -> list[dict[str, float]]:
    """Every combination in cfg.train.hparam_grid, or the defaults if there is no grid."""
    defaults = {
        "lr": cfg["train"]["lr"],
        "weight_decay": cfg["train"]["weight_decay"],
        "dropout": cfg["model"]["dropout"],
    }
    grid = cfg["train"].get("hparam_grid")
    if not grid:
        return [defaults]
    return [{**defaults, **dict(zip(grid, values))} for values in itertools.product(*grid.values())]


def hparam_candidates(
    cfg: dict[str, Any], seed: int, out: Path
) -> tuple[list[dict[str, float]], str]:
    """Hyperparameters to train in this run, and where they come from.

    Without a grid: the defaults. At cfg.train.hparam_seed: every grid combination. Any
    other seed reuses the choice of the hparam_seed run of the same (stage, dataset, n,
    variant), which must already exist.
    """
    if not cfg["train"].get("hparam_grid"):
        return hparam_configs(cfg), "default"
    hparam_seed = cfg["train"]["hparam_seed"]
    if seed == hparam_seed:
        return hparam_configs(cfg), "search"
    source = out.parent / f"seed{hparam_seed}" / "metrics.json"
    if not source.exists():
        raise FileNotFoundError(f"{source} missing: run seed {hparam_seed} first, it selects hyperparameters")
    selected = json.loads(source.read_text())
    if selected.get("hparams_from") != "search":
        raise ValueError(f"{source} did not search the hyperparameter grid")
    return [selected["hparams"]], f"seed{hparam_seed}"


@torch.no_grad()
def per_column_nll(
    model: MaskedTabularTransformer, x: torch.Tensor, attn_mask: torch.Tensor, batch_size: int = 4096
) -> np.ndarray:
    """-log p(x_j | x_C(j)) in eval mode, shape [N, d] in position order."""
    model.eval()
    return torch.cat([-model.log_prob(xb, attn_mask) for xb in x.split(batch_size)]).cpu().numpy()


def train_model(
    x_train: torch.Tensor,
    x_val: torch.Tensor,
    cardinalities: list[int],
    attn_mask: torch.Tensor,
    hparams: dict[str, float],
    cfg: dict[str, Any],
) -> tuple[MaskedTabularTransformer, dict[str, Any]]:
    """AdamW on mean NLL (nats/row) with early stopping on validation NLL; returns the
    best-validation model and a training summary. The model lives on `x_train`'s device;
    batch order is drawn on the CPU, so it is the same on every device."""
    model_cfg, train_cfg = cfg["model"], cfg["train"]
    model = MaskedTabularTransformer(
        cardinalities,
        dim=model_cfg["dim"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        dropout=hparams["dropout"],
    ).to(x_train.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hparams["lr"], weight_decay=hparams["weight_decay"]
    )
    best_nll, best_epoch, best_state = math.inf, 0, None
    start = time.perf_counter()
    for epoch in range(1, train_cfg["max_epochs"] + 1):
        model.train()
        for idx in torch.randperm(len(x_train)).split(train_cfg["batch_size"]):
            loss = -model.log_prob(x_train[idx], attn_mask).sum(1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        nll_val = float(per_column_nll(model, x_val, attn_mask).sum(1).mean())
        if nll_val < best_nll:
            best_nll, best_epoch, best_state = nll_val, epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= train_cfg["patience"]:
            break
    model.load_state_dict(best_state)
    return model, {
        "nll_val": best_nll,
        "best_epoch": best_epoch,
        "epochs": epoch,
        "train_seconds": time.perf_counter() - start,
    }


def fit_and_evaluate(
    variant: str,
    splits: DataSplits,
    graph: Graph,
    cfg: dict[str, Any],
    out: Path,
    seed: int,
    candidates: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Train (early stopping on splits.val), sample, compute metrics, write artifacts to `out`.

    Every hyperparameter combination in `candidates` (default: `hparam_configs(cfg)`) is
    trained and the one with the lowest validation NLL is kept (train and val data only).
    """
    if variant not in NEURAL_VARIANTS and not RANDOM_DAG.fullmatch(variant):
        raise NotImplementedError(f"variant {variant!r} is not implemented yet")

    order = graph.order
    columns = [splits.columns[c] for c in order]
    cardinalities = [splits.cardinalities[c] for c in order]
    device = resolve_device(cfg["train"]["device"])
    x = {
        split: torch.as_tensor(getattr(splits, split).to_numpy()[:, order], device=device)
        for split in ("train", "val", "test")
    }
    attn_mask = build_attention_mask(graph.parents).to(device)

    search, best = [], None
    for hparams in candidates or hparam_configs(cfg):
        torch.manual_seed(seed)  # same init and batch order for every combination
        model, info = train_model(x["train"], x["val"], cardinalities, attn_mask, hparams, cfg)
        search.append({**hparams, **info})
        if best is None or info["nll_val"] < best[1]["nll_val"]:
            best = (model, info, hparams)
    model, info, hparams = best

    nll = {split: per_column_nll(model, xs, attn_mask) for split, xs in x.items()}
    np.save(out / "per_column_nll.npy", nll["test"])
    test = summarize_nll(nll["test"], columns)

    samples = ancestral_sample(
        model, attn_mask, len(splits.train), torch.Generator(device).manual_seed(seed)
    ).cpu().numpy()
    samples_original = np.empty_like(samples)
    samples_original[:, order] = samples
    pd.DataFrame(samples_original, columns=splits.columns).to_csv(out / "samples.csv", index=False)

    metrics: dict[str, Any] = {
        "variant": variant,
        "n_train": len(splits.train),
        "n_val": len(splits.val),
        "n_test": len(splits.test),
        "d": len(order),
        "n_params": sum(p.numel() for p in model.parameters()),
        "device": device.type,
        "hparams": hparams,
        "best_epoch": info["best_epoch"],
        "epochs": info["epochs"],
        "train_seconds": sum(s["train_seconds"] for s in search),
        "nll_train": float(nll["train"].sum(1).mean()),
        "nll_val": float(nll["val"].sum(1).mean()),
        "nll_test": test["nll"],
        "nll_test_per_column": test["per_column"],
    }
    if splits.true_model is not None:
        true_log_p = bnlearn.true_log_likelihood(
            splits.true_model, bnlearn.decode(splits.true_model, splits.test)
        )
        metrics["true_nll_test"] = float(-true_log_p.mean())
        metrics["excess_nll_test"] = test["nll"] - metrics["true_nll_test"]
    if len(search) > 1:
        metrics["hparam_search"] = search
    return metrics


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    rng = set_seed(args.seed)
    out = run_dir(
        cfg["paths"]["results_dir"], args.stage, args.dataset, args.n, args.variant, args.seed
    )
    save_yaml({**cfg, "run": vars(args)}, out / "config.yaml")

    splits = make_splits(args.dataset, args.n, args.seed, cfg)
    graph = build_graph(args.variant, splits, cfg, rng)
    save_json(graph.to_dict(), out / "graph.json")

    candidates, hparams_from = hparam_candidates(cfg, args.seed, out)
    metrics = fit_and_evaluate(args.variant, splits, graph, cfg, out, args.seed, candidates)
    metrics["hparams_from"] = hparams_from
    save_json(metrics, out / "metrics.json")
    print(f"{out}: test NLL {metrics['nll_test']:.4f} nats/row, best epoch {metrics['best_epoch']}")


if __name__ == "__main__":
    main()
