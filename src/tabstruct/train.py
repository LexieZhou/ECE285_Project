"""Single-run entry point: one (dataset, n, variant, seed) -> one results directory.

    python -m tabstruct.train --config configs/default.yaml \
        --dataset adult --n 500 --variant bn --seed 0

Writes config.yaml, graph.json, metrics.json, per_column_nll.npy and samples.csv to
results/<stage>/<dataset>/n<n>/<variant>/seed<s>/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from tabstruct.data import DATASETS
from tabstruct.data.splits import DataSplits, make_splits
from tabstruct.structure.graphs import Graph
from tabstruct.utils import load_config, run_dir, save_json, save_yaml, set_seed


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
    `full_prefix_trueorder` use the true DAG's order (bnlearn only).
    """
    raise NotImplementedError


def fit_and_evaluate(
    variant: str, splits: DataSplits, graph: Graph, cfg: dict[str, Any], out: Path
) -> dict[str, Any]:
    """Train (early stopping on splits.val), sample, compute metrics, write artifacts to `out`."""
    raise NotImplementedError


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

    metrics = fit_and_evaluate(args.variant, splits, graph, cfg, out)
    save_json(metrics, out / "metrics.json")


if __name__ == "__main__":
    main()
