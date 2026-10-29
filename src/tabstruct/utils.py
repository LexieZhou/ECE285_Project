"""Shared helpers: seeding, config loading, run directories, serialization."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> np.random.Generator:
    """Seed python, numpy and torch; return a numpy Generator for explicit use."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into a copy of `base`."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a YAML config. A config may inherit from another via `base: <relative path>`."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    base = cfg.pop("base", None)
    if base is not None:
        cfg = deep_update(load_config(path.parent / base), cfg)
    return deep_update(cfg, overrides or {})


def run_dir(
    results_dir: str | Path, stage: str, dataset: str, n: int | None, variant: str, seed: int
) -> Path:
    """Create and return `results/<stage>/<dataset>/n<n>/<variant>/seed<s>/`."""
    n_tag = "full" if n is None else str(n)
    path = Path(results_dir) / stage / dataset / f"n{n_tag}" / variant / f"seed{seed}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=float))


def save_yaml(obj: Any, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(obj, sort_keys=False))
