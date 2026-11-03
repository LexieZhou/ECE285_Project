"""Membership-inference risk: DOMIAS (van Breugel et al., 2023) and record vulnerability
(Meeus et al., 2023)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def vulnerability_scores(train: pd.DataFrame, k: int = 5, block: int = 1024) -> np.ndarray:
    """Mean cosine distance of each training row to its k nearest training neighbours
    (one-hot encoding). Higher = more isolated = more vulnerable.

    Every one-hot row has exactly d ones, so cosine similarity is the fraction of
    matching columns; similarities are computed in blocks of rows to bound memory.
    """
    x = train.to_numpy()
    n, d = x.shape
    offsets = np.concatenate([[0], np.cumsum(x.max(axis=0) + 1)[:-1]])
    onehot = np.zeros((n, int(x.max(axis=0).sum()) + d), dtype=np.float32)
    onehot[np.arange(n)[:, None], x + offsets] = 1.0
    scores = np.empty(n)
    for start in range(0, n, block):
        sim = onehot[start : start + block] @ onehot.T / d
        rows = np.arange(len(sim))
        sim[rows, start + rows] = -np.inf  # a row is not its own neighbour
        nearest = np.partition(sim, -k, axis=1)[:, -k:]
        scores[start : start + len(sim)] = 1.0 - nearest.mean(axis=1)
    return scores


def domias_scores(
    targets: pd.DataFrame, synth: pd.DataFrame, reference: pd.DataFrame, estimator: str = "bn"
) -> np.ndarray:
    """Density-ratio scores p_G(x) / p_R(x), with the same estimator fit on `synth` (p_G)
    and on `reference` (p_R)."""
    raise NotImplementedError


def tpr_at_fpr(scores: np.ndarray, is_member: np.ndarray, fpr: float = 0.01) -> float:
    """True-positive rate at the threshold giving false-positive rate `fpr`."""
    raise NotImplementedError
