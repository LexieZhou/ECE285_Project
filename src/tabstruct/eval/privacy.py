"""Membership-inference risk: DOMIAS (van Breugel et al., 2023) and record vulnerability
(Meeus et al., 2023)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def vulnerability_scores(train: pd.DataFrame, k: int = 5) -> np.ndarray:
    """Mean cosine distance of each training row to its k nearest training neighbours
    (one-hot encoding). Higher = more isolated = more vulnerable."""
    raise NotImplementedError


def domias_scores(
    targets: pd.DataFrame, synth: pd.DataFrame, reference: pd.DataFrame, estimator: str = "bn"
) -> np.ndarray:
    """Density-ratio scores p_G(x) / p_R(x), with the same estimator fit on `synth` (p_G)
    and on `reference` (p_R)."""
    raise NotImplementedError


def tpr_at_fpr(scores: np.ndarray, is_member: np.ndarray, fpr: float = 0.01) -> float:
    """True-positive rate at the threshold giving false-positive rate `fpr`."""
    raise NotImplementedError
