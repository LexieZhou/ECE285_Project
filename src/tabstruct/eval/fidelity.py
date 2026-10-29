"""Fidelity of synthetic data: marginal TVD and train-synthetic-test-real (TSTR)."""

from __future__ import annotations

import pandas as pd


def tvd_marginals(real: pd.DataFrame, synth: pd.DataFrame) -> dict[str, float]:
    """Average total variation distance over all 1-way and 2-way marginals."""
    raise NotImplementedError


def tstr(synth: pd.DataFrame, test: pd.DataFrame, target: str) -> dict[str, float]:
    """Train a gradient-boosted classifier on `synth`, report accuracy/AUC on real `test`."""
    raise NotImplementedError
