"""bnlearn benchmark networks with known DAGs (CHILD, INSURANCE, ALARM), via pgmpy."""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_network(name: str):
    """Return the ground-truth pgmpy model (`pgmpy.example_models.load_model`)."""
    from pgmpy.example_models import load_model

    return load_model(f"bnlearn/{name}")


def state_names(model) -> dict[str, list[str]]:
    """Schema vocabulary of every node, in `model.nodes()` order."""
    return {node: list(model.get_cpds(node).state_names[node]) for node in model.nodes()}


def sample_network(model, n: int, seed: int) -> pd.DataFrame:
    """Forward-sample `n` rows; columns in `model.nodes()` order, values as state names."""
    from pgmpy.sampling import BayesianModelSampling

    df = BayesianModelSampling(model).forward_sample(size=n, seed=seed, show_progress=False)
    return df[list(model.nodes())].astype(str)


def true_log_likelihood(model, df: pd.DataFrame) -> np.ndarray:
    """Per-row log p(x) under the true network, shape [N]. Gives the NLL lower bound."""
    log_p = np.zeros(len(df))
    for node in model.nodes():
        cpd = model.get_cpds(node)
        # cpd.values has one axis per variable in cpd.variables = [node, *parents].
        index = tuple(
            df[var].map({s: i for i, s in enumerate(cpd.state_names[var])}).to_numpy()
            for var in cpd.variables
        )
        log_p += np.log(cpd.values[index])
    return log_p
