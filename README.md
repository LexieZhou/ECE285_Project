# When Does Structure Help Generative Models of Small Tables?

## Overview

Autoregressive tabular generators can overfit and memorize rare records when training
data is scarce. Structural priors such as Bayesian networks (BNs) may help, but it is
unclear whether they help through **sparsity alone** or through **informative
dependencies**. This project answers that with a controlled study:

- One masked Transformer at a fixed parameter budget, where only the attention mask
  changes between variants.
- Conditioning sets compared: **full prefix**, **learned BN parents**,
  **degree-matched random DAGs**, and the **true DAG** (when known).
- Held-out negative log-likelihood (NLL) and membership-inference (MI) risk are analysed
  as a function of each row's **support** — the number of training rows that share its
  conditioning values.

Building on this analysis we propose **Support-Conditioned Structural Attention (SCSA)**,
which mixes three conditionals per column — BN parents, full prefix, and the marginal —
with a small gate driven by support:

```
p(x_j | x_<j) = Σ_r π_{j,r}(x_<j) · p_r(x_j | x_{C_r(j)}),    r ∈ {BN, full, marginal}
π_j = softmax(g_φ([log(1 + s_{j,BN}), log(1 + s_{j,full})]))
```

The components share backbone weights and differ only in their attention masks.

## Research questions

| | Question | Stage |
|---|---|---|
| RQ1 | Does BN conditioning improve held-out NLL over full-prefix conditioning? | 1 |
| RQ2 | Does the gain come from sparse conditioning or from informative edges? | 1 |
| RQ3 | How do held-out NLL and MI risk vary with a row's support? | 2 |
| RQ4 | Can support-adaptive conditioning reduce MI risk without hurting NLL? | 3 |

## Data

| Dataset | Source | Columns | Training sizes n |
|---|---|---|---|
| Adult | UCI ML Repository | 14 (numerical columns quantile-binned) | 500, 1k, 5k, full |
| CHILD | bnlearn repository (known DAG) | 20 | 500, 1k, 5k |
| INSURANCE | bnlearn repository (known DAG) | 27 | 500, 1k, 5k |
| ALARM | bnlearn repository (known DAG) | 37 | 500, 1k, 5k |

Each dataset has a fixed test set (10k rows) and a fixed reference set (5k rows) for
DOMIAS. bnlearn tables are sampled from the ground-truth networks, so the true test NLL
is available as a lower bound.

## Models and baselines

- **Masked Transformer variants:** `marginal`, `full_prefix`, `bn` (hill-climbing + BIC,
  in-degree ≤ 3), `random_dag` (≥ 5 per seed, in-degree matched to the BN per column),
  `true_dag` (bnlearn only), and `scsa`.
- **Baselines:** BN with smoothed CPTs, BN with HDP smoothing, SMOTE, TVAE, and TabSyn
  (time permitting).

## Evaluation

| Metric | What it measures | Models |
|---|---|---|
| Held-out NLL (nats/row), overall and per support bucket | Distribution fit | Likelihood-based models |
| TVD on 1- and 2-way marginals | Fidelity of synthetic data | All |
| TSTR (train on synthetic, test on real) | Downstream utility | All |
| DOMIAS TPR @ 1% FPR, random and outlier targets | Membership-inference risk | All |

All metrics are reported as mean ± 95% CI over 5 seeds.

## Repository layout

```
src/tabstruct/
  data/        # loaders, discretization, splits
  structure/   # BN learning, random DAGs, graph utilities
  models/      # attention masks, masked Transformer, SCSA, support counting, sampler
  baselines/   # BN-CPT, BN-HDP, SMOTE, TVAE wrappers
  eval/        # NLL, TVD, TSTR, DOMIAS, vulnerability scores
  train.py     # single-run entry point
  utils.py     # seeding, config loading, run directories
configs/       # default.yaml + smoke / stage1 / stage3 grids
scripts/       # grid runner, Stage 2 support analysis, aggregation, figures
tests/         # unit tests (incl. mask-invariance test)
reports/       # mid-term and final reports
PLAN.md        # step-by-step plan and progress
CLAUDE.md      # notes for Claude Code
```

## Setup

Requires Python 3.11.

```bash
uv sync
```

Or with pip:

```bash
pip install -e ".[dev,baselines]"
```

## Usage

Run a single experiment:

```bash
uv run python -m tabstruct.train --config configs/default.yaml --dataset adult --n 500 --variant bn --seed 0
```

Run a stage's grid (add `--dry-run` to list the runs), the Stage 2 support analysis,
aggregate results, and make figures:

```bash
uv run python scripts/run_grid.py --config configs/stage1.yaml --jobs 8
```

```bash
uv run python scripts/analyze_support.py
```

```bash
uv run python scripts/aggregate.py
```

```bash
uv run python scripts/make_figures.py
```

Run tests:

```bash
uv run pytest -q
```

Each run writes to `results/<stage>/<dataset>/n<n>/<variant>/seed<s>/`
(`config.yaml`, `metrics.json`, `per_column_nll.npy`, `graph.json`, `samples.csv`).

## Timeline

| Dates (2026) | Milestone |
|---|---|
| Oct 29 – Nov 2 | Data pipelines, masked backbone, evaluation code |
| Nov 3 – Nov 9 | Stages 1–2; mid-term report (decide whether SCSA is the main contribution) |
| Nov 10 – Nov 23 | SCSA implementation and MI evaluation |
| Nov 24 – Dec 12 | Full experiments and baselines; final report and presentation |

## Key references

- Vowels et al., 2024. *CaTs and DAGs: Integrating directed acyclic graphs with transformers for causally constrained predictions.*
- van Breugel et al., 2023. *Membership inference attacks against synthetic data through overfitting detection* (DOMIAS).
- Meeus et al., 2023. *Achilles' heels: Vulnerable record identification in synthetic data publishing.*
- Scutari, 2010. *Learning Bayesian networks with the bnlearn R package.*
- Petitjean et al., 2018. *Accurate parameter estimation for Bayesian network classifiers using hierarchical Dirichlet processes.*
- Xu et al., 2019. *Modeling tabular data using conditional GAN* (CTGAN/TVAE).
- Yao et al., 2025. *The DCR delusion: Measuring the privacy risk of synthetic data.*

See the proposal for the full reference list.
