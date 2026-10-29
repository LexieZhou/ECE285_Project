# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

**"When Does Structure Help Generative Models of Small Tables?"**

We study autoregressive generators for small discrete tables (n ≈ 10²–10³) that
factorize `p(x) = Π_j p(x_j | x_{C(j)})`, and compare conditioning sets `C(j)`:
full prefix, learned Bayesian-network (BN) parents, degree-matched random DAGs, and
the true DAG. We then propose **SCSA** (Support-Conditioned Structural Attention),
which mixes BN-parent, full-prefix and marginal conditionals with a gate driven by
row-level *support* (number of training rows sharing the conditioning values).

Research questions:
- RQ1: Does BN conditioning beat full-prefix conditioning on held-out NLL?
- RQ2: Is the gain from sparsity or from informative edges (BN vs random DAGs)?
- RQ3: How do held-out NLL and membership-inference (MI) risk vary with support?
- RQ4: Can support-adaptive conditioning (SCSA) lower MI risk without hurting NLL?

The step-by-step plan and current status live in `PLAN.md`. Update its checkboxes
when you complete a task.

## Repository layout

```
src/tabstruct/
  data/        # adult.py, bnlearn.py, discretize.py, splits.py
  structure/   # learn_bn.py (pgmpy hill-climb + BIC), random_dag.py, graphs.py
  models/      # masks.py, masked_transformer.py, scsa.py, support.py, sampler.py
  baselines/   # bn_cpt.py, bn_hdp.py, smote.py, tvae.py
  eval/        # nll.py, fidelity.py (TVD, TSTR), privacy.py (DOMIAS, vulnerability)
  train.py     # single-run entry point
  utils.py     # seeding, config loading (`base:` inheritance), run dirs
configs/       # default.yaml + smoke/stage1/stage3.yaml (each defines a `grid`)
scripts/       # run_grid.py, analyze_support.py (Stage 2), aggregate.py, make_figures.py
tests/         # pytest unit tests
data/          # downloaded/sampled data (gitignored)
results/       # run outputs (gitignored)
reports/       # mid-term / final report sources and figures
```

## Commands

```bash
uv sync                                  # install (or: pip install -e ".[dev,baselines]")
uv run pytest -q                         # all tests
uv run pytest tests/test_masks.py -q     # mask-invariance test only
uv run python -m tabstruct.train --config configs/default.yaml \
    --dataset adult --n 500 --variant bn --seed 0
uv run python scripts/run_grid.py --config configs/smoke.yaml --dry-run   # list runs
uv run python scripts/run_grid.py --config configs/stage1.yaml --jobs 8   # run a grid
uv run python scripts/aggregate.py       # results/**/metrics.json -> results/summary.csv
uv run python scripts/make_figures.py    # regenerate all figures from summary.csv
```

## Invariants — do not break these

1. **Mask correctness.** Column j's prediction may depend only on `x_{C(j)}`, at every
   layer. Content tokens attend only to themselves; query token j attends to content
   tokens in `C(j)` plus itself. `tests/test_masks.py` checks this by perturbing
   non-parent inputs; it must pass before any experiment is trusted.
2. **Capacity control.** All neural variants (full, BN, random, true, SCSA components)
   share one architecture and parameter count; they differ only in attention masks.
   SCSA adds only the small gate `g_φ`. Log the parameter count in `metrics.json`.
3. **Same column order.** Within a (dataset, n, seed), full-prefix, BN and random-DAG
   variants use the same order (a topological order of the learned BN). The true-DAG
   variant uses its own order and is paired with a full-prefix run in that order.
4. **No leakage from held-out data.** Bin edges, the learned BN, random DAGs, support
   count tables and hyperparameter selection use only the training split (and the
   validation split for early stopping/model selection). The test set and the DOMIAS
   reference set are never seen during training or tuning.
5. **Leave-one-out support for training rows.** When computing `s_{j,r}` for a row that
   is in the training set, subtract that row itself. SCSA must be trained with LOO
   support.
6. **Discrete space everywhere.** All models, including baselines, are trained and
   evaluated on the same discretized integer-coded tables, so NLL/TVD/DOMIAS are
   comparable.
7. **Reproducibility.** Every run takes an explicit `--seed`; the seed determines the
   training subset, learned BN, random DAGs and initialization. Every run saves its
   resolved config next to its metrics.

## Conventions

- Python 3.11, PyTorch, pgmpy, pandas, scikit-learn. Type hints on public functions.
- Graphs are `parents: list[list[int]]` indexed by **position in the generation order**,
  plus an `order: list[int]` mapping positions to original column indices.
- NLL is reported in **nats per row** (sum over columns, mean over rows). Per-column NLL
  arrays have shape `[n_rows, d]` in position order.
- Variant names: `marginal`, `full_prefix`, `bn`, `random_dag_<k>`, `true_dag`,
  `full_prefix_trueorder`, `scsa`, `scsa_<ablation>`, `bn_cpt`, `bn_hdp`, `smote`, `tvae`.
- Run outputs go to `results/<stage>/<dataset>/n<n>/<variant>/seed<s>/` containing
  `config.yaml`, `metrics.json`, `per_column_nll.npy`, `graph.json`, `samples.csv`.
- Support buckets (fixed, do not tune): zero = 0, low = 1–4, medium = 5–19, high ≥ 20.
- Keep experiment scripts thin; logic belongs in `src/tabstruct/`.
- Unimplemented functions raise `NotImplementedError`; `tests/conftest.py` reports tests
  that hit one as skipped, so a test starts counting as soon as its code exists.
- Never commit `data/`, `results/`, checkpoints, or large CSVs.

## Working style for Claude

- Before changing the model or masks, run `tests/test_masks.py`; after changing them,
  add or update a test that would have caught the change being wrong.
- Do not change the evaluation protocol (splits, buckets, metrics, target selection)
  silently. If a change is needed, note it in `PLAN.md` under the relevant phase.
- Prefer small, verifiable runs (e.g. Adult n=500, 1 seed, few epochs) before launching
  a grid. Report numbers with the seed count and confidence interval.
- Do not tune anything on the test set or the DOMIAS reference set.
