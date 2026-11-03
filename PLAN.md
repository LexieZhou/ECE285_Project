# Project Plan: When Does Structure Help Generative Models of Small Tables?

## Timeline at a glance

| Phase | Dates (2026) | Goal | Exit criterion |
|---|---|---|---|
| 0. Foundation | Oct 29 – Nov 2 | Repo, data, discretization, structures, masked backbone, NLL | Mask-invariance test passes; full-prefix model trains on Adult-500 |
| 1. Stage 1: graph study (RQ1–2) | Nov 3 – Nov 6 | Full-prefix vs learned BN vs random DAGs vs true DAG | Results table with mean ± CI over seeds |
| 2. Stage 2: support analysis (RQ3) | Nov 5 – Nov 8 | Per-column NLL by support bucket; support vs vulnerability | Bucket plots + correlation numbers |
| Mid-term report | Nov 9 | Write up Stages 1–2; go/no-go on SCSA as main contribution | Report submitted |
| 3. Stage 3: SCSA (RQ4) | Nov 10 – Nov 23 | SCSA model, gate, ablations; MI evaluation pipeline | SCSA trains stably; DOMIAS numbers for all neural variants |
| 4. Full experiments | Nov 24 – Dec 5 | All datasets × n × seeds, baselines | Final aggregated tables and figures |
| 5. Writing | Dec 1 – Dec 12 | Final report + presentation | Submitted |

Suggested split into two parallel workstreams (swap as you like):

- **Workstream A — data, structure, baselines, evaluation:** loaders, discretization, BN
  learning, random DAGs, BN/SMOTE/TVAE baselines, TVD/TSTR/DOMIAS, vulnerability scores.
- **Workstream B — neural models:** masked Transformer, masks, training loop, sampling,
  support counting, SCSA, ablations.

Both meet at a shared interface: `(train_df, val_df, test_df, schema, graph)` in and
`metrics.json` out (see Phase 0.7).

---

## Phase 0 — Foundation (Oct 29 – Nov 2)

### 0.1 Repo and environment
- [x] Create layout from `README.md` (`src/tabstruct/`, `configs/`, `scripts/`, `tests/`).
      Module stubs define the interfaces; tests encode the contracts (mask invariance,
      LOO support, degree matching, discretizer range) and auto-skip until implemented.
- [x] `pyproject.toml` with: `torch`, `numpy`, `pandas`, `scikit-learn`, `pgmpy`,
      `ucimlrepo`, `pyyaml`, `scipy`, `matplotlib`, `joblib`; extras `dev` (`pytest`,
      `ruff`) and `baselines` (`sdv`).
- [x] `.gitignore`: `data/`, `results/`, `*.pt`, `.venv/`, `wandb/`.
- [x] Global seeding helper `tabstruct.utils.set_seed` (`numpy`, `torch`, `random`).

### 0.2 Datasets
- [x] **Adult** (UCI, ~48k rows): load via `ucimlrepo` (id=2). Drop `fnlwgt`; treat `?` as
      its own category. Hold out a fixed **test set** and a fixed
      **reference set (5k rows)** for DOMIAS, never used for training.
      *Protocol change (2026-10-29):* the Adult test set is the **official `adult.test`
      (16,281 rows)** instead of a random 10k from all 48,842 rows, so test metrics (e.g.
      TSTR) line up with the literature. Reference (5k) and the train/val pool (27,561)
      are drawn from the official `adult.data` (32,561). bnlearn keeps test 10k.
      *Protocol change (2026-11-02):* 2,000 more `adult.data` rows form a fixed
      **binning set** (bin edges only, see 0.3), so the pool is now 25,561.
- [x] Training subsets `n ∈ {500, 1k, 5k, full}` drawn per seed from the remaining pool;
      validation set of size `max(0.25n, 200)` drawn disjointly from the same pool.
      For a fixed seed, smaller train sets are prefixes of larger ones (nested).
- [x] **bnlearn networks** CHILD (20 nodes), INSURANCE (27), ALARM (37): load with
      `pgmpy.example_models.load_model("bnlearn/<name>")`, sample train/val/test/reference with
      forward sampling. Sizes `n ∈ {500, 1k, 5k}`; test 10k, reference 5k.
- [x] For bnlearn, compute the **ground-truth test NLL** under the true BN; report
      every model's *excess NLL* over it (great sanity check and lower bound).
      `metrics.json` has `true_nll_test` and `excess_nll_test` (CHILD test: 12.17 nats/row).

### 0.3 Discretization and schema
- [x] Quantile-bin numerical columns using **training split only** (default K = 10,
      reduce to K = 5 for n ≤ 1k; record the choice in config).
      *Superseded (2026-11-02), see the decision below.*
- [x] Categorical vocabularies come from the known schema (all categories), so
      test-time categories never go out of vocabulary. Bin edges extend to ±∞.
- [x] Output: integer-coded DataFrames + column names and cardinalities (`DataSplits`).
- [x] **Bug fixed (2026-11-02): `capital-gain` and `capital-loss` were constant after
      binning.** ~92% / ~95% of values are 0, so every quantile edge was 0, `np.unique`
      left the single edge [0], and every value (0 or positive) landed in bin 1; the
      columns carried no information in every Adult run before this fix.
      *Rule now (`quantile_edges`):* if more than 1/K of the training values equal the
      column minimum, the minimum gets bin 0 to itself and the values above it are
      quantile-binned into K − 1 bins (still fit on the training split only). Adult-500:
      capital-gain 445 zeros + 55 positives over bins 1–4.
- [x] **Decided (2026-11-02): fixed bins for Adult.** Previously bin edges depended on
      the training split and K changed between n ≤ 1k and n > 1k, so Adult test NLL was
      not comparable across seeds/n. Now edges are fit **once** on the fixed binning set
      (2,000 `adult.data` rows disjoint from train/val/test/reference), with **K = 10 for
      every n**, so all Adult runs share one discrete space (CLAUDE.md invariant 4 updated).
      This follows the fixed-domain convention of the private-pgm / MST / AIM Adult
      benchmarks. `education-num` (16 levels, one per `education` value) is kept as-is
      instead of binned. NLL is still not comparable with other papers (every paper
      discretizes differently); literature comparisons go through TSTR and DOMIAS.
      NLL relative to `marginal` can be reported as a secondary number.

### 0.4 Structures
- [x] **Column order:** for each (dataset, n, seed), the generation order is a
      topological order of the learned BN. Full-prefix and random DAGs use the **same**
      order, so only the conditioning sets differ.
      `train.build_graph` re-learns the BN (deterministic) in every run, so `marginal`,
      `full_prefix`, `bn` and `random_dag_<k>` share its order. `random_dag_<k>` uses the
      k-th spawned stream of the seed's RNG, so it depends only on (seed, k).
- [x] **Learned BN:** `pgmpy` `HillClimbSearch` + `BicScore`, `max_indegree=3`, fit on
      the training split only. Uses `pgmpy.causal_discovery.HillClimbSearch` with
      `scoring_method="bic-d"` (the `pgmpy.estimators` version is deprecated in 1.1).
      Sanity check, seed 0: CHILD n=1k recovers 22/25 true skeleton edges, ALARM n=5k
      41/46; each fit takes a few seconds or less.
- [x] **Degree-matched random DAGs (≥5):** for column at position j with BN in-degree
      k_j, sample k_j parents uniformly from positions `< j`. Same in-degree
      per column by construction.
- [x] **True DAG (bnlearn only):** uses its own topological order. Also run
      full-prefix under that order as its matched control (order confound).
      Variants `true_dag` and `full_prefix_trueorder` in `build_graph`.
- [x] Represent every graph as `parents: list[list[int]]` in position space.

### 0.5 Masked Transformer backbone (capacity-controlled)
The key requirement: the prediction for column j must depend **only** on `x_{C(j)}`,
at every layer (CaTs-style). Recommended design, two token streams:

- **Content tokens** `k = 1..d`: `embed_value(x_k) + embed_col(k)`. Content tokens attend
  **only to themselves** (so no information mixes across columns in this stream).
- **Query tokens** `j = 1..d`: `embed_col(j)` only (no value). Query token j attends to
  content tokens in `C(j)` plus itself (self-attention avoids empty rows → NaNs).
- Output head on query token j → logits over column j's categories (per-column heads
  or one shared head with per-column masking of invalid classes).
- Variants differ **only** in the attention mask → identical parameter count.
- Small default: dim 64, 4 heads, 3 layers, dropout 0.1. Log parameter count per run.

Why not let content tokens attend to their own parents? Then query j would see the
ancestors of its parents, so the model would no longer be `p(x_j | x_{pa(j)})`.

- [x] `masks.py`: build `(2d × 2d)` boolean masks from a `parents` list (full prefix,
      BN, random, empty/marginal).
- [x] `masked_transformer.py`: forward returns per-column log-probs `[B, d]`.
- [x] **Unit test (must pass before any experiment):** perturb `x_m` for every
      `m ∉ C(j)` and assert the logits for column j are unchanged (to 1e-6); perturb
      some `m ∈ C(j)` and assert they change (for a randomly initialized net).
- [x] Ancestral sampler: generate columns in order, one query at a time.

### 0.6 Training loop and NLL
- [x] Loss = mean over rows of `Σ_j −log p(x_j | x_C(j))` (nats/row). AdamW,
      early stopping on validation NLL (patience 20 epochs), keep best checkpoint.
- [x] Report held-out test NLL (nats/row) and per-column NLL `[N_test, d]` saved to
      disk (Stage 2 needs it). `metrics.json` also has train/val NLL and per-column
      test NLL keyed by column name (positions differ across seeds).
- [x] Hyperparameters: a **small shared grid** (lr ∈ {1e-3, 3e-4}, weight decay ∈
      {0, 1.0}, dropout ∈ {0.1, 0.3}) selected on validation NLL **per variant**.
      This matters for RQ2: a well-regularized full-prefix model is the fair control
      for "is BN just a regularizer?".
      **Decided and implemented (2026-11-02):** the run at `train.hparam_seed` (0) trains
      every combination and keeps the lowest validation NLL; other seeds of the same
      (stage, dataset, n, variant) read its `metrics.json` and reuse that choice
      (`hparams_from` records which). `run_grid.py` runs seed 0 first. Caveat: at n = 500
      the selection uses only 200 val rows.
      *Grid change (2026-11-02): weight decay {0, 1e-2} → {0, 1.0}.* AdamW shrinks weights
      by (1 − lr·wd) per step and Adult-500 trains only ~320 steps, so 1e-2 shrank them by
      0.3% in total and gave val NLL identical to wd = 0 (the grid was effectively 4
      combinations). Sweep on Adult-500, seed 0, dropout 0.1: test NLL moves ≤ 0.05 for
      wd ∈ {0, 0.1, 1.0} for both `full_prefix` and `bn`. 1.0 is kept because it does bite
      at larger n (~1,600 steps at n = 5000 → ×0.2), and val selection falls back to 0 if
      it is too strong. Observation for RQ2: `full_prefix` overfits (train 11.7–12.0 vs
      test 13.4; `bn` gap ~0.3) and neither weight decay nor dropout 0.3 closes the gap;
      early stopping is the effective regularizer at n = 500.

### 0.7 Run interface and results layout
- [x] One entry point: `python -m tabstruct.train --config configs/....yaml
      --dataset adult --n 500 --variant bn --seed 0`.
- [x] Each run writes `results/<stage>/<dataset>/n<n>/<variant>/seed<s>/` with
      `config.yaml`, `metrics.json`, `per_column_nll.npy`, `graph.json`, `samples.csv`.
      *Added (2026-11-02):* `train_per_column_nll.npy` `[n_train, d]` (row r = row r of the
      training split), so Stage 2 loss-based MI and per-row support/vulnerability analysis
      need no reruns.
      Neural variants only so far; baselines and SCSA raise `NotImplementedError`.
- [x] `scripts/aggregate.py` collects all `metrics.json` into one `results/summary.csv`.

**Phase 0 done when:** mask test passes; full-prefix on Adult-500 trains, early-stops,
and reports a test NLL lower than the independent-marginals model.

*Status (2026-11-02): met.* Default hyperparameters, seed 0, 1 seed only (no CI yet),
test NLL in nats/row:

| | marginal | full_prefix | bn |
|---|---|---|---|
| Adult n=500 (fixed bins K=10, grid-selected hparams) | 17.38 | 13.37 | 14.05 |
| CHILD n=500 (true BN 12.17) | 17.20 | 12.68 | 12.72 |

CHILD used default hyperparameters. Adult history: 14.99 / 11.72 / 12.28 before the
capital-column fix, 15.64 / 12.34 / 12.85 after it (per-run bins, K = 5); NLL rises with
each change because the discrete space gets finer (more information per row), not
because models got worse. With fixed bins, seed 1 (reusing seed 0's hyperparameters)
gives 17.39 / 13.37 / 14.07, within 0.02 of seed 0.
Early stopping kicked in after 32–49 epochs. Adult test NLL was 0.3–0.45 nats/row above
val NLL for every variant. *Diagnosed (2026-11-02): not a shift between `adult.data` and
`adult.test`.* Under a Laplace-smoothed marginal model fit on train, `adult.test` and
16k unused `adult.data` rows differ by only 0.02 nats/row (n=500) and 0.004 (n=5000);
raw per-column TVD between the two files is ≤ 0.012. The gap comes from the small val
set (200 rows at n=500: val 14.70 vs 14.96 on both large held-out sets; at n=5000 with
1,250 val rows the gap disappears) plus the optimistic bias of early-stopping on val.
So val NLL is not a reliable estimate at small n; report test NLL only.

---

## Phase 1 — Stage 1: graph study, RQ1–RQ2 (Nov 3 – Nov 6)

- [ ] Variants: `full_prefix`, `bn`, `random_dag × 5`, `true_dag` (bnlearn),
      `marginal` (empty graph, floor reference).
- [ ] Grid: Adult × {500, 1k, 5k, full}; CHILD/INSURANCE/ALARM × {500, 1k, 5k};
      5 seeds each (seed changes the training subset, BN, random DAGs and init).
- [ ] Classical baselines with likelihoods, on the same splits:
  - [x] BN with Laplace / BDeu-smoothed CPTs (pgmpy `BayesianEstimator`).
        `baselines/bn_cpt.py` counts CPTs with numpy (tested equal to pgmpy's
        `BayesianEstimator` to 1e-10). Smoothing (BDeu ess ∈ {0.1, 1, 10, 100} or K2/Laplace)
        is selected on val NLL in every run, since a fit takes milliseconds.
        Seed 0, n=500: Adult 14.04 (neural `bn` 14.05), CHILD 12.71 (neural `bn` 12.72,
        default hparams); CHILD n=5000: 12.28, excess 0.11. So the neural model with BN
        masks matches counting on the same graph.
  - [x] BN with HDP smoothing (Petitjean et al., 2018). Reference code is Java;
        if wrapping it is too costly, implement hierarchical back-off smoothing of
        each CPT row toward its parent-subset/marginal and document the substitution.
        Note: this baseline is conceptually close to SCSA (back-off by support), so it
        is important for the RQ4 story.
        *Substitution (2026-11-02), `baselines/bn_hdp.py`:* deterministic hierarchical
        Dirichlet back-off, not the Java Gibbs sampler. Each CPT is a tree over its
        parents (added in decreasing mutual information with the child, train only):
        level l is smoothed toward level l − 1 with one concentration per (column, level),
        marginal toward uniform. Differences from the paper: raw counts are passed up (no
        table counts), and concentrations are point estimates maximizing the leave-one-out
        log-likelihood of the training rows, not sampled. LOO alone drives a concentration
        to ~0 when a CPT row has no singletons (unseen values then cost a lot on test), so
        its lower bound ∈ {0.01, 0.3, 1, 2} is selected on val NLL, like `bn_cpt`'s ess.
        Seed 0 test NLL, `bn_cpt` → `bn_hdp`: Adult 500 14.042 → 14.041, 5k 13.131 → 13.122;
        CHILD 500 12.705 → 12.698, 5k equal; INSURANCE 500 14.404 → 14.367, 5k 13.273 → 13.272;
        ALARM 500 11.450 → 11.493 (backing off to the marginal hurts ALARM's near-deterministic
        CPTs with rare parent configurations), 5k equal. Fits in seconds.
- [ ] Analysis:
  - **RQ1:** `bn − full_prefix` test NLL, per (dataset, n), mean ± 95% CI over seeds.
  - **RQ2:** `bn − random_dag` (average over the 5 random DAGs per seed). If BN ≈ random,
        the gain is sparsity; if BN ≪ random, edges are informative. On bnlearn also
        compare `bn` vs `true_dag`.
  - Paired tests across seeds (Wilcoxon or paired bootstrap).
  - *Implemented (2026-11-03):* `eval/compare.py` + `scripts/make_figures.py` →
    `reports/tables/stage1_{variants,comparisons}.csv`. Paired per-seed differences,
    t-CI, Wilcoxon (two-sided; with 5 seeds its minimum is 0.0625, so read the CI and the
    count of seeds favouring each side). `random_dag_<k>` averaged within a seed.
  - *Results, n ∈ {500, 1k}, 5 seeds (n ≥ 5k still running):* **RQ1 is negative** —
    `bn − full_prefix` > 0 everywhere: Adult +0.70 ± 0.03 (n=500) / +0.52 ± 0.12 (1k);
    INSURANCE +0.37 / +0.17; CHILD +0.07 / ≈0; ALARM +0.14 / ≈0 (CI crosses 0).
    **RQ2: edges are informative, sparsity alone hurts** — `bn − random_dag` −2 to −7,
    `random_dag − full_prefix` +3 to +7.5 (5/5 seeds). Why BN loses: `bn − true_dag`
    +0.19 to +0.90 (the BIC-learned BN misses true edges at small n; Adult-500's BN has 7
    of 14 columns without parents), while `true_dag − full_prefix_trueorder` is −0.11 to
    −0.48 (5/5): correct sparse structure does beat the full prefix. Ranking:
    true_dag < full_prefix < bn ≪ random_dag. `bn ≈ bn_cpt` (within 0.03): the neural
    model matches counting on the same graph, so this is the graph, not training.
- [x] Figure: NLL vs n, one line per variant, per dataset (`stage1_nll_vs_n.{pdf,png}`).

## Phase 2 — Stage 2: support analysis, RQ3 (Nov 5 – Nov 8)

- [x] `support.py`: for each column j and conditioning set C(j), count training rows
      that match `x_{C(j)}` (hash tuples → counts). Training rows use leave-one-out
      counts (subtract 1). Unit-test against a brute-force count.
      Full prefix on 22k rows × 37 columns takes ~3 s.
- [x] Buckets on `s_{j,1}` (support under **BN parents**, same buckets for every graph):
      zero (0), low (1–4), medium (5–19), high (≥20). Fix thresholds now; do not tune.
- [x] For every test row × column: assign bucket, then average per-column NLL per
      (bucket, variant). Plot NLL by bucket with one bar per variant.
      *Implemented (2026-11-03):* `eval/support_analysis.py` + `scripts/analyze_support.py`
      rebuild each (dataset, n, seed)'s splits from the `bn` run's `config.yaml`, map every
      variant's per-column NLL to original column order, and write
      `results/stage2/<support>/` (per-seed tables, paired `bucket_comparisons.csv`,
      train/test `generalization_gap.csv`). The figure shows the paired difference
      `bn − full_prefix` per bucket (one bar per n) rather than raw NLL bars per variant:
      buckets mix columns, so the paired difference on the same cells is the readable
      quantity; raw per-variant NLL is in `cells.csv`.
      *Protocol addition (2026-11-03): secondary view bucketed on full-prefix support
      `s_{j,2}` (`--support full_prefix`).* The learned BN at small n is so sparse that
      almost every cell has high `s_{j,1}` (Adult-500 seed 0: 0 zero-support test cells,
      97% high), so `s_{j,1}` buckets barely resolve anything on Adult. Under `s_{j,2}`
      the Adult-500 cells split 32 / 18 / 11 / 39%. The `s_{j,1}` view stays the primary,
      pre-registered analysis; thresholds unchanged.
- [x] Vulnerability: Meeus et al. (2023) score = mean cosine distance of a training
      row to its k = 5 nearest training neighbours (one-hot encoding).
      Spearman correlation between each training row's `min_j s_{j,1}` and its score.
      (`eval/privacy.vulnerability_scores`, tested against sklearn cosine distances.)
- [x] Optional cheap MI signal for likelihood models: per-row loss-based MI
      (train vs. held-out NLL) by bucket, before DOMIAS is ready.
      AUC of training rows (members) vs test rows (non-members) scored by −row NLL,
      per row bucket of `min_j s_{j,1}` (training rows LOO). Under `s_{j,2}` the row
      minimum is almost always 0, so that view's row-level MI is degenerate.
- [ ] Run on the Stage 1 grid and fill in the go/no-go below.

## Mid-term report (due ~Nov 9)

Contents: problem, setup, Stage 1 table + NLL-vs-n figure, Stage 2 bucket figure,
correlation result, and the SCSA plan.

**Go/no-go for SCSA as the main contribution** — make it the headline if both hold:
1. The full-prefix vs BN gap is concentrated in specific support buckets (e.g. BN wins
   in zero/low, full-prefix wins or ties in high), and
2. Low support is associated with higher vulnerability or loss-based MI risk.

Otherwise, make the controlled graph study (sparsity vs knowledge × support) the main
contribution and present SCSA as an exploratory extension.

---

## Phase 3 — Stage 3: SCSA, RQ4 (Nov 10 – Nov 23)

### 3.1 Model
- [ ] Three components share backbone weights and differ only in masks:
      `C1 = BN parents`, `C2 = full prefix`, `C3 = ∅ (marginal)`.
      Implementation: one forward with the batch replicated 3× (one mask each).
- [ ] Gate `g_φ`: small MLP (per-column embedding + `u_j = [log(1+s_{j,1}), log(1+s_{j,2})]`)
      → softmax over 3 components → `π_j`.
- [ ] Loss: `−Σ_j log Σ_r π_{j,r} p_r(x_j | x_{C_r(j)})` computed with `logsumexp`.
      Add an auxiliary term `λ · mean_r NLL_r` (λ ≈ 0.1) so every component stays a good
      model even when the gate rarely selects it.
- [ ] **Training rows must use leave-one-out support.** Without it, every training row
      looks supported under the full prefix, the gate learns to trust it, and SCSA
      memorizes exactly the rows it is meant to protect.
- [ ] Sampling: compute supports on the fly from the training-count tables as each
      column is generated.

### 3.2 Ablations
- [ ] Gate input = constant (learned global mixture, no support) → tests whether
      *support* drives the benefit.
- [ ] Two-component gate {BN, full} without the marginal.
- [ ] Hard back-off rule (full if `s_{j,2} ≥ τ`, else BN if `s_{j,1} ≥ τ`, else marginal).
- [ ] No-LOO support during training (expected to hurt privacy).
- [ ] Inspect learned `π_j` vs support: does it match the expected pattern?

### 3.3 Membership-inference pipeline
- [ ] Targets: (a) **random** — 100 members + 100 non-members; (b) **outlier** — top
      members/non-members by vulnerability score. Non-members drawn from held-out data.
- [ ] DOMIAS (van Breugel et al., 2023): score = `p_G(x) / p_R(x)`, with `p_G` fit on
      synthetic samples and `p_R` fit on the reference set using the **same** density
      estimator. For discrete tables use a smoothed BN or KDE on one-hot encodings;
      cross-check one setting against `synthcity`'s DOMIAS implementation if feasible.
- [ ] Report **TPR @ 1% FPR** and AUC. At n = 500 the 1%-FPR point is very noisy: pool
      scores across seeds and report bootstrap CIs.

## Phase 4 — Full experiments and baselines (Nov 24 – Dec 5)

- [ ] Non-likelihood baselines on the same discretized tables:
  - [ ] SMOTE-style generator (interpolate a row with one of its k-NN; for
        categoricals pick either parent's value). ~30 lines, no label needed.
  - [ ] TVAE (`sdv` `TVAESynthesizer` or `synthcity`), default hyperparameters.
  - [ ] TabSyn — only if time permits.
- [ ] Metrics for **every** model (synthetic set size = n):
  - TVD: average over 1-way and 2-way marginals.
  - TSTR: train a gradient-boosted classifier on synthetic, test on real test set
    (Adult target = `income`; for bnlearn pick one fixed node per network and record it).
  - DOMIAS TPR @ 1% FPR (random and outlier targets).
  - Held-out NLL for likelihood models only.
- [ ] Final grid = Stage 1 variants + SCSA + ablations + baselines, 5 seeds.
- [ ] Compute estimate: models are tiny (≤ 37 columns); one run is ~1–3 min on CPU/MPS
      for n ≤ 5k. ~600–900 runs total → parallelize with a simple job runner
      (`joblib` or a shell loop); use a rented GPU only for the full-Adult runs if needed.
- [ ] `scripts/make_figures.py` regenerates every figure/table from `results/summary.csv`.

## Phase 5 — Final report and presentation (Dec 1 – Dec 12)

- [ ] Main table: NLL / TVD / TSTR / DOMIAS per model × dataset at n = 500 and 5k.
- [ ] Figures: NLL vs n; NLL by support bucket; fidelity–privacy scatter
      (NLL or TVD vs DOMIAS TPR); gate weights vs support.
- [ ] Discuss limitations: discretization, single column order, DOMIAS noise at small n,
      HDP substitution (if any).
- [ ] Release: clean README, pinned environment, one command per figure.

---

## Key risks and mitigations

| Risk | Mitigation |
|---|---|
| Mask leaks information across layers | Two-stream design + invariance unit test (Phase 0.5) |
| Column-order confound between variants | Same topological order for full/BN/random; matched full-prefix control for true DAG |
| "BN helps" is really "full-prefix under-regularized" | Per-variant hyperparameter grid incl. dropout/weight decay |
| HDP baseline hard to reproduce | Hierarchical back-off substitute, documented |
| DOMIAS unstable at n = 500 | Pool across seeds, report AUC + bootstrap CIs, add loss-based MI |
| Falling further behind schedule | Cut order: TabSyn → INSURANCE → n = full for Adult → extra ablations |
