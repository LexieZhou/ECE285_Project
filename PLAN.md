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
      its own category. Hold out a fixed **test set (10k rows)** and a fixed
      **reference set (5k rows)** for DOMIAS, never used for training.
- [x] Training subsets `n ∈ {500, 1k, 5k, full}` drawn per seed from the remaining pool;
      validation set of size `max(0.25n, 200)` drawn disjointly from the same pool.
      For a fixed seed, smaller train sets are prefixes of larger ones (nested).
- [x] **bnlearn networks** CHILD (20 nodes), INSURANCE (27), ALARM (37): load with
      `pgmpy.example_models.load_model("bnlearn/<name>")`, sample train/val/test/reference with
      forward sampling. Sizes `n ∈ {500, 1k, 5k}`; test 10k, reference 5k.
- [ ] For bnlearn, compute the **ground-truth test NLL** under the true BN; report
      every model's *excess NLL* over it (great sanity check and lower bound).
      `bnlearn.true_log_likelihood` is done; reporting it happens in `fit_and_evaluate`.

### 0.3 Discretization and schema
- [x] Quantile-bin numerical columns using **training split only** (default K = 10,
      reduce to K = 5 for n ≤ 1k; record the choice in config).
- [x] Categorical vocabularies come from the known schema (all categories), so
      test-time categories never go out of vocabulary. Bin edges extend to ±∞.
- [x] Output: integer-coded DataFrames + column names and cardinalities (`DataSplits`).
- [ ] **Open question (decide before Stage 1 figures):** Adult bin edges depend on the
      training split and K changes between n ≤ 1k and n > 1k, so Adult test NLL is not
      comparable across seeds/n in absolute terms. Paired comparisons within a
      (dataset, n, seed) are unaffected. Options: fixed K for all n, or report NLL
      relative to the `marginal` variant.

### 0.4 Structures
- [ ] **Column order:** for each (dataset, n, seed), the generation order is a
      topological order of the learned BN. Full-prefix and random DAGs use the **same**
      order, so only the conditioning sets differ.
- [ ] **Learned BN:** `pgmpy` `HillClimbSearch` + `BicScore`, `max_indegree=3`, fit on
      the training split only.
- [ ] **Degree-matched random DAGs (≥5):** for column at position j with BN in-degree
      k_j, sample k_j parents uniformly from positions `< j`. Same in-degree
      per column by construction.
- [ ] **True DAG (bnlearn only):** uses its own topological order. Also run
      full-prefix under that order as its matched control (order confound).
- [ ] Represent every graph as `parents: list[list[int]]` in position space.

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
- [ ] Loss = mean over rows of `Σ_j −log p(x_j | x_C(j))` (nats/row). AdamW,
      early stopping on validation NLL (patience 20 epochs), keep best checkpoint.
- [ ] Report held-out test NLL (nats/row) and per-column NLL `[N_test, d]` saved to
      disk (Stage 2 needs it).
- [ ] Hyperparameters: a **small shared grid** (lr ∈ {1e-3, 3e-4}, weight decay ∈
      {0, 1e-2}, dropout ∈ {0.1, 0.3}) selected on validation NLL **per variant**.
      This matters for RQ2: a well-regularized full-prefix model is the fair control
      for "is BN just a regularizer?".

### 0.7 Run interface and results layout
- [ ] One entry point: `python -m tabstruct.train --config configs/....yaml
      --dataset adult --n 500 --variant bn --seed 0`.
- [ ] Each run writes `results/<stage>/<dataset>/n<n>/<variant>/seed<s>/` with
      `config.yaml`, `metrics.json`, `per_column_nll.npy`, `graph.json`, `samples.csv`.
- [ ] `scripts/aggregate.py` collects all `metrics.json` into one `results/summary.csv`.

**Phase 0 done when:** mask test passes; full-prefix on Adult-500 trains, early-stops,
and reports a test NLL lower than the independent-marginals model.

---

## Phase 1 — Stage 1: graph study, RQ1–RQ2 (Nov 3 – Nov 6)

- [ ] Variants: `full_prefix`, `bn`, `random_dag × 5`, `true_dag` (bnlearn),
      `marginal` (empty graph, floor reference).
- [ ] Grid: Adult × {500, 1k, 5k, full}; CHILD/INSURANCE/ALARM × {500, 1k, 5k};
      5 seeds each (seed changes the training subset, BN, random DAGs and init).
- [ ] Classical baselines with likelihoods, on the same splits:
  - [ ] BN with Laplace / BDeu-smoothed CPTs (pgmpy `BayesianEstimator`).
  - [ ] BN with HDP smoothing (Petitjean et al., 2018). Reference code is Java;
        if wrapping it is too costly, implement hierarchical back-off smoothing of
        each CPT row toward its parent-subset/marginal and document the substitution.
        Note: this baseline is conceptually close to SCSA (back-off by support), so it
        is important for the RQ4 story.
- [ ] Analysis:
  - **RQ1:** `bn − full_prefix` test NLL, per (dataset, n), mean ± 95% CI over seeds.
  - **RQ2:** `bn − random_dag` (average over the 5 random DAGs per seed). If BN ≈ random,
        the gain is sparsity; if BN ≪ random, edges are informative. On bnlearn also
        compare `bn` vs `true_dag`.
  - Paired tests across seeds (Wilcoxon or paired bootstrap).
- [ ] Figure: NLL vs n, one line per variant, per dataset.

## Phase 2 — Stage 2: support analysis, RQ3 (Nov 5 – Nov 8)

- [ ] `support.py`: for each column j and conditioning set C(j), count training rows
      that match `x_{C(j)}` (hash tuples → counts). Training rows use leave-one-out
      counts (subtract 1). Unit-test against a brute-force count.
- [ ] Buckets on `s_{j,1}` (support under **BN parents**, same buckets for every graph):
      zero (0), low (1–4), medium (5–19), high (≥20). Fix thresholds now; do not tune.
- [ ] For every test row × column: assign bucket, then average per-column NLL per
      (bucket, variant). Plot NLL by bucket with one bar per variant.
- [ ] Vulnerability: Meeus et al. (2023) score = mean cosine distance of a training
      row to its k = 5 nearest training neighbours (one-hot encoding).
      Spearman correlation between each training row's `min_j s_{j,1}` and its score.
- [ ] Optional cheap MI signal for likelihood models: per-row loss-based MI
      (train vs. held-out NLL) by bucket, before DOMIAS is ready.

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
