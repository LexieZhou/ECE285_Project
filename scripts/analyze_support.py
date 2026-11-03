"""Stage 2 (RQ3): per-column NLL by support bucket and support-vs-vulnerability.

Reads Stage 1 outputs (per_column_nll.npy, train_per_column_nll.npy, graph.json,
config.yaml) and writes to results/stage2/<support>/, where --support bn (default, the
Stage 2 protocol: buckets on learned-BN-parent support s_{j,1}) or full_prefix (buckets on
full-prefix support s_{j,2}, a secondary view):

    cells.csv, mi.csv, vulnerability.csv      per (dataset, n, seed) [and variant, bucket]
    bucket_comparisons.csv                    paired a - b test NLL per bucket, over seeds
    generalization_gap.csv                    test - train NLL per variant and bucket
    mi_summary.csv, vulnerability_summary.csv means and 95% CIs over seeds

    python scripts/analyze_support.py --results results --jobs 4
    python scripts/analyze_support.py --support full_prefix --jobs 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from tabstruct.eval.support_analysis import (
    analyze_group,
    average_random_dags,
    bucket_comparisons,
    find_groups,
    over_seeds,
)

SHOWN = ["full_prefix", "bn", "random_dag", "true_dag", "full_prefix_trueorder"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default="results")
    parser.add_argument("--stage", default="stage1", help="grid whose runs are analysed")
    parser.add_argument("--support", default="bn", choices=["bn", "full_prefix"],
                        help="conditioning sets that define the support buckets")
    parser.add_argument("--out", default=None, help="default: results/stage2/<support>")
    parser.add_argument("--datasets", nargs="+", help="only these datasets")
    parser.add_argument("--n", nargs="+", dest="ns", help="only these n values, e.g. 500 1000")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    groups = [g for g in find_groups(args.results, args.stage)
              if (not args.datasets or g.dataset in args.datasets) and (not args.ns or g.n in args.ns)]
    if not groups:
        raise SystemExit(f"no finished bn runs under {args.results}/{args.stage}")
    print(f"analysing {len(groups)} (dataset, n, seed) groups, {args.support} support", flush=True)
    parts = Parallel(n_jobs=args.jobs)(delayed(analyze_group)(g, args.support) for g in groups)
    tables = {name: pd.concat([p[name] for p in parts], ignore_index=True)
              for name in ("cells", "mi", "vulnerability")}

    out = Path(args.out or f"results/stage2/{args.support}")
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    comparisons = bucket_comparisons(tables["cells"])
    comparisons.to_csv(out / "bucket_comparisons.csv", index=False)
    cells = average_random_dags(tables["cells"])
    if "nll_train" in cells:
        gap = cells.assign(gap=cells["nll_test"] - cells["nll_train"])
        over_seeds(gap, "gap", ["dataset", "n", "variant", "bucket"]).to_csv(
            out / "generalization_gap.csv", index=False)
    mi = over_seeds(average_random_dags(tables["mi"]), "auc", ["dataset", "n", "variant", "bucket"])
    mi.to_csv(out / "mi_summary.csv", index=False)
    vuln = tables["vulnerability"]
    vuln_summary = over_seeds(vuln, "spearman_rho", ["dataset", "n"])
    vuln_summary.to_csv(out / "vulnerability_summary.csv", index=False)

    with pd.option_context("display.width", 160, "display.max_rows", None, "display.precision", 3):
        print(f"\nTest NLL difference by {args.support} support bucket (a - b; negative = a better):")
        shown = comparisons[comparisons["comparison"].str.startswith(("RQ1", "RQ2", "order"))]
        print(shown.to_string(index=False))
        print("\nTest cells per bucket (seed mean):")
        counts = tables["cells"][tables["cells"]["variant"] == "bn"]
        print(counts.pivot_table(index=["dataset", "n"], columns="bucket", values="count_test")
              .reindex(columns=list(dict.fromkeys(counts["bucket"]))).round(0).to_string())
        print("\nLoss-based MI AUC by row bucket (min_j support; 0.5 = no signal):")
        mi_shown = mi[mi["variant"].isin(SHOWN)]
        print(mi_shown.pivot_table(index=["dataset", "n", "variant"], columns="bucket", values="mean",
                                   sort=False).to_string())
        print("\nSpearman(min_j support, vulnerability) over seeds (negative = rare rows more isolated):")
        print(vuln_summary.to_string(index=False))
    print(f"\nwrote {out}/")


if __name__ == "__main__":
    main()
