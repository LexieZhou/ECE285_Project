"""Regenerate every report figure and table from results/summary.csv into reports/.

    python scripts/make_figures.py
    python scripts/make_figures.py --summary results/summary.csv --stage stage1

Stage 1 writes reports/tables/stage1_variants.csv (mean and 95% CI per variant),
reports/tables/stage1_comparisons.csv (paired differences, see tabstruct.eval.compare)
and reports/figures/stage1_nll_vs_n.{pdf,png}. If scripts/analyze_support.py has run, it
also draws reports/figures/stage2_<support>_support_{rq1,true_dag}.{pdf,png} from
results/stage2/<support>/bucket_comparisons.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

from tabstruct.eval.compare import n_sort_key, paired_comparisons, variant_table

# Fixed categorical slots (validated: CVD-separable on white); every series also has its
# own marker, since three slots are below 3:1 contrast. `marginal` is far above the rest
# and is left out of the figure (it is in the table).
SERIES = {
    "full_prefix": ("#2a78d6", "o"),
    "bn": ("#eb6834", "s"),
    "random_dag": ("#1baf7a", "^"),
    "true_dag": ("#eda100", "D"),
    "bn_cpt": ("#e87ba4", "v"),
    "bn_hdp": ("#008300", "P"),
}
LABELS = {"random_dag": "random_dag (mean of 5)"}
DATASET_ORDER = ["adult", "child", "insurance", "alarm"]  # bnlearn by network size
# Stage 2: one ordinal blue ramp for n (validated light -> dark), fixed buckets.
N_SHADES = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
BUCKETS = ["zero", "low", "medium", "high"]
BUCKET_LABELS = ["zero\n0", "low\n1–4", "medium\n5–19", "high\n≥20"]
BUCKET_FIGURES = [("RQ1: bn - full_prefix", "rq1"),
                  ("order: true_dag - full_prefix_trueorder", "true_dag")]
INK, MUTED, GRID = "#1f1f1e", "#5f5e5a", "#e6e5e0"


def plot_nll_vs_n(summary: pd.DataFrame, path: Path) -> None:
    """One panel per dataset: test NLL (Adult) or excess NLL over the true BN (bnlearn)
    against training size, mean and 95% CI over seeds."""
    datasets = sorted(summary["dataset"].unique(),
                      key=lambda d: DATASET_ORDER.index(d) if d in DATASET_ORDER else len(DATASET_ORDER))
    n_train = summary.assign(n=summary["n"].astype(str)).groupby(["dataset", "n"])["n_train"].first()
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.4 * len(datasets), 3.2), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        rows = summary[summary["dataset"] == dataset]
        excess = rows["excess_nll_test"].notna().all() if "excess_nll_test" in rows else False
        table = variant_table(rows, "excess_nll_test" if excess else "nll_test")
        for i, (variant, (color, marker)) in enumerate(SERIES.items()):
            t = table[table["variant"] == variant]
            if t.empty:
                continue
            x = [n_train[(dataset, n)] * 1.04 ** (i - 2.5) for n in t["n"]]  # dodge CI bars
            ax.errorbar(x, t["mean"], yerr=t["ci95"], color=color, marker=marker, markersize=6,
                        linewidth=2, capsize=0, elinewidth=1.2, label=LABELS.get(variant, variant))
        ax.set_xscale("log")
        sizes = n_train[dataset].sort_values()
        ax.xaxis.set_major_locator(FixedLocator(sizes.to_numpy()))
        ax.xaxis.set_major_formatter(FixedFormatter(
            [f"full\n({v:,})" if n == "full" else f"{v:,}" for n, v in sizes.items()]
        ))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_title(dataset, color=INK, fontsize=11)
        ax.set_xlabel("training rows", color=MUTED, fontsize=9)
        ax.set_ylabel("excess test NLL (nats/row)" if excess else "test NLL (nats/row)",
                      color=MUTED, fontsize=9)
        _style(ax)
    handles, labels = axes[0][-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               fontsize=9, labelcolor=INK, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    for suffix in (".pdf", ".png"):
        fig.savefig(path.with_suffix(suffix), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_bucket_comparison(comparisons: pd.DataFrame, comparison: str, path: Path) -> None:
    """One panel per dataset: mean (95% CI over seeds) per-cell test NLL difference of
    `comparison` in each support bucket, one bar per n (light = small n)."""
    comp = comparisons[comparisons["comparison"] == comparison]
    datasets = [d for d in DATASET_ORDER if d in set(comp["dataset"])]
    ns = sorted(comp["n"].astype(str).unique(), key=n_sort_key)
    shades = dict(zip(ns, N_SHADES[-len(ns):] if len(ns) <= len(N_SHADES) else N_SHADES))
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.4 * len(datasets), 3.0), squeeze=False)
    width = 0.8 / len(ns)
    for ax, dataset in zip(axes[0], datasets):
        rows = comp[comp["dataset"] == dataset].assign(n=lambda df: df["n"].astype(str))
        for i, n in enumerate(ns):
            r = rows[rows["n"] == n].set_index("bucket").reindex(BUCKETS)
            x = [b + (i - (len(ns) - 1) / 2) * width for b in range(len(BUCKETS))]
            ax.bar(x, r["mean_diff"], width=width * 0.9, yerr=r["ci95"], color=shades[n],
                   error_kw={"elinewidth": 1.2, "ecolor": INK}, label=f"n = {n}")
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_xticks(range(len(BUCKETS)), BUCKET_LABELS)
        ax.set_title(dataset, color=INK, fontsize=11)
        ax.set_ylabel(f"{comparison.split(': ')[-1]}\n(test NLL, nats/cell)", color=MUTED, fontsize=9)
        _style(ax, grid_axis="y")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               fontsize=9, labelcolor=INK, bbox_to_anchor=(0.5, 1.07))
    fig.text(0.5, -0.04, "below 0: the first variant has lower NLL in that bucket", ha="center",
             color=MUTED, fontsize=8)
    fig.tight_layout()
    for suffix in (".pdf", ".png"):
        fig.savefig(path.with_suffix(suffix), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _style(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", default="results/summary.csv")
    parser.add_argument("--stage", default="stage1")
    parser.add_argument("--stage2", default="results/stage2", help="analyze_support.py output")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()

    summary = pd.read_csv(args.summary)
    summary = summary[summary["stage"] == args.stage]
    if summary.empty:
        raise SystemExit(f"no {args.stage} runs in {args.summary}")
    tables, figures = Path(args.out) / "tables", Path(args.out) / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    variants = variant_table(summary)
    comparisons = paired_comparisons(summary)
    variants.to_csv(tables / f"{args.stage}_variants.csv", index=False)
    comparisons.to_csv(tables / f"{args.stage}_comparisons.csv", index=False)
    plot_nll_vs_n(summary, figures / f"{args.stage}_nll_vs_n")

    with pd.option_context("display.width", 140, "display.max_rows", None, "display.precision", 3):
        print(comparisons.to_string(index=False))
    print(f"\nwrote {tables}/{args.stage}_*.csv and {figures}/{args.stage}_nll_vs_n.{{pdf,png}}")

    for support_dir in sorted(Path(args.stage2).glob("*/bucket_comparisons.csv")):
        support = support_dir.parent.name
        bucket = pd.read_csv(support_dir)
        for comparison, tag in [(c, t) for c, t in BUCKET_FIGURES if c in set(bucket["comparison"])]:
            path = figures / f"stage2_{support}_support_{tag}"
            plot_bucket_comparison(bucket, comparison, path)
            print(f"wrote {path}.{{pdf,png}}")


if __name__ == "__main__":
    main()
