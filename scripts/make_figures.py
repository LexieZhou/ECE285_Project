"""Regenerate every report figure and table from results/summary.csv into reports/.

    python scripts/make_figures.py
    python scripts/make_figures.py --summary results/summary.csv --stage stage1

Stage 1 writes reports/tables/stage1_variants.csv (mean and 95% CI per variant),
reports/tables/stage1_comparisons.csv (paired differences, see tabstruct.eval.compare)
and reports/figures/stage1_nll_vs_n.{pdf,png}.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

from tabstruct.eval.compare import paired_comparisons, variant_table

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
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    handles, labels = axes[0][-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               fontsize=9, labelcolor=INK, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    for suffix in (".pdf", ".png"):
        fig.savefig(path.with_suffix(suffix), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", default="results/summary.csv")
    parser.add_argument("--stage", default="stage1")
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


if __name__ == "__main__":
    main()
