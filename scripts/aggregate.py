"""Collect results/<stage>/<dataset>/n<n>/<variant>/seed<s>/metrics.json into one CSV.

    python scripts/aggregate.py --results results --out results/summary.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def collect(results: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(results.glob("*/*/n*/*/seed*/metrics.json")):
        stage, dataset, n_tag, variant, seed_tag = path.parent.relative_to(results).parts
        metrics = pd.json_normalize(json.loads(path.read_text()), sep=".").iloc[0].to_dict()
        rows.append({
            "stage": stage,
            "dataset": dataset,
            "n": n_tag.removeprefix("n"),
            "variant": variant,
            "seed": int(seed_tag.removeprefix("seed")),
            **metrics,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/summary.csv")
    args = parser.parse_args()

    summary = collect(Path(args.results))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(f"wrote {len(summary)} runs to {args.out}")


if __name__ == "__main__":
    main()
