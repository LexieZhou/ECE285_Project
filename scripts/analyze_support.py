"""Stage 2 (RQ3): per-column NLL by support bucket and support-vs-vulnerability.

Reads Stage 1 outputs (per_column_nll.npy, graph.json) and writes bucket tables and the
Spearman correlation between each training row's min BN-parent support and its
vulnerability score to results/stage2/.

    python scripts/analyze_support.py --results results
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default="results")
    parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
