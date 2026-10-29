"""Run every (dataset, n, variant, seed) in a stage config's `grid`.

    python scripts/run_grid.py --config configs/stage1.yaml --jobs 8
    python scripts/run_grid.py --config configs/smoke.yaml --dry-run
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys

from joblib import Parallel, delayed

from tabstruct.utils import load_config


def grid_commands(config_path: str) -> list[list[str]]:
    cfg = load_config(config_path)
    grid = cfg["grid"]
    commands = []
    for dataset, spec in grid["datasets"].items():
        variants = list(grid["variants"]) + list(spec.get("extra_variants", []))
        for n, variant, seed in itertools.product(spec["n"], variants, grid["seeds"]):
            commands.append([
                sys.executable, "-m", "tabstruct.train",
                "--config", config_path, "--stage", cfg["stage"],
                "--dataset", dataset, "--n", str(n), "--variant", variant, "--seed", str(seed),
            ])
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="print commands only")
    args = parser.parse_args()

    commands = grid_commands(args.config)
    if args.dry_run:
        for cmd in commands:
            print(" ".join(cmd[1:]))
        print(f"{len(commands)} runs")
        return
    codes = Parallel(n_jobs=args.jobs)(delayed(subprocess.call)(cmd) for cmd in commands)
    failed = sum(code != 0 for code in codes)
    print(f"{len(commands) - failed}/{len(commands)} runs succeeded")


if __name__ == "__main__":
    main()
