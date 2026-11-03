"""Run every (dataset, n, variant, seed) in a stage config's `grid`.

    python scripts/run_grid.py --config configs/stage1.yaml --jobs 4 --skip-existing
    python scripts/run_grid.py --config configs/stage1.yaml --n 500 1000 --dry-run
    python scripts/run_grid.py --config configs/smoke.yaml --dry-run

Each run's output goes to train.log in its results directory. With --skip-existing, runs
that already wrote metrics.json are skipped, so a killed grid can be restarted.
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from joblib import Parallel, delayed

from tabstruct.utils import load_config, resolve_device, run_dir


@dataclass
class Run:
    cmd: list[str]
    out: Path
    seed: int

    @property
    def done(self) -> bool:
        return (self.out / "metrics.json").exists()


def grid_runs(
    config_path: str,
    datasets: list[str] | None = None,
    ns: list[str] | None = None,
    variants: list[str] | None = None,
    seeds: list[int] | None = None,
    results_dir: str | None = None,
) -> list[Run]:
    """Every run in the config's grid, optionally restricted to the given datasets, n
    values (as strings, e.g. "500" or "full"), variants and seeds."""
    cfg = load_config(config_path)
    grid = cfg["grid"]
    results_dir = results_dir or cfg["paths"]["results_dir"]
    runs = []
    for dataset, spec in grid["datasets"].items():
        if datasets and dataset not in datasets:
            continue
        all_variants = list(grid["variants"]) + list(spec.get("extra_variants", []))
        for n, variant, seed in itertools.product(spec["n"], all_variants, grid["seeds"]):
            if (ns and str(n) not in ns) or (variants and variant not in variants) or (
                seeds is not None and seed not in seeds
            ):
                continue
            cmd = [
                sys.executable, "-m", "tabstruct.train",
                "--config", config_path, "--stage", cfg["stage"],
                "--dataset", dataset, "--n", str(n), "--variant", variant, "--seed", str(seed),
            ]
            n_value = None if n == "full" else int(n)
            out = run_dir(results_dir, cfg["stage"], dataset, n_value, variant, seed, create=False)
            runs.append(Run(cmd, out, seed))
    return runs


def execute(run: Run, env: dict[str, str] | None, lock: threading.Lock) -> int:
    run.out.mkdir(parents=True, exist_ok=True)
    with open(run.out / "train.log", "w") as log:
        code = subprocess.call(run.cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    with lock:
        print(f"[{'ok' if code == 0 else 'FAILED'}] {run.out}", flush=True)
    return code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--jobs", type=int, default=None, help="parallel runs (default: 4 on a GPU, else 1)"
    )
    parser.add_argument("--dry-run", action="store_true", help="print commands only")
    parser.add_argument(
        "--skip-existing", action="store_true", help="skip runs that already wrote metrics.json"
    )
    parser.add_argument("--datasets", nargs="+", help="only these datasets")
    parser.add_argument("--n", nargs="+", dest="ns", help="only these n values, e.g. 500 full")
    parser.add_argument("--variants", nargs="+", help="only these variants")
    parser.add_argument("--seeds", nargs="+", type=int, help="only these seeds")
    args = parser.parse_args()

    runs = grid_runs(args.config, args.datasets, args.ns, args.variants, args.seeds)
    n_total = len(runs)
    if args.skip_existing:
        runs = [run for run in runs if not run.done]
    skipped = n_total - len(runs)
    if args.dry_run:
        for run in runs:
            print(" ".join(run.cmd[1:]))
        print(f"{len(runs)} runs" + (f" ({skipped} already done, skipped)" if skipped else ""))
        return
    if args.jobs is None:
        # One run uses a small fraction of a GPU, so several share it.
        device = resolve_device(load_config(args.config)["train"]["device"])
        args.jobs = 1 if device.type == "cpu" else 4
    print(f"{len(runs)} runs to do, {skipped} already done", flush=True)
    # The models are tiny: one thread per run is as fast as many, and avoids oversubscription.
    env = {**os.environ, "OMP_NUM_THREADS": "1"} if args.jobs > 1 else None
    # Seeds reuse the hyperparameters selected at train.hparam_seed, so that seed runs first.
    # Each run is a subprocess, so threads are enough to run them in parallel.
    hparam_seed = load_config(args.config)["train"]["hparam_seed"]
    lock = threading.Lock()
    failed = []
    for phase in ([r for r in runs if r.seed == hparam_seed], [r for r in runs if r.seed != hparam_seed]):
        codes = Parallel(n_jobs=args.jobs, prefer="threads")(
            delayed(execute)(run, env, lock) for run in phase
        )
        failed += [run for run, code in zip(phase, codes) if code != 0]
    print(f"{len(runs) - len(failed)}/{len(runs)} runs succeeded")
    for run in failed:
        print(f"  failed: {run.out / 'train.log'}")


if __name__ == "__main__":
    main()
