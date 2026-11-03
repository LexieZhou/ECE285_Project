import importlib.util
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_grid", ROOT / "scripts/run_grid.py")
run_grid = importlib.util.module_from_spec(spec)
sys.modules["run_grid"] = run_grid  # dataclasses look up their module
spec.loader.exec_module(run_grid)

STAGE1 = str(ROOT / "configs/stage1.yaml")


def test_grid_size_and_filters(tmp_path):
    assert len(run_grid.grid_runs(STAGE1, results_dir=tmp_path)) == 740
    # Adult: 10 variants x 5 seeds per n.
    adult = run_grid.grid_runs(STAGE1, datasets=["adult"], ns=["500", "full"], results_dir=tmp_path)
    assert len(adult) == 2 * 10 * 5
    assert {run.out.parts[-3] for run in adult} == {"n500", "nfull"}
    small = run_grid.grid_runs(STAGE1, ns=["500", "1000"], seeds=[0], results_dir=tmp_path)
    assert len(small) == 2 * (10 + 3 * 12)
    assert not any(p.is_dir() for p in tmp_path.iterdir())  # listing runs creates nothing


def test_out_matches_train_layout_and_done(tmp_path):
    run = run_grid.grid_runs(
        STAGE1, datasets=["child"], ns=["500"], variants=["true_dag"], seeds=[3], results_dir=tmp_path
    )[0]
    assert run.out == tmp_path / "stage1" / "child" / "n500" / "true_dag" / "seed3"
    assert run.cmd[-8:] == ["--dataset", "child", "--n", "500", "--variant", "true_dag", "--seed", "3"]
    assert not run.done
    run.out.mkdir(parents=True)
    (run.out / "metrics.json").write_text("{}")
    assert run.done


def test_execute_writes_log(tmp_path):
    ok = run_grid.Run([sys.executable, "-c", "print('hello')"], tmp_path / "a", seed=0)
    bad = run_grid.Run([sys.executable, "-c", "raise SystemExit(3)"], tmp_path / "b", seed=0)
    lock = threading.Lock()
    assert run_grid.execute(ok, None, lock) == 0
    assert (tmp_path / "a" / "train.log").read_text().strip() == "hello"
    assert run_grid.execute(bad, None, lock) == 3
