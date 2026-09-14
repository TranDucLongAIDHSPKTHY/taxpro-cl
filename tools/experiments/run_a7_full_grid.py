"""A7 full-grid orchestrator: comparable-budget baseline tuning grid.

Supersedes run_a7_matched_budget.py (an earlier 1D temperature-only design
that produced 1 valid cell before the full grid scope below was settled on).

A7's requirement: tune SimGCL and XSimGCL on the full temperature x epsilon
grid
  tau in {0.05, 0.10, 0.15, 0.20}, epsilon in {0.05, 0.10, 0.20}
(12 combinations each), all 4 datasets, 3 seeds (42, 0, 1).

NCL is architecturally different (cluster-based, no perturbation-direction/
magnitude concept), so its comparable-budget axis is temperature x k
(number of clusters) instead of temperature x epsilon:
  tau in {0.05, 0.10, 0.15, 0.20}, k in {1000, 2000, 3000}
Amazon-Book only, matching where NCL's prior ad hoc exploration already
lives; that ad hoc exploration used temperature values 0.03/0.08 outside
this grid and is left as informal context, not part of this formal grid.

Reuse (no retraining): each model's own already-completed default-
hyperparameter run from the main A1/A2 comparison table covers exactly one
grid cell -- SimGCL (temperature=0.2, epsilon=0.05), XSimGCL
(temperature=0.15, epsilon=0.2), NCL (temperature=0.05, k=2000). All three
are detected automatically via DEFAULT_DIRS below (verified 3/3 seeds
completed, 2026-09-12). No other historical directory (e.g. the killed
1D-orchestrator's leftover single-axis runs, or NCL's 0.03/0.08 ad hoc
exploration) is special-cased -- the idempotent manifest check will simply
retrain those few cells fresh rather than risk a brittle path-mapping.

Runs strictly sequentially (one GPU); safe to re-launch after interruption
(re-checks run_manifest.json status before deciding to skip, resume, or
start a cell fresh).

Usage:
    python -m tools.experiments.run_a7_full_grid
    python -m tools.experiments.run_a7_full_grid --dry-run
    python -m tools.experiments.run_a7_full_grid --skip-ncl
    python -m tools.experiments.run_a7_full_grid --skip-simxsim
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATASETS = ["musical-instruments", "arts-crafts-and-sewing", "yelp2018", "amazon-book"]

SIMXSIM_TEMPS = [0.05, 0.10, 0.15, 0.20]
SIMXSIM_EPS = [0.05, 0.10, 0.20]
SEEDS = [42, 0, 1]

NCL_TEMPS = [0.05, 0.10, 0.15, 0.20]
NCL_KS = [1000, 2000, 3000]

# Each model's already-completed default-hyperparameter run (from the main
# A1/A2 comparison), keyed by dataset -> directory name under
# log/p0/baseline/<Model>/<dataset>/. Verified 3/3 seeds completed,
# 2026-09-12. (Musical-Instruments/SimGCL had two duplicate hash dirs on
# disk; simgcl-f263d2bff497 is the one with all 3 seeds completed.)
DEFAULT_DIRS = {
    "SimGCL": {
        "amazon-book": "simgcl-69bb0902f42f",
        "yelp2018": "simgcl-20c63ca8d16c",
        "musical-instruments": "simgcl-f263d2bff497",
        "arts-crafts-and-sewing": "simgcl-c8b32441f415",
    },
    "XSimGCL": {
        "amazon-book": "xsimgcl-7c3a56b5abcd",
        "yelp2018": "xsimgcl-daf2ce02a033",
        "musical-instruments": "xsimgcl-d60a996d56f1",
        "arts-crafts-and-sewing": "xsimgcl-fd0112231ebb",
    },
}
SIMXSIM_DEFAULT = {"SimGCL": (0.20, 0.05), "XSimGCL": (0.15, 0.20)}
NCL_DEFAULT_DIR = "log/p0/baseline/NCL/amazon-book"  # seed{0,1,42}, no subdir
NCL_DEFAULT = (0.05, 2000)

LOG_PATH = ROOT / "results" / "a7_full_grid_progress.log"


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def manifest_status(run_dir: Path) -> str | None:
    manifest = run_dir / "run_manifest.json"
    if not manifest.exists():
        return None
    try:
        return json.loads(manifest.read_text(encoding="utf-8")).get("status")
    except Exception:
        return None


def is_default_reused(model: str, dataset: str, param1: float, param2) -> bool:
    if model in ("SimGCL", "XSimGCL"):
        if (param1, param2) != SIMXSIM_DEFAULT[model]:
            return False
        default_dir = ROOT / "log" / "p0" / "baseline" / model / dataset / DEFAULT_DIRS[model][dataset]
        return all(manifest_status(default_dir / f"seed{s}") == "completed" for s in SEEDS)
    if model == "NCL":
        if (param1, param2) != NCL_DEFAULT:
            return False
        default_dir = ROOT / NCL_DEFAULT_DIR
        return all(manifest_status(default_dir / f"seed{s}") == "completed" for s in SEEDS)
    return False


def cell_dir(model: str, dataset: str, param1: float, param2, seed: int) -> Path:
    if model in ("SimGCL", "XSimGCL"):
        return ROOT / "log" / "p0" / "baseline" / model / dataset / f"A7-temp{param1}-eps{param2}" / f"seed{seed}"
    return ROOT / "log" / "p0" / "baseline" / model / dataset / f"A7-temp{param1}-k{param2}" / f"seed{seed}"


def run_one(model: str, dataset: str, param1: float, param2, seed: int, device: str) -> None:
    run_dir = cell_dir(model, dataset, param1, param2, seed)
    status = manifest_status(run_dir)
    if status == "completed":
        log(f"SKIP (already completed): {model} {dataset} p1={param1} p2={param2} seed={seed}")
        return

    resume_ckpt = run_dir / "last_model.pt"
    cmd = [sys.executable, "main.py", "--model", model, "--dataset", dataset,
           "--seed", str(seed), "--device", device, "--temperature", str(param1)]
    if model in ("SimGCL", "XSimGCL"):
        cmd += ["--epsilon", str(param2)]
    else:
        cmd += ["--k", str(param2)]

    if status == "running" and resume_ckpt.exists():
        cmd += ["--resume_checkpoint", str(resume_ckpt.relative_to(ROOT))]
        log(f"RESUME: {model} {dataset} p1={param1} p2={param2} seed={seed} from {resume_ckpt.name}")
    else:
        log(f"START: {model} {dataset} p1={param1} p2={param2} seed={seed}")

    env = os.environ.copy()
    env["TAXPROCL_RUN_OUTPUT_DIR"] = str(run_dir.relative_to(ROOT))

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT), env=env)
    elapsed = time.time() - t0
    if result.returncode != 0:
        log(f"FAILED (exit={result.returncode}) after {elapsed:.0f}s: {model} {dataset} p1={param1} p2={param2} seed={seed}")
    else:
        log(f"DONE ({elapsed:.0f}s): {model} {dataset} p1={param1} p2={param2} seed={seed}")


def iter_simxsim_cells():
    for model in ("SimGCL", "XSimGCL"):
        for dataset in DATASETS:
            for temp in SIMXSIM_TEMPS:
                for eps in SIMXSIM_EPS:
                    if is_default_reused(model, dataset, temp, eps):
                        continue
                    for seed in SEEDS:
                        yield model, dataset, temp, eps, seed


def iter_ncl_cells():
    for temp in NCL_TEMPS:
        for k in NCL_KS:
            if is_default_reused("NCL", "amazon-book", temp, k):
                continue
            for seed in SEEDS:
                yield "NCL", "amazon-book", temp, k, seed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-ncl", action="store_true", help="run only the SimGCL/XSimGCL grid")
    parser.add_argument("--skip-simxsim", action="store_true", help="run only the NCL grid")
    args = parser.parse_args(argv)

    cells = []
    if not args.skip_simxsim:
        cells += list(iter_simxsim_cells())
    if not args.skip_ncl:
        cells += list(iter_ncl_cells())

    log(f"A7 full grid: {len(cells)} cells total "
        f"(SimGCL/XSimGCL: {len(SIMXSIM_TEMPS)} temps x {len(SIMXSIM_EPS)} eps x {len(DATASETS)} datasets x {len(SEEDS)} seeds x 2 models minus reused defaults; "
        f"NCL: {len(NCL_TEMPS)} temps x {len(NCL_KS)} k x {len(SEEDS)} seeds, amazon-book only, minus reused default)")

    if args.dry_run:
        for model, dataset, p1, p2, seed in cells:
            run_dir = cell_dir(model, dataset, p1, p2, seed)
            status = manifest_status(run_dir)
            print(f"{model:10s} {dataset:24s} p1={p1:<5} p2={p2:<6} seed={seed:<3} status={status}")
        return 0

    for model, dataset, p1, p2, seed in cells:
        run_one(model, dataset, p1, p2, seed, args.device)

    log("A7 full grid: all cells processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
