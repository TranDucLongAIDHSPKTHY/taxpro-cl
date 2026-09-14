"""A7 comparable-budget baseline tuning orchestrator.

Only TaxPro-CL is tuned per dataset in the main comparison (Table 7);
SimGCL, XSimGCL, and NCL are evaluated at each model's own published
default. This script gives the three SSL-CF baselines a tuning budget
comparable to what TaxPro-CL itself received: the identical 3-point
temperature grid used for TaxPro-CL's own A4 sensitivity check (Online
Resource 1, Table S4), {0.05, 0.10, 0.20}, swept for every (model, dataset)
pair, 3 seeds each (42, 0, 1) -- this formal grid supersedes an earlier ad
hoc search that produced the now-superseded NCL/Amazon-Book temp=0.03/0.08
runs (left on disk, no longer part of the reported grid).

Two cells of the grid coincide with runs already on disk and used
elsewhere in the paper (no retraining, no duplicate compute):
  - SimGCL @ temperature=0.2 is each dataset's existing A1/A2 baseline run
    (SimGCL's own published default).
  - NCL @ temperature=0.05 is each dataset's existing A1/A2 baseline run
    (NCL's own published default).
Everything else in the grid is fresh and is written under
log/p0/baseline/<model>/<dataset>/A7-temp<value>/seed<seed>/, resuming from
last_model.pt if a partial run is already there (e.g. NCL/amazon-book/
A7-temp0.1/seed42, paused mid-training on 2026-09-09).

Runs strictly sequentially (one GPU) and is idempotent / safe to re-launch
after an interruption -- it always re-checks run_manifest.json status
before deciding to skip, resume, or start a cell fresh.

Usage:
    python -m tools.experiments.run_a7_matched_budget
    python -m tools.experiments.run_a7_matched_budget --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TEMPS = [0.05, 0.1, 0.2]
SEEDS = [42, 0, 1]
# Dataset order: cheapest catalog first ("cheap-first-expensive-later").
DATASETS = ["musical-instruments", "arts-crafts-and-sewing", "yelp2018", "amazon-book"]
MODELS = ["SimGCL", "XSimGCL", "NCL"]

# (model, temperature) cells already covered by each model's own default-
# hyperparameter A1/A2 baseline run -- do not retrain these.
REUSE_DEFAULT = {
    ("SimGCL", 0.2),
    ("NCL", 0.05),
}

LOG_PATH = ROOT / "results" / "a7_matched_budget_progress.log"


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def cell_dir(model: str, dataset: str, temp: float, seed: int) -> Path:
    return ROOT / "log" / "p0" / "baseline" / model / dataset / f"A7-temp{temp}" / f"seed{seed}"


def manifest_status(run_dir: Path) -> str | None:
    manifest = run_dir / "run_manifest.json"
    if not manifest.exists():
        return None
    try:
        return json.loads(manifest.read_text(encoding="utf-8")).get("status")
    except Exception:
        return None


def run_one(model: str, dataset: str, temp: float, seed: int, device: str) -> None:
    run_dir = cell_dir(model, dataset, temp, seed)
    status = manifest_status(run_dir)
    if status == "completed":
        log(f"SKIP (already completed): {model} {dataset} temp={temp} seed={seed}")
        return

    resume_ckpt = run_dir / "last_model.pt"
    cmd = [
        sys.executable, "main.py",
        "--model", model,
        "--dataset", dataset,
        "--seed", str(seed),
        "--device", device,
        "--temperature", str(temp),
    ]
    if status == "running" and resume_ckpt.exists():
        cmd += ["--resume_checkpoint", str(resume_ckpt.relative_to(ROOT))]
        log(f"RESUME: {model} {dataset} temp={temp} seed={seed} from {resume_ckpt.name}")
    else:
        log(f"START: {model} {dataset} temp={temp} seed={seed}")

    env_output_dir = str(run_dir.relative_to(ROOT))
    import os
    env = os.environ.copy()
    env["TAXPROCL_RUN_OUTPUT_DIR"] = env_output_dir

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT), env=env)
    elapsed = time.time() - t0
    if result.returncode != 0:
        log(f"FAILED (exit={result.returncode}) after {elapsed:.0f}s: {model} {dataset} temp={temp} seed={seed}")
    else:
        log(f"DONE ({elapsed:.0f}s): {model} {dataset} temp={temp} seed={seed}")


def iter_cells():
    for model in MODELS:
        for dataset in DATASETS:
            for temp in TEMPS:
                if (model, temp) in REUSE_DEFAULT:
                    continue
                for seed in SEEDS:
                    yield model, dataset, temp, seed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    cells = list(iter_cells())
    log(f"A7 matched-budget grid: {len(cells)} cells total "
        f"({len(MODELS)} models x {len(DATASETS)} datasets x {len(TEMPS)} temps x {len(SEEDS)} seeds, "
        f"minus {len(REUSE_DEFAULT)} reused-default (model,temp) pairs x {len(DATASETS)} datasets x {len(SEEDS)} seeds)")

    if args.dry_run:
        for model, dataset, temp, seed in cells:
            run_dir = cell_dir(model, dataset, temp, seed)
            status = manifest_status(run_dir)
            print(f"{model:10s} {dataset:24s} temp={temp:<5} seed={seed:<3} status={status}")
        return 0

    for model, dataset, temp, seed in cells:
        run_one(model, dataset, temp, seed, args.device)

    log("A7 matched-budget grid: all cells processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
