"""B2/Limitation-5 interim re-analysis (JIIS V58 review): for the datasets
where the ongoing A7 comparable-budget grid (tau in {0.05,0.10,0.15,0.20} x
epsilon in {0.05,0.10,0.20}, SimGCL, 3 seeds) has already finished as of
this check, select the validation-best (tau, epsilon) cell per dataset and
report its test Near-Cold/Long-Tail/Overall/Warm Recall@20 against
TaxPro-CL's main configuration.

This does NOT resolve Limitation 5 -- the A7 grid was still running (on
Amazon-Book's SimGCL cells, and had not yet started XSimGCL or NCL) at the
time this script was run; see results/a7_full_grid_progress.log for the
live status. This script only extracts what is already complete and
usable without retraining anything itself, exactly as instructed: use
existing evidence before proposing new experiments.

Model selection matches the paper's own rule everywhere else: for each
(tau, epsilon) cell, the checkpoint is chosen by best validation Recall@20
Overall (max over recorded epochs in validation_metrics.json); the cell
chosen for this report is the one with the highest mean best-validation
Recall@20 Overall across its 3 seeds -- i.e. comparable-budget tuning
selects a temperature/epsilon combination the same way the paper selects
an epoch, on validation only, never on test.
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = ROOT / "log" / "p0" / "baseline" / "SimGCL"
METRICS_SEED_CSV = ROOT / "results" / "metrics_seed.csv"

TEMPS = [0.05, 0.10, 0.15, 0.20]
EPS = [0.05, 0.10, 0.20]
SEEDS = ["42", "0", "1"]


def cell_dirs(dataset):
    """Yield (temp, eps, cell_dir) for the 11 A7-trained cells (grid minus
    the reused default temp=0.2/eps=0.05, which is SimGCL-main itself)."""
    ds_dir = BASELINE_DIR / dataset
    for temp in TEMPS:
        for eps in EPS:
            if abs(temp - 0.20) < 1e-9 and abs(eps - 0.05) < 1e-9:
                continue  # reused default = SimGCL-main, handled separately
            cell_dir = ds_dir / f"A7-temp{temp}-eps{eps}"
            yield temp, eps, cell_dir


def best_val_recall20_overall(seed_dir):
    path = seed_dir / "validation_metrics.json"
    if not path.exists():
        return None
    entries = json.loads(path.read_text(encoding="utf-8"))
    values = [e["primary_value"] for e in entries if e.get("primary_metric") == "recall@20"]
    return max(values) if values else None


def cell_is_complete(cell_dir):
    return all((cell_dir / f"seed{s}" / "final_test_group_metrics.json").exists() for s in SEEDS)


def cell_test_means(cell_dir):
    groups = {"near_cold": [], "long_tail": [], "overall": [], "warm": []}
    val_scores = []
    for s in SEEDS:
        seed_dir = cell_dir / f"seed{s}"
        metrics = json.loads((seed_dir / "final_test_group_metrics.json").read_text(encoding="utf-8"))
        for g in groups:
            groups[g].append(metrics[g]["recall"]["20"])
        v = best_val_recall20_overall(seed_dir)
        if v is not None:
            val_scores.append(v)
    return {
        g: (statistics.fmean(v), statistics.stdev(v) if len(v) > 1 else 0.0)
        for g, v in groups.items()
    }, (statistics.fmean(val_scores) if val_scores else None)


def simgcl_main_from_metrics_seed(dataset):
    """SimGCL-main (temp=0.2, eps=0.05 default) test means, reused not retrained."""
    rows = list(csv.DictReader(METRICS_SEED_CSV.open(encoding="utf-8")))
    groups = {"near_cold": [], "long_tail": [], "overall": [], "warm": []}
    for row in rows:
        if row["model"] == "SimGCL" and row["dataset"] == dataset and row["variant"] == "SimGCL-main" and row["group"] in groups:
            groups[row["group"]].append(float(row["recall_at_20"]))
    return {g: (statistics.fmean(v), statistics.stdev(v) if len(v) > 1 else 0.0) for g, v in groups.items()}


def taxpro_main_from_metrics_seed(dataset):
    rows = list(csv.DictReader(METRICS_SEED_CSV.open(encoding="utf-8")))
    groups = {"near_cold": [], "long_tail": [], "overall": [], "warm": []}
    for row in rows:
        if row["model"] == "TaxPro-CL" and row["dataset"] == dataset and row["variant"] == "TaxPro-CL-main" and row["group"] in groups:
            groups[row["group"]].append(float(row["recall_at_20"]))
    return {g: (statistics.fmean(v), statistics.stdev(v) if len(v) > 1 else 0.0) for g, v in groups.items()}


def analyze_dataset(dataset):
    completed_cells = []
    incomplete_cells = []
    for temp, eps, cell_dir in cell_dirs(dataset):
        if cell_is_complete(cell_dir):
            means, val_mean = cell_test_means(cell_dir)
            completed_cells.append({"temp": temp, "eps": eps, "means": means, "val_recall20_overall": val_mean})
        else:
            incomplete_cells.append({"temp": temp, "eps": eps})

    # Add the reused-default cell (SimGCL-main) as the 12th grid point.
    default_means = simgcl_main_from_metrics_seed(dataset)
    completed_cells.append({"temp": 0.20, "eps": 0.05, "means": default_means, "val_recall20_overall": None,
                             "note": "reused SimGCL-main default, not part of A7 retraining"})

    grid_complete = len(incomplete_cells) == 0
    scored = [c for c in completed_cells if c["val_recall20_overall"] is not None]
    best = max(scored, key=lambda c: c["val_recall20_overall"]) if scored else None

    return {
        "dataset": dataset,
        "grid_complete": grid_complete,
        "n_cells_complete": len(completed_cells),
        "n_cells_incomplete": len(incomplete_cells),
        "incomplete_cells": incomplete_cells,
        "best_cell_by_validation_overall": best,
        "taxpro_cl_main": taxpro_main_from_metrics_seed(dataset),
        "simgcl_default": default_means,
    }


def main():
    datasets_to_check = ["musical-instruments", "arts-crafts-and-sewing", "yelp2018", "amazon-book"]
    results = {ds: analyze_dataset(ds) for ds in datasets_to_check}
    out_path = ROOT / "results" / "b2_interim_matched_budget.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Wrote", out_path)
    for ds, r in results.items():
        print(f"\n=== {ds} === grid_complete={r['grid_complete']} "
              f"({r['n_cells_complete']} complete / {r['n_cells_incomplete']} incomplete)")
        if r["best_cell_by_validation_overall"]:
            b = r["best_cell_by_validation_overall"]
            print(f"  best cell: temp={b['temp']} eps={b['eps']} "
                  f"val_recall20_overall={b['val_recall20_overall']}")
            for g in ("near_cold", "long_tail", "overall", "warm"):
                bm, bs = b["means"][g]
                tm, ts = r["taxpro_cl_main"][g]
                print(f"    {g}: best-SimGCL-cell={bm*1000:.4f}e-3 (+-{bs*1000:.4f})  "
                      f"TaxPro-CL-main={tm*1000:.4f}e-3 (+-{ts*1000:.4f})")
    return results


if __name__ == "__main__":
    main()
