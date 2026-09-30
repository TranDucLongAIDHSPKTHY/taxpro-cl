"""TaxPro-CL against comparable-budget-tuned baselines (A7 grid; main paper Section 5.3,
Online Resource 1, Section S24).

For every dataset whose grid is complete, the tuned cell is selected from validation
Recall@20 alone (three-seed means at each run's best-validation checkpoint), with two
rules:
  overall  highest validation Overall Recall@20 (the primary A7 rule);
  R        highest validation Long-Tail among cells within 2% of the best validation
           Overall (rule R of docs/confirmatory_protocol.md, 2026-09-27, specified
           after the overall-rule results had been viewed; a sensitivity check).
The default-hyperparameter run (the main comparison's baseline) is one of the 12 cells.
For each selected cell, Near-Cold and Long-Tail test Recall@20 are compared with the
main TaxPro-CL checkpoints exactly as in main paper Table 11: seed-matched pairs,
per-user differences averaged over the three pairs, percentile bootstrap (2,000
resamples) and a two-sided paired sign-flip test (10^5 permutations, seed 42); Holm
within the 8-cell family (4 datasets x {NC, LT}) of each rule.

Inference only, no retraining.  Usage:
    python -m tools.analysis.a7_tuned_comparison [--models SimGCL XSimGCL]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analysis.factorial_multiplicity import holm_bonferroni, sign_flip_pvalue  # noqa: E402
from tools.analysis.seed_matched_bootstrap import (  # noqa: E402
    bootstrap_ci, compute_hits_for_seed_pair, user_diffs)

DATASETS = ("amazon-book", "yelp2018", "musical-instruments", "arts-crafts-and-sewing")
SEEDS = ("42", "0", "1")
TEMPS = (0.05, 0.1, 0.15, 0.2)
EPS = (0.05, 0.1, 0.2)
DEFAULT = {"SimGCL": (0.2, 0.05), "XSimGCL": (0.15, 0.2)}
GROUPS = ("near_cold", "long_tail", "warm", "overall")
OUT = ROOT / "results" / "a7_tuned_comparison.json"


def main_run_dirs(model, dataset):
    rows = csv.DictReader((ROOT / "results" / "results_manifest.csv").open(encoding="utf-8"))
    dirs = {r["seed"]: ROOT / r["run_dir"].replace("\\", "/") for r in rows
            if r["model"] == model and r["dataset"] == dataset and r["variant"].endswith("-main")}
    return [dirs[s] for s in SEEDS]


def cell_run_dirs(model, dataset, temp, eps):
    if (temp, eps) == DEFAULT[model]:
        return main_run_dirs(model, dataset)
    base = ROOT / "log" / "p0" / "baseline" / model / dataset / f"A7-temp{temp}-eps{eps}"
    return [base / f"seed{s}" for s in SEEDS]


def cell_means(run_dirs):
    """Three-seed means of test and best-checkpoint validation Recall@20, or None."""
    test = {g: [] for g in GROUPS}
    val = {g: [] for g in GROUPS}
    for d in run_dirs:
        if not (d / "final_test_group_metrics.json").exists():
            return None
        t = json.loads((d / "final_test_group_metrics.json").read_text(encoding="utf-8"))
        v = json.loads((d / "final_test_metrics.json").read_text(encoding="utf-8"))["best_validation_metrics"]
        v = v.get("groupwise", v)
        for g in GROUPS:
            test[g].append(float(t[g]["recall"]["20"]))
            val[g].append(float(v[g]["recall"]["20"]))
    return {"test": {g: statistics.fmean(x) for g, x in test.items()},
            "test_std": {g: statistics.stdev(x) for g, x in test.items()},
            "validation": {g: statistics.fmean(x) for g, x in val.items()}}


def select(cells):
    best_overall = max(cells, key=lambda k: cells[k]["validation"]["overall"])
    top = cells[best_overall]["validation"]["overall"]
    eligible = [k for k in cells if cells[k]["validation"]["overall"] >= 0.98 * top]
    best_r = max(eligible, key=lambda k: cells[k]["validation"]["long_tail"])
    return {"overall": best_overall, "R": best_r}


def compare(dataset, taxpro_dirs, baseline_dirs, device):
    per_seed = [compute_hits_for_seed_pair(dataset, str(t), str(b), device, 20, 256)
                for t, b in zip(taxpro_dirs, baseline_dirs)]
    out = {}
    for group, idx in (("near_cold", (0, 1, 2)), ("long_tail", (3, 4, 5))):
        pooled = {}
        for per_user in per_seed:
            for user, diff in user_diffs(per_user, *idx).items():
                pooled.setdefault(user, []).append(diff)
        values = [float(np.mean(v)) for v in pooled.values()]
        stat = bootstrap_ci(values, 2000, np.random.default_rng(42))
        stat["p_signflip"] = float(sign_flip_pvalue(values, 100_000, np.random.default_rng(42)))
        out[group] = stat
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["SimGCL", "XSimGCL"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--reuse", action="store_true",
                        help="reuse vs_taxpro statistics already in the output file for the same selected cell")
    args = parser.parse_args(argv)
    previous = json.loads(OUT.read_text(encoding="utf-8")) if (args.reuse and OUT.exists()) else {"models": {}}
    results = {"rules": {"overall": "highest validation Overall Recall@20",
                         "R": "highest validation Long-Tail within 2% of the best validation Overall"},
               "models": {}}
    for model in args.models:
        per_dataset = {}
        for dataset in DATASETS:
            cells = {}
            for t in TEMPS:
                for e in EPS:
                    m = cell_means(cell_run_dirs(model, dataset, t, e))
                    if m is not None:
                        cells[(t, e)] = m
            taxpro_dirs = main_run_dirs("TaxPro-CL", dataset)
            entry = {"cells_complete": len(cells), "taxpro_cl": cell_means(taxpro_dirs),
                     "default": {"cell": list(DEFAULT[model]), **cells.get(DEFAULT[model], {})}}
            if len(cells) == 12:
                chosen = select(cells)
                for rule, key in chosen.items():
                    print(f"{model} {dataset} rule {rule}: cell {key}")
                    old = previous["models"].get(model, {}).get(dataset, {}).get(f"selected_{rule}", {})
                    if old.get("cell") == list(key) and "vs_taxpro" in old:
                        stats = old["vs_taxpro"]
                    else:
                        stats = compare(dataset, taxpro_dirs, cell_run_dirs(model, dataset, *key), args.device)
                    entry[f"selected_{rule}"] = {"cell": list(key), **cells[key], "vs_taxpro": stats}
                entry["all_cells"] = [{"cell": list(k), **v} for k, v in sorted(cells.items())]
            per_dataset[dataset] = entry
        for rule in ("overall", "R"):
            family = [{"dataset": d, "group": g, "p": per_dataset[d][f"selected_{rule}"]["vs_taxpro"][g]["p_signflip"]}
                      for d in DATASETS if f"selected_{rule}" in per_dataset[d] for g in ("near_cold", "long_tail")]
            if len(family) == 8:
                holm_bonferroni(family, "p", "holm_significant")
                for cell in family:
                    per_dataset[cell["dataset"]][f"selected_{rule}"]["vs_taxpro"][cell["group"]].update(
                        holm_significant=cell["holm_significant"], holm_threshold=cell["holm_significant_threshold"])
        results["models"][model] = per_dataset
    OUT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
