"""Export the evidence behind Online Resource 1, Section S35 (Tables S35-S35c), as CSV.

1. Re-runs the six validation-only rules of tools/analysis/validation_reselection_audit.py
   on the retained runs and checks that every selection equals the one recorded in
   results/validation_reselection_audit.json.
2. results/s35_rules.csv       one row per rule: score, direction, tie-break, pool, and the
                                configuration it selects on each dataset.
3. results/s35_candidates.csv  every candidate configuration (run family) per dataset with
                                its tuned difference from the main configuration and its
                                three-seed validation and test Recall@20.
4. results/s35_results.csv       per rule and dataset: selected configuration, its test Recall@20
                                change vs. SimGCL and TaxPro-CL's rank of six (Table S35).
5. results/s35_overall_rule_cells.csv  the four AB/Yelp x NC/LT cells under the
                                validation-Overall rule: per-seed and mean test Recall@20 of
                                the selected TaxPro-CL configuration and of default SimGCL,
                                unrounded, with the absolute and relative difference.

Usage: python -m tools.analysis.s35_export
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from tools.analysis.validation_reselection_audit import (
    DATASETS, GROUPS, LOG, RULES, SEEDS, build_pools, load_families, main_family_name, select)

ROOT = Path(__file__).resolve().parents[2]
RULE_TEXT = {
    "overall": ("validation Overall Recall@20", "maximize", "first candidate in scan order"),
    "lt": ("validation Long-Tail Recall@20", "maximize", "higher validation Overall"),
    "nc": ("validation Near-Cold Recall@20", "maximize", "higher validation Long-Tail"),
    "lt_within_1": ("validation Long-Tail Recall@20 among candidates with Overall >= 99% of the best",
                    "maximize", "higher validation Overall"),
    "lt_within_2": ("validation Long-Tail Recall@20 among candidates with Overall >= 98% of the best",
                    "maximize", "higher validation Overall"),
    "lt_within_3": ("validation Long-Tail Recall@20 among candidates with Overall >= 97% of the best",
                    "maximize", "higher validation Overall"),
}


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def simgcl_main_dirs(dataset):
    with open(ROOT / "results" / "results_manifest.csv", encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r["model"] == "SimGCL" and r["dataset"] == dataset and r["variant"].endswith("-main")]
    return {f"seed{r['seed']}": ROOT / r["run_dir"].replace("\\", "/") for r in rows}


def test_recall(run_dir, group):
    m = json.loads((run_dir / "final_test_metrics.json").read_text(encoding="utf-8"))
    return float(m["test_metrics"]["groupwise"][group]["recall"]["20"])


def main():
    recorded = json.loads((ROOT / "results" / "validation_reselection_audit.json").read_text(encoding="utf-8"))
    rules_rows, cand_rows, cell_rows, results_rows = [], [], [], []
    selected = {}
    for ds in DATASETS:
        families = load_families(ds)
        _main, method, _with_components = build_pools(families, main_family_name(ds))
        for fam in method:
            cand_rows.append({"dataset": ds, "config_id": fam["family"],
                              "differs_from_main": ";".join(f"{k}={v}" for k, v in fam["tuned_difference"].items()) or "main",
                              **{f"val_{g}": fam["val"][g] for g in GROUPS}, **{f"test_{g}": fam["test"][g] for g in GROUPS}})
        for rule in RULES:
            pick = select(method, rule)
            assert pick["family"] == recorded[ds]["selections"][rule]["family"], (ds, rule, pick["family"])
            selected[(ds, rule)] = pick["family"]
    for rule in RULES:
        score, direction, tie = RULE_TEXT[rule]
        rules_rows.append({"rule_id": rule, "score": score, "direction": direction, "tie_break": tie,
                           "pool": "TaxPro-CL runs equal to the main configuration except tuned keys; all 3 seeds",
                           "missing_metric": "run family excluded unless all three seeds have validation and test metrics",
                           "seed_aggregation": "three-seed mean at each run's best-validation epoch; one configuration for all seeds",
                           **{f"selected_{ds}": selected[(ds, rule)] for ds in DATASETS}})
        for ds in DATASETS:
            s = recorded[ds]["selections"][rule]
            vb = s["vs_baselines"]
            results_rows.append({"dataset": ds, "rule_id": rule, "config_id": s["family"],
                                 "is_main_configuration": s["is_main_configuration"], "configuration": s["configuration"],
                                 **{f"test_{g}_pct_vs_simgcl": vb[g]["pct_vs_simgcl"] for g in ("near_cold", "long_tail", "overall")},
                                 **{f"rank_of_six_{g}": vb[g]["rank_of_six"] for g in ("near_cold", "long_tail", "overall")}})
    for ds in ("amazon-book", "yelp2018"):
        fam = recorded[ds]["selections"]["overall"]["family"]
        sim = simgcl_main_dirs(ds)
        for g in ("near_cold", "long_tail"):
            tax = {s: test_recall(LOG / ds / fam / s, g) for s in SEEDS}
            base = {s: test_recall(sim[s], g) for s in SEEDS}
            tm, bm = statistics.fmean(tax.values()), statistics.fmean(base.values())
            cell_rows.append({"dataset": ds, "group": g, "rule": "overall", "config_id": fam,
                              **{f"taxpro_{s}": tax[s] for s in SEEDS}, "taxpro_mean": tm,
                              **{f"simgcl_{s}": base[s] for s in SEEDS}, "simgcl_mean": bm,
                              "diff_abs": tm - bm, "diff_pct": 100 * (tm / bm - 1),
                              "seeds_taxpro_below": sum(tax[s] < base[s] for s in SEEDS)})
            assert tm < bm, (ds, g)
    write_csv(ROOT / "results" / "s35_rules.csv", rules_rows)
    write_csv(ROOT / "results" / "s35_candidates.csv", cand_rows)
    write_csv(ROOT / "results" / "s35_overall_rule_cells.csv", cell_rows)
    write_csv(ROOT / "results" / "s35_results.csv", results_rows)
    print("all 24 selections reproduced; candidates:", {ds: sum(r["dataset"] == ds for r in cand_rows) for ds in DATASETS})
    for r in cell_rows:
        print(r["dataset"], r["group"], r["config_id"], f"{r['taxpro_mean']:.6f} vs {r['simgcl_mean']:.6f}",
              f"{r['diff_pct']:+.2f}%", "seeds below:", r["seeds_taxpro_below"])


if __name__ == "__main__":
    main()
