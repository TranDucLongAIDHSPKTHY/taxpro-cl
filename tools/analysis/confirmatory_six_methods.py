"""Descriptive six-method comparison on the held-out dataset (docs/confirmatory_protocol.md,
Section 7, "descriptive only"): three-seed mean and std of test Recall@20 and NDCG@20 by
group for the selected TaxPro-CL and the five default baselines, the rank of each method
(1 = highest mean), and, for context, the tuned SimGCL (rule R).

Usage: python -m tools.analysis.confirmatory_six_methods --dataset office-products
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDS = ("42", "0", "1")
GROUPS = ("near_cold", "long_tail", "warm", "overall")


def summarize(run_dir):
    per = [json.loads((run_dir / f"seed{s}" / "final_test_group_metrics.json").read_text(encoding="utf-8")) for s in SEEDS]
    return {g: {m: {"mean": statistics.fmean(float(p[g][m]["20"]) for p in per),
                    "std": statistics.stdev(float(p[g][m]["20"]) for p in per)} for m in ("recall", "ndcg")} for g in GROUPS}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="office-products")
    args = ap.parse_args(argv)
    base = ROOT / "log" / "p0" / "confirmatory" / args.dataset
    sel = json.loads((base / "confirmatory_selection.json").read_text(encoding="utf-8"))
    policy, tax = sel["step1_policy"]["selected"], sel["step2_taxpro"]["selected"]
    dirs = {
        "TaxPro-CL": base / "TaxPro-CL" / f"step2-{policy}-{tax}",
        "LightGCN": base / "LightGCN" / "default",
        "SGL-ED": base / "SGL" / "default",
        "SimGCL": base / "SimGCL" / f"grid-{sel['step3_tuned_simgcl']['default_cell']}",
        "XSimGCL": base / "XSimGCL" / "default",
        "NCL": base / "NCL" / "default",
    }
    table = {name: summarize(d) for name, d in dirs.items()}
    ranks = {m: {g: {name: 1 + sum(1 for o in table if table[o][g][m]["mean"] > table[name][g][m]["mean"]) for name in table}
                 for g in GROUPS} for m in ("recall", "ndcg")}
    context = {"SimGCL tuned (rule R)": summarize(base / "SimGCL" / f"grid-{sel['step3_tuned_simgcl']['selected']}")}
    out = {"dataset": args.dataset, "run_dirs": {k: str(v.relative_to(ROOT)) for k, v in dirs.items()},
           "six_methods": table, "ranks_of_six": ranks, "context": context}
    path = ROOT / "results" / "confirmatory" / f"six_methods_{args.dataset}.json"
    path.write_bytes((json.dumps(out, indent=2) + "\n").encode("utf-8"))
    for name in table:
        print(f"{name:10s}", {g: (round(1e3 * table[name][g]['recall']['mean'], 3), ranks['recall'][g][name]) for g in GROUPS})
    c = context["SimGCL tuned (rule R)"]
    print("SimGCL-R  ", {g: round(1e3 * c[g]["recall"]["mean"], 3) for g in GROUPS})
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
