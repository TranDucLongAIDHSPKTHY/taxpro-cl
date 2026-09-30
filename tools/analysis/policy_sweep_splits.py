"""Taxonomy-policy sweep on both splits (Online Resource 1, Tables S3 and S30d;
main paper Section 4.1).

For every dataset's four-policy sweep (no_merge, merge_t5, merge_t10,
merge_t15; 3 seeds each), reads each run's best-validation checkpoint snapshot
(final_test_metrics.json["best_validation_metrics"]) and its test metrics
(final_test_group_metrics.json), reports mean +- sample std per group, and
applies the retrospective validation-only rule of main paper Section 4.1: keep
the nominal top scorer on validation Overall Recall@20, substituting no_merge
when its difference from the top scorer is smaller than the top scorer's
seed-to-seed standard deviation.

Reads run outputs only.

Usage:
    python -m tools.analysis.policy_sweep_splits
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TAXPRO = ROOT / "log" / "p0" / "taxprocl"
GROUPS = ("near_cold", "long_tail", "overall", "warm")
POLICIES = ("no_merge", "merge_t5", "merge_t10", "merge_t15")
RETAINED = {"amazon-book": "merge_t10", "yelp2018": "no_merge", "musical-instruments": "no_merge",
            "arts-crafts-and-sewing": "no_merge", "cds-and-vinyl": "no_merge"}
# Sweep runs: the four retained datasets keep theirs under a "chon_nguong_taxonomy"
# folder (the Arts-Crafts-and-Sewing folder name carries a Vietnamese diacritic);
# CDs-and-Vinyl's sweep is the no_merge run of Section S30 plus the three
# merge policies trained under the same configuration.
SWEEP_GLOB = {ds: "*nguong_taxonomy/*" for ds in RETAINED if ds != "cds-and-vinyl"}
SWEEP_GLOB["cds-and-vinyl"] = "taxpro-cl-*"


def policy_of(run_dir: Path) -> str:
    return next(p for p in ("merge_t10", "merge_t15", "merge_t5", "no_merge") if p in run_dir.name)


def summarise(run_dir: Path):
    val = {g: [] for g in GROUPS}
    test = {g: [] for g in GROUPS}
    for seed_dir in sorted(run_dir.glob("seed*")):
        final = json.loads((seed_dir / "final_test_metrics.json").read_text(encoding="utf-8"))
        groups = json.loads((seed_dir / "final_test_group_metrics.json").read_text(encoding="utf-8"))
        for g in GROUPS:
            val[g].append(final["best_validation_metrics"]["groupwise"][g]["recall"]["20"])
            test[g].append(groups[g]["recall"]["20"])
    stat = lambda xs: [statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0]  # noqa: E731
    return {"n_seeds": len(val["overall"]), "val": {g: stat(v) for g, v in val.items()},
            "test": {g: stat(v) for g, v in test.items()}, "run_dir": str(run_dir.relative_to(ROOT)).replace("\\", "/")}


def main() -> int:
    out = {}
    for dataset, pattern in SWEEP_GLOB.items():
        runs = [d for d in (TAXPRO / dataset).glob(pattern) if d.is_dir() and any(d.glob("seed*/final_test_metrics.json"))]
        by_policy = {}
        for run_dir in runs:
            by_policy.setdefault(policy_of(run_dir), []).append(run_dir)
        if set(by_policy) != set(POLICIES) or any(len(v) != 1 for v in by_policy.values()):
            print(f"{dataset}: sweep incomplete or ambiguous ({ {k: len(v) for k, v in by_policy.items()} }), skipped")
            continue
        result = {p: summarise(by_policy[p][0]) for p in POLICIES}
        overall = {p: result[p]["val"]["overall"] for p in POLICIES}
        top = max(overall, key=lambda p: overall[p][0])
        diff = overall[top][0] - overall["no_merge"][0]
        picks = "no_merge" if (top == "no_merge" or diff < overall[top][1]) else top
        result["_rule"] = {"validation_top": top, "top_minus_no_merge": diff, "top_sd": overall[top][1],
                           "rule_picks": picks, "retained": RETAINED[dataset],
                           "test_top": max(POLICIES, key=lambda p: result[p]["test"]["overall"][0])}
        out[dataset] = result
        print(dataset, result["_rule"])
    path = ROOT / "results" / "policy_sweep_splits.json"
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("Saved to", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
