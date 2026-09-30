"""CDs-and-Vinyl six-method, three-seed summary (Online Resource 1, Table S30b).

Reads each run's final_test_metrics.json["best_validation_metrics"] (the
group-wise validation snapshot at the epoch at which best_validation_model.pt
was saved; see tools/ranking/checkpoint_selection.py) and reports
per-method, per-group Recall@20 mean and std across seeds 0/1/42, plus a
win/loss tally of TaxPro-CL against each of the five baselines on Near-Cold
and Long-Tail.

Usage: python -m tools.analysis.cds_and_vinyl_summary"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RUN_DIRS = {
    "LightGCN": "log/p0/baseline/LightGCN/cds-and-vinyl/lightgcn-3b0f9c415824",
    "SGL-ED": "log/p0/baseline/SGL/cds-and-vinyl/sgl-8c2fb16d3f73",
    "SimGCL": "log/p0/baseline/SimGCL/cds-and-vinyl/simgcl-9981deb5004c",
    "XSimGCL": "log/p0/baseline/XSimGCL/cds-and-vinyl/xsimgcl-4063661ba733",
    "NCL": "log/p0/baseline/NCL/cds-and-vinyl/ncl-226da98203d3",
    "TaxPro-CL": "log/p0/taxprocl/cds-and-vinyl/taxpro-cl-no_merge-213d252b5ac8",
}
SEEDS = (0, 1, 42)
GROUPS = ("overall", "near_cold", "long_tail", "warm")


def load_group_recall20(run_dir, seed):
    path = PROJECT_ROOT / run_dir / "seed{}".format(seed) / "final_test_metrics.json"
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    groupwise = payload["best_validation_metrics"]["groupwise"]
    return {group: groupwise[group]["recall"]["20"] for group in GROUPS}


def main():
    per_method = {}
    for method, run_dir in RUN_DIRS.items():
        per_seed = {seed: load_group_recall20(run_dir, seed) for seed in SEEDS}
        per_method[method] = {
            group: [per_seed[seed][group] for seed in SEEDS] for group in GROUPS
        }

    print("Recall@20 (validation, best checkpoint), mean +- std over seeds {0,1,42}:")
    print()
    header = "{:<10}".format("Method") + "".join(
        "{:>22}".format(group) for group in GROUPS
    )
    print(header)
    for method, groups in per_method.items():
        row = "{:<10}".format(method)
        for group in GROUPS:
            values = groups[group]
            mean = statistics.mean(values)
            std = statistics.stdev(values)
            row += "{:>22}".format("{:.4f} +- {:.4f}".format(mean, std))
        print(row)

    print()
    print("TaxPro-CL vs. each baseline, mean Recall@20 (Near-Cold, Long-Tail):")
    wins, losses = 0, 0
    tax_means = {
        group: statistics.mean(per_method["TaxPro-CL"][group]) for group in GROUPS
    }
    for method in RUN_DIRS:
        if method == "TaxPro-CL":
            continue
        for group in ("near_cold", "long_tail"):
            base_mean = statistics.mean(per_method[method][group])
            outcome = "WIN" if tax_means[group] > base_mean else "LOSS"
            wins += outcome == "WIN"
            losses += outcome == "LOSS"
            print(
                "  TaxPro-CL {:.4f} vs {:<10} {:.4f} on {:<9} -> {}".format(
                    tax_means[group], method, base_mean, group, outcome
                )
            )
    print()
    print("Tally: TaxPro-CL wins {}/{} baseline-group comparisons".format(wins, wins + losses))

    out_path = PROJECT_ROOT / "results" / "cds_and_vinyl_summary.json"
    serializable = {
        method: {group: values for group, values in groups.items()}
        for method, groups in per_method.items()
    }
    with out_path.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "seeds": list(SEEDS),
                "metric": "recall@20",
                "split": "validation (best checkpoint)",
                "per_method_per_seed": serializable,
                "taxprocl_win_loss_near_cold_long_tail": {"wins": wins, "losses": losses},
            },
            stream,
            indent=2,
        )
    print()
    print("Wrote {}".format(out_path))


if __name__ == "__main__":
    main()
