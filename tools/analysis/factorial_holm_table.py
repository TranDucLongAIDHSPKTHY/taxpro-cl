"""Extended Holm table for the RQ5 factorial (Online Resource 1, Table S13e;
main paper Section 5.4).

From results/factorial_multiplicity.json (no new inference), lists for
each of the four 8-cell families (factor: direction, epsilon-adaptivity x
metric: Recall@20, NDCG@20; cells: dataset x {near_cold, long_tail}) the two
conditional contrasts' sign-flip p-values and signs, the intersection-union
cell p (their maximum), the Holm rank, threshold, adjusted p and decision, and
the Holm-adjusted p under a single pooled family of all 32 cells (first
threshold 0.05/32 = 0.0015625).

Usage:
    python -m tools.analysis.factorial_holm_table
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("amazon-book", "yelp2018", "musical-instruments", "arts-crafts-and-sewing")
PAIRS = {"direction": ("direction_fixed_eps__V1-V0", "direction_adaptive_eps__V3-V2"),
         "epsilon": ("eps_random_dir__V2-V0", "eps_taxonomy_dir__V3-V1")}
P_KEY = "p_signflip_1e5_seed42"


def holm(rows, alpha=0.05, prefix=""):
    rows = sorted(rows, key=lambda r: r["cell_p"])
    m = len(rows)
    running = 0.0
    still = True
    for k, row in enumerate(rows):
        threshold = alpha / (m - k)
        running = max(running, (m - k) * row["cell_p"])
        row[prefix + "rank"] = k + 1
        row[prefix + "threshold"] = threshold
        row[prefix + "adjusted_p"] = min(1.0, running)
        row[prefix + "significant"] = still and row["cell_p"] <= threshold
        still = row[prefix + "significant"]
    return rows


def main() -> int:
    data = json.loads((ROOT / "results" / "factorial_multiplicity.json").read_text(encoding="utf-8"))
    rows = []
    for factor, (a, b) in PAIRS.items():
        for metric in ("recall", "ndcg"):
            family = []
            for dataset in DATASETS:
                for group in ("near_cold", "long_tail"):
                    c1 = data["cells"][f"{dataset}|current"][f"{a}|{metric}|{group}"]
                    c2 = data["cells"][f"{dataset}|current"][f"{b}|{metric}|{group}"]
                    family.append({"factor": factor, "metric": metric, "dataset": dataset, "group": group,
                                   "p_first": c1[P_KEY], "p_second": c2[P_KEY],
                                   "diff_first": c1["mean_diff"], "diff_second": c2["mean_diff"],
                                   "cell_p": max(c1[P_KEY], c2[P_KEY])})
            rows += holm(family)
    holm(rows, prefix="pooled32_")
    out = ROOT / "results" / "factorial_holm_table.json"
    out.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    for row in sorted(rows, key=lambda r: r["cell_p"])[:3]:
        print(row["factor"], row["metric"], row["dataset"], row["group"], f"p={row['cell_p']:.5f}",
              f"family adj={row['adjusted_p']:.3f} sig={row['significant']}",
              f"pooled adj={row['pooled32_adjusted_p']:.3f} sig={row['pooled32_significant']}")
    print("Saved to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
