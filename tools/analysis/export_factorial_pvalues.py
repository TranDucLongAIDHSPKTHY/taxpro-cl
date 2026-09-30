"""Export the full p-value vector of the RQ5 factorial (main paper Section 5.4; Online Resource 1,
Table S13e).

For every checkpoint set (current: main-configuration policy; nomerge: AB under no_merge),
dataset, factor (direction, epsilon-adaptivity), metric (Recall@20, NDCG@20), and group (NC, LT):
the two conditional contrasts' sign-flip p-values and signs, the intersection-union cell p (the
larger of the two), and, for the four 8-cell families of the current set (factor x metric), the
Holm rank, step threshold, decision, and Holm-adjusted p, plus the adjusted p in the pooled
32-test family. Also checks the numbers quoted in the main text.

Reads results/factorial_multiplicity.json; writes results/factorial_pvalues.csv.
Usage: python -m tools.analysis.export_factorial_pvalues
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAIRS = {"direction": ("direction_fixed_eps__V1-V0", "direction_adaptive_eps__V3-V2"),
         "epsilon": ("eps_random_dir__V2-V0", "eps_taxonomy_dir__V3-V1")}


def holm_adjusted(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    m, adj, running = len(ps), [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        adj[i] = running
    ranks = {i: r + 1 for r, i in enumerate(order)}
    return adj, ranks


def main():
    d = json.loads((ROOT / "results" / "factorial_multiplicity.json").read_text(encoding="utf-8"))
    rows = []
    for key, groups in d["combined_cells"].items():
        dataset, cset, factor, metric = key.split("|")
        for group, cell in groups.items():
            c1, c2 = PAIRS[factor]
            s1 = d["cells"][f"{dataset}|{cset}"][f"{c1}|{metric}|{group}"]
            s2 = d["cells"][f"{dataset}|{cset}"][f"{c2}|{metric}|{group}"]
            p_cell = max(s1["p_signflip_1e5_seed42"], s2["p_signflip_1e5_seed42"])
            assert abs(p_cell - cell["p_signflip_1e5_seed42"]) < 1e-15, key
            rows.append({"checkpoint_set": cset, "dataset": dataset, "factor": factor, "metric": metric, "group": group,
                         "contrast_1": c1.split("__")[1], "p_1": s1["p_signflip_1e5_seed42"], "sign_1": "+" if s1["mean_diff"] > 0 else ("-" if s1["mean_diff"] < 0 else "0"),
                         "contrast_2": c2.split("__")[1], "p_2": s2["p_signflip_1e5_seed42"], "sign_2": "+" if s2["mean_diff"] > 0 else ("-" if s2["mean_diff"] < 0 else "0"),
                         "p_cell": p_cell, "p_cell_1e6_seed123": cell.get("p_signflip_1e6_seed123", ""),
                         "holm_significant_reported": cell.get("holm_significant", "")})
    current = [r for r in rows if r["checkpoint_set"] == "current"]
    for factor in PAIRS:
        for metric in ("recall", "ndcg"):
            fam = [r for r in current if r["factor"] == factor and r["metric"] == metric]
            assert len(fam) == 8, (factor, metric, len(fam))
            adj, ranks = holm_adjusted([r["p_cell"] for r in fam])
            still = True
            for i in sorted(range(8), key=lambda k: ranks[k]):
                r = fam[i]
                r["holm_rank_8"], r["holm_threshold_8"] = ranks[i], 0.05 / (8 - ranks[i] + 1)
                still = still and r["p_cell"] <= r["holm_threshold_8"]
                r["holm_significant_8"], r["holm_adjusted_p_8"] = still, adj[i]
                assert r["holm_significant_8"] == r["holm_significant_reported"], (r["dataset"], factor, metric, r["group"])
    adj32, ranks32 = holm_adjusted([r["p_cell"] for r in current])
    for i, r in enumerate(current):
        r["holm_rank_32"], r["holm_adjusted_p_32"] = ranks32[i], adj32[i]

    ab = {(r["factor"], r["metric"], r["group"]): r for r in current if r["dataset"] == "amazon-book"}
    x = ab[("direction", "recall", "long_tail")]
    assert round(x["p_cell"], 3) == 0.016 and round(x["holm_adjusted_p_8"], 2) == 0.13 and not x["holm_significant_8"]
    y = ab[("direction", "ndcg", "long_tail")]
    assert round(y["p_cell"], 4) == 0.0019 and round(y["holm_adjusted_p_8"], 3) == 0.015 and y["holm_significant_8"]
    assert round(y["holm_adjusted_p_32"], 3) == 0.061
    assert sum(r["holm_significant_8"] for r in current) == 1

    fields = ["checkpoint_set", "dataset", "factor", "metric", "group", "contrast_1", "p_1", "sign_1", "contrast_2", "p_2", "sign_2",
              "p_cell", "p_cell_1e6_seed123", "holm_rank_8", "holm_threshold_8", "holm_significant_8", "holm_adjusted_p_8",
              "holm_rank_32", "holm_adjusted_p_32"]
    with open(ROOT / "results" / "factorial_pvalues.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["checkpoint_set"], r["factor"], r["metric"], r["dataset"], r["group"])))
    print(f"wrote {len(rows)} cells; quoted values reproduced: AB/LT Recall p_cell {x['p_cell']:.4f} adj {x['holm_adjusted_p_8']:.3f}; "
          f"NDCG p_cell {y['p_cell']:.4f} adj8 {y['holm_adjusted_p_8']:.3f} adj32 {y['holm_adjusted_p_32']:.3f}")


if __name__ == "__main__":
    main()
