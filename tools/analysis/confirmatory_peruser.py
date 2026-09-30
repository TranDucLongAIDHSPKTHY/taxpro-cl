"""Per-user test metrics of the held-out evaluation (docs/confirmatory_protocol.md; main paper
Table 12; Online Resource 1, Section S36): Recall@20 and NDCG@20 of every Near-Cold and Long-Tail user for each run whose
test split was opened for the analysis (selected TaxPro-CL, V0-V2, SimGCL rule R / rule Ov / default),
three seeds. Recomputed from the saved checkpoints with the evaluator's ranking; the per-run group means
are checked against final_test_group_metrics.json.

Writes results/confirmatory/peruser_office-products.csv.gz.
Usage: python -m tools.analysis.confirmatory_peruser --dataset office-products
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics
from pathlib import Path

import torch

from tools.analysis.factorial_multiplicity import SEEDS, score_checkpoint_set

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="office-products")
    args = ap.parse_args(argv)
    an = json.loads((ROOT / "results" / "confirmatory" / f"confirmatory_analysis_{args.dataset}.json").read_text(encoding="utf-8"))
    runs = an["run_dirs"]  # variant -> relative run family
    runs = {k: v for k, v in runs.items() if k != "V3"}  # V3 is the selected TaxPro-CL run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores = score_checkpoint_set(args.dataset, runs, device)
    out = ROOT / "results" / "confirmatory" / f"peruser_{args.dataset}.csv.gz"
    n = 0
    with gzip.open(out, "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["variant", "run_family", "seed", "group", "user_id", "recall_at_20", "ndcg_at_20"])
        for variant, rel in runs.items():
            for seed in SEEDS:
                ref = json.loads((ROOT / rel / f"seed{seed}" / "final_test_group_metrics.json").read_text(encoding="utf-8"))
                for group in ("near_cold", "long_tail"):
                    rec, ndcg = scores[variant][seed]["recall"][group], scores[variant][seed]["ndcg"][group]
                    assert abs(statistics.fmean(rec.values()) - float(ref[group]["recall"]["20"])) < 1e-9, (variant, seed, group)
                    for user in sorted(rec):
                        w.writerow([variant, rel, seed, group, user, repr(rec[user]), repr(ndcg[user])])
                        n += 1
    print("wrote", out, n, "rows; group means match final_test_group_metrics.json")


if __name__ == "__main__":
    main()
