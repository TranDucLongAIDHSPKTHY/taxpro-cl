"""Bootstrap 95% CI on per-user paired Recall@20 difference (TaxPro-CL vs a
baseline), for Near-Cold and Long-Tail, built directly from
tests/Recommendation_system/result/ranking_details.csv (already-computed
live inference from checkpoints -- no retraining needed). This is the
paper's primary statistical-significance evidence (Section 5.3): it has far
higher power than the n=3 paired-seed Wilcoxon test (tools/analysis/
statistics.py) since it resamples thousands of users instead of 3 seeds, and
it covers every dataset regardless of whether per-seed training logs still
exist on disk.

Scope note (see the paper's Section 5.3/5.4 for the full disclosure): this
CI is checkpoint-conditional. Each model contributes exactly one checkpoint
(its best-validation checkpoint among the seeds trained), not necessarily
matching seeds across models, so the interval measures per-user sampling
uncertainty for that specific pair of trained checkpoints -- it does not
marginalize over training-seed randomness.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ["amazon-book", "yelp2018", "musical-instruments", "arts-crafts-and-sewing"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "tests" / "Recommendation_system" / "result" / "ranking_details.csv",
    )
    parser.add_argument("--baseline", default="SimGCL", help="Baseline column value to filter on")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for resampling")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "week6" / "bootstrap_ci_per_user.json",
    )
    return parser.parse_args(argv)


def load_per_user_diffs(input_path, baseline, k):
    """Return {dataset: {group: [per-user recall diffs]}} for near_cold/long_tail."""
    agg = {ds: {} for ds in DATASETS}
    with open(input_path, encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if row["Baseline"] != baseline or row["K"] != str(k):
                continue
            if row["Is_Ground_Truth"] != "True":
                continue
            dataset = row["Dataset"]
            if dataset not in agg:
                continue
            user = row["User"]
            # [nc_total, nc_hit_tax, nc_hit_base, lt_total, lt_hit_tax, lt_hit_base]
            record = agg[dataset].setdefault(user, [0, 0, 0, 0, 0, 0])
            rank_candidate = int(row["Rank_TaxProCL"])
            rank_baseline = int(row["Rank_Baseline"])
            hit_candidate = 1 if rank_candidate <= k else 0
            hit_baseline = 1 if rank_baseline <= k else 0
            if row["Is_Near_Cold"] == "True":
                record[0] += 1
                record[1] += hit_candidate
                record[2] += hit_baseline
            if row["Is_Long_Tail"] == "True":
                record[3] += 1
                record[4] += hit_candidate
                record[5] += hit_baseline
    return agg


def bootstrap_dataset_group(diffs, n_boot, rng):
    diffs_arr = np.asarray(diffs, dtype=np.float64)
    n_users = len(diffs_arr)
    if n_users == 0:
        return None
    observed_mean = float(diffs_arr.mean())
    boot_means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n_users, size=n_users)
        boot_means[b] = diffs_arr[idx].mean()
    boot_means.sort()
    lo = float(boot_means[int(0.025 * n_boot)])
    hi = float(boot_means[int(0.975 * n_boot) - 1])
    return {
        "n_users": n_users,
        "mean_diff_recall20": observed_mean,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "excludes_zero": bool((lo > 0) or (hi < 0)),
    }


def main(argv=None):
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    agg = load_per_user_diffs(args.input, args.baseline, args.k)

    results = {}
    for dataset in DATASETS:
        results[dataset] = {}
        for group, idx_total, idx_candidate, idx_baseline in [
            ("near_cold", 0, 1, 2),
            ("long_tail", 3, 4, 5),
        ]:
            diffs = []
            for record in agg[dataset].values():
                total = record[idx_total]
                if total == 0:
                    continue
                recall_candidate = record[idx_candidate] / total
                recall_baseline = record[idx_baseline] / total
                diffs.append(recall_candidate - recall_baseline)
            stat = bootstrap_dataset_group(diffs, args.n_boot, rng)
            results[dataset][group] = stat
            if stat is not None:
                print(
                    f"{dataset}/{group}: n_users={stat['n_users']} "
                    f"mean_diff={stat['mean_diff_recall20']:.6f} "
                    f"95% CI=[{stat['ci95_lo']:.6f}, {stat['ci95_hi']:.6f}] "
                    f"excludes_zero={stat['excludes_zero']}"
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
