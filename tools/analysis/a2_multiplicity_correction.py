"""A2 re-analysis (JIIS V58 review, issue A2): Holm-Bonferroni-corrected
version of Main Paper Table 11 (the 8-cell TaxPro-CL vs. SimGCL bootstrap
that anchors the Abstract's headline claim).

Reuses the exact same checkpoints, evaluation protocol, and per-user diff
computation as tools/analysis/seed_matched_bootstrap.py (which produced
Table 11); the only addition is a two-sided bootstrap p-value per cell and
a Holm-Bonferroni step-down correction across the 8 dataset x group cells.
Inference only -- no retraining.

Bootstrap p-value convention: p = 2 * min(P(boot_mean <= 0), P(boot_mean >= 0)),
floored at 1/(n_boot+1) to avoid a reported p of exactly 0 from a finite
resample (matches the standard percentile-bootstrap p-value convention).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config_path.config_path import evaluation_protocol_dir
from tests.Recommendation_system import inference
from utility.utility_train.group_evaluator import load_targets

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ["amazon-book", "yelp2018", "musical-instruments", "arts-crafts-and-sewing"]


def compute_hits_for_seed_pair(dataset_name, taxpro_run_dir, simgcl_run_dir, device, k, batch_size):
    taxpro_model, taxpro_dataset, _, _ = inference.load_model(taxpro_run_dir, device)
    simgcl_model, simgcl_dataset, _, _ = inference.load_model(simgcl_run_dir, device)

    protocol_dir = evaluation_protocol_dir(dataset_name)
    targets = load_targets(protocol_dir, "test")
    nc_targets = targets["near_cold"]
    lt_targets = targets["long_tail"]
    users = sorted(set(nc_targets) | set(lt_targets))

    per_user = {}
    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        _, rank_tax = inference.compute_batch_order_and_rank(taxpro_model, taxpro_dataset, device, batch, split="test")
        _, rank_sim = inference.compute_batch_order_and_rank(simgcl_model, simgcl_dataset, device, batch, split="test")
        rank_tax_np = rank_tax.cpu().numpy()
        rank_sim_np = rank_sim.cpu().numpy()
        for row, user in enumerate(batch):
            record = per_user.setdefault(user, [0, 0, 0, 0, 0, 0])
            for item in nc_targets.get(user, ()):
                record[0] += 1
                record[1] += 1 if rank_tax_np[row, item] <= k else 0
                record[2] += 1 if rank_sim_np[row, item] <= k else 0
            for item in lt_targets.get(user, ()):
                record[3] += 1
                record[4] += 1 if rank_tax_np[row, item] <= k else 0
                record[5] += 1 if rank_sim_np[row, item] <= k else 0
    return per_user


def user_diffs(per_user, idx_total, idx_tax, idx_sim):
    diffs = {}
    for user, record in per_user.items():
        total = record[idx_total]
        if total == 0:
            continue
        diffs[user] = record[idx_tax] / total - record[idx_sim] / total
    return diffs


def bootstrap_stat(values, n_boot, rng):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    observed_mean = float(values.mean())
    boot_means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = values[idx].mean()
    boot_means.sort()
    lo = float(boot_means[int(0.025 * n_boot)])
    hi = float(boot_means[int(0.975 * n_boot) - 1])
    p_le = float(np.mean(boot_means <= 0.0))
    p_ge = float(np.mean(boot_means >= 0.0))
    p_two_sided = 2 * min(p_le, p_ge)
    p_floor = 1.0 / (n_boot + 1)
    p_two_sided = max(p_two_sided, p_floor)
    p_two_sided = min(p_two_sided, 1.0)
    return {
        "n": n, "mean_diff_recall20": observed_mean,
        "ci95_lo": lo, "ci95_hi": hi,
        "excludes_zero_uncorrected": bool((lo > 0) or (hi < 0)),
        "p_two_sided_bootstrap": p_two_sided,
    }


def holm_bonferroni(cells, alpha=0.05):
    """cells: list of dicts with 'p_two_sided_bootstrap'; adds 'holm_significant'."""
    order = sorted(range(len(cells)), key=lambda i: cells[i]["p_two_sided_bootstrap"])
    m = len(cells)
    still_significant = True
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        p = cells[idx]["p_two_sided_bootstrap"]
        if still_significant and p <= threshold:
            cells[idx]["holm_significant"] = True
            cells[idx]["holm_threshold"] = threshold
        else:
            still_significant = False
            cells[idx]["holm_significant"] = False
            cells[idx]["holm_threshold"] = threshold
    return cells


def main():
    import argparse
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "a2_multiplicity_correction.json")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    import csv
    rows = list(csv.DictReader(open(ROOT / "results" / "results_manifest.csv", encoding="utf-8")))

    def seed_dirs(dataset, variant):
        d = {}
        for row in rows:
            if row["dataset"] == dataset and row["variant"] == variant:
                d[int(row["seed"])] = ROOT / row["run_dir"].replace("\\", "/")
        return d

    all_cells = []
    for dataset_name in args.datasets:
        taxpro_dirs = seed_dirs(dataset_name, "TaxPro-CL-main")
        simgcl_dirs = seed_dirs(dataset_name, "SimGCL-main")
        seeds = sorted(set(taxpro_dirs) & set(simgcl_dirs))
        assert len(seeds) == 3, f"{dataset_name}: expected 3 seeds, got {len(seeds)}"

        per_seed_hits = []
        for s in seeds:
            print(f"[{dataset_name}] scoring seed={s}")
            per_seed_hits.append(
                compute_hits_for_seed_pair(dataset_name, taxpro_dirs[s], simgcl_dirs[s], args.device, args.k, args.batch_size)
            )

        for group, idx_total, idx_tax, idx_sim in [("near_cold", 0, 1, 2), ("long_tail", 3, 4, 5)]:
            per_user_across_seeds = {}
            for per_user in per_seed_hits:
                for user, diff in user_diffs(per_user, idx_total, idx_tax, idx_sim).items():
                    per_user_across_seeds.setdefault(user, []).append(diff)
            averaged = [float(np.mean(vals)) for vals in per_user_across_seeds.values()]
            stat = bootstrap_stat(averaged, args.n_boot, rng)
            stat["dataset"] = dataset_name
            stat["group"] = group
            all_cells.append(stat)
            print(f"  {group}: n={stat['n']} mean_diff={stat['mean_diff_recall20']:.6f} "
                  f"CI=[{stat['ci95_lo']:.6f},{stat['ci95_hi']:.6f}] "
                  f"p_boot={stat['p_two_sided_bootstrap']:.5f} "
                  f"excl0(uncorrected)={stat['excludes_zero_uncorrected']}")

    all_cells = holm_bonferroni(all_cells, alpha=0.05)

    print("\n=== Holm-Bonferroni summary (alpha=0.05, m=8) ===")
    for c in sorted(all_cells, key=lambda x: x["p_two_sided_bootstrap"]):
        print(f"  {c['dataset']:25s} {c['group']:10s} p={c['p_two_sided_bootstrap']:.5f} "
              f"threshold={c['holm_threshold']:.5f} holm_significant={c['holm_significant']} "
              f"(uncorrected_excl0={c['excludes_zero_uncorrected']})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(all_cells, f, indent=2)
    print("\nWrote", args.output)


if __name__ == "__main__":
    main()
