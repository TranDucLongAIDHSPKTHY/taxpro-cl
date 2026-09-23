"""Multiplicity-corrected re-analysis of Main Paper Table 11 (the 8-cell
TaxPro-CL vs. SimGCL comparison that anchors the Abstract's headline claim).

Reuses the exact same checkpoints, evaluation protocol, and per-user diff
computation as tools/analysis/seed_matched_bootstrap.py (which produced
Table 11) and adds, per cell:

- the percentile-bootstrap 95% CI (unchanged; a descriptive interval);
- a paired sign-flip randomization p-value on the seed-averaged per-user
  differences: under H0 (the per-user difference is symmetric about zero) the
  sign of each user's difference is exchangeable, so the p-value is the share
  of random sign assignments whose mean is at least as extreme as the
  observed mean (Monte Carlo, floored at 1/(n_perm+1)). This is a null-
  distribution p-value and is the one fed to Holm-Bonferroni;
- a null-centred bootstrap p-value (resample the differences after subtracting
  their mean) as a second, independent check;
- the earlier percentile-bootstrap tail p-value, p = 2 * min(P(boot<=0),
  P(boot>=0)), kept only for comparison with the previous analysis: it is the
  CI inverted, not a null-distribution p-value.

All three describe uncertainty over users conditional on the three trained
checkpoints; none reflects training-seed randomness. Inference only.
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


def sign_flip_pvalue(values, n_perm, rng, chunk=200):
    """Two-sided paired sign-flip randomization test for mean(values) == 0."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    observed = abs(float(values.sum()))
    extreme = 0
    done = 0
    while done < n_perm:
        m = min(chunk, n_perm - done)
        signs = rng.integers(0, 2, size=(m, n), dtype=np.int8) * 2 - 1
        sums = np.abs(signs @ values)
        extreme += int(np.sum(sums >= observed - 1e-12))
        done += m
    return (extreme + 1) / (n_perm + 1)


def centred_bootstrap_pvalue(values, n_boot, rng):
    """Two-sided p-value from a bootstrap of the mean-centred differences."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    observed = abs(float(values.mean()))
    centred = values - values.mean()
    extreme = 0
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if abs(float(centred[idx].mean())) >= observed - 1e-15:
            extreme += 1
    return (extreme + 1) / (n_boot + 1)


def bootstrap_stat(values, n_boot, rng, n_perm=None):
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
    p_tail = min(max(2 * min(p_le, p_ge), 1.0 / (n_boot + 1)), 1.0)
    n_perm = n_perm or n_boot
    p_flip = sign_flip_pvalue(values, n_perm, rng)
    p_centred = centred_bootstrap_pvalue(values, n_boot, rng)
    return {
        "n": n, "mean_diff_recall20": observed_mean,
        "ci95_lo": lo, "ci95_hi": hi,
        "excludes_zero_uncorrected": bool((lo > 0) or (hi < 0)),
        "p_signflip": p_flip,
        "p_centred_bootstrap": p_centred,
        "p_percentile_tail_legacy": p_tail,
        "n_perm": n_perm,
        "n_boot": n_boot,
    }


def holm_bonferroni(cells, p_key, out_key, alpha=0.05):
    """Holm step-down over cells using cells[i][p_key]; writes cells[i][out_key]
    (bool) and cells[i][out_key + '_threshold']."""
    order = sorted(range(len(cells)), key=lambda i: cells[i][p_key])
    m = len(cells)
    still_significant = True
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        significant = still_significant and cells[idx][p_key] <= threshold
        still_significant = significant
        cells[idx][out_key] = bool(significant)
        cells[idx][out_key + "_threshold"] = threshold
    return cells


def main():
    import argparse
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-perm", type=int, default=None,
                        help="Sign-flip permutations per cell (default: --n-boot).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "a2_multiplicity_correction.json")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--save-diffs", type=Path, default=None,
                        help="Write the seed-averaged per-user differences of every cell to this .npz.")
    parser.add_argument("--load-diffs", type=Path, default=None,
                        help="Reuse per-user differences saved by --save-diffs instead of re-scoring checkpoints.")
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

    cached = dict(np.load(args.load_diffs)) if args.load_diffs else {}
    saved = {}
    all_cells = []
    for dataset_name in args.datasets:
        taxpro_dirs = seed_dirs(dataset_name, "TaxPro-CL-main")
        simgcl_dirs = seed_dirs(dataset_name, "SimGCL-main")
        seeds = sorted(set(taxpro_dirs) & set(simgcl_dirs))
        assert len(seeds) == 3, f"{dataset_name}: expected 3 seeds, got {len(seeds)}"

        per_seed_hits = []
        if not cached:
            for s in seeds:
                print(f"[{dataset_name}] scoring seed={s}", flush=True)
                per_seed_hits.append(
                    compute_hits_for_seed_pair(dataset_name, taxpro_dirs[s], simgcl_dirs[s], args.device, args.k, args.batch_size)
                )

        for group, idx_total, idx_tax, idx_sim in [("near_cold", 0, 1, 2), ("long_tail", 3, 4, 5)]:
            key = f"{dataset_name}|{group}"
            if cached:
                averaged = cached[key]
            else:
                per_user_across_seeds = {}
                for per_user in per_seed_hits:
                    for user, diff in user_diffs(per_user, idx_total, idx_tax, idx_sim).items():
                        per_user_across_seeds.setdefault(user, []).append(diff)
                averaged = [float(np.mean(vals)) for vals in per_user_across_seeds.values()]
            saved[key] = np.asarray(averaged, dtype=np.float64)
            stat = bootstrap_stat(averaged, args.n_boot, rng, args.n_perm)
            stat["dataset"] = dataset_name
            stat["group"] = group
            all_cells.append(stat)
            print(f"  {group}: n={stat['n']} mean_diff={stat['mean_diff_recall20']:.6f} "
                  f"CI=[{stat['ci95_lo']:.6f},{stat['ci95_hi']:.6f}] "
                  f"p_signflip={stat['p_signflip']:.5f} p_centred={stat['p_centred_bootstrap']:.5f} "
                  f"p_tail_legacy={stat['p_percentile_tail_legacy']:.5f} "
                  f"excl0(uncorrected)={stat['excludes_zero_uncorrected']}", flush=True)

    for p_key, out_key in (
        ("p_signflip", "holm_significant_signflip"),
        ("p_centred_bootstrap", "holm_significant_centred_bootstrap"),
        ("p_percentile_tail_legacy", "holm_significant_percentile_tail_legacy"),
    ):
        all_cells = holm_bonferroni(all_cells, p_key, out_key, alpha=0.05)

    print("\n=== Holm-Bonferroni summary (alpha=0.05, m=8), primary = sign-flip ===")
    for c in sorted(all_cells, key=lambda x: x["p_signflip"]):
        print(f"  {c['dataset']:25s} {c['group']:10s} p_flip={c['p_signflip']:.5f} "
              f"thr={c['holm_significant_signflip_threshold']:.5f} "
              f"holm(flip)={c['holm_significant_signflip']} "
              f"holm(centred)={c['holm_significant_centred_bootstrap']} "
              f"holm(legacy tail)={c['holm_significant_percentile_tail_legacy']} "
              f"(uncorrected_excl0={c['excludes_zero_uncorrected']})")

    if args.save_diffs:
        args.save_diffs.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_diffs, **saved)
        print("Wrote per-user differences to", args.save_diffs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(all_cells, f, indent=2)
    print("\nWrote", args.output)


if __name__ == "__main__":
    main()
