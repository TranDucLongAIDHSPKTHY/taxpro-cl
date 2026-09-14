"""Seed-matched bootstrap CI, TaxPro-CL vs a baseline, for one dataset.

Unlike tests/Recommendation_system/rank_comparison.py (used by
tools/analysis/bootstrap_ci.py), which auto-selects ONE best-validation
checkpoint per model independently -- so the two models' seeds need not
match -- this pairs each of the 3 trained seeds of TaxPro-CL with the SAME
seed of the baseline, computes per-user Recall@K hit indicators for each of
the 3 seed-matched pairs, then reports:

  1. a bootstrap CI per seed pair (consistency check across seeds), and
  2. one pooled CI where each user's diff is first AVERAGED across the
     (up to 3) seed pairs they appear in, then bootstrap-resampled over
     users -- this is the number that should be reported as the headline
     "seed-matched" result: it avoids pseudo-replication (a user is one
     resampling unit, not 3), while still folding in all 3 training seeds
     instead of one arbitrarily-picked checkpoint pair.

Motivated by the paper's Section 5.3/5.4 disclosure: the existing
Yelp2018 bootstrap compared TaxPro-CL's declared-winner checkpoint
(v12/seed1) against SimGCL's auto-selected checkpoint (seed42) -- a genuine
seed mismatch, not just checkpoint-conditional evidence. Inference-only
(reuses already-trained checkpoints), no retraining needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from config_path.config_path import evaluation_protocol_dir
from tests.Recommendation_system import inference
from utility.utility_train.group_evaluator import load_targets

ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--taxpro-run-dirs", nargs=3, required=True,
        metavar=("SEED_A", "SEED_B", "SEED_C"),
        help="3 TaxPro-CL run directories, in the same seed order as --baseline-run-dirs",
    )
    parser.add_argument(
        "--baseline-run-dirs", nargs=3, required=True,
        metavar=("SEED_A", "SEED_B", "SEED_C"),
    )
    parser.add_argument("--baseline-name", default="SimGCL")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for resampling")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def compute_hits_for_seed_pair(dataset_name, taxpro_run_dir, baseline_run_dir, device, k, batch_size):
    """Returns {user: [nc_total, nc_hit_tax, nc_hit_base, lt_total, lt_hit_tax, lt_hit_base]}."""
    taxpro_model, taxpro_dataset, _, _ = inference.load_model(taxpro_run_dir, device)
    baseline_model, baseline_dataset, _, _ = inference.load_model(baseline_run_dir, device)

    protocol_dir = evaluation_protocol_dir(dataset_name)
    targets = load_targets(protocol_dir, "test")
    users = sorted(targets["overall"])
    near_cold_targets = targets["near_cold"]
    long_tail_targets = targets["long_tail"]

    per_user = {}
    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        _, rank_tax = inference.compute_batch_order_and_rank(
            taxpro_model, taxpro_dataset, device, batch, split="test"
        )
        _, rank_base = inference.compute_batch_order_and_rank(
            baseline_model, baseline_dataset, device, batch, split="test"
        )
        rank_tax_np = rank_tax.cpu().numpy()
        rank_base_np = rank_base.cpu().numpy()
        for row, user in enumerate(batch):
            record = per_user.setdefault(user, [0, 0, 0, 0, 0, 0])
            for item in near_cold_targets.get(user, ()):
                record[0] += 1
                record[1] += 1 if rank_tax_np[row, item] <= k else 0
                record[2] += 1 if rank_base_np[row, item] <= k else 0
            for item in long_tail_targets.get(user, ()):
                record[3] += 1
                record[4] += 1 if rank_tax_np[row, item] <= k else 0
                record[5] += 1 if rank_base_np[row, item] <= k else 0
    return per_user


def user_diffs(per_user, idx_total, idx_tax, idx_base):
    diffs = {}
    for user, record in per_user.items():
        total = record[idx_total]
        if total == 0:
            continue
        diffs[user] = record[idx_tax] / total - record[idx_base] / total
    return diffs


def bootstrap_ci(values, n_boot, rng):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return None
    observed_mean = float(values.mean())
    boot_means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = values[idx].mean()
    boot_means.sort()
    lo = float(boot_means[int(0.025 * n_boot)])
    hi = float(boot_means[int(0.975 * n_boot) - 1])
    return {
        "n": n,
        "mean_diff_recall20": observed_mean,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "excludes_zero": bool((lo > 0) or (hi < 0)),
    }


def _sha256_of(path, chunk_size=1 << 20):
    import hashlib
    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_label_from_run_dir(run_dir):
    name = Path(run_dir).name
    if name.startswith("seed"):
        try:
            return int(name[len("seed"):])
        except ValueError:
            pass
    return None


def _git_commit():
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return None


def _build_metadata(args):
    """Self-describing run record so Table 12 can be audited from the JSON
    alone, without reverse-engineering seed_index against metric files
    (a reproducibility gap an external review flagged 2026-09-05)."""
    import datetime

    pairs = []
    for taxpro_dir, baseline_dir in zip(args.taxpro_run_dirs, args.baseline_run_dirs):
        pairs.append({
            "seed": _seed_label_from_run_dir(taxpro_dir),
            "taxpro_run_dir": str(taxpro_dir),
            "baseline_run_dir": str(baseline_dir),
            "taxpro_checkpoint_sha256": _sha256_of(Path(taxpro_dir) / "best_validation_model.pt"),
            "baseline_checkpoint_sha256": _sha256_of(Path(baseline_dir) / "best_validation_model.pt"),
        })
    return {
        "dataset": args.dataset,
        "baseline_name": args.baseline_name,
        "k": args.k,
        "n_boot": args.n_boot,
        "rng_seed": args.seed,
        "batch_size": args.batch_size,
        "device": args.device,
        "seed_pairs": pairs,
        "git_commit": _git_commit(),
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "script": "tools/analysis/seed_matched_bootstrap.py",
    }


def main(argv=None):
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    device = args.device

    per_seed_hits = []
    for taxpro_dir, baseline_dir in zip(args.taxpro_run_dirs, args.baseline_run_dirs):
        print(f"Scoring seed pair: taxpro={taxpro_dir} baseline={baseline_dir}")
        per_seed_hits.append(
            compute_hits_for_seed_pair(args.dataset, taxpro_dir, baseline_dir, device, args.k, args.batch_size)
        )

    results = {"metadata": _build_metadata(args), "per_seed": [], "pooled_seed_averaged": {}}

    for group, idx_total, idx_tax, idx_base in [("near_cold", 0, 1, 2), ("long_tail", 3, 4, 5)]:
        # 1) one CI per seed pair
        for seed_idx, per_user in enumerate(per_seed_hits):
            diffs = list(user_diffs(per_user, idx_total, idx_tax, idx_base).values())
            stat = bootstrap_ci(diffs, args.n_boot, rng)
            if stat is not None:
                seed_label = _seed_label_from_run_dir(args.taxpro_run_dirs[seed_idx])
                results["per_seed"].append({
                    "seed_index": seed_idx, "seed": seed_label, "group": group, **stat
                })
                print(
                    f"[seed {seed_idx}] {group}: n={stat['n']} "
                    f"mean_diff={stat['mean_diff_recall20']:.6f} "
                    f"CI=[{stat['ci95_lo']:.6f}, {stat['ci95_hi']:.6f}] "
                    f"excludes_zero={stat['excludes_zero']}"
                )

        # 2) pooled: average each user's diff across the seed pairs they appear in
        per_user_across_seeds = {}
        for per_user in per_seed_hits:
            for user, diff in user_diffs(per_user, idx_total, idx_tax, idx_base).items():
                per_user_across_seeds.setdefault(user, []).append(diff)
        averaged = [float(np.mean(vals)) for vals in per_user_across_seeds.values()]
        stat = bootstrap_ci(averaged, args.n_boot, rng)
        results["pooled_seed_averaged"][group] = stat
        if stat is not None:
            print(
                f"[pooled, seed-averaged-per-user] {group}: n={stat['n']} "
                f"mean_diff={stat['mean_diff_recall20']:.6f} "
                f"CI=[{stat['ci95_lo']:.6f}, {stat['ci95_hi']:.6f}] "
                f"excludes_zero={stat['excludes_zero']}"
            )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as stream:
            json.dump(results, stream, indent=2)
        print("Wrote", args.output)

    return results


if __name__ == "__main__":
    main()
