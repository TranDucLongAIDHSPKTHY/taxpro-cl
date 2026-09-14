"""Near-Cold/Long-Tail bootstrap restricted to items that are sparse under BOTH
the train-time degree
definition (used everywhere else in the paper) AND the pre-split pool
degree definition (train+validation for Protocol A; train+validation+test
for Protocol B) -- i.e. "intrinsically" sparse items, excluding the
split-carve artifact documented in Online Resource 1 Table S24.

Reuses the exact same checkpoints, evaluation protocol, and bootstrap
procedure as tools/analysis/seed_matched_bootstrap.py (which produced
Main Paper Table 11); the only change is the target item set. Inference
only -- no retraining.

Sanity check: this script also reports, per dataset, the artifact rate it
computes independently (items in train-degree Near-Cold whose pool degree
is >5), which must match Online Resource 1 Table S24's published
percentages before its bootstrap output can be trusted.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from config_path.config_path import evaluation_protocol_dir, verified_dataset_dir
from tests.Recommendation_system import inference
from utility.utility_train.group_evaluator import load_targets

ROOT = Path(__file__).resolve().parents[2]

# Protocol A datasets: pool = train+validation only (test file is a disjoint
# holdout copied unfiltered; the main paper's own S24 artifact calculation
# for Protocol A also excludes test -- matched here for consistency).
PROTOCOL_A = {"amazon-book", "yelp2018"}
# Protocol B datasets: whole pool (train+validation+test) was 5-core filtered
# together before splitting, so pool = all three files.
PROTOCOL_B = {"musical-instruments", "arts-crafts-and-sewing"}

DATASETS = ["amazon-book", "yelp2018", "musical-instruments", "arts-crafts-and-sewing"]

# Published Table S24 artifact rates (train-degree Near-Cold items whose
# pool degree > 5), used only as a sanity check on this script's own
# independently-computed item-degree counts.
PUBLISHED_ARTIFACT_RATE = {
    "amazon-book": 0.4879,
    "yelp2018": 0.5297,
    "musical-instruments": 0.5612,
    "arts-crafts-and-sewing": 0.5509,
}


def _parse_adjacency_file(path):
    """NGCF-style 'user item item item ...' per line -> Counter of item -> degree."""
    counts = Counter()
    with open(path, "r", encoding="utf-8") as stream:
        for line in stream:
            parts = line.strip().split()
            if not parts:
                continue
            items = parts[1:]
            counts.update(int(item) for item in items)
    return counts


def compute_item_degrees(dataset_name):
    """Returns (train_degree, pool_degree) dicts, item -> int degree (0 if absent)."""
    data_dir = Path(verified_dataset_dir(dataset_name))
    train_counts = _parse_adjacency_file(data_dir / "train.txt")
    val_counts = _parse_adjacency_file(data_dir / "validation.txt")
    pool_counts = Counter(train_counts)
    pool_counts.update(val_counts)
    if dataset_name in PROTOCOL_B:
        test_counts = _parse_adjacency_file(data_dir / "test.txt")
        pool_counts.update(test_counts)
    elif dataset_name not in PROTOCOL_A:
        raise ValueError("Unknown protocol family for dataset {}".format(dataset_name))
    return train_counts, pool_counts


def intrinsic_item_sets(train_degree, pool_degree):
    """Item ids that are sparse under BOTH degree definitions."""
    all_items = set(train_degree) | set(pool_degree)
    intrinsic_near_cold = {
        item for item in all_items
        if 1 <= train_degree.get(item, 0) <= 5 and 1 <= pool_degree.get(item, 0) <= 5
    }
    intrinsic_long_tail = {
        item for item in all_items
        if 1 <= train_degree.get(item, 0) <= 10 and 1 <= pool_degree.get(item, 0) <= 10
    }
    return intrinsic_near_cold, intrinsic_long_tail


def artifact_rate_check(train_degree, pool_degree):
    """Independent recomputation of the Table S24 Near-Cold artifact rate."""
    near_cold_items = [item for item, deg in train_degree.items() if 1 <= deg <= 5]
    if not near_cold_items:
        return None
    artifact = sum(1 for item in near_cold_items if pool_degree.get(item, 0) > 5)
    return artifact / len(near_cold_items), len(near_cold_items), artifact


def restrict_targets(targets_by_user, allowed_items):
    """targets_by_user: {user: [item,...]} -> filtered, dropping empty users."""
    restricted = {}
    for user, items in targets_by_user.items():
        kept = [item for item in items if item in allowed_items]
        if kept:
            restricted[user] = kept
    return restricted


def compute_hits_for_seed_pair(dataset_name, taxpro_run_dir, simgcl_run_dir, device, k,
                                batch_size, intrinsic_nc_items, intrinsic_lt_items):
    taxpro_model, taxpro_dataset, _, _ = inference.load_model(taxpro_run_dir, device)
    simgcl_model, simgcl_dataset, _, _ = inference.load_model(simgcl_run_dir, device)

    protocol_dir = evaluation_protocol_dir(dataset_name)
    targets = load_targets(protocol_dir, "test")
    nc_targets = restrict_targets(targets["near_cold"], intrinsic_nc_items)
    lt_targets = restrict_targets(targets["long_tail"], intrinsic_lt_items)
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
    return per_user, {"near_cold_eligible_users": len(nc_targets), "long_tail_eligible_users": len(lt_targets)}


def user_diffs(per_user, idx_total, idx_tax, idx_sim):
    diffs = {}
    for user, record in per_user.items():
        total = record[idx_total]
        if total == 0:
            continue
        diffs[user] = record[idx_tax] / total - record[idx_sim] / total
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


def _seed_dirs_from_manifest(dataset_name):
    import csv
    rows = list(csv.DictReader(open(ROOT / "results" / "results_manifest.csv", encoding="utf-8")))
    taxpro_dirs, simgcl_dirs = {}, {}
    for row in rows:
        if row["dataset"] != dataset_name:
            continue
        if row["variant"] == "TaxPro-CL-main":
            taxpro_dirs[int(row["seed"])] = ROOT / row["run_dir"].replace("\\", "/")
        elif row["variant"] == "SimGCL-main":
            simgcl_dirs[int(row["seed"])] = ROOT / row["run_dir"].replace("\\", "/")
    seeds = sorted(set(taxpro_dirs) & set(simgcl_dirs))
    return [(s, taxpro_dirs[s], simgcl_dirs[s]) for s in seeds]


def run_dataset(dataset_name, device, k, n_boot, batch_size, rng):
    train_degree, pool_degree = compute_item_degrees(dataset_name)
    check = artifact_rate_check(train_degree, pool_degree)
    intrinsic_nc, intrinsic_lt = intrinsic_item_sets(train_degree, pool_degree)

    seed_pairs = _seed_dirs_from_manifest(dataset_name)
    if len(seed_pairs) != 3:
        raise RuntimeError(f"{dataset_name}: expected 3 seed-matched pairs, found {len(seed_pairs)}")

    per_seed_hits = []
    supports = []
    for seed, taxpro_dir, simgcl_dir in seed_pairs:
        print(f"[{dataset_name}] scoring seed={seed} taxpro={taxpro_dir.name} simgcl={simgcl_dir.name}")
        hits, support = compute_hits_for_seed_pair(
            dataset_name, taxpro_dir, simgcl_dir, device, k, batch_size, intrinsic_nc, intrinsic_lt
        )
        per_seed_hits.append(hits)
        supports.append(support)

    result = {
        "dataset": dataset_name,
        "artifact_rate_check": {
            "computed_rate": check[0] if check else None,
            "near_cold_train_items": check[1] if check else None,
            "artifact_items": check[2] if check else None,
            "published_table_s24_rate": PUBLISHED_ARTIFACT_RATE[dataset_name],
            "matches_published": (
                check is not None and abs(check[0] - PUBLISHED_ARTIFACT_RATE[dataset_name]) < 0.001
            ),
        },
        "intrinsic_near_cold_item_count": len(intrinsic_nc),
        "intrinsic_long_tail_item_count": len(intrinsic_lt),
        "per_seed_support": supports,
        "per_seed": [],
        "pooled_seed_averaged": {},
    }

    for group, idx_total, idx_tax, idx_sim in [("near_cold", 0, 1, 2), ("long_tail", 3, 4, 5)]:
        for seed_idx, per_user in enumerate(per_seed_hits):
            diffs = list(user_diffs(per_user, idx_total, idx_tax, idx_sim).values())
            stat = bootstrap_ci(diffs, n_boot, rng)
            if stat is not None:
                result["per_seed"].append({"seed": seed_pairs[seed_idx][0], "group": group, **stat})

        per_user_across_seeds = {}
        for per_user in per_seed_hits:
            for user, diff in user_diffs(per_user, idx_total, idx_tax, idx_sim).items():
                per_user_across_seeds.setdefault(user, []).append(diff)
        averaged = [float(np.mean(vals)) for vals in per_user_across_seeds.values()]
        stat = bootstrap_ci(averaged, n_boot, rng)
        result["pooled_seed_averaged"][group] = stat
        if stat is not None:
            print(
                f"[{dataset_name}][pooled] {group}: n={stat['n']} mean_diff={stat['mean_diff_recall20']:.6f} "
                f"CI=[{stat['ci95_lo']:.6f}, {stat['ci95_hi']:.6f}] excludes_zero={stat['excludes_zero']}"
            )
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "a1_intrinsic_sparsity_bootstrap.json")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    all_results = {}
    for dataset_name in args.datasets:
        all_results[dataset_name] = run_dataset(
            dataset_name, args.device, args.k, args.n_boot, args.batch_size, rng
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(all_results, stream, indent=2)
    print("Wrote", args.output)
    return all_results


if __name__ == "__main__":
    main()
