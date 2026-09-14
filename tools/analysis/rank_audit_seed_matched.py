"""Seed-matched rank-level audit (GVHD review of manuscript V56, B7).

The existing rank-level audit (tests/Recommendation_system/rank_comparison.py,
manuscript Tables 7-8) pairs each baseline's auto-selected ("best validation")
checkpoint against TaxPro-CL's auto-selected checkpoint independently -- for
Yelp2018 these can land on different training seeds (the exact gap GVHD's
review flags). This instead pairs the SAME seed's checkpoint on both sides
(42-42, 0-0, 1-1 -- the same three seeds already used for the main results
table) and averages the per-group summary statistic across all three pairs.

Does not retrain, and does not write a new per-item detail-row CSV (that file
is 3.6GB for one unmatched pass; three seed-matched passes of it would not be
a useful thing to keep around). Only the already-small summary statistics
(one row per dataset x baseline x k x item-group) are kept, tagged with how
many of the three seed pairs actually contributed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_path.config_path import evaluation_protocol_dir
from utility.utility_train.group_evaluator import load_targets
from tests.Recommendation_system import checkpoint_selection, config, inference, rank_comparison

SEEDS = (42, 0, 1)
AVERAGED_FIELDS = (
    "Increased_Ratio", "Avg_Delta_Rank", "Median_Delta_Rank", "Std_Delta_Rank",
    "Enter_TopK", "Leave_TopK", "Increased", "Decreased", "Unchanged",
    "N_Items", "N_Users",
)


def seed_matched_dir(model_name, dataset_name, seed):
    """The (model, dataset)'s official checkpoint *family*'s directory for a
    specific seed -- not necessarily the auto-selected "best" seed."""
    choice = checkpoint_selection.select_checkpoint(model_name, dataset_name)
    candidate = choice.run_dir.parent / "seed{}".format(seed)
    manifest_path = candidate / "run_manifest.json"
    if not manifest_path.is_file():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        return None
    return candidate


def score_one_seed_pair(dataset_name, baseline_name, baseline_dir, taxpro_dir, k_values, device, batch_size):
    """Same core computation as rank_comparison.compare_dataset_baseline, but
    accumulates group summary statistics in memory only -- no per-item CSV."""
    max_k = max(k_values)
    baseline_model, baseline_dataset, _config, _name = inference.load_model(baseline_dir, device)
    taxpro_model, taxpro_dataset, _config2, _name2 = inference.load_model(taxpro_dir, device)

    protocol_dir = evaluation_protocol_dir(dataset_name)
    overall_targets = load_targets(protocol_dir, "test")["overall"]
    item_degrees = rank_comparison.load_item_degrees(dataset_name)
    users = sorted(overall_targets)

    group_state = {
        k: {
            "delta_ranks": [], "in_b": [], "in_t": [],
            "near_cold": [], "long_tail": [], "warm": [], "user": [],
        }
        for k in k_values
    }

    for start in range(0, len(users), batch_size):
        batch_users = users[start : start + batch_size]
        order_b, rank_b = inference.compute_batch_order_and_rank(
            baseline_model, baseline_dataset, device, batch_users, split="test"
        )
        order_t, rank_t = inference.compute_batch_order_and_rank(
            taxpro_model, taxpro_dataset, device, batch_users, split="test"
        )
        order_b_np, order_t_np = order_b.cpu().numpy(), order_t.cpu().numpy()
        rank_b_np, rank_t_np = rank_b.cpu().numpy(), rank_t.cpu().numpy()

        for row_idx, user in enumerate(batch_users):
            positives = {int(item) for item in overall_targets.get(user, ())}
            topk_b_full = order_b_np[row_idx, :max_k]
            topk_t_full = order_t_np[row_idx, :max_k]
            for k in k_values:
                candidate_set = set(topk_b_full[:k].tolist()) | set(topk_t_full[:k].tolist()) | positives
                if not candidate_set:
                    continue
                candidates_arr = np.asarray(sorted(candidate_set), dtype=np.int64)
                rb_arr = rank_b_np[row_idx, candidates_arr]
                rt_arr = rank_t_np[row_idx, candidates_arr]
                delta_arr = rb_arr - rt_arr
                in_b_arr = rb_arr <= k
                in_t_arr = rt_arr <= k
                degree_arr = item_degrees[candidates_arr]
                near_cold_arr = (degree_arr >= 1) & (degree_arr <= rank_comparison.NEAR_COLD_MAX_DEGREE)
                long_tail_arr = (degree_arr >= 1) & (degree_arr <= rank_comparison.LONG_TAIL_MAX_DEGREE)
                warm_arr = degree_arr > rank_comparison.LONG_TAIL_MAX_DEGREE

                acc = group_state[k]
                acc["delta_ranks"].extend(delta_arr.tolist())
                acc["in_b"].extend(in_b_arr.tolist())
                acc["in_t"].extend(in_t_arr.tolist())
                acc["near_cold"].extend(near_cold_arr.tolist())
                acc["long_tail"].extend(long_tail_arr.tolist())
                acc["warm"].extend(warm_arr.tolist())
                acc["user"].extend([user] * len(candidates_arr))

    rows = []
    for k in k_values:
        acc = group_state[k]
        rows.extend(
            rank_comparison.build_group_summary_rows(
                dataset_name, baseline_name, k,
                acc["delta_ranks"], acc["in_b"], acc["in_t"],
                acc["near_cold"], acc["long_tail"], acc["warm"], acc["user"],
                str(baseline_dir), str(taxpro_dir), None, None,
            )
        )
    return rows


def average_rows(rows_by_seed):
    """rows_by_seed: list of (seed, [row, ...]) -- one row list per seed that
    actually had both checkpoints available. Returns one averaged row per
    (K, Item_Group), tagged with n_seed_pairs."""
    grouped = defaultdict(list)
    for _seed, rows in rows_by_seed:
        for row in rows:
            grouped[(row["K"], row["Item_Group"])].append(row)

    averaged = []
    for (k, group), rows in sorted(grouped.items()):
        out = {"K": k, "Item_Group": group, "N_Seed_Pairs": len(rows)}
        for field in AVERAGED_FIELDS:
            out[field] = float(np.mean([row[field] for row in rows]))
        averaged.append(out)
    return averaged


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=config.DATASETS, choices=config.DATASETS)
    parser.add_argument("--baselines", nargs="+", default=config.WITHOUT_TAXONOMY_MODELS)
    parser.add_argument("--k-values", nargs="+", type=int, default=config.K_VALUES)
    parser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "rank_audit_seed_matched.json")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    results = []

    for dataset_name, baseline_name in itertools.product(args.datasets, args.baselines):
        rows_by_seed = []
        used_seeds = []
        for seed in SEEDS:
            baseline_dir = seed_matched_dir(baseline_name, dataset_name, seed)
            taxpro_dir = seed_matched_dir(config.TAXONOMY_MODEL, dataset_name, seed)
            if baseline_dir is None or taxpro_dir is None:
                print(
                    "SKIP {}/{} seed={}: missing completed checkpoint (baseline={}, taxpro={})".format(
                        baseline_name, dataset_name, seed, baseline_dir, taxpro_dir
                    ),
                    file=sys.stderr,
                )
                continue
            print("{} vs TaxPro-CL on {}, seed={}-{}".format(baseline_name, dataset_name, seed, seed))
            rows = score_one_seed_pair(
                dataset_name, baseline_name, baseline_dir, taxpro_dir, args.k_values, device, args.batch_size
            )
            rows_by_seed.append((seed, rows))
            used_seeds.append(seed)

        if not rows_by_seed:
            print("SKIP {}/{}: no seed-matched pairs available at all".format(baseline_name, dataset_name), file=sys.stderr)
            continue

        averaged = average_rows(rows_by_seed)
        for row in averaged:
            row["Dataset"] = dataset_name
            row["Baseline"] = baseline_name
            row["Seeds_Used"] = used_seeds
        results.extend(averaged)
        print(json.dumps({"dataset": dataset_name, "baseline": baseline_name, "seeds_used": used_seeds}, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Saved {} averaged rows to {}".format(len(results), args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
