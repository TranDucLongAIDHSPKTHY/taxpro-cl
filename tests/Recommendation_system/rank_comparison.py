"""Per-(dataset, baseline) core comparison: TaxPro-CL vs a non-taxonomy baseline.

For every eligible test user, both models score the exact same candidate
catalog under the exact same train+validation exclusion (utility.utility_data
.data_loader.Data.get_user_pos_items_for_evaluation, split="test" -- the
project's existing protocol-defined exclusion, reused unchanged). Candidate
items compared per K are the union of each model's own top-K plus the user's
ground-truth test-positive items, so "entered/left Top-K" and "true positive
still buried outside Top-K" are both observable.

Every candidate item is also labelled with its train-degree item group
(near_cold / long_tail / warm, same thresholds and same source file --
preprocessed/evaluation_protocol/split_seed_42/<dataset>/item_degrees_train
.json -- as the project's own final_test_group_metrics.json), because
TaxPro-CL's whole point is improving near_cold/long_tail ranking, not
Overall (see project memory project_taxprocl_v15_breakthrough.md). near_cold
is a subset of long_tail by the project's own definition, so an item can be
True for both -- these are independent flags, not a single exclusive label.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np

from config_path.config_path import evaluation_protocol_dir, relative_to_project
from utility.utility_train.group_evaluator import load_targets

from . import checkpoint_selection, config, inference

logger = logging.getLogger(__name__)

DETAIL_FIELDNAMES = [
    "Dataset", "User", "Item", "Baseline", "K",
    "Rank_Baseline", "Rank_TaxProCL", "Delta_Rank", "Status",
    "In_TopK_Baseline", "In_TopK_TaxProCL", "Is_Ground_Truth",
    "Item_Degree", "Is_Near_Cold", "Is_Long_Tail", "Is_Warm",
]

SUMMARY_FIELDNAMES = [
    "Dataset", "Baseline", "K", "Item_Group",
    "Increased", "Decreased", "Unchanged", "Enter_TopK", "Leave_TopK",
    "Increased_Ratio", "Avg_Delta_Rank", "Median_Delta_Rank", "Std_Delta_Rank",
    "N_Items", "N_Users",
    "Checkpoint_Baseline", "Checkpoint_TaxProCL",
    "Baseline_Validation_Score", "TaxProCL_Validation_Score",
]

# Same thresholds as utility.utility_train.group_evaluator.GROUP_DEFINITIONS.
# near_cold is a *subset* of long_tail (both start at degree 1), not disjoint.
NEAR_COLD_MAX_DEGREE = 5
LONG_TAIL_MAX_DEGREE = 10

ITEM_GROUPS = ("overall", "near_cold", "long_tail", "warm")

# Vietnamese labels with diacritics, matching the spec's own example table.
_STATUS_LABELS = {
    1: "Tăng hạng",
    -1: "Giảm hạng",
    0: "Không đổi",
}


def _status_for(delta):
    if delta > 0:
        return _STATUS_LABELS[1]
    if delta < 0:
        return _STATUS_LABELS[-1]
    return _STATUS_LABELS[0]


def load_item_degrees(dataset_name):
    """Train-degree per item id (index-aligned), the same source the
    project's own near_cold/long_tail/warm group metrics are built from."""
    path = evaluation_protocol_dir(dataset_name) / "item_degrees_train.json"
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return np.asarray(payload["degrees"], dtype=np.int64)


def build_group_summary_rows(
    dataset_name, baseline_name, k, deltas, in_b, in_t, near_cold, long_tail, warm, users,
    checkpoint_baseline, checkpoint_taxpro, baseline_validation_score, taxpro_validation_score,
):
    """One summary row per Item_Group ("overall" plus each degree-based
    group) for a single (dataset, baseline, k). All arg arrays share one
    length (one entry per detail row for this k); users is user-per-row.
    Shared by both the live scoring path and any offline recomputation from
    an already-written ranking_details.csv, so the two stay identical.
    """
    deltas = np.asarray(deltas, dtype=np.int64)
    in_b = np.asarray(in_b, dtype=bool)
    in_t = np.asarray(in_t, dtype=bool)
    near_cold = np.asarray(near_cold, dtype=bool)
    long_tail = np.asarray(long_tail, dtype=bool)
    warm = np.asarray(warm, dtype=bool)
    users = np.asarray(users, dtype=np.int64)

    masks = {
        "overall": np.ones(deltas.shape[0], dtype=bool),
        "near_cold": near_cold,
        "long_tail": long_tail,
        "warm": warm,
    }

    rows = []
    for group in ITEM_GROUPS:
        mask = masks[group]
        n_items = int(np.count_nonzero(mask))
        group_deltas = deltas[mask]
        group_in_b = in_b[mask]
        group_in_t = in_t[mask]
        increased = int(np.count_nonzero(group_deltas > 0))
        decreased = int(np.count_nonzero(group_deltas < 0))
        unchanged = int(np.count_nonzero(group_deltas == 0))
        enter_topk = int(np.count_nonzero(group_in_t & ~group_in_b))
        leave_topk = int(np.count_nonzero(group_in_b & ~group_in_t))
        n_users = int(np.unique(users[mask]).size) if n_items else 0
        rows.append({
            "Dataset": dataset_name,
            "Baseline": baseline_name,
            "K": k,
            "Item_Group": group,
            "Increased": increased,
            "Decreased": decreased,
            "Unchanged": unchanged,
            "Enter_TopK": enter_topk,
            "Leave_TopK": leave_topk,
            "Increased_Ratio": (increased / n_items) if n_items else 0.0,
            "Avg_Delta_Rank": float(np.mean(group_deltas)) if n_items else 0.0,
            "Median_Delta_Rank": float(np.median(group_deltas)) if n_items else 0.0,
            "Std_Delta_Rank": float(np.std(group_deltas)) if n_items else 0.0,
            "N_Items": n_items,
            "N_Users": n_users,
            "Checkpoint_Baseline": checkpoint_baseline,
            "Checkpoint_TaxProCL": checkpoint_taxpro,
            "Baseline_Validation_Score": baseline_validation_score,
            "TaxProCL_Validation_Score": taxpro_validation_score,
        })
    return rows


def compare_dataset_baseline(
    dataset_name,
    baseline_name,
    k_values,
    device,
    details_tmp_path,
    batch_size=None,
    num_users=None,
    seed=42,
):
    """Score one (dataset, baseline) pair, streaming detail rows to disk.

    Returns (summary_rows, baseline_choice, taxpro_choice, n_users).
    """
    batch_size = batch_size or config.DEFAULT_BATCH_SIZE
    k_values = sorted({int(k) for k in k_values})
    max_k = max(k_values)

    baseline_choice = checkpoint_selection.select_checkpoint(baseline_name, dataset_name)
    taxpro_choice = checkpoint_selection.select_checkpoint(config.TAXONOMY_MODEL, dataset_name)
    logger.info(
        "%s vs TaxPro-CL on %s: baseline=%s (val=%.4f) taxpro=%s (val=%.4f)",
        baseline_name, dataset_name,
        relative_to_project(baseline_choice.run_dir), baseline_choice.validation_score,
        relative_to_project(taxpro_choice.run_dir), taxpro_choice.validation_score,
    )

    baseline_model, baseline_dataset, _, _ = inference.load_model(baseline_choice.run_dir, device)
    taxpro_model, taxpro_dataset, _, _ = inference.load_model(taxpro_choice.run_dir, device)

    protocol_dir = evaluation_protocol_dir(dataset_name)
    overall_targets = load_targets(protocol_dir, "test")["overall"]
    item_degrees = load_item_degrees(dataset_name)
    users = sorted(overall_targets)
    if num_users is not None and num_users < len(users):
        import random

        users = sorted(random.Random(seed).sample(users, num_users))

    group_state = {
        k: {
            "delta_ranks": [], "in_b": [], "in_t": [],
            "near_cold": [], "long_tail": [], "warm": [], "user": [],
        }
        for k in k_values
    }

    details_tmp_path = Path(details_tmp_path)
    details_tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with details_tmp_path.open("w", encoding="utf-8", newline="") as out_stream:
        writer = csv.DictWriter(out_stream, fieldnames=DETAIL_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        n_batches = (len(users) + batch_size - 1) // batch_size
        for batch_index, start in enumerate(range(0, len(users), batch_size)):
            batch_users = users[start:start + batch_size]
            order_b, rank_b = inference.compute_batch_order_and_rank(
                baseline_model, baseline_dataset, device, batch_users, split="test"
            )
            order_t, rank_t = inference.compute_batch_order_and_rank(
                taxpro_model, taxpro_dataset, device, batch_users, split="test"
            )
            order_b_np = order_b.cpu().numpy()
            order_t_np = order_t.cpu().numpy()
            rank_b_np = rank_b.cpu().numpy()
            rank_t_np = rank_t.cpu().numpy()

            for row_idx, user in enumerate(batch_users):
                positives = {int(item) for item in overall_targets.get(user, ())}
                topk_b_full = order_b_np[row_idx, :max_k]
                topk_t_full = order_t_np[row_idx, :max_k]
                for k in k_values:
                    candidate_set = (
                        set(topk_b_full[:k].tolist())
                        | set(topk_t_full[:k].tolist())
                        | positives
                    )
                    if not candidate_set:
                        continue
                    candidates = sorted(candidate_set)
                    candidates_arr = np.asarray(candidates, dtype=np.int64)
                    rb_arr = rank_b_np[row_idx, candidates_arr]
                    rt_arr = rank_t_np[row_idx, candidates_arr]
                    delta_arr = rb_arr - rt_arr
                    in_b_arr = rb_arr <= k
                    in_t_arr = rt_arr <= k
                    degree_arr = item_degrees[candidates_arr]
                    near_cold_arr = (degree_arr >= 1) & (degree_arr <= NEAR_COLD_MAX_DEGREE)
                    long_tail_arr = (degree_arr >= 1) & (degree_arr <= LONG_TAIL_MAX_DEGREE)
                    warm_arr = degree_arr > LONG_TAIL_MAX_DEGREE

                    acc = group_state[k]
                    acc["delta_ranks"].extend(delta_arr.tolist())
                    acc["in_b"].extend(in_b_arr.tolist())
                    acc["in_t"].extend(in_t_arr.tolist())
                    acc["near_cold"].extend(near_cold_arr.tolist())
                    acc["long_tail"].extend(long_tail_arr.tolist())
                    acc["warm"].extend(warm_arr.tolist())
                    acc["user"].extend([user] * len(candidates))

                    for item, rb, rt, delta, in_b, in_t, degree, near_cold, long_tail, warm in zip(
                        candidates,
                        rb_arr.tolist(),
                        rt_arr.tolist(),
                        delta_arr.tolist(),
                        in_b_arr.tolist(),
                        in_t_arr.tolist(),
                        degree_arr.tolist(),
                        near_cold_arr.tolist(),
                        long_tail_arr.tolist(),
                        warm_arr.tolist(),
                    ):
                        writer.writerow({
                            "Dataset": dataset_name,
                            "User": user,
                            "Item": item,
                            "Baseline": baseline_name,
                            "K": k,
                            "Rank_Baseline": rb,
                            "Rank_TaxProCL": rt,
                            "Delta_Rank": delta,
                            "Status": _status_for(delta),
                            "In_TopK_Baseline": in_b,
                            "In_TopK_TaxProCL": in_t,
                            "Is_Ground_Truth": item in positives,
                            "Item_Degree": degree,
                            "Is_Near_Cold": near_cold,
                            "Is_Long_Tail": long_tail,
                            "Is_Warm": warm,
                        })
            if (batch_index + 1) % 20 == 0 or batch_index + 1 == n_batches:
                logger.info(
                    "%s/%s: batch %d/%d (%d users)",
                    dataset_name, baseline_name, batch_index + 1, n_batches, len(batch_users),
                )

    summary_rows = []
    for k in k_values:
        acc = group_state[k]
        summary_rows.extend(build_group_summary_rows(
            dataset_name, baseline_name, k,
            acc["delta_ranks"], acc["in_b"], acc["in_t"],
            acc["near_cold"], acc["long_tail"], acc["warm"], acc["user"],
            relative_to_project(baseline_choice.run_dir),
            relative_to_project(taxpro_choice.run_dir),
            baseline_choice.validation_score,
            taxpro_choice.validation_score,
        ))

    return summary_rows, baseline_choice, taxpro_choice, len(users)
