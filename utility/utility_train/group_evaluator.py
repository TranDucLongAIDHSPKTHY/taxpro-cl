"""Generic full-catalog group-wise evaluator for TaxPro-CL.

Groups are defined only from train item degree.  Ranking always uses the same
full candidate catalog as overall evaluation; group evaluation filters only
the relevant positives and excludes users with no positive in that slice.
"""

from __future__ import annotations

import csv
import json
import math
import numbers
from pathlib import Path


GROUP_DEFINITIONS = {
    "near_cold": "1 <= train_item_degree <= 5",
    "long_tail": "1 <= train_item_degree <= 10",
    "warm": "train_item_degree > 10",
}

CORE_CSV_COLUMNS = [
    "recall_at_10_overall",
    "recall_at_20_overall",
    "ndcg_at_10_overall",
    "ndcg_at_20_overall",
    "recall_at_10_near_cold",
    "recall_at_20_near_cold",
    "ndcg_at_10_near_cold",
    "ndcg_at_20_near_cold",
    "recall_at_10_long_tail",
    "recall_at_20_long_tail",
    "ndcg_at_10_long_tail",
    "ndcg_at_20_long_tail",
    "recall_at_10_warm",
    "recall_at_20_warm",
    "ndcg_at_10_warm",
    "ndcg_at_20_warm",
]
PRECISION_CSV_COLUMNS = [
    "precision_at_10_overall",
    "precision_at_20_overall",
    "precision_at_10_near_cold",
    "precision_at_20_near_cold",
    "precision_at_10_long_tail",
    "precision_at_20_long_tail",
    "precision_at_10_warm",
    "precision_at_20_warm",
]
CSV_COLUMNS = CORE_CSV_COLUMNS + PRECISION_CSV_COLUMNS
SUPPORT_COLUMNS = (
    "group",
    "eligible_users",
    "relevant_items",
    "positive_interactions",
)


def _normalize_targets(targets):
    return {
        int(user): tuple(int(item) for item in items)
        for user, items in targets.items()
        if items
    }


def intersect_targets(overall_targets, group_items):
    """Return exact ground-truth intersection for one item group."""
    mask = set(int(item) for item in group_items)
    return {
        int(user): [int(item) for item in items if int(item) in mask]
        for user, items in overall_targets.items()
        if any(int(item) in mask for item in items)
    }


def metric_support(targets):
    normalized = _normalize_targets(targets)
    return {
        "eligible_users": len(normalized),
        "relevant_items": len(
            {item for items in normalized.values() for item in items}
        ),
        "positive_interactions": sum(len(items) for items in normalized.values()),
    }


def _user_recall(ranking, positives, k):
    positive_set = set(positives)
    return sum(item in positive_set for item in ranking[:k]) / float(len(positive_set))


def _user_ndcg(ranking, positives, k):
    positive_set = set(positives)
    dcg = sum(
        (1.0 / math.log2(rank + 2)) if item in positive_set else 0.0
        for rank, item in enumerate(ranking[:k])
    )
    ideal_length = min(len(positive_set), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_length))
    return dcg / idcg if idcg else 0.0


def _user_precision(ranking, positives, k):
    positive_set = set(positives)
    return sum(item in positive_set for item in ranking[:k]) / float(k)


def evaluate_target_slice(rankings, targets, ks=(10, 20)):
    """Evaluate one target slice and average only over eligible users."""
    normalized = _normalize_targets(targets)
    missing = sorted(set(normalized) - set(int(user) for user in rankings))
    if missing:
        raise ValueError("Missing rankings for eligible users: {}".format(missing[:10]))
    eligible = sorted(normalized)
    support = metric_support(normalized)
    recalls = {}
    ndcgs = {}
    precisions = {}
    for k in ks:
        k = int(k)
        if k <= 0:
            raise ValueError("K must be positive")
        if eligible:
            recalls[str(k)] = sum(
                _user_recall(rankings[user], normalized[user], k) for user in eligible
            ) / len(eligible)
            ndcgs[str(k)] = sum(
                _user_ndcg(rankings[user], normalized[user], k) for user in eligible
            ) / len(eligible)
            precisions[str(k)] = sum(
                _user_precision(rankings[user], normalized[user], k)
                for user in eligible
            ) / len(eligible)
        else:
            recalls[str(k)] = 0.0
            ndcgs[str(k)] = 0.0
            precisions[str(k)] = 0.0
    return {
        **support,
        "recall": recalls,
        "ndcg": ndcgs,
        "precision": precisions,
    }


def evaluate_rankings(
    rankings,
    overall_targets,
    targets_by_group,
    ks=(10, 20),
    definitions=None,
):
    """Return the locked stakeholder-facing JSON schema."""
    rankings = {
        int(user): tuple(int(item) for item in items)
        for user, items in rankings.items()
    }
    definitions = definitions or GROUP_DEFINITIONS
    overall_slice = evaluate_target_slice(rankings, overall_targets, ks)
    result = {
        "overall": {
            "recall": overall_slice["recall"],
            "ndcg": overall_slice["ndcg"],
            "precision": overall_slice["precision"],
        }
    }
    for group in ("near_cold", "long_tail", "warm"):
        evaluated = evaluate_target_slice(rankings, targets_by_group[group], ks)
        result[group] = {
            "definition": definitions[group],
            **evaluated,
        }
    validate_result_schema(result, ks)
    return result


def validate_result_schema(result, ks=(10, 20), require_protocol=False):
    """Reject incomplete/non-finite group metric artifacts before persistence."""
    expected_groups = ("overall", "near_cold", "long_tail", "warm")
    missing_groups = [group for group in expected_groups if group not in result]
    if missing_groups:
        raise ValueError("Missing evaluation groups: {}".format(missing_groups))
    for group in expected_groups:
        payload = result[group]
        for metric in ("precision", "recall", "ndcg"):
            if metric not in payload:
                raise ValueError("Missing {} for {}".format(metric, group))
            for k in ks:
                key = str(int(k))
                if key not in payload[metric]:
                    raise ValueError("Missing {}@{} for {}".format(metric, k, group))
                value = payload[metric][key]
                if not isinstance(value, numbers.Real) or not math.isfinite(float(value)):
                    raise ValueError("Non-finite {}@{} for {}".format(metric, k, group))
                if not 0.0 <= float(value) <= 1.0:
                    raise ValueError("Out-of-range {}@{} for {}".format(metric, k, group))
        if group != "overall":
            for support in (
                "eligible_users",
                "relevant_items",
                "positive_interactions",
            ):
                value = payload.get(support)
                if not isinstance(value, numbers.Integral) or int(value) < 0:
                    raise ValueError("Missing/invalid {} for {}".format(support, group))
    if require_protocol and "protocol" not in result:
        raise ValueError("Group result lacks protocol evidence")
    return result


def flatten_metrics(result):
    """Flatten the locked 16 Recall/NDCG metrics plus 8 Precision metrics."""
    flattened = {}
    for group in ("overall", "near_cold", "long_tail", "warm"):
        suffix = group
        for metric in ("recall", "ndcg"):
            for k in ("10", "20"):
                flattened["{}_at_{}_{}".format(metric, k, suffix)] = float(
                    result[group][metric][k]
                )
    for group in ("overall", "near_cold", "long_tail", "warm"):
        for k in ("10", "20"):
            flattened["precision_at_{}_{}".format(k, group)] = float(
                result[group]["precision"][k]
            )
    if list(flattened) != CSV_COLUMNS:
        raise AssertionError("Unexpected metric column order")
    return flattened


def write_result_json(path, result):
    validate_result_schema(result, require_protocol=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def write_result_csv(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = flatten_metrics(result)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


def support_rows(result):
    validate_result_schema(result)
    return [
        {
            "group": group,
            "eligible_users": int(result[group]["eligible_users"]),
            "relevant_items": int(result[group]["relevant_items"]),
            "positive_interactions": int(result[group]["positive_interactions"]),
        }
        for group in ("near_cold", "long_tail", "warm")
    ]


def write_support_csv(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=SUPPORT_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(support_rows(result))


def load_targets(protocol_directory, split):
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    path = Path(protocol_directory) / ("{}_targets_by_group.json".format(split))
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return {
        "overall": _normalize_targets(payload["overall"]),
        "near_cold": _normalize_targets(payload["near_cold"]),
        "long_tail": _normalize_targets(payload["long_tail"]),
        "warm": _normalize_targets(payload["warm"]),
    }


def attach_protocol_evidence(result, protocol_directory, split):
    """Attach mask hash and verify saved support counts for every item slice."""
    protocol_directory = Path(protocol_directory)
    with (protocol_directory / "manifest.json").open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    with (protocol_directory / "group_statistics.json").open(
        "r", encoding="utf-8"
    ) as stream:
        statistics = json.load(stream)
    mask_hash = manifest["output_hashes"]["group_masks.npz"]
    for group in ("near_cold", "long_tail", "warm"):
        expected = statistics["groups"][group][split]
        for key in ("eligible_users", "relevant_items", "positive_interactions"):
            if int(result[group][key]) != int(expected[key]):
                raise AssertionError(
                    "{} {} support mismatch for {}: {} != {}".format(
                        split, group, key, result[group][key], expected[key]
                    )
                )
        result[group]["source_group_mask_hash"] = mask_hash
    result["protocol"] = {
        "version": manifest["protocol_version"],
        "split_seed": manifest["split_seed"],
        "group_mask_sha256": mask_hash,
        "candidate_catalog": "full",
        "target_split": split,
    }
    validate_result_schema(result, require_protocol=True)
    return result


def evaluate_model_full_catalog(
    dataset,
    model,
    device,
    protocol_directory,
    split="test",
    ks=(10, 20),
    batch_size=256,
):
    """Evaluate any repository model without changing its forward/rating method.

    Torch is imported lazily so the metric helpers remain unit-testable with the
    Python standard library only.
    """
    import torch

    targets = load_targets(protocol_directory, split)
    overall_targets = targets.pop("overall")
    users = sorted(overall_targets)
    rankings = {}
    model = model.eval()
    with torch.no_grad():
        for start in range(0, len(users), int(batch_size)):
            batch_users = users[start : start + int(batch_size)]
            batch_tensor = torch.tensor(batch_users, dtype=torch.long, device=device)
            rating = model.get_rating_for_test(batch_tensor)
            excluded = dataset.get_user_pos_items_for_evaluation(batch_users, split)
            for row, items in enumerate(excluded):
                if len(items):
                    rating[row, list(items)] = float("-inf")
            _, top_items = torch.topk(rating, k=max(int(k) for k in ks))
            for user, items in zip(batch_users, top_items.detach().cpu().tolist()):
                rankings[user] = items
    result = evaluate_rankings(rankings, overall_targets, targets, ks=ks)
    return attach_protocol_evidence(result, protocol_directory, split)
