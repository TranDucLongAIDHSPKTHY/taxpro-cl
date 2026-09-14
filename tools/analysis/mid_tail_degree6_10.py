"""Post-hoc evaluation of a new disjoint item group: mid_tail (train degree 6-10).

The project's locked evaluation protocol defines its item groups purely from
train degree (utility/utility_train/group_evaluator.py GROUP_DEFINITIONS):

    strict_cold : degree == 0
    near_cold   : 1 <= degree <= 5
    long_tail   : 1 <= degree <= 10      (near_cold is a *subset*, not disjoint)
    warm        : degree > 10

`mid_tail` fills the only gap in that ladder: 6 <= degree <= 10, i.e. exactly
long_tail minus near_cold. It is therefore disjoint from near_cold by
construction, and near_cold + mid_tail must reconstitute long_tail exactly --
this module asserts that per user before evaluating anything.

Nothing is retrained. Per-user full-catalog rankings are not persisted by the
training pipeline (only aggregated final_test_group_metrics.json is), so they
are regenerated with one inference pass per checkpoint, reusing
tests/Recommendation_system/inference.py's checkpoint loader and scorer
unchanged. The same regenerated rankings are also scored against the existing
near_cold/long_tail targets and diffed against each run's own on-disk
final_test_group_metrics.json; a large diff there means the inference-reuse
plumbing is wrong and the mid_tail numbers must not be trusted.

Usage (from the repository root):
    python -m tools.analysis.mid_tail_degree6_10
    python -m tools.analysis.mid_tail_degree6_10 --datasets amazon-book \
        --models TaxPro-CL --device cuda --resume
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from config_path.config_path import (
    RESULT_DIR,
    evaluation_protocol_dir,
    relative_to_project,
)
from tests.Recommendation_system import checkpoint_selection, inference
from utility.utility_train.group_evaluator import evaluate_target_slice

logger = logging.getLogger("mid_tail")

MID_TAIL_MIN_DEGREE = 6
MID_TAIL_MAX_DEGREE = 10
MID_TAIL_DEFINITION = "{} <= train_item_degree <= {}".format(
    MID_TAIL_MIN_DEGREE, MID_TAIL_MAX_DEGREE
)

DATASETS = [
    "amazon-book",
    "yelp2018",
    "musical-instruments",
    "arts-crafts-and-sewing",
]

# Checkpoint-directory / model_result_dir names (NOT the paper's display names).
MODELS = ["LightGCN", "SGL", "SimGCL", "NCL", "XSimGCL", "TaxPro-CL"]

# Paper-facing display names used as the JSON keys.
DISPLAY_NAMES = {
    "LightGCN": "LightGCN",
    "SGL": "SGL-ED",
    "SimGCL": "SimGCL",
    "NCL": "NCL",
    "XSimGCL": "XSimGCL",
    "TaxPro-CL": "TaxPro-CL",
}

SEEDS = [0, 1, 2, 3, 4, 42]
KS = (10, 20)

# Cross-check tolerance on Recall/NDCG/Precision@K for the *existing* groups.
# The rankings are regenerated through the same scorer, exclusion rule and
# evaluate_target_slice that wrote final_test_group_metrics.json, so the
# expected difference is exactly zero; the tolerance only absorbs JSON
# round-tripping of float64.
CROSS_CHECK_TOLERANCE = 1e-12

OUTPUT_PATH = RESULT_DIR / "week6" / "mid_tail_degree6_10.json"


# --------------------------------------------------------------------------
# protocol / target construction
# --------------------------------------------------------------------------
def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_item_degrees(dataset_name):
    """Train degree per item id, index-aligned (the group masks' own source)."""
    path = evaluation_protocol_dir(dataset_name) / "item_degrees_train.json"
    payload = _read_json(path)
    if payload.get("dataset") not in (None, dataset_name):
        raise ValueError(
            "{}: item_degrees_train.json is for dataset {!r}".format(
                dataset_name, payload.get("dataset")
            )
        )
    return [int(value) for value in payload["degrees"]]


def _normalize(targets):
    return {
        int(user): [int(item) for item in items]
        for user, items in targets.items()
        if items
    }


def build_targets(dataset_name):
    """Return (overall, near_cold, long_tail, mid_tail) test-target dicts.

    mid_tail is derived from `overall` (verified below to hold every user's
    full test-positive list) by filtering to degree in [6, 10]. The partition
    identity near_cold U mid_tail == long_tail is asserted for EVERY user --
    if it does not hold exactly this raises rather than reporting a number
    built on a broken group definition.
    """
    protocol_dir = evaluation_protocol_dir(dataset_name)
    payload = _read_json(protocol_dir / "test_targets_by_group.json")
    degrees = load_item_degrees(dataset_name)

    overall = _normalize(payload["overall"])
    near_cold = _normalize(payload["near_cold"])
    long_tail = _normalize(payload["long_tail"])

    def in_mid(item):
        return MID_TAIL_MIN_DEGREE <= degrees[item] <= MID_TAIL_MAX_DEGREE

    mid_tail = {}
    for user, items in overall.items():
        kept = [item for item in items if in_mid(item)]
        if kept:
            mid_tail[user] = kept

    # `overall` must be the superset every other group was carved out of.
    for name, group in (("near_cold", near_cold), ("long_tail", long_tail)):
        for user, items in group.items():
            if not set(items) <= set(overall.get(user, ())):
                raise AssertionError(
                    "{}: {} targets for user {} are not a subset of overall "
                    "targets -- 'overall' is not the full test-positive list".format(
                        dataset_name, name, user
                    )
                )

    # Exact partition: near_cold (1-5) U mid_tail (6-10) == long_tail (1-10).
    users = set(near_cold) | set(mid_tail) | set(long_tail)
    mismatches = []
    for user in users:
        reconstructed = set(near_cold.get(user, ())) | set(mid_tail.get(user, ()))
        if reconstructed != set(long_tail.get(user, ())):
            mismatches.append(user)
        if set(near_cold.get(user, ())) & set(mid_tail.get(user, ())):
            mismatches.append(user)
    if mismatches:
        raise AssertionError(
            "{}: near_cold U mid_tail != long_tail for {} user(s), e.g. {}".format(
                dataset_name, len(mismatches), sorted(mismatches)[:10]
            )
        )

    logger.info(
        "%s: mid_tail targets built -- %d eligible users, %d positive "
        "interactions (partition vs near_cold/long_tail verified for all "
        "%d users)",
        dataset_name,
        len(mid_tail),
        sum(len(items) for items in mid_tail.values()),
        len(users),
    )
    return overall, near_cold, long_tail, mid_tail


# --------------------------------------------------------------------------
# checkpoint discovery
# --------------------------------------------------------------------------
def resolve_seed_run_dirs(model_name, dataset_name):
    """Return {seed: run_dir} for the LOCKED run family of (model, dataset).

    checkpoint_selection.select_checkpoint pins the run family the paper
    reports (including config.CHECKPOINT_OVERRIDES for TaxPro-CL); its
    .run_dir.parent is that family, whose seed* subdirectories are the
    project's standard 3-seed protocol. Missing seeds are reported, never
    silently skipped or padded.
    """
    choice = checkpoint_selection.select_checkpoint(model_name, dataset_name)
    family_dir = choice.run_dir.parent
    run_dirs = {}
    missing = []
    for seed in SEEDS:
        run_dir = family_dir / "seed{}".format(seed)
        if (
            run_dir.is_dir()
            and (run_dir / "run_manifest.json").exists()
            and (run_dir / "best_validation_model.pt").exists()
        ):
            run_dirs[seed] = run_dir
        else:
            missing.append(seed)
    return choice, family_dir, run_dirs, missing


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------
def compute_rankings(model, dataset, device, users, batch_size, max_k):
    """Top-`max_k` full-catalog ranking per user, train+validation excluded.

    Scoring and the candidate-exclusion rule are exactly
    utility.utility_train.group_evaluator.evaluate_model_full_catalog's --
    model.get_rating_for_test, then -inf on
    dataset.get_user_pos_items_for_evaluation(..., "test"), then
    torch.topk(max_k). That is the code path that produced every run's saved
    final_test_group_metrics.json, so the near_cold/long_tail cross-check
    reproduces those numbers bit-exactly (verified: 0.0 abs diff).

    inference.compute_batch_order_and_rank offers the same scores via a full
    argsort, but argsort and topk break score ties in different orders; that
    leaves recall/precision (set-based) identical while perturbing NDCG at the
    ~1e-5 level, which would blunt the cross-check. The genuinely hard,
    model-class-specific part -- loading a checkpoint for any of the six
    architectures -- is still reused unchanged from
    tests.Recommendation_system.inference.load_model.
    """
    rankings = {}
    total = len(users)
    start = 0
    with torch.no_grad():
        while start < total:
            size = batch_size
            while True:
                batch_users = users[start:start + size]
                try:
                    batch_tensor = torch.tensor(
                        batch_users, dtype=torch.long, device=device
                    )
                    rating = model.get_rating_for_test(batch_tensor)
                    excluded = dataset.get_user_pos_items_for_evaluation(
                        batch_users, "test"
                    )
                    for row, items in enumerate(excluded):
                        if len(items):
                            rating[row, list(items)] = float("-inf")
                    _, top_items = torch.topk(rating, k=max_k)
                    break
                except torch.cuda.OutOfMemoryError:
                    if size <= 8:
                        raise
                    size = max(8, size // 2)
                    torch.cuda.empty_cache()
                    logger.warning("CUDA OOM -- retrying with batch_size=%d", size)
            for user, items in zip(batch_users, top_items.detach().cpu().tolist()):
                rankings[int(user)] = items
            del rating, top_items
            start += len(batch_users)
    return rankings


def _slice_payload(result):
    return {
        "eligible_users": int(result["eligible_users"]),
        "relevant_items": int(result["relevant_items"]),
        "positive_interactions": int(result["positive_interactions"]),
        "recall": {k: float(v) for k, v in result["recall"].items()},
        "ndcg": {k: float(v) for k, v in result["ndcg"].items()},
        "precision": {k: float(v) for k, v in result["precision"].items()},
    }


def cross_check(run_dir, recomputed):
    """Diff recomputed near_cold/long_tail against the run's saved metrics."""
    path = Path(run_dir) / "final_test_group_metrics.json"
    if not path.exists():
        return {"status": "missing_reference", "max_abs_diff": None, "file": None}
    saved = _read_json(path)
    max_diff = 0.0
    worst = None
    details = {}
    for group in ("near_cold", "long_tail"):
        if group not in saved:
            continue
        for metric in ("recall", "ndcg", "precision"):
            for k in KS:
                key = str(int(k))
                reference = float(saved[group][metric][key])
                observed = float(recomputed[group][metric][key])
                diff = abs(reference - observed)
                details["{}_{}_at_{}".format(group, metric, key)] = {
                    "saved": reference,
                    "recomputed": observed,
                    "abs_diff": diff,
                }
                if diff > max_diff:
                    max_diff = diff
                    worst = "{}_{}_at_{}".format(group, metric, key)
    return {
        "status": "pass" if max_diff <= CROSS_CHECK_TOLERANCE else "FAIL",
        "max_abs_diff": max_diff,
        "worst_metric": worst,
        "file": relative_to_project(path),
        "metrics": details,
    }


def evaluate_run(run_dir, targets, device, batch_size):
    """One checkpoint -> mid_tail metrics + near_cold/long_tail cross-check."""
    overall, near_cold, long_tail, mid_tail = targets
    model, dataset, _config, _name = inference.load_model(run_dir, device)
    try:
        users = sorted(overall)
        started = time.time()
        rankings = compute_rankings(
            model, dataset, device, users, batch_size, max(KS)
        )
        elapsed = time.time() - started
        logger.info(
            "  scored %d users in %.1fs (%s)",
            len(users), elapsed, relative_to_project(run_dir),
        )
        recomputed = {
            "mid_tail": _slice_payload(evaluate_target_slice(rankings, mid_tail, KS)),
            "near_cold": _slice_payload(evaluate_target_slice(rankings, near_cold, KS)),
            "long_tail": _slice_payload(evaluate_target_slice(rankings, long_tail, KS)),
        }
    finally:
        del model, dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return recomputed, cross_check(run_dir, recomputed), elapsed


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------
def aggregate(per_seed):
    """mean and sample (ddof=1) std across seeds for every metric/K."""
    summary = {}
    for metric in ("recall", "ndcg", "precision"):
        for k in KS:
            key = str(int(k))
            values = [
                float(payload["mid_tail"][metric][key]) for payload in per_seed.values()
            ]
            base = "{}{}".format(metric, key)
            summary["{}_mean".format(base)] = (
                statistics.fmean(values) if values else None
            )
            summary["{}_std".format(base)] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
    return summary


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device", choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--resume", action="store_true",
        help="Keep (model, dataset) entries already present in --output.",
    )
    parser.add_argument(
        "--retries", type=int, default=2,
        help="Retries per checkpoint on transient failure (shared GPU).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    device = torch.device(args.device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    document = {}
    if args.resume and output_path.exists():
        document = _read_json(output_path)
        logger.info("Resuming from %s", relative_to_project(output_path))
    document.pop("_metadata", None)

    checkpoint_dirs = {}
    cross_checks = {}
    missing_report = []
    failures = []
    started = time.time()

    for dataset_name in args.datasets:
        targets = build_targets(dataset_name)
        overall, near_cold, long_tail, mid_tail = targets
        dataset_block = document.setdefault(dataset_name, {})
        dataset_block["_mid_tail_support"] = {
            "definition": MID_TAIL_DEFINITION,
            "eligible_users": len(mid_tail),
            "relevant_items": len({i for v in mid_tail.values() for i in v}),
            "positive_interactions": sum(len(v) for v in mid_tail.values()),
            "near_cold_eligible_users": len(near_cold),
            "long_tail_eligible_users": len(long_tail),
            "overall_eligible_users": len(overall),
            "partition_check": "near_cold U mid_tail == long_tail verified for all users",
        }

        for model_name in args.models:
            display = DISPLAY_NAMES[model_name]
            if args.resume and display in dataset_block and dataset_block[display].get("per_seed"):
                logger.info("%s/%s: already present, skipping", dataset_name, display)
                continue

            try:
                choice, family_dir, run_dirs, missing = resolve_seed_run_dirs(
                    model_name, dataset_name
                )
            except Exception as error:  # noqa: BLE001 - reported, not swallowed
                logger.error("%s/%s: checkpoint resolution failed: %s",
                             dataset_name, model_name, error)
                failures.append({
                    "dataset": dataset_name, "model": display,
                    "stage": "checkpoint_resolution", "error": str(error),
                })
                continue

            if missing:
                missing_report.append({
                    "dataset": dataset_name,
                    "model": display,
                    "family_dir": relative_to_project(family_dir),
                    "missing_seeds": missing,
                })
                logger.error(
                    "%s/%s: MISSING seed(s) %s under %s",
                    dataset_name, display, missing, relative_to_project(family_dir),
                )

            per_seed = {}
            per_seed_dirs = {}
            for seed, run_dir in sorted(run_dirs.items()):
                logger.info("%s/%s seed%s: %s", dataset_name, display, seed,
                            relative_to_project(run_dir))
                last_error = None
                for attempt in range(args.retries + 1):
                    try:
                        recomputed, check, _elapsed = evaluate_run(
                            run_dir, targets, device, args.batch_size
                        )
                        per_seed[str(seed)] = recomputed
                        per_seed_dirs[str(seed)] = relative_to_project(run_dir)
                        cross_checks[
                            "{}|{}|seed{}".format(dataset_name, display, seed)
                        ] = check
                        if check["status"] == "FAIL":
                            logger.error(
                                "  CROSS-CHECK FAIL: max abs diff %.3e on %s",
                                check["max_abs_diff"], check["worst_metric"],
                            )
                        else:
                            logger.info(
                                "  cross-check %s (max abs diff %.3e)",
                                check["status"], check["max_abs_diff"] or 0.0,
                            )
                        last_error = None
                        break
                    except Exception as error:  # noqa: BLE001
                        last_error = error
                        logger.warning(
                            "  attempt %d/%d failed: %s",
                            attempt + 1, args.retries + 1, error,
                        )
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                if last_error is not None:
                    logger.exception("%s/%s seed%s: giving up", dataset_name, display, seed)
                    failures.append({
                        "dataset": dataset_name, "model": display, "seed": seed,
                        "stage": "inference", "error": str(last_error),
                    })

            if not per_seed:
                continue

            entry = aggregate(per_seed)
            entry["seeds_evaluated"] = sorted(int(s) for s in per_seed)
            entry["missing_seeds"] = missing
            entry["checkpoint_family_dir"] = relative_to_project(family_dir)
            entry["checkpoint_dirs"] = per_seed_dirs
            entry["per_seed"] = per_seed
            dataset_block[display] = entry
            checkpoint_dirs["{}|{}".format(dataset_name, display)] = per_seed_dirs

            # Persist incrementally: a shared-GPU crash mid-sweep must not
            # throw away the passes already completed.
            _write(output_path, document, cross_checks, missing_report,
                   failures, args, started, partial=True)

    _write(output_path, document, cross_checks, missing_report, failures,
           args, started, partial=False)
    logger.info("Wrote %s", output_path)
    return 1 if failures or missing_report else 0


def _write(path, document, cross_checks, missing_report, failures, args, started, partial):
    checks = list(cross_checks.values())
    diffs = [c["max_abs_diff"] for c in checks if c["max_abs_diff"] is not None]
    per_dataset = {}
    for key, check in cross_checks.items():
        dataset_name = key.split("|", 1)[0]
        bucket = per_dataset.setdefault(
            dataset_name, {"status": "pass", "max_abs_diff": 0.0, "checkpoints": 0}
        )
        bucket["checkpoints"] += 1
        if check["max_abs_diff"] is not None:
            bucket["max_abs_diff"] = max(bucket["max_abs_diff"], check["max_abs_diff"])
        if check["status"] != "pass":
            bucket["status"] = check["status"]

    payload = dict(document)
    payload["_metadata"] = {
        "script": "tools/analysis/mid_tail_degree6_10.py",
        "generated_at": datetime.datetime.now().astimezone().isoformat(),
        "partial": bool(partial),
        "git_commit": git_commit(),
        "group": {
            "name": "mid_tail",
            "definition": MID_TAIL_DEFINITION,
            "disjoint_from": "near_cold (1 <= degree <= 5)",
            "identity": "near_cold U mid_tail == long_tail (asserted per user)",
            "source": "preprocessed/evaluation_protocol/split_seed_42/<dataset>/"
                      "item_degrees_train.json + test_targets_by_group.json",
        },
        "protocol": {
            "split": "test",
            "split_seed": 42,
            "candidate_catalog": "full (train+validation positives excluded)",
            "checkpoint": "best_validation_model.pt",
            "ks": list(KS),
            "seeds": SEEDS,
            "std": "sample standard deviation (ddof=1)",
        },
        "device": str(args.device),
        "batch_size": int(args.batch_size),
        "checkpoint_dirs": _checkpoint_index(document),
        "cross_check": {
            "description": "near_cold/long_tail recomputed from the same "
                           "regenerated rankings, diffed against each run's "
                           "own final_test_group_metrics.json",
            "tolerance": CROSS_CHECK_TOLERANCE,
            "checkpoints_checked": len(checks),
            "max_abs_diff_observed": max(diffs) if diffs else None,
            "status": "pass" if all(c["status"] == "pass" for c in checks) and checks else (
                "FAIL" if any(c["status"] == "FAIL" for c in checks) else "no_reference"
            ),
            "per_dataset": per_dataset,
            "per_checkpoint": cross_checks,
        },
        "missing_checkpoints": missing_report,
        "failures": failures,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    tmp = Path(path).with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    tmp.replace(path)


def _checkpoint_index(document):
    index = {}
    for dataset_name, block in document.items():
        for display, entry in block.items():
            if display.startswith("_"):
                continue
            index["{}|{}".format(dataset_name, display)] = entry.get("checkpoint_dirs", {})
    return index


if __name__ == "__main__":
    raise SystemExit(main())
