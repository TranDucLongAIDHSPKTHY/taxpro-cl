"""Evaluate sealed runs on the test split, once (docs/confirmatory_protocol.md, Section 6).

Runs trained with TAXPRO_DEFER_TEST=1 stop after validation and write no test
metric. This script rebuilds each run's model from its run_manifest.json, then
calls the unchanged batch_test.final_test, which loads best_validation_model.pt
and writes final_test_metrics.json and final_test_group_metrics.json exactly as a
normal run would. In the same opening it writes final_test_per_user.npz, so paired
analyses never read the test split again. It refuses any run that already has a test
result, so the test split cannot be opened twice for the same run.

Usage:
    python -m tools.experiments.evaluate_sealed_test --selection confirmatory_selection.json
    python -m tools.experiments.evaluate_sealed_test --runs <run_dir> [<run_dir> ...]
    python -m tools.experiments.evaluate_sealed_test --complete <run_dir> [<run_dir> ...]
"""
from __future__ import annotations

import argparse
import ast
import importlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import utility.utility_train.batch_test as batch_test  # noqa: E402
from config_path.config_path import PREPROCESSED_DIR, checkpoint_file  # noqa: E402
from tools.analysis.factorial_multiplicity import GROUPS  # noqa: E402
from utility.utility_data.data_loader import Data  # noqa: E402
from utility.utility_function.tools import mini_batch  # noqa: E402
from utility.utility_train.group_evaluator import (  # noqa: E402
    _normalize_targets,
    _user_ndcg,
    _user_recall,
    load_targets,
)

K = 20


def protocol_directory(config):
    """The evaluation protocol that batch_test.Test reads for this run."""
    root = Path(str(config.get("evaluation_protocol_path",
                               PREPROCESSED_DIR / "evaluation_protocol" / "split_seed_42")))
    if not root.is_absolute():
        root = ROOT / root
    return root / str(config["dataset"])


def ranked_lists(model, dataset, device, config, split="test"):
    """Top-K lists built as batch_test.Test builds them: users in the order of its target dict,
    its batch size, seen items set to -1 and torch.topk. Saturated sigmoid scores tie, and
    another sort or another batching can order tied items differently, which changes NDCG."""
    target_dict = dataset.validation_dict if split == "validation" else dataset.test_dict
    users = list(target_dict.keys())
    top_k = max(ast.literal_eval(str(config["top_K"])))
    rankings = {}
    model = model.eval()
    with torch.no_grad():
        for batch_users in mini_batch(users, batch_size=int(config["test_batch_size"])):
            rows, cols = [], []
            for index, items in enumerate(dataset.get_user_pos_items_for_evaluation(batch_users, split)):
                rows.extend([index] * len(items))
                cols.extend(items)
            rating = model.get_rating_for_test(torch.Tensor(batch_users).long().to(device))
            rating[rows, cols] = -1
            _, top = torch.topk(rating, k=top_k)
            for user, items in zip(batch_users, top.cpu().tolist()):
                rankings[int(user)] = items
    return rankings


def per_user_arrays(model, dataset, device, config, split="test"):
    """Recall@20 and NDCG@20 of every user with a target in each group, from the same rankings,
    targets and metric functions as the stored group metrics."""
    rankings = ranked_lists(model, dataset, device, config, split)
    targets = load_targets(protocol_directory(config), split)
    arrays = {}
    for group in GROUPS:
        positives = _normalize_targets(targets[group])
        users = sorted(positives)
        arrays[f"{group}_users"] = np.asarray(users, dtype=np.int64)
        arrays[f"{group}_recall"] = np.asarray([_user_recall(rankings[u], positives[u], K) for u in users])
        arrays[f"{group}_ndcg"] = np.asarray([_user_ndcg(rankings[u], positives[u], K) for u in users])
    return arrays


def check_against_group_metrics(arrays, group_metrics):
    """The per-user means must reproduce the stored group metrics."""
    for group in GROUPS:
        for name in ("recall", "ndcg"):
            stored = float(group_metrics[group][name]["20"])
            if abs(float(arrays[f"{group}_{name}"].mean()) - stored) > 1e-6:
                raise RuntimeError(f"per-user {name} of {group} does not reproduce the stored value")


def build(run_dir: Path, device: torch.device, logger: logging.Logger):
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise RuntimeError(f"{run_dir} is not a completed run")
    config = manifest["configuration"]
    dataset_root = Path(str(config.get("dataset_path", "./dataset_verify/")))
    if not dataset_root.is_absolute():
        dataset_root = ROOT / dataset_root
    dataset = Data(str(dataset_root / str(config["dataset"])), config, logger=logger)
    dataset.training_output_dir = run_dir
    trainer_class = getattr(importlib.import_module("models." + manifest["model"]), "Trainer")
    trainer = trainer_class(SimpleNamespace(seed=manifest.get("training_seed")), config, dataset, device, logger)
    return trainer.model, dataset, config


def store_per_user(run_dir: Path, arrays, stamp) -> None:
    check_against_group_metrics(
        arrays, json.loads((run_dir / "final_test_group_metrics.json").read_text(encoding="utf-8"))
    )
    np.savez_compressed(run_dir / "final_test_per_user.npz", **arrays)
    stamp = {**stamp, "evaluated_by": "tools/experiments/evaluate_sealed_test.py"}
    (run_dir / "sealed_test_evaluation.json").write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")


def evaluate(run_dir: Path, device: torch.device, logger: logging.Logger) -> None:
    if (run_dir / "final_test_metrics.json").exists():
        raise RuntimeError(f"{run_dir} already has a test result; the test split is opened once per run")
    model, dataset, config = build(run_dir, device, logger)
    batch_test.final_test(dataset, model, device, config, logger)
    store_per_user(run_dir, per_user_arrays(model, dataset, device, config),
                   {"evaluated_at_utc": datetime.now(timezone.utc).isoformat()})


def complete(run_dir: Path, device: torch.device, logger: logging.Logger) -> None:
    """A run whose test result was written but whose per-user values were not. As for a reference
    run, they are computed once from best_validation_model.pt and kept only if their means
    reproduce the stored group metrics; the test result is not written again."""
    for name in ("final_test_metrics.json", "final_test_group_metrics.json"):
        if not (run_dir / name).exists():
            raise RuntimeError(f"{run_dir} has no {name}")
    for name in ("final_test_per_user.npz", "sealed_test_evaluation.json"):
        if (run_dir / name).exists():
            raise RuntimeError(f"{run_dir} already has {name}")
    model, dataset, config = build(run_dir, device, logger)
    checkpoint = batch_test._load_checkpoint(checkpoint_file(run_dir, "best_validation"), device)
    if hasattr(model, "validate_checkpoint_metadata"):
        model.validate_checkpoint_metadata(checkpoint.get("taxprocl_metadata", {}))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    opened = datetime.fromtimestamp((run_dir / "final_test_metrics.json").stat().st_mtime, timezone.utc)
    store_per_user(run_dir, per_user_arrays(model, dataset, device, config),
                   {"evaluated_at_utc": opened.isoformat(),
                    "per_user_values_at_utc": datetime.now(timezone.utc).isoformat()})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--selection", type=Path, help="confirmatory_selection.json listing test_runs")
    group.add_argument("--runs", type=Path, nargs="+")
    group.add_argument("--complete", type=Path, nargs="+",
                       help="runs with a test result but no per-user values; see complete()")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("sealed_test")
    if args.complete:
        for run_dir in args.complete:
            logger.info("Per-user values of an opened run: %s", run_dir)
            complete(Path(run_dir), torch.device(args.device), logger)
        return 0
    if args.selection:
        runs = [ROOT / r for r in json.loads(args.selection.read_text(encoding="utf-8"))["test_runs"]]
    else:
        runs = args.runs
    missing = [r for r in runs if not (r / "best_validation_model.pt").exists()]
    if missing:
        raise SystemExit(f"missing checkpoints: {missing}")
    device = torch.device(args.device)
    for run_dir in runs:
        logger.info("Sealed test: %s", run_dir)
        evaluate(Path(run_dir), device, logger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
