"""Evaluate sealed runs on the test split, once (docs/confirmatory_protocol.md, Section 6).

Runs trained with TAXPRO_DEFER_TEST=1 stop after validation and write no test
metric. This script rebuilds each run's model from its run_manifest.json, then
calls the unchanged batch_test.final_test, which loads best_validation_model.pt
and writes final_test_metrics.json and final_test_group_metrics.json exactly as a
normal run would. It refuses any run that already has a test result, so the test
split cannot be opened twice for the same run.

Usage:
    python -m tools.experiments.evaluate_sealed_test --selection confirmatory_selection.json
    python -m tools.experiments.evaluate_sealed_test --runs <run_dir> [<run_dir> ...]
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import utility.utility_train.batch_test as batch_test  # noqa: E402
from utility.utility_data.data_loader import Data  # noqa: E402


def evaluate(run_dir: Path, device: torch.device, logger: logging.Logger) -> None:
    if (run_dir / "final_test_metrics.json").exists():
        raise RuntimeError(f"{run_dir} already has a test result; the test split is opened once per run")
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
    batch_test.final_test(dataset, trainer.model, device, config, logger)
    stamp = {"evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
             "evaluated_by": "tools/experiments/evaluate_sealed_test.py"}
    (run_dir / "sealed_test_evaluation.json").write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--selection", type=Path, help="confirmatory_selection.json listing test_runs")
    group.add_argument("--runs", type=Path, nargs="+")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("sealed_test")
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
