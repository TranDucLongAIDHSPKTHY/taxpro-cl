"""Realized item-side perturbation norm, TaxPro-CL vs SimGCL, by train-degree bin.

Addresses the specific unverified assumption flagged in the paper's A2
confound discussion: dividing epsilon_max by GCN_layer (TaxPro-CL, Section
3.2) only bounds the *nominal* per-layer draw; it says nothing about the
*realized* embedding displacement produced by the full aggregate() pipeline
(per-layer perturbation interleaved with graph propagation, then mean-pooled
across layers), nor about how that compares to SimGCL's fixed per-layer
epsilon. This script measures the realized quantity directly, on the
already-trained checkpoints (no retraining):

    displacement_i = || item_embedding_perturbed[i] - item_embedding_clean[i] ||_2

using each model's own real aggregate(perturbed=True/False) forward pass
(same code path used during training), averaged over repeated stochastic
draws to get a stable per-item mean (both models re-sample noise on every
call). Items are then grouped by train degree into the paper's own bins
(Strict-Cold, Near-Cold, Mid-Tail, Warm) so the comparison is reported at
the same granularity as the main results.

Usage (from the repository root):
    python -m tools.analysis.realized_perturbation_norm
    python -m tools.analysis.realized_perturbation_norm --datasets amazon-book --repeats 30
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

from config_path.config_path import RESULT_DIR, evaluation_protocol_dir, relative_to_project
from tests.Recommendation_system import checkpoint_selection, inference

logger = logging.getLogger("realized_perturbation_norm")

DATASETS = ["amazon-book", "yelp2018", "musical-instruments", "arts-crafts-and-sewing"]
SEEDS = [0, 1, 42]
REPEATS_DEFAULT = 30

BIN_ORDER = ["strict_cold", "near_cold", "mid_tail", "warm"]


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_item_degrees(dataset_name):
    """Train degree per item id, index-aligned -- same source the paper's
    group definitions (near_cold/long_tail/warm/strict_cold) are built from."""
    path = evaluation_protocol_dir(dataset_name) / "item_degrees_train.json"
    payload = _read_json(path)
    if payload.get("dataset") not in (None, dataset_name):
        raise ValueError(
            "{}: item_degrees_train.json is for dataset {!r}".format(
                dataset_name, payload.get("dataset")
            )
        )
    return [int(value) for value in payload["degrees"]]


def bin_of(degree):
    if degree == 0:
        return "strict_cold"
    if 1 <= degree <= 5:
        return "near_cold"
    if 6 <= degree <= 10:
        return "mid_tail"
    return "warm"


def resolve_seed_run_dirs(model_name, dataset_name):
    choice = checkpoint_selection.select_checkpoint(model_name, dataset_name)
    family_dir = choice.run_dir.parent
    run_dirs = {}
    missing = []
    for seed in SEEDS:
        run_dir = family_dir / "seed{}".format(seed)
        if (run_dir / "run_manifest.json").exists() and (
            run_dir / "best_validation_model.pt"
        ).exists():
            run_dirs[seed] = run_dir
        else:
            missing.append(seed)
    return family_dir, run_dirs, missing


def measure_checkpoint(run_dir, degrees, device, repeats):
    """Return {bin_name: {mean, std, n_items}} of the per-item mean realized
    item-embedding displacement under that checkpoint's own aggregate()."""
    model, dataset, config, model_name = inference.load_model(run_dir, device)
    try:
        model.eval()
        num_items = dataset.num_items
        if len(degrees) != num_items:
            raise ValueError(
                "{}: item_degrees_train.json has {} items, dataset has {}".format(
                    run_dir, len(degrees), num_items
                )
            )
        with torch.no_grad():
            _, clean_items = model.aggregate(perturbed=False)
            clean_items = clean_items.detach()
            accum = torch.zeros(num_items, dtype=torch.float64, device=device)
            for _ in range(repeats):
                _, pert_items = model.aggregate(perturbed=True)
                diff = (pert_items.detach() - clean_items).double()
                accum += torch.linalg.vector_norm(diff, dim=1)
            mean_displacement = (accum / repeats).cpu().tolist()

        buckets = {name: [] for name in BIN_ORDER}
        for item_id, deg in enumerate(degrees):
            buckets[bin_of(deg)].append(mean_displacement[item_id])

        out = {}
        for name in BIN_ORDER:
            values = buckets[name]
            out[name] = {
                "n_items": len(values),
                "mean": statistics.fmean(values) if values else None,
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            }
        return out, model_name, config
    finally:
        del model, dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()


def aggregate_across_seeds(per_seed):
    """Mean across seeds of each seed's own per-bin mean displacement (a
    mean of means -- each seed already averages over `repeats` stochastic
    perturbation draws and over all items in the bin)."""
    out = {}
    for name in BIN_ORDER:
        vals = [p[name]["mean"] for p in per_seed.values() if p[name]["mean"] is not None]
        out[name] = {
            "mean_across_seeds": statistics.fmean(vals) if vals else None,
            "std_across_seeds": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "n_items": next(iter(per_seed.values()))[name]["n_items"] if per_seed else 0,
            "n_seeds": len(vals),
        }
    return out


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--models", nargs="+", default=["TaxPro-CL", "SimGCL"],
                         choices=["TaxPro-CL", "SimGCL"])
    parser.add_argument("--repeats", type=int, default=REPEATS_DEFAULT)
    parser.add_argument(
        "--device", choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output", type=Path,
        default=RESULT_DIR / "week6" / "realized_perturbation_norm.json",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(args.device)
    started = time.time()

    document = {}
    missing_report = []
    failures = []

    for dataset_name in args.datasets:
        degrees = load_item_degrees(dataset_name)
        dataset_block = document.setdefault(dataset_name, {})

        for model_name in args.models:
            family_dir, run_dirs, missing = resolve_seed_run_dirs(model_name, dataset_name)
            if missing:
                missing_report.append({
                    "dataset": dataset_name, "model": model_name,
                    "family_dir": relative_to_project(family_dir),
                    "missing_seeds": missing,
                })

            per_seed = {}
            per_seed_dirs = {}
            config_seen = None
            for seed, run_dir in sorted(run_dirs.items()):
                logger.info("%s/%s seed%s: measuring realized displacement (%d repeats)",
                            dataset_name, model_name, seed, args.repeats)
                t0 = time.time()
                try:
                    result, resolved_name, config = measure_checkpoint(
                        run_dir, degrees, device, args.repeats
                    )
                    per_seed[str(seed)] = result
                    per_seed_dirs[str(seed)] = relative_to_project(run_dir)
                    config_seen = config
                    logger.info("  done in %.1fs -- near_cold mean=%.5f, warm mean=%.5f",
                                time.time() - t0,
                                result["near_cold"]["mean"] or float("nan"),
                                result["warm"]["mean"] or float("nan"))
                except Exception as error:  # noqa: BLE001 - reported, not swallowed
                    logger.exception("%s/%s seed%s: FAILED", dataset_name, model_name, seed)
                    failures.append({
                        "dataset": dataset_name, "model": model_name, "seed": seed,
                        "error": str(error),
                    })

            if not per_seed:
                continue

            entry = {
                "per_seed": per_seed,
                "aggregate": aggregate_across_seeds(per_seed),
                "seeds_evaluated": sorted(int(s) for s in per_seed),
                "missing_seeds": missing,
                "checkpoint_family_dir": relative_to_project(family_dir),
                "checkpoint_dirs": per_seed_dirs,
            }
            if model_name == "SimGCL" and config_seen is not None:
                entry["epsilon_fixed_per_layer"] = float(config_seen.get("epsilon"))
            if model_name == "TaxPro-CL" and config_seen is not None:
                entry["epsilon_max"] = float(config_seen.get("epsilon_max"))
                entry["GCN_layer"] = int(config_seen.get("GCN_layer"))
                entry["gamma_cold"] = float(config_seen.get("gamma_cold", 1.0))
                entry["gamma_warm"] = float(config_seen.get("gamma_warm", 1.0))
                entry["use_adaptive_epsilon"] = bool(config_seen.get("use_adaptive_epsilon", True))
            dataset_block[model_name] = entry

    payload = dict(document)
    payload["_metadata"] = {
        "script": "tools/analysis/realized_perturbation_norm.py",
        "generated_at": datetime.datetime.now().astimezone().isoformat(),
        "git_commit": git_commit(),
        "definition": {
            "quantity": "mean over `repeats` stochastic aggregate(perturbed=True) "
                        "draws of ||item_embedding_perturbed - item_embedding_clean||_2 "
                        "per item, using each checkpoint's own trained weights and graph; "
                        "final embeddings are the model's own mean-pooled-across-layers "
                        "output (post aggregate()), not a single-layer analytical bound.",
            "bins": "strict_cold: degree=0; near_cold: 1<=degree<=5; "
                    "mid_tail: 6<=degree<=10; warm: degree>10 (same definitions as "
                    "the paper's main evaluation groups).",
            "repeats_per_checkpoint": args.repeats,
        },
        "device": str(args.device),
        "missing_checkpoints": missing_report,
        "failures": failures,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    tmp.replace(args.output)
    logger.info("Wrote %s", args.output)
    return 1 if failures or missing_report else 0


if __name__ == "__main__":
    raise SystemExit(main())
