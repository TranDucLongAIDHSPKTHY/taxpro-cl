"""Run protocol-scoped TaxProCL experiments sequentially."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.utility_train.group_evaluator import validate_result_schema

POLICIES = ("no_merge", "merge_t5", "merge_t10", "merge_t15")
DATASETS = ("amazon-book", "yelp2018", "musical-instruments", "arts-crafts-and-sewing")
FULL_TRAINING_EPOCHS = 200
LOCKED_P0_DEFAULTS = {
    "dataset_path": "./dataset_verify/",
    "evaluation_protocol_path": "./preprocessed/evaluation_protocol/split_seed_42/",
    "top_K": "[10, 20]",
    "selection_K": 20,
    "early_stopping": 20,
    "interval": 1,
    "embedding_size": 64,
    "batch_size": 2048,
    "test_batch_size": 2048,
    "learn_rate": 0.001,
    "reg_lambda": 0.0001,
    "GCN_layer": 3,
    "mu": 0.9,
    "delta": 1e-8,
    "unknown_taxonomy_policy": "exclude_cl",
    "prototype_mode": "leaf",
    "prototype_update": "ema",
    "mixture_alpha": 0.5,
    "symmetric_info_nce": "false",
    "resume_checkpoint": "none",
    "sparsity_test": 0,
}


def canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--taxonomy-policy", choices=POLICIES, default=None,
        help="Default: the policy selected for the dataset in the paper "
             "(Table 3): merge_t10 for amazon-book, no_merge otherwise.",
    )
    parser.add_argument(
        "--taxonomy-granularity", choices=("leaf", "parent"), default="leaf"
    )
    parser.add_argument(
        "--prototype-mode", choices=("leaf", "parent", "mixture"), default=None
    )
    parser.add_argument(
        "--prototype-update", choices=("ema", "fixed"), default="ema"
    )
    parser.add_argument("--mixture-alpha", type=float, default=0.5)
    parser.add_argument(
        "--augmentation-direction", choices=("taxonomy", "random"), default="taxonomy"
    )
    parser.add_argument(
        "--use-adaptive-epsilon", choices=("true", "false"), default="true",
        help="Degree-adaptive (true, default) vs. fixed (false) perturbation "
             "magnitude. Together with --augmentation-direction, this is the "
             "second factor of the RQ5 2x2 factorial (paper Section 5.4) -- "
             "see tools/README.md for the V0-V3 <-> (direction, epsilon) mapping.",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-worker", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--config-id")
    parser.add_argument("--output-root", type=Path, default=ROOT / "log")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--warm-start-epochs", type=int)
    # Defaults match configure/TaxPro-CL.txt (the reported Amazon-Book main
    # config and every V0-V3 factorial cell); A3/A4 sweeps override these
    # explicitly per sweep point, so changing the bare default only affects
    # invocations that omit the flag entirely.
    parser.add_argument("--epsilon-max", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--temperature-user", type=float, default=0.2,
        help="Item/user-side temperatures are independent; this overrides "
             "the user-side value only (default matches configure/TaxPro-CL.txt).",
    )
    parser.add_argument(
        "--gamma-cold", type=float, default=1.5,
        help="Degree-adaptive-epsilon floor exponent (default matches "
             "configure/TaxPro-CL.txt; Musical-Instruments' main config and "
             "V0-V3 factorial both use 5.0, see tools/README.md).",
    )
    parser.add_argument("--ssl-lambda", type=float, default=0.5)  # matches configure/TaxPro-CL.txt
    parser.add_argument("--mu", type=float, default=0.9)
    parser.add_argument("--embedding-size", type=int, default=64)
    parser.add_argument("--dataset-path", default="./dataset_verify/")
    parser.add_argument(
        "--evaluation-protocol-path",
        default="./preprocessed/evaluation_protocol/split_seed_42/",
    )
    return parser.parse_args(argv)


PAPER_TAXONOMY_POLICY = {
    "amazon-book": "merge_t10",
    "yelp2018": "no_merge",
    "musical-instruments": "no_merge",
    "arts-crafts-and-sewing": "no_merge",
}


def effective_configuration(args):
    if args.taxonomy_policy is None:
        args.taxonomy_policy = PAPER_TAXONOMY_POLICY[args.dataset]
    epochs = (
        args.epochs
        if args.epochs is not None
        else (5 if args.smoke else FULL_TRAINING_EPOCHS)
    )
    warm = (
        args.warm_start_epochs
        if args.warm_start_epochs is not None
        # 20 matches every reported TaxPro-CL run (all datasets, all V0-V3
        # factorial cells); the sole exception is the warm-start-removal
        # companion check (ESM Table S20), which explicitly passes 0.
        else (1 if args.smoke else 20)
    )
    if warm >= epochs:
        raise ValueError("warm-start epochs must be smaller than total epochs")
    prototype_mode = args.prototype_mode or args.taxonomy_granularity
    if (args.taxonomy_granularity == "parent" or prototype_mode in {"parent", "mixture"}) and args.dataset != "amazon-book":
        raise ValueError("Parent granularity A5 is defined only for amazon-book")
    if not (0.0 < args.epsilon_max):
        raise ValueError("epsilon-max must be positive")
    if not (0.0 < args.temperature):
        raise ValueError("temperature must be positive")
    if not (0.0 <= args.ssl_lambda):
        raise ValueError("ssl-lambda must be non-negative")
    if not (0.0 <= args.mu < 1.0):
        raise ValueError("mu must be in [0, 1)")
    if args.embedding_size <= 0:
        raise ValueError("embedding-size must be positive")
    return {
        **LOCKED_P0_DEFAULTS,
        "dataset_path": args.dataset_path,
        "evaluation_protocol_path": args.evaluation_protocol_path,
        "dataset": args.dataset,
        "taxonomy_policy": args.taxonomy_policy,
        "prototype_mode": prototype_mode,
        "prototype_update": args.prototype_update,
        "mixture_alpha": float(args.mixture_alpha),
        "augmentation_direction": args.augmentation_direction,
        "use_adaptive_epsilon": args.use_adaptive_epsilon,
        "epsilon_max": float(args.epsilon_max),
        "temperature": float(args.temperature),
        "temperature_user": float(args.temperature_user),
        "gamma_cold": float(args.gamma_cold),
        "ssl_lambda": float(args.ssl_lambda),
        "mu": float(args.mu),
        "embedding_size": int(args.embedding_size),
        "training_epochs": int(epochs),
        "warm_start_epochs": int(warm),
        "num_worker": int(args.num_worker),
    }


def output_directory(args, config, seed):
    identifier = args.config_id or (
        "taxprocl-{}-{}-{}".format(
            args.dataset,
            args.taxonomy_policy,
            canonical_hash(config)[:12],
        )
    )
    return (
        args.output_root
        / "p0"
        / "taxprocl"
        / args.dataset
        / identifier
        / ("seed" + str(seed))
    )


def is_completed(directory):
    manifest = directory / "run_manifest.json"
    final = directory / "final_test_metrics.json"
    group = directory / "final_test_group_metrics.json"
    if not (manifest.is_file() and final.is_file() and group.is_file()):
        return False
    try:
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        group_payload = json.loads(group.read_text(encoding="utf-8"))
        validate_result_schema(
            group_payload, ks=(10, 20), require_protocol=True
        )
        protocol = group_payload["protocol"]
        return (
            manifest_payload.get("status") == "completed"
            and protocol.get("candidate_catalog") == "full"
            and protocol.get("target_split") == "test"
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def build_command(args, config, seed, directory):
    command = [
        sys.executable,
        str(ROOT / "main.py"),
        "--model",
        "TaxPro-CL",
        "--seeds",
        str(seed),
        "--gpu_id",
        str(args.gpu_id),
        "--device",
        args.device,
    ]
    for key in sorted(config):
        if key == "resume_checkpoint":
            continue
        command.extend(["--" + key, str(config[key])])
    if args.resume:
        checkpoint = directory / "last_model.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError("Resume checkpoint not found: {}".format(checkpoint))
        command.extend(["--resume_checkpoint", str(checkpoint.resolve())])
    return command


def main(argv=None):
    args = parse_args(argv)
    config = effective_configuration(args)
    rows = []
    for seed in dict.fromkeys(args.seeds):
        directory = output_directory(args, config, seed)
        status = "completed" if is_completed(directory) else "pending"
        command = build_command(args, config, seed, directory)
        rows.append((seed, directory, status, command))
        print(
            "TaxProCL dataset={} policy={} granularity={} direction={} seed={} status={} output={}".format(
                args.dataset,
                args.taxonomy_policy,
                args.taxonomy_granularity,
                args.augmentation_direction,
                seed,
                status,
                directory,
            )
        )
        print("  command={}".format(subprocess.list2cmdline(command)))
    if args.dry_run:
        return 0

    failures = 0
    for seed, directory, status, command in rows:
        if status == "completed" and args.skip_completed:
            continue
        if directory.exists() and any(directory.iterdir()) and not args.resume:
            print(
                "Refusing non-empty output without --resume: {}".format(directory),
                file=sys.stderr,
            )
            failures += 1
            continue
        directory.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        environment["TAXPROCL_RUN_OUTPUT_DIR"] = str(directory.resolve())
        completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
        failures += int(completed.returncode != 0)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
