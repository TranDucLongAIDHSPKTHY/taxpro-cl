"""Sequential smoke/full runner for the six inherited baselines."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.utility_train.group_evaluator import validate_result_schema

MODELS = ("LightGCN", "SimGCL", "XSimGCL", "SGL", "NCL")


def is_completed(directory):
    manifest_path = directory / "run_manifest.json"
    overall_path = directory / "final_test_metrics.json"
    group_path = directory / "final_test_group_metrics.json"
    if not (
        manifest_path.is_file()
        and overall_path.is_file()
        and group_path.is_file()
    ):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        group = json.loads(group_path.read_text(encoding="utf-8"))
        validate_result_schema(group, ks=(10, 20), require_protocol=True)
        protocol = group["protocol"]
        return (
            manifest.get("status") == "completed"
            and protocol.get("candidate_catalog") == "full"
            and protocol.get("target_split") == "test"
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("amazon-book", "yelp2018", "musical-instruments", "arts-crafts-and-sewing"),
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-worker", type=int, default=2)
    parser.add_argument(
        "--config-id",
        help="artifact configuration directory (default: paper-v1 or smoke-v1)",
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "log")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dataset-path", default="./dataset_verify/")
    parser.add_argument(
        "--evaluation-protocol-path",
        default="./preprocessed/evaluation_protocol/split_seed_42/",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    failures = 0
    config_id = args.config_id or ("smoke-v1" if args.smoke else "paper-v1")
    for model in args.models:
        for seed in dict.fromkeys(args.seeds):
            directory = (
                args.output_root
                / "p0"
                / "baseline"
                / model
                / args.dataset
                / config_id
                / ("seed" + str(seed))
            )
            completed = is_completed(directory)
            status = "completed" if completed else "pending"
            command = [
                sys.executable,
                str(ROOT / "main.py"),
                "--model",
                model,
                "--seeds",
                str(seed),
                "--dataset",
                args.dataset,
                "--dataset_path",
                args.dataset_path,
                "--evaluation_protocol_path",
                args.evaluation_protocol_path,
                "--gpu_id",
                str(args.gpu_id),
                "--device",
                args.device,
                "--num_worker",
                str(args.num_worker),
                "--top_K",
                "[10, 20]",
                "--selection_K",
                "20",
            ]
            if args.smoke:
                command.extend(["--training_epochs", "1", "--interval", "1"])
            elif model in ("SGL", "SimGCL"):
                # configure/SGL.txt and configure/SimGCL.txt default to 50
                # epochs with no early stopping strong enough to reliably
                # converge on the larger datasets (README.md, "Known
                # epoch-cap pitfall"). Every reported SGL/SimGCL run used
                # 200; apply it here too so this runner matches the
                # manually-documented main.py commands instead of silently
                # producing an undertrained baseline.
                command.extend(["--training_epochs", "200"])
            if args.resume:
                checkpoint = directory / "last_model.pt"
                if not checkpoint.is_file():
                    raise FileNotFoundError(
                        "Resume checkpoint not found: {}".format(checkpoint)
                    )
                command.extend(["--resume_checkpoint", str(checkpoint.resolve())])
            print(
                "{} {} seed={} status={} output={}".format(
                    model, args.dataset, seed, status, directory
                )
            )
            if args.dry_run or (completed and args.skip_completed):
                continue
            if directory.exists() and any(directory.iterdir()) and not args.resume:
                print("Refusing non-empty output: {}".format(directory), file=sys.stderr)
                failures += 1
                if not args.continue_on_error:
                    return 1
                continue
            directory.mkdir(parents=True, exist_ok=True)
            environment = dict(os.environ)
            environment["TAXPROCL_RUN_OUTPUT_DIR"] = str(directory.resolve())
            result = subprocess.run(command, cwd=ROOT, env=environment, check=False)
            if result.returncode:
                failures += 1
                if not args.continue_on_error:
                    return 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
