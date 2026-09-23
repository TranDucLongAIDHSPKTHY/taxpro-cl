"""PyTorch implementation of ID-based graph recommender systems.

Adapted from ID-GRec (https://github.com/BlueGhostYi/ID-GRec, Yi Zhang et al.);
extended for the TaxPro-CL experiments (see README.md, Acknowledgements And
Upstream Code).
"""

__author__ = "Yi Zhang"  # upstream ID-GRec author

import importlib
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone
from time import time

import torch

import Parser
import utility.utility_data.data_loader as data_loader
import utility.utility_function.tools as tools
import utility.utility_train.batch_test as batch_test
from config_path.config_path import (
    model_config_file,
    p0_run_dir,
    training_log_file,
    verified_dataset_dir,
)


MODEL_LIST = {
    "1": "LightGCN",
    "3": "SimGCL",
    "4": "XSimGCL",
    "5": "SGL",
    "6": "NCL",
    "7": "TaxPro-CL",
}
AVAILABLE_MODELS = frozenset(MODEL_LIST.values())


def resolve_device(requested="auto", gpu_id=0, legacy_cuda=None, torch_module=torch):
    """Resolve an explicit CPU/CUDA policy without silently changing requests."""
    requested = str(requested).lower()
    if legacy_cuda is not None and requested == "auto":
        requested = "cuda" if legacy_cuda else "cpu"
    if requested == "cpu":
        return torch_module.device("cpu")
    if requested not in {"auto", "cuda"}:
        raise ValueError("Unknown device policy: {}".format(requested))
    if not torch_module.cuda.is_available():
        if requested == "cuda":
            raise RuntimeError(
                "CUDA was requested but is not available. Use --device cpu or auto."
            )
        return torch_module.device("cpu")
    device_count = int(torch_module.cuda.device_count())
    if int(gpu_id) < 0 or int(gpu_id) >= device_count:
        raise ValueError(
            "GPU id {} is outside the available range 0..{}.".format(
                gpu_id, device_count - 1
            )
        )
    return torch_module.device("cuda:{}".format(int(gpu_id)))


def git_commit():
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def runtime_environment(device, requested_device):
    details = {
        "requested_device": requested_device,
        "resolved_device": str(device),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "git_commit": git_commit(),
    }
    if device.type == "cuda":
        details["gpu_name"] = torch.cuda.get_device_name(device)
        details["gpu_memory_bytes"] = int(
            torch.cuda.get_device_properties(device).total_memory
        )
    else:
        details["cpu"] = platform.processor() or "unknown"
    return details


def default_run_output_dir(model_name, config, seed):
    canonical_config = json.dumps(
        config, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    config_hash = hashlib.sha256(canonical_config).hexdigest()[:12]
    if model_name == "TaxPro-CL":
        identifier = "{}-{}-{}".format(
            model_name.lower(), config.get("taxonomy_policy", "no_merge"), config_hash
        )
    else:
        identifier = "{}-{}".format(model_name.lower(), config_hash)
    return p0_run_dir(model_name, config["dataset"], identifier, seed)


def resolve_run_output_dir(model_name, config, seed):
    explicit = os.environ.get("TAXPROCL_RUN_OUTPUT_DIR")
    return Path(explicit) if explicit else default_run_output_dir(model_name, config, seed)


def configure_training_logging(model_name=None, dataset_name=None, seed=None, output_dir=None):
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(module)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    log_path = None
    if model_name is not None and dataset_name is not None:
        log_path = (
            Path(output_dir) / "training.log"
            if output_dir is not None
            else training_log_file(model_name, dataset_name, seed)
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logfile = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        logfile.setFormatter(formatter)
        root_logger.addHandler(logfile)
    return logging.getLogger("training"), log_path


def select_model(args, logger):
    if args.model != "unknown":
        if args.model not in AVAILABLE_MODELS:
            raise ValueError(
                "Model '{}' is not available in models/: {}".format(
                    args.model, ", ".join(sorted(AVAILABLE_MODELS))
                )
            )
        return args.model
    logger.info("Available models: %s", ", ".join(MODEL_LIST.values()))
    while True:
        selected_num = input("Please input the identifier of the model:")
        if selected_num in MODEL_LIST and selected_num != "0":
            return MODEL_LIST[selected_num]
        logger.error("Input Error. Please select a valid implemented model identifier.")


def run_training_seed(
    model_name, args, config, default_config, device, seed, environment
):
    started_at = time()
    dataset = None
    run_manifest_path = None
    run_manifest = {}
    args.seed = seed
    output_dir = resolve_run_output_dir(model_name, config, seed)
    logger, log_path = configure_training_logging(
        model_name, config["dataset"], seed, output_dir
    )
    logger.info(
        "Start Training: model=%s dataset=%s seed=%d",
        model_name,
        config["dataset"],
        seed,
    )
    logger.info("Training log: %s", log_path)
    try:
        for key in config:
            if config[key] != default_config[key]:
                logger.info(
                    "CLI configuration override: %s=%s (default=%s)",
                    key,
                    config[key],
                    default_config[key],
                )

        if args.seed_flag:
            tools.set_seed(seed)
        logger.info("Random seed: %d (enabled=%s)", seed, args.seed_flag)

        dataset_root = Path(str(config.get("dataset_path", "./dataset_verify/")))
        if not dataset_root.is_absolute():
            dataset_root = Path(__file__).resolve().parent / dataset_root
        dataset_path = dataset_root / str(config["dataset"])
        dataset = data_loader.Data(str(dataset_path), config, logger=logger)
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset.training_output_dir = output_dir

        with (output_dir / "config_resolved.json").open(
            "w", encoding="utf-8", newline="\n"
        ) as stream:
            json.dump(config, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        with (output_dir / "environment.json").open(
            "w", encoding="utf-8", newline="\n"
        ) as stream:
            json.dump(environment, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")

        logger.info("Dataset Loaded: %s", dataset_path)
        logger.info("Train Dataset Size: %d", dataset.num_train)
        logger.info("Validation Dataset Size: %d", dataset.num_validation)
        logger.info("Test Dataset Size: %d", dataset.num_test)
        logger.info(dataset.get_statistics())

        trainer_class = getattr(importlib.import_module("models." + model_name), "Trainer")
        recommender = trainer_class(args, config, dataset, device, logger)
        run_manifest_path = output_dir / "run_manifest.json"
        run_manifest = {
            "status": "running",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": model_name,
            "dataset": config["dataset"],
            "training_seed": int(seed),
            "environment": environment,
            "configuration": dict(config),
            "selection_metric": "validation Recall@20 Overall",
            "test_policy": "once after loading best validation checkpoint",
            "model_metadata": (
                recommender.model.checkpoint_metadata()
                if hasattr(recommender.model, "checkpoint_metadata")
                else None
            ),
        }
        with run_manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(run_manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        logger.info("Model Initialized: %s on %s", model_name, device)
        effective_config = (
            recommender.model.effective_config()
            if hasattr(recommender.model, "effective_config")
            else config
        )
        for key, value in effective_config.items():
            logger.info("Configuration %s=%s", key, value)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        recommender.train()
        logger.info("Loading Best Model")
        batch_test.final_test(dataset, recommender.model, device, config, logger)
        run_manifest["status"] = "completed"
        run_manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        run_manifest["duration_seconds"] = round(time() - started_at, 6)
        run_manifest["completed_epochs"] = int(
            getattr(dataset, "current_epoch", config.get("training_epochs", 0))
        )
        run_manifest["peak_gpu_memory_bytes"] = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        )
        run_manifest["model_metadata"] = (
            recommender.model.checkpoint_metadata()
            if hasattr(recommender.model, "checkpoint_metadata")
            else None
        )
        with run_manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(run_manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        logger.info("Training Finished")
        return 0
    except Exception:
        epoch = getattr(dataset, "current_epoch", "not started") if dataset is not None else "not started"
        logger.exception("Training failed; seed=%d epoch=%s", seed, epoch)
        if run_manifest_path is not None and run_manifest_path.parent.is_dir():
            try:
                run_manifest["status"] = "failed"
                with run_manifest_path.open(
                    "w", encoding="utf-8", newline="\n"
                ) as stream:
                    json.dump(run_manifest, stream, ensure_ascii=False, indent=2)
                    stream.write("\n")
            except Exception:
                logger.exception("Could not update failed run manifest")
        return 1
    finally:
        logger.info("Seed %d Training Time: %.3f seconds", seed, time() - started_at)


def main():
    started_at = time()
    logger, _ = configure_training_logging()
    try:
        preliminary_args = Parser.parse_model_args()
        model_name = select_model(preliminary_args, logger)
        config_path = model_config_file(model_name)
        default_config = tools.read_configuration(str(config_path), model_name)
        args = Parser.parse_args(default_config)
        args.model = model_name
        config = Parser.apply_config_overrides(default_config, args)

        requested_device = args.device
        if args.cuda is not None and args.device == "auto":
            requested_device = "cuda" if args.cuda else "cpu"
        device = resolve_device(args.device, args.gpu_id, args.cuda)
        environment = runtime_environment(device, requested_device)
        logger.info(
            "Device selected: %s (requested=%s)", device, requested_device
        )
        seeds = [args.seed] if args.seed is not None else list(args.seeds)
        seeds = list(dict.fromkeys(seeds))
        logger.info("Training seeds: %s", seeds)

        failed_seeds = []
        for seed_index, seed in enumerate(seeds):
            seed_config = dict(config)
            # A checkpoint belongs to one training seed.  In a multi-seed
            # invocation it resumes only the first requested seed; later
            # seeds are deliberately fresh independent runs.
            if seed_index > 0:
                seed_config["resume_checkpoint"] = "none"
            result = run_training_seed(
                model_name,
                args,
                seed_config,
                default_config,
                device,
                seed,
                environment,
            )
            if result != 0:
                failed_seeds.append(seed)

        logger, _ = configure_training_logging()
        if failed_seeds:
            logger.error("Training failed for seeds: %s", failed_seeds)
            return 1
        logger.info("Training completed for seeds: %s", seeds)
        return 0
    except Exception:
        logger.exception("Training setup failed")
        return 1
    finally:
        logger.info("Total Multi-seed Training Time: %.3f seconds", time() - started_at)

if __name__ == "__main__":
    sys.exit(main())
