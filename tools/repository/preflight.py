"""Validate local data, protocol artifacts, dependencies and model imports."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_path.config_path import (
    evaluation_protocol_dir,
    model_config_file,
    taxonomy_variant_dir,
    verified_dataset_dir,
)


MODELS = ("LightGCN", "SimGCL", "XSimGCL", "SGL", "NCL", "TaxPro-CL")
DATASETS = ("amazon-book", "yelp2018", "musical-instruments", "arts-crafts-and-sewing")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args(argv)


def check_paths(dataset, model):
    failures = []
    dataset_dir = verified_dataset_dir(dataset)
    for split in ("train", "validation", "test"):
        if not (dataset_dir / (split + ".txt")).is_file():
            failures.append("missing_dataset:{}:{}".format(dataset, split))
    protocol_dir = evaluation_protocol_dir(dataset)
    for filename in ("manifest.json", "group_masks.npz", "test_targets_by_group.json"):
        if not (protocol_dir / filename).is_file():
            failures.append("missing_protocol:{}:{}".format(dataset, filename))
    if not model_config_file(model).is_file():
        failures.append("missing_config:{}".format(model))
    if model == "TaxPro-CL":
        for policy in ("no_merge", "merge_t5", "merge_t10", "merge_t15"):
            variant = taxonomy_variant_dir(dataset, policy)
            if not (variant / "manifest.json").is_file():
                failures.append("missing_taxonomy:{}:{}".format(dataset, policy))
    return failures


def check_imports(models):
    failures = []
    for model in models:
        try:
            importlib.import_module("models." + model)
        except Exception as error:
            failures.append("import:{}:{}".format(model, error))
    return failures


def main(argv=None):
    args = parse_args(argv)
    failures = []
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        device = torch.cuda.get_device_name(0) if cuda_available else "cpu"
    except Exception as error:
        cuda_available = False
        device = "unavailable:{}".format(error)
    if args.require_cuda and not cuda_available:
        failures.append("cuda_required_but_unavailable")
    for dataset in args.datasets:
        for model in args.models:
            failures.extend(check_paths(dataset, model))
    failures.extend(check_imports(args.models))
    if "NCL" in args.models:
        try:
            import faiss

            faiss_gpus = int(getattr(faiss, "get_num_gpus", lambda: 0)())
        except Exception as error:
            faiss_gpus = 0
            failures.append("NCL_FAISS_unavailable:{}".format(error))
    else:
        faiss_gpus = 0
    payload = {
        "datasets": args.datasets,
        "models": args.models,
        "torch_cuda_available": cuda_available,
        "device": device,
        "faiss_gpu_count": faiss_gpus,
        "failures": sorted(set(failures)),
        "status": "PASS" if not failures else "FAIL",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
