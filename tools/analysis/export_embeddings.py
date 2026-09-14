"""Export deterministic item-embedding bundles from completed run artifacts."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from utility.utility_data.data_loader import Data
from utility.utility_data.taxonomy import load_taxonomy


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    manifest = json.loads((args.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("Embedding export requires a completed run")
    config = manifest["configuration"]
    model_name = manifest["model"]
    device = torch.device(args.device)
    dataset_root = Path(str(config.get("dataset_path", "./dataset_verify/")))
    if not dataset_root.is_absolute():
        dataset_root = Path(__file__).resolve().parents[2] / dataset_root
    dataset = Data(
        str(dataset_root / str(config["dataset"])), config,
        logger=logging.getLogger(__name__),
    )
    dataset.training_output_dir = args.run_dir
    trainer = getattr(importlib.import_module("models." + model_name), "Trainer")(
        SimpleNamespace(seed=manifest.get("training_seed")), config, dataset, device, logging.getLogger(__name__)
    )
    checkpoint = torch.load(args.run_dir / "best_validation_model.pt", map_location=device)
    trainer.model.load_state_dict(checkpoint["model_state_dict"])
    trainer.model.to(device).eval()
    with torch.no_grad():
        _, items = trainer.model.aggregate()
    policy = config.get("taxonomy_policy", "no_merge")
    granularity = config.get("taxonomy_granularity", "leaf")
    if granularity == "parent" and config["dataset"] != "amazon-book":
        granularity = "leaf"
    taxonomy = load_taxonomy(config["dataset"], policy, granularity)
    payload = {
        "embeddings": items.detach().cpu().numpy().astype(np.float32),
        "labels": taxonomy.item_to_leaf_id.astype(np.int64),
    }
    bank = getattr(trainer.model, "prototype_bank", None)
    if bank is not None and bool(bank.initialized.item()):
        payload["prototypes"] = bank.prototypes.detach().cpu().numpy().astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print("Wrote {}".format(args.output)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
