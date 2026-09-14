"""Mean cosine similarity between a model's two
contrastive views (item side), split by train-degree group, for TaxPro-CL
and for SimGCL side-by-side. Runs on an already-completed checkpoint -- no
retraining -- calling each model's own aggregate(perturbed=True) twice,
exactly mirroring what its own forward() does during training.

TaxPro-CL already logs a running view_displacement_cosine_mean during
training (a single number per epoch, all perturbed items pooled together);
this script produces the missing breakdown by degree group, and the
SimGCL-side number, which nothing currently logs.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_path.config_path import evaluation_protocol_dir
from utility.utility_data.data_loader import Data

DEGREE_NEAR_COLD_MAX = 5
DEGREE_LONG_TAIL_MAX = 10


def load_item_degrees(dataset_name):
    path = evaluation_protocol_dir(dataset_name) / "item_degrees_train.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(payload["degrees"], dtype=np.int64)


def load_trainer(run_dir, device):
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("Run must be completed: {}".format(run_dir))
    config = manifest["configuration"]
    model_name = manifest["model"]
    dataset_root = Path(str(config.get("dataset_path", "./dataset_verify/")))
    if not dataset_root.is_absolute():
        dataset_root = ROOT / dataset_root
    dataset = Data(str(dataset_root / str(config["dataset"])), config, logger=logging.getLogger(__name__))
    dataset.training_output_dir = run_dir
    trainer = getattr(importlib.import_module("models." + model_name), "Trainer")(
        SimpleNamespace(seed=manifest.get("training_seed")), config, dataset, device, logging.getLogger(__name__)
    )
    checkpoint = torch.load(run_dir / "best_validation_model.pt", map_location=device)
    trainer.model.load_state_dict(checkpoint["model_state_dict"])
    trainer.model.to(device).eval()
    return trainer.model, dataset, config["dataset"]


def group_of(degree):
    if degree <= 0:
        return None
    if degree <= DEGREE_NEAR_COLD_MAX:
        return "near_cold"
    if degree <= DEGREE_LONG_TAIL_MAX:
        return "mid_tail"
    return "warm"


def cosine_by_degree_taxprocl(model, dataset_name, device):
    degrees = load_item_degrees(dataset_name)
    with torch.no_grad():
        _u_a, item_a = model.aggregate(perturbed=True, view_id=0)
        _u_b, item_b = model.aggregate(perturbed=True, view_id=1)
    cos = torch.nn.functional.cosine_similarity(item_a, item_b, dim=1).cpu().numpy()
    return aggregate_by_group(cos, degrees, valid_mask=model.valid_train_mask.cpu().numpy())


def cosine_by_degree_simgcl(model, dataset_name, device):
    degrees = load_item_degrees(dataset_name)
    with torch.no_grad():
        _u_a, item_a = model.aggregate(perturbed=True)
        _u_b, item_b = model.aggregate(perturbed=True)
    cos = torch.nn.functional.cosine_similarity(item_a, item_b, dim=1).cpu().numpy()
    valid_mask = degrees > 0
    return aggregate_by_group(cos, degrees, valid_mask=valid_mask)


def aggregate_by_group(cos, degrees, valid_mask):
    groups = {"near_cold": [], "mid_tail": [], "warm": []}
    for i in range(len(degrees)):
        if not valid_mask[i]:
            continue
        g = group_of(int(degrees[i]))
        if g is None:
            continue
        groups[g].append(float(cos[i]))
    return {
        g: {"mean_cosine": float(np.mean(vals)) if vals else None, "n_items": len(vals)}
        for g, vals in groups.items()
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxprocl-run", type=Path, required=True)
    parser.add_argument("--simgcl-run", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "a4_view_cosine_by_degree.json")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    torch.manual_seed(42)

    tax_model, _tax_ds, _ = load_trainer(args.taxprocl_run, device)
    tax_result = cosine_by_degree_taxprocl(tax_model, args.dataset, device)

    sim_model, _sim_ds, _ = load_trainer(args.simgcl_run, device)
    sim_result = cosine_by_degree_simgcl(sim_model, args.dataset, device)

    payload = {"dataset": args.dataset, "TaxPro-CL": tax_result, "SimGCL": sim_result}
    print(json.dumps(payload, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if args.output.is_file():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
    existing = [r for r in existing if r.get("dataset") != args.dataset]
    existing.append(payload)
    args.output.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Saved to", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
