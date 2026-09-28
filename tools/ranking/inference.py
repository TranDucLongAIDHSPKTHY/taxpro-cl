"""Load a trained checkpoint and compute full-catalog rankings for inference.

Mirrors the checkpoint-loading pattern in tools/analysis/export_embeddings.py
(the project's existing reference implementation for post-hoc, read-only use
of a completed run), extended with a full per-item rank (not just top-K).
"""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import torch

from utility.utility_data.data_loader import Data

REPO_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)


def _load_state_dict_tolerant(model, state_dict, run_dir):
    """load_state_dict that tolerates missing *buffers* only.

    Model code evolves (e.g. models/TaxPro-CL.py gained the
    warm_observed_mask buffer after some checkpoints were trained); that
    buffer only matters for prototype re-initialization during training, not
    for get_rating_for_test, so an older checkpoint missing it is safe to
    load as long as every missing key is a non-parameter buffer -- any
    missing *parameter*, or any unexpected key, means the checkpoint does
    not actually match this model architecture and must not be trusted.
    """
    result = model.load_state_dict(state_dict, strict=False)
    missing, unexpected = result.missing_keys, result.unexpected_keys
    if unexpected:
        raise RuntimeError(
            "{}: checkpoint has keys not present in the current model "
            "architecture: {}".format(run_dir, unexpected)
        )
    if missing:
        parameter_names = {name for name, _ in model.named_parameters()}
        bad = [name for name in missing if name in parameter_names]
        if bad:
            raise RuntimeError(
                "{}: checkpoint is missing learned parameters (not just "
                "buffers), refusing to load: {}".format(run_dir, bad)
            )
        logger.warning(
            "%s: checkpoint predates buffer(s) %s; using freshly-initialized "
            "default (safe for inference-only use).",
            run_dir, missing,
        )


def load_model(run_dir, device):
    """Load a completed run's model + dataset exactly as it was trained.

    Returns (model, dataset, config, model_name).
    """
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("{} is not a completed run".format(run_dir))
    config = manifest["configuration"]
    model_name = manifest["model"]

    dataset_root = Path(str(config.get("dataset_path", "./dataset_verify/")))
    if not dataset_root.is_absolute():
        dataset_root = REPO_ROOT / dataset_root
    dataset = Data(
        str(dataset_root / str(config["dataset"])),
        config,
        logger=logging.getLogger(__name__),
    )
    dataset.training_output_dir = run_dir

    trainer = getattr(importlib.import_module("models." + model_name), "Trainer")(
        SimpleNamespace(seed=manifest.get("training_seed")),
        config,
        dataset,
        device,
        logging.getLogger(__name__),
    )
    checkpoint = torch.load(run_dir / "best_validation_model.pt", map_location=device)
    _load_state_dict_tolerant(trainer.model, checkpoint["model_state_dict"], run_dir)
    trainer.model.to(device).eval()
    return trainer.model, dataset, config, model_name


def compute_batch_order_and_rank(model, dataset, device, batch_users, split="test"):
    """Full-catalog scores for one batch of users, train/val positives excluded.

    Returns (order, rank): both [len(batch_users), num_items] long tensors.
    order[row] lists item ids best-to-worst; rank[row, item] is that item's
    1-indexed position (1 = top recommendation).
    """
    batch_tensor = torch.tensor(batch_users, dtype=torch.long, device=device)
    with torch.no_grad():
        rating = model.get_rating_for_test(batch_tensor)
        excluded = dataset.get_user_pos_items_for_evaluation(batch_users, split)
        for row, items in enumerate(excluded):
            if len(items):
                rating[row, list(items)] = float("-inf")
        order = torch.argsort(rating, dim=1, descending=True)
        num_items = rating.shape[1]
        rank = torch.empty_like(order)
        positions = torch.arange(1, num_items + 1, device=device).unsqueeze(0).expand(order.shape[0], -1)
        rank.scatter_(1, order, positions)
    return order, rank
