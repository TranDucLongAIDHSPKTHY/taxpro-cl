"""B2 : direct top-20 set/order overlap for the
prototype-construction variants (leaf_uniform, leave_one_out, rescue) against
TaxPro-CL-main on Amazon-Book, seed-matched (0, 1, 42).

Section 5.5 / Online Resource 1 Table S17 report per-user Recall@20 equality
(or near-equality) between each variant and TaxPro-CL-main, but Recall@20
equality does not by itself establish that the two checkpoints recommend the
same top-20 *set* per user, let alone the same order -- two models can share
a Recall@20 count while disagreeing on which items fill the list. This
script measures that directly, mirroring the methodology already used for
the A5 granularity comparison (Section 5.5, Online Resource 1 Section S7:
"98.74% of top-20 items on average, 76% of users identical top-20 sets").

Inference only; no retraining. Checkpoints:
  log/p0/taxprocl/amazon-book/proto-leafuniform/seed{0,1,42}
  log/p0/taxprocl/amazon-book/proto-leaveoneout/seed{0,1,42}
  log/p0/taxprocl/amazon-book/proto-rescue/seed{0,1,42}
  log/p0/taxprocl/amazon-book/<main-config-dir>/seed{0,1,42}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_path.config_path import evaluation_protocol_dir
from utility.utility_train.group_evaluator import load_targets
from tests.Recommendation_system import inference


def load_model_dropping_stale_buffers(run_dir, device):
    """Like inference.load_model, but drops checkpoint keys absent from the
    current model class (e.g. prototype_bank.last_snapshot_epoch, a
    bookkeeping buffer removed from a later models/TaxPro-CL.py revision)
    instead of refusing to load. Only ever drops non-parameter buffers --
    any missing/unexpected *parameter* still raises, exactly as
    inference.load_model does, so this cannot silently load onto the wrong
    architecture."""
    try:
        return inference.load_model(run_dir, device)
    except RuntimeError as exc:
        if "checkpoint has keys not present" not in str(exc):
            raise
    import json as _json
    manifest = _json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    config = manifest["configuration"]
    model_name = manifest["model"]
    from types import SimpleNamespace
    import importlib
    import logging
    from utility.utility_data.data_loader import Data

    dataset_root = Path(str(config.get("dataset_path", "./dataset_verify/")))
    if not dataset_root.is_absolute():
        dataset_root = ROOT / dataset_root
    dataset = Data(str(dataset_root / str(config["dataset"])), config, logger=logging.getLogger(__name__))
    dataset.training_output_dir = run_dir
    trainer = getattr(importlib.import_module("models." + model_name), "Trainer")(
        SimpleNamespace(seed=manifest.get("training_seed")), config, dataset, device, logging.getLogger(__name__)
    )
    checkpoint = torch.load(run_dir / "best_validation_model.pt", map_location=device)
    state_dict = checkpoint["model_state_dict"]
    model_keys = set(trainer.model.state_dict().keys())
    dropped = [k for k in state_dict if k not in model_keys]
    filtered = {k: v for k, v in state_dict.items() if k in model_keys}
    result = trainer.model.load_state_dict(filtered, strict=False)
    parameter_names = {name for name, _ in trainer.model.named_parameters()}
    bad = [name for name in result.missing_keys if name in parameter_names]
    if bad:
        raise RuntimeError("{}: checkpoint is missing learned parameters: {}".format(run_dir, bad))
    if dropped:
        print("NOTE {}: dropped stale checkpoint-only buffer(s) {}".format(run_dir, dropped), file=sys.stderr)
    trainer.model.to(device).eval()
    return trainer.model, dataset, config, model_name

DATASET = "amazon-book"
SEEDS = (0, 1, 42)
K = 20

MAIN_DIR_NAME = (
    "taxpro-cl-v15-prototype-leaf-lambda0.5-same_leaf_weight0-user_ssl-"
    "warmstart20-noblend-temp0.1-DONE-overall-2.30pct-BEST-overall-nearcold-longtail-positive"
)
VARIANTS = {
    "leaf_uniform": "proto-leafuniform",
    "leave_one_out": "proto-leaveoneout",
    "rescue": "proto-rescue",
}

BASE = ROOT / "log" / "p0" / "taxprocl" / DATASET


def score_pair(variant_dir, main_dir, device, batch_size=256):
    variant_model, variant_dataset, _c1, _n1 = load_model_dropping_stale_buffers(variant_dir, device)
    main_model, main_dataset, _c2, _n2 = load_model_dropping_stale_buffers(main_dir, device)

    protocol_dir = evaluation_protocol_dir(DATASET)
    overall_targets = load_targets(protocol_dir, "test")["overall"]
    users = sorted(overall_targets)

    identical_set_count = 0
    identical_order_count = 0
    overlap_fractions = []

    for start in range(0, len(users), batch_size):
        batch_users = users[start : start + batch_size]
        order_v, _ = inference.compute_batch_order_and_rank(variant_model, variant_dataset, device, batch_users, split="test")
        order_m, _ = inference.compute_batch_order_and_rank(main_model, main_dataset, device, batch_users, split="test")
        order_v_np = order_v[:, :K].cpu().numpy()
        order_m_np = order_m[:, :K].cpu().numpy()

        for row in range(len(batch_users)):
            top_v = order_v_np[row].tolist()
            top_m = order_m_np[row].tolist()
            set_v, set_m = set(top_v), set(top_m)
            overlap = len(set_v & set_m) / K
            overlap_fractions.append(overlap)
            if set_v == set_m:
                identical_set_count += 1
                if top_v == top_m:
                    identical_order_count += 1

    n_users = len(users)
    return {
        "n_users": n_users,
        "mean_topk_overlap": sum(overlap_fractions) / n_users,
        "frac_identical_set": identical_set_count / n_users,
        "frac_identical_order_given_identical_set": (
            identical_order_count / identical_set_count if identical_set_count else None
        ),
        "frac_identical_order_overall": identical_order_count / n_users,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for variant_name, variant_subdir in VARIANTS.items():
        per_seed = []
        for seed in SEEDS:
            variant_dir = BASE / variant_subdir / "seed{}".format(seed)
            main_dir = BASE / MAIN_DIR_NAME / "seed{}".format(seed)
            if not (variant_dir / "run_manifest.json").is_file() or not (main_dir / "run_manifest.json").is_file():
                print("SKIP {} seed={}: missing checkpoint".format(variant_name, seed), file=sys.stderr)
                continue
            print("Scoring {} vs main, seed={}".format(variant_name, seed))
            try:
                row = score_pair(variant_dir, main_dir, device)
            except Exception as exc:
                print("ERROR {} seed={}: {}".format(variant_name, seed, exc), file=sys.stderr)
                continue
            row["seed"] = seed
            per_seed.append(row)
            print(json.dumps(row, indent=2))
        results[variant_name] = per_seed

    out_path = ROOT / "results" / "b2_prototype_variant_overlap.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Saved to {}".format(out_path))


if __name__ == "__main__":
    main()
