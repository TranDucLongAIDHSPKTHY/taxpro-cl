"""Bootstrap CI for the leaf-uniform and leave-one-out prototype-construction
variants vs TaxPro-CL-main, Amazon-Book. Each user's diff is averaged
across the 3 seed pairs it appears in BEFORE bootstrapping (the previous
version pseudo-replicated each user up to 3x by appending per-seed diffs
directly into the pooled list inside the seed loop), same corrected
methodology as tools/analysis/rescue_vs_variants_bootstrap.py and
tools/analysis/seed_matched_bootstrap.py.

The reference model was previously A2-V3, which was later discovered to
run under taxonomy_policy=no_merge while leaf_uniform/leave_one_out (like
TaxPro-CL-main) run under merge_t10 -- a taxonomy-policy mismatch that
confounded the comparison. This version compares against the real
TaxPro-CL-main checkpoint instead (same merge_t10 policy, and identical
to A2-V3 in every other hyperparameter), which is a controlled comparison.
"""
from __future__ import annotations
import json, logging, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from config_path.config_path import evaluation_protocol_dir
from utility.utility_train.group_evaluator import load_targets
from tests.Recommendation_system import inference as _inference_mod
from tests.Recommendation_system.inference import compute_batch_order_and_rank


def load_model(run_dir, device):
    """Same as tests.Recommendation_system.inference.load_model, but strips
    prototype_bank.last_snapshot_epoch from the checkpoint before loading:
    this buffer was persistent=True when these variant checkpoints were
    saved and is persistent=False in the current model code
    (a tracking-only buffer, not used by get_rating_for_test/inference) --
    a known, already-diagnosed benign mismatch, not a real architecture
    difference. The stricter _load_state_dict_tolerant used by the shared
    load_model correctly refuses this as an unexpected key by design; this
    local wrapper is the targeted, narrow bypass for that one known key.
    """
    import json, importlib, logging
    from types import SimpleNamespace
    import torch as _torch
    from utility.utility_data.data_loader import Data

    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("{} is not a completed run".format(run_dir))
    config = manifest["configuration"]
    model_name = manifest["model"]
    dataset_root = Path(str(config.get("dataset_path", "./dataset_verify/")))
    if not dataset_root.is_absolute():
        dataset_root = ROOT / dataset_root
    dataset = Data(str(dataset_root / str(config["dataset"])), config, logger=logging.getLogger(__name__))
    dataset.training_output_dir = run_dir
    trainer = getattr(importlib.import_module("models." + model_name), "Trainer")(
        SimpleNamespace(seed=manifest.get("training_seed")), config, dataset, device, logging.getLogger(__name__),
    )
    checkpoint = _torch.load(run_dir / "best_validation_model.pt", map_location=device)
    state_dict = checkpoint["model_state_dict"]
    state_dict.pop("prototype_bank.last_snapshot_epoch", None)
    _inference_mod._load_state_dict_tolerant(trainer.model, state_dict, run_dir)
    trainer.model.to(device).eval()
    return trainer.model, dataset, config, model_name

DATASET = "amazon-book"
SEEDS = ["42", "0", "1"]
GROUPS = ["near_cold", "long_tail", "overall", "warm"]
K = 20
BATCH_SIZE = 256
N_BOOT = 5000

VARIANTS = {
    "leaf_uniform": "log/p0/taxprocl/amazon-book/gvhd-A3-leafuniform",
    "leave_one_out": "log/p0/taxprocl/amazon-book/gvhd-A3-leaveoneout",
}
MAIN_DIR = "log/p0/taxprocl/amazon-book/taxpro-cl-v15-prototype-leaf-lambda0.5-same_leaf_weight0-user_ssl-warmstart20-noblend-temp0.1-DONE-overall-2.30pct-BEST-overall-nearcold-longtail-positive"


def per_user_recall(model, dataset, device, targets_by_group):
    all_users = sorted(set().union(*[set(t.keys()) for t in targets_by_group.values()]))
    per_user = {g: {} for g in GROUPS}
    for start in range(0, len(all_users), BATCH_SIZE):
        batch = all_users[start:start + BATCH_SIZE]
        order, _rank = compute_batch_order_and_rank(model, dataset, device, batch, split="test")
        order_np = order[:, :K].cpu().numpy()
        for row_idx, user in enumerate(batch):
            topk_items = set(order_np[row_idx].tolist())
            for g in GROUPS:
                positives = targets_by_group[g].get(user)
                if not positives:
                    continue
                hits = len(topk_items & set(positives))
                per_user[g][user] = hits / len(positives)
    return per_user


def bootstrap(diffs, n_boot, rng):
    arr = np.asarray(diffs, dtype=np.float64)
    n = len(arr)
    observed = float(arr.mean())
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = arr[idx].mean()
    boot.sort()
    lo = float(boot[int(0.025 * n_boot)])
    hi = float(boot[int(0.975 * n_boot) - 1])
    return {"n_users": n, "mean_diff": observed, "ci95_lo": lo, "ci95_hi": hi,
            "excludes_zero": bool((lo > 0) or (hi < 0))}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)
    logging.basicConfig(level=logging.WARNING)
    protocol_dir = evaluation_protocol_dir(DATASET)
    targets_by_group = load_targets(protocol_dir, "test")

    results = {}
    for variant_name, variant_dir in VARIANTS.items():
        per_user = {g: {} for g in GROUPS}
        for seed in SEEDS:
            print(f"=== {variant_name} seed {seed} ===", flush=True)
            mv, dsv, _c, _n = load_model(ROOT / variant_dir / f"seed{seed}", device)
            pu_v = per_user_recall(mv, dsv, device, targets_by_group)
            del mv
            mm, dsm, _c, _n = load_model(ROOT / MAIN_DIR / f"seed{seed}", device)
            pu_m = per_user_recall(mm, dsm, device, targets_by_group)
            del mm
            for g in GROUPS:
                for u in set(pu_v[g]) & set(pu_m[g]):
                    per_user[g].setdefault(u, []).append(pu_v[g][u] - pu_m[g][u])
        # Average each user's diff across the seed pairs they appear in.
        pooled = {g: [float(np.mean(vals)) for vals in per_user[g].values()] for g in GROUPS}
        rng = np.random.default_rng(42)
        variant_results = {}
        print(f"-- {variant_name} vs TaxPro-CL-main --")
        for g in GROUPS:
            stat = bootstrap(pooled[g], N_BOOT, rng)
            variant_results[g] = stat
            print(f"{g}: n={stat['n_users']} mean_diff={stat['mean_diff']:+.6f} "
                  f"95% CI=[{stat['ci95_lo']:+.6f}, {stat['ci95_hi']:+.6f}] excludes_zero={stat['excludes_zero']}")
        results[variant_name] = variant_results

    out = ROOT / "results" / "a3_variants_bootstrap.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Saved to", out)


if __name__ == "__main__":
    main()
