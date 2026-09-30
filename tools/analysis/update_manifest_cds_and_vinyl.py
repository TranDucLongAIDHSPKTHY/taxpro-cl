"""Record the CDs-and-Vinyl runs (Online Resource 1, Section S30) in
results_manifest.csv and metrics_seed.csv: the 6-method x 3-seed comparison
(Tables S30b/S30c) and the TaxPro-CL four-policy taxonomy sweep (Table S30d;
the no_merge arm is the TaxPro-CL run of the comparison).

These runs are not part of the four-dataset main comparison. Two metric rows
per run and group are written, because metrics_seed.csv has no split column:
"<variant>-validation" from final_test_metrics.json["best_validation_metrics"]
(the groupwise validation snapshot at the epoch best_validation_model.pt was
checkpointed on -- see tools/ranking/checkpoint_selection.py)
and "<variant>-test" from final_test_group_metrics.json. The test split of
this dataset is not held out: test results informed the decision to drop it
(main paper Section 4.1).

Idempotent: existing rows tagged with the "-cdvinyl" variant suffix are
replaced.

Usage: python -m tools.analysis.update_manifest_cds_and_vinyl
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from tools.analysis.build_results_manifest import config_hash, dataset_split_hashes, sha256_file

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "results" / "results_manifest.csv"
METRICS = ROOT / "results" / "metrics_seed.csv"
DATASET = "cds-and-vinyl"
SEEDS = (0, 1, 42)
GROUPS = ("overall", "near_cold", "long_tail", "warm")
USED_IN = "Online Resource 1, Tables S30b/S30c"
USED_IN_SWEEP = "Online Resource 1, Table S30d"
SWEEP_POLICIES = ("merge_t5", "merge_t10", "merge_t15")

RUN_DIRS = {
    "LightGCN": "log/p0/baseline/LightGCN/cds-and-vinyl/lightgcn-3b0f9c415824",
    "SGL-ED": "log/p0/baseline/SGL/cds-and-vinyl/sgl-8c2fb16d3f73",
    "SimGCL": "log/p0/baseline/SimGCL/cds-and-vinyl/simgcl-9981deb5004c",
    "XSimGCL": "log/p0/baseline/XSimGCL/cds-and-vinyl/xsimgcl-4063661ba733",
    "NCL": "log/p0/baseline/NCL/cds-and-vinyl/ncl-226da98203d3",
    "TaxPro-CL": "log/p0/taxprocl/cds-and-vinyl/taxpro-cl-no_merge-213d252b5ac8",
}


def variant_name(method):
    return "{}-cdvinyl".format(method)


def metrics_variant_name(variant, split):
    # metrics_seed.csv has no split column and holds test-split metrics for every
    # other variant; mark the split of these rows explicitly in the key.
    return "{}-{}".format(variant, split)


def sweep_runs():
    """TaxPro-CL taxonomy-policy sweep runs with all three seeds completed."""
    runs = {}
    for policy in SWEEP_POLICIES:
        found = [d for d in (ROOT / "log/p0/taxprocl" / DATASET).glob("taxpro-cl-{}-*".format(policy))
                 if all((d / "seed{}".format(s) / "final_test_metrics.json").is_file() for s in SEEDS)]
        if len(found) == 1:
            runs[policy] = str(found[0].relative_to(ROOT)).replace("\\", "/")
    return runs


def manifest_row(fields, method, seed_dir, variant=None, used_in=USED_IN):
    manifest = json.loads((seed_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise SystemExit("run not completed: {}".format(seed_dir))
    config = manifest.get("configuration", {})
    meta = manifest.get("model_metadata") or {}
    split_hashes = meta.get("split_hashes") or {}
    split_hash_source = "model_metadata"
    if not split_hashes:
        split_hashes = dataset_split_hashes(DATASET)
        split_hash_source = "dataset split_manifest.json (shared; model's own manifest had none)"
    ckpt = seed_dir / "best_validation_model.pt"
    seed = int(seed_dir.name[len("seed"):])
    variant = variant or variant_name(method)
    row = {f: "" for f in fields}
    row.update({
        "run_id": "{}|{}|{}|seed{}".format(method, DATASET, variant, seed),
        "model": method, "dataset": DATASET, "variant": variant, "seed": str(seed),
        "status": manifest.get("status", ""),
        "run_dir": str(seed_dir.relative_to(ROOT)).replace("/", "\\"),
        "git_commit": manifest.get("environment", {}).get("git_commit", ""),
        "config_sha256": config_hash(config),
        "train_split_sha256": split_hashes.get("train", ""),
        "validation_split_sha256": split_hashes.get("validation", ""),
        "test_split_sha256": split_hashes.get("test", ""),
        "split_hash_source": split_hash_source,
        "taxonomy_hash": meta.get("taxonomy_hash", ""),
        "evaluation_protocol_hash": meta.get("evaluation_protocol_hash", ""),
        "checkpoint_sha256_best_validation_model": sha256_file(ckpt) if ckpt.is_file() else "",
        "v3_is_same_checkpoint_as_taxprocl_main": "",
        "selection_metric": manifest.get("selection_metric", ""),
        "test_policy": manifest.get("test_policy", ""),
        "completed_epochs": str(manifest.get("completed_epochs", "")),
        "duration_seconds": str(manifest.get("duration_seconds", "")),
        "temperature": str(config.get("temperature", "")),
        "temperature_user": str(config.get("temperature_user", "")),
        "epsilon_max": str(config.get("epsilon_max", "")),
        "warm_start_epochs": str(config.get("warm_start_epochs", "")),
        "augmentation_direction": str(config.get("augmentation_direction", "")),
        "use_adaptive_epsilon": str(config.get("use_adaptive_epsilon", "")),
        "taxonomy_policy": str(config.get("taxonomy_policy", "")),
        "used_in_tables": used_in,
    })
    return row


def metric_rows(method, seed_dir, base_variant, split):
    if split == "validation":
        payload = json.loads((seed_dir / "final_test_metrics.json").read_text(encoding="utf-8"))
        groupwise = payload["best_validation_metrics"]["groupwise"]
    else:
        groupwise = json.loads((seed_dir / "final_test_group_metrics.json").read_text(encoding="utf-8"))
    seed = int(seed_dir.name[len("seed"):])
    variant = metrics_variant_name(base_variant, split)
    run_id = "{}|{}|{}|seed{}".format(method, DATASET, variant, seed)
    rows = []
    for group in GROUPS:
        g = groupwise[group]
        rows.append({
            "run_id": run_id, "model": method, "dataset": DATASET, "variant": variant,
            "seed": str(seed), "group": group,
            "eligible_users": str(g.get("eligible_users", "")),
            "relevant_items": str(g.get("relevant_items", "")),
            "recall_at_10": str(g["recall"]["10"]),
            "recall_at_20": str(g["recall"]["20"]),
            "ndcg_at_10": str(g["ndcg"]["10"]),
            "ndcg_at_20": str(g["ndcg"]["20"]),
        })
    return rows


def main():
    manifest_rows_existing = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    fields = list(manifest_rows_existing[0].keys())
    manifest_rows_existing = [r for r in manifest_rows_existing if "-cdvinyl" not in r["variant"]]

    metric_rows_existing = list(csv.DictReader(open(METRICS, encoding="utf-8")))
    metric_fields = list(metric_rows_existing[0].keys())
    metric_rows_existing = [r for r in metric_rows_existing if "-cdvinyl" not in r["variant"]]

    new_manifest_rows = []
    new_metric_rows = []
    jobs = [(method, run_dir, variant_name(method), USED_IN) for method, run_dir in RUN_DIRS.items()]
    jobs += [("TaxPro-CL", run_dir, "TaxPro-CL-cdvinyl-{}".format(policy), USED_IN_SWEEP)
             for policy, run_dir in sweep_runs().items()]
    for method, run_dir, variant, used_in in jobs:
        for seed in SEEDS:
            seed_dir = ROOT / run_dir / "seed{}".format(seed)
            new_manifest_rows.append(manifest_row(fields, method, seed_dir, variant, used_in))
            for split in ("validation", "test"):
                new_metric_rows.extend(metric_rows(method, seed_dir, variant, split))
    print("CDs-and-Vinyl runs recorded:", len(jobs), "(sweep policies:", sorted(sweep_runs()), ")")

    manifest_rows_existing.extend(new_manifest_rows)
    with open(MANIFEST, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(manifest_rows_existing)

    metric_rows_existing.extend(new_metric_rows)
    with open(METRICS, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=metric_fields, lineterminator="\n")
        w.writeheader()
        w.writerows(metric_rows_existing)

    print("results_manifest.csv: +{} cdvinyl rows ({} total)".format(
        len(new_manifest_rows), len(manifest_rows_existing)))
    print("metrics_seed.csv: +{} cdvinyl rows ({} total)".format(
        len(new_metric_rows), len(metric_rows_existing)))


if __name__ == "__main__":
    main()
