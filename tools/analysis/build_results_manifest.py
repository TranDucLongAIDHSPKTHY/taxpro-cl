# -*- coding: utf-8 -*-
"""GVHD A1: build results_manifest.csv and metrics_seed.csv from actual run
manifests and metrics files -- no invented values, everything read directly
from each run's own run_manifest.json / final_test_group_metrics.json.

Scope: the runs directly relevant to A1's concern (V3-factorial vs. main-
table checkpoint provenance) -- TaxPro-CL main config, SimGCL main config,
and the V0-V3 factorial variants, all four datasets, 3 seeds each.
"""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDS = [42, 0, 1]

RUNS = [
    # (variant_label, dataset, model, run_dir)
    ("TaxPro-CL-main", "amazon-book", "TaxPro-CL",
     "log/p0/taxprocl/amazon-book/taxpro-cl-v15-prototype-leaf-lambda0.5-same_leaf_weight0-user_ssl-warmstart20-noblend-temp0.1-DONE-overall-2.30pct-BEST-overall-nearcold-longtail-positive"),
    ("TaxPro-CL-main", "yelp2018", "TaxPro-CL",
     "log/p0/taxprocl/yelp2018/taxpro-cl-FINAL-no_merge-temp0.125-tempuser0.15-gammacold1.5"),
    ("TaxPro-CL-main", "musical-instruments", "TaxPro-CL",
     "log/p0/taxprocl/musical-instruments/taxpro-cl-FINAL-no_merge-temp0.15-gammacold5.0"),
    ("TaxPro-CL-main", "arts-crafts-and-sewing", "TaxPro-CL",
     "log/p0/taxprocl/arts-crafts-and-sewing/taxpro-cl-FINAL-no_merge-temp0.125-gammacold1.5-sameleaf0"),

    ("SimGCL-main", "amazon-book", "SimGCL", "log/p0/baseline/SimGCL/amazon-book/simgcl-69bb0902f42f"),
    ("SimGCL-main", "yelp2018", "SimGCL", "log/p0/baseline/SimGCL/yelp2018/simgcl-20c63ca8d16c"),
    ("SimGCL-main", "musical-instruments", "SimGCL", "log/p0/baseline/SimGCL/musical-instruments/simgcl-f263d2bff497"),
    ("SimGCL-main", "arts-crafts-and-sewing", "SimGCL", "log/p0/baseline/SimGCL/arts-crafts-and-sewing/simgcl-c8b32441f415"),

    ("V0", "amazon-book", "TaxPro-CL", "log/p0/taxprocl/amazon-book/A2-V0"),
    ("V1", "amazon-book", "TaxPro-CL", "log/p0/taxprocl/amazon-book/A2-V1"),
    ("V2", "amazon-book", "TaxPro-CL", "log/p0/taxprocl/amazon-book/A2-V2"),
    ("V3", "amazon-book", "TaxPro-CL", "log/p0/taxprocl/amazon-book/A2-V3"),

    ("V0", "yelp2018", "TaxPro-CL", "log/p0/taxprocl/yelp2018/A2-V0"),
    ("V1", "yelp2018", "TaxPro-CL", "log/p0/taxprocl/yelp2018/A2-V1"),
    ("V2", "yelp2018", "TaxPro-CL", "log/p0/taxprocl/yelp2018/A2-V2"),
    ("V3", "yelp2018", "TaxPro-CL", "log/p0/taxprocl/yelp2018/A2-V3"),

    ("V0", "musical-instruments", "TaxPro-CL", "log/p0/taxprocl/musical-instruments/gvhd-taxctrl-V0"),
    ("V1", "musical-instruments", "TaxPro-CL", "log/p0/taxprocl/musical-instruments/gvhd-taxctrl-V1"),
    ("V2", "musical-instruments", "TaxPro-CL", "log/p0/taxprocl/musical-instruments/gvhd-taxctrl-V2"),
    ("V3", "musical-instruments", "TaxPro-CL",
     "log/p0/taxprocl/musical-instruments/taxpro-cl-FINAL-no_merge-temp0.15-gammacold5.0"),

    ("V0", "arts-crafts-and-sewing", "TaxPro-CL", "log/p0/taxprocl/arts-crafts-and-sewing/gvhd-taxctrl-V0"),
    ("V1", "arts-crafts-and-sewing", "TaxPro-CL", "log/p0/taxprocl/arts-crafts-and-sewing/gvhd-taxctrl-V1"),
    ("V2", "arts-crafts-and-sewing", "TaxPro-CL", "log/p0/taxprocl/arts-crafts-and-sewing/gvhd-taxctrl-V2"),
    ("V3", "arts-crafts-and-sewing", "TaxPro-CL",
     "log/p0/taxprocl/arts-crafts-and-sewing/taxpro-cl-FINAL-no_merge-temp0.125-gammacold1.5-sameleaf0"),
]

# V3 variants that are literally the same checkpoint dir as TaxPro-CL-main
# (Musical-Instruments, Arts-Crafts-and-Sewing) vs. a separately-trained
# checkpoint (Amazon-Book, Yelp2018) -- this IS the A1 finding, made
# mechanically checkable via identical checkpoint_sha256 in the CSV.


def sha256_file(path: Path, chunk_size=1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def dataset_split_hashes(dataset: str) -> dict:
    path = ROOT / "dataset_verify" / dataset / "split_manifest.json"
    if not path.is_file():
        return {}
    d = json.loads(path.read_text(encoding="utf-8"))
    return d.get("output_hashes") or {}


def main():
    manifest_rows = []
    metric_rows = []
    checkpoint_hash_cache = {}
    dataset_hash_cache = {}

    for variant, dataset, model, run_dir_rel in RUNS:
        run_dir = ROOT / run_dir_rel
        for seed in SEEDS:
            seed_dir = run_dir / f"seed{seed}"
            manifest_path = seed_dir / "run_manifest.json"
            metrics_path = seed_dir / "final_test_group_metrics.json"
            ckpt_path = seed_dir / "best_validation_model.pt"
            if not manifest_path.is_file():
                print("MISSING manifest:", manifest_path)
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            config = manifest.get("configuration", {})
            meta = manifest.get("model_metadata") or {}
            split_hashes = meta.get("split_hashes") or {}
            split_hash_source = "model_metadata"
            if not split_hashes:
                if dataset not in dataset_hash_cache:
                    dataset_hash_cache[dataset] = dataset_split_hashes(dataset)
                split_hashes = dataset_hash_cache[dataset]
                split_hash_source = "dataset split_manifest.json (shared; model's own manifest had none)"

            run_id = f"{model}|{dataset}|{variant}|seed{seed}"

            ckpt_key = str(ckpt_path)
            if ckpt_key not in checkpoint_hash_cache:
                if ckpt_path.is_file():
                    checkpoint_hash_cache[ckpt_key] = sha256_file(ckpt_path)
                else:
                    checkpoint_hash_cache[ckpt_key] = ""
            ckpt_hash = checkpoint_hash_cache[ckpt_key]

            manifest_rows.append({
                "run_id": run_id,
                "model": model,
                "dataset": dataset,
                "variant": variant,
                "seed": seed,
                "status": manifest.get("status", ""),
                "run_dir": str(seed_dir.relative_to(ROOT)),
                "git_commit": manifest.get("environment", {}).get("git_commit", ""),
                "config_sha256": config_hash(config),
                "train_split_sha256": split_hashes.get("train", ""),
                "validation_split_sha256": split_hashes.get("validation", ""),
                "test_split_sha256": split_hashes.get("test", ""),
                "split_hash_source": split_hash_source,
                "taxonomy_hash": meta.get("taxonomy_hash", ""),
                "evaluation_protocol_hash": meta.get("evaluation_protocol_hash", ""),
                "checkpoint_sha256_best_validation_model": ckpt_hash,
                "selection_metric": manifest.get("selection_metric", ""),
                "test_policy": manifest.get("test_policy", ""),
                "completed_epochs": manifest.get("completed_epochs", ""),
                "duration_seconds": manifest.get("duration_seconds", ""),
                "temperature": config.get("temperature", ""),
                "temperature_user": config.get("temperature_user", ""),
                "epsilon_max": config.get("epsilon_max", ""),
                "warm_start_epochs": config.get("warm_start_epochs", ""),
                "augmentation_direction": config.get("augmentation_direction", ""),
                "use_adaptive_epsilon": config.get("use_adaptive_epsilon", ""),
                "taxonomy_policy": config.get("taxonomy_policy", ""),
                "used_in_tables": "",  # filled in pass 2 below
            })

            if metrics_path.is_file():
                gm = json.loads(metrics_path.read_text(encoding="utf-8"))
                for group in ["near_cold", "long_tail", "overall", "warm"]:
                    g = gm.get(group)
                    if g is None:
                        continue
                    metric_rows.append({
                        "run_id": run_id,
                        "model": model,
                        "dataset": dataset,
                        "variant": variant,
                        "seed": seed,
                        "group": group,
                        "eligible_users": g.get("eligible_users", ""),
                        "relevant_items": g.get("relevant_items", ""),
                        "recall_at_10": g.get("recall", {}).get("10", ""),
                        "recall_at_20": g.get("recall", {}).get("20", ""),
                        "ndcg_at_10": g.get("ndcg", {}).get("10", ""),
                        "ndcg_at_20": g.get("ndcg", {}).get("20", ""),
                    })
            else:
                print("MISSING metrics:", metrics_path)

    # Pass 2: annotate which main-paper tables each variant feeds, and flag
    # the A1 finding mechanically (same checkpoint hash as TaxPro-CL-main?).
    main_ckpt_hash_by_dataset = {}
    for row in manifest_rows:
        if row["variant"] == "TaxPro-CL-main":
            main_ckpt_hash_by_dataset.setdefault(row["dataset"], set()).add(
                row["checkpoint_sha256_best_validation_model"]
            )
    for row in manifest_rows:
        tables = []
        if row["variant"] == "TaxPro-CL-main":
            tables = ["Table S1", "Table 7 (deltas)", "Table 11 (bootstrap)"]
        elif row["variant"] == "SimGCL-main":
            tables = ["Table S1", "Table 7 (deltas)", "Table 11 (bootstrap)"]
        elif row["variant"] in ("V0", "V1", "V2", "V3"):
            tables = ["Table 6 (factorial %)", "Table S16", "Table S20", "Table S22"]
        row["used_in_tables"] = "; ".join(tables)
        if row["variant"] == "V3":
            is_same = row["checkpoint_sha256_best_validation_model"] in main_ckpt_hash_by_dataset.get(row["dataset"], set())
            row["v3_is_same_checkpoint_as_taxprocl_main"] = "YES" if is_same else "NO (separately trained)"
        else:
            row["v3_is_same_checkpoint_as_taxprocl_main"] = ""

    manifest_fields = [
        "run_id", "model", "dataset", "variant", "seed", "status", "run_dir",
        "git_commit", "config_sha256", "train_split_sha256", "validation_split_sha256",
        "test_split_sha256", "split_hash_source", "taxonomy_hash", "evaluation_protocol_hash",
        "checkpoint_sha256_best_validation_model", "v3_is_same_checkpoint_as_taxprocl_main",
        "selection_metric", "test_policy", "completed_epochs", "duration_seconds",
        "temperature", "temperature_user", "epsilon_max", "warm_start_epochs",
        "augmentation_direction", "use_adaptive_epsilon", "taxonomy_policy", "used_in_tables",
    ]
    metric_fields = [
        "run_id", "model", "dataset", "variant", "seed", "group",
        "eligible_users", "relevant_items", "recall_at_10", "recall_at_20", "ndcg_at_10", "ndcg_at_20",
    ]

    out_manifest = ROOT / "results" / "results_manifest.csv"
    out_metrics = ROOT / "results" / "metrics_seed.csv"
    with out_manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=manifest_fields)
        w.writeheader()
        w.writerows(manifest_rows)
    with out_metrics.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=metric_fields)
        w.writeheader()
        w.writerows(metric_rows)

    print(f"Wrote {len(manifest_rows)} rows to {out_manifest}")
    print(f"Wrote {len(metric_rows)} rows to {out_metrics}")

    # Print the A1 headline check directly
    print("\n--- A1 check: is V3's checkpoint the same as TaxPro-CL-main's, per dataset? ---")
    seen = set()
    for row in manifest_rows:
        if row["variant"] == "V3" and row["dataset"] not in seen:
            seen.add(row["dataset"])
            print(f"  {row['dataset']}: {row['v3_is_same_checkpoint_as_taxprocl_main']} (seed {row['seed']})")


if __name__ == "__main__":
    main()
