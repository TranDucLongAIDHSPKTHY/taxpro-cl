"""Record the LightGCN / SGL-ED / XSimGCL / NCL main-comparison runs in
results_manifest.csv and (re)export their per-seed metrics to metrics_seed.csv.

results_manifest.csv originally covered only TaxPro-CL-main, SimGCL-main and the
V0-V3 factorial runs, while metrics_seed.csv also held the four remaining
baselines. This script closes that gap: every run behind the main comparison
(Table S1, Table 7) now has a manifest row with its run directory, resolved
configuration hash, split hashes and best-validation checkpoint hash.

The run families are read from results/week6/mid_tail_degree6_10.json (the
checkpoint-resolved source of the main table; the resolver never selects the
exploratory A7 families, see tests/Recommendation_system/checkpoint_selection.py).
Each family must be the framework default for its dataset: the script refuses
to record a run whose resolved configuration names an A7 family.

Idempotent: existing rows for the four baseline variants are replaced.

Usage:
    python -m tools.analysis.update_manifest_baselines_main
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from tools.analysis.build_results_manifest import config_hash, dataset_split_hashes, sha256_file
from tools.analysis.export_baseline_metrics_seed import (
    BASELINES_TO_ADD,
    CSV_FIELDS,
    MID_TAIL_FILE,
    VARIANT_NAME,
    build_rows,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "results" / "results_manifest.csv"
METRICS = ROOT / "results" / "metrics_seed.csv"
DATASETS = ["amazon-book", "yelp2018", "musical-instruments", "arts-crafts-and-sewing"]
USED_IN = "Table S1 (main comparison); Table 7 (deltas)"


def manifest_row(fields, method, dataset, seed_dir):
    manifest = json.loads((seed_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise SystemExit(f"run not completed: {seed_dir}")
    if "A7-" in seed_dir.as_posix():
        raise SystemExit(f"exploratory A7 run must not enter the main comparison: {seed_dir}")
    config = manifest.get("configuration", {})
    meta = manifest.get("model_metadata") or {}
    split_hashes = meta.get("split_hashes") or {}
    split_hash_source = "model_metadata"
    if not split_hashes:
        split_hashes = dataset_split_hashes(dataset)
        split_hash_source = "dataset split_manifest.json (shared; model's own manifest had none)"
    ckpt = seed_dir / "best_validation_model.pt"
    seed = int(seed_dir.name[len("seed"):])
    variant = VARIANT_NAME[method]
    row = {f: "" for f in fields}
    row.update({
        "run_id": f"{method}|{dataset}|{variant}|seed{seed}",
        "model": method, "dataset": dataset, "variant": variant, "seed": str(seed),
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
        "used_in_tables": USED_IN,
    })
    return row


def main() -> int:
    doc = json.loads(MID_TAIL_FILE.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    fields = list(rows[0].keys())
    variants = set(VARIANT_NAME.values())
    rows = [r for r in rows if r["variant"] not in variants]

    new_rows = []
    for dataset in DATASETS:
        for method in BASELINES_TO_ADD:
            for seed_str, rel in sorted(doc[dataset][method]["checkpoint_dirs"].items(), key=lambda kv: int(kv[0])):
                new_rows.append(manifest_row(fields, method, dataset, ROOT / rel))
    rows.extend(new_rows)
    with open(MANIFEST, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    mrows = list(csv.DictReader(open(METRICS, encoding="utf-8")))
    mrows = [r for r in mrows if r["variant"] not in variants]
    exported, skipped = build_rows(doc)
    if skipped:
        raise SystemExit(f"skipped entries: {skipped}")
    mrows.extend(exported)
    with open(METRICS, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(mrows)
    print(f"results_manifest.csv: {len(new_rows)} baseline rows ({len(rows)} total); "
          f"metrics_seed.csv: {len(exported)} baseline rows ({len(mrows)} total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
