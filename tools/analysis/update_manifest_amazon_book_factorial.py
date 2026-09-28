"""Add results_manifest.csv / metrics_seed.csv rows for the Amazon-Book
factorial checkpoints run under taxonomy_policy=merge_t10 (the main
configuration's policy), the independent V3 run, and the warm-start-removal
pair, and mark the earlier no_merge Amazon-Book V0-V3 rows as not reported.

Idempotent: rows already present (same dataset/variant/seed, or same
run_id/group in metrics_seed.csv) are not duplicated.

Usage:
    python -m tools.analysis.update_manifest_a2_mergedt10
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "results" / "results_manifest.csv"
METRICS = ROOT / "results" / "metrics_seed.csv"
BASE = ROOT / "log" / "p0" / "taxprocl" / "amazon-book"
SEEDS = (42, 0, 1)

# directory -> variant label in the manifest
RUNS = {
    "A2-V0-mergedt10": "V0-mergedt10",
    "A2-V1-mergedt10": "V1-mergedt10",
    "A2-V2-mergedt10": "V2-mergedt10",
    "A2-V3-warm20": "V3-warm20",
    "A2-V0-nowarm": "V0-nowarm",
    "A2-V3-nowarm": "V3-nowarm",
}
USED_IN = {
    "V0-mergedt10": "Table 8; Table 9; Figure 2; Online Resource 1 Tables S13/S13b/S17/S17b/S19/S20/S27 (Amazon-Book, merge_t10)",
    "V1-mergedt10": "Table 8; Table 9; Figure 2; Online Resource 1 Tables S13/S13b/S17/S17b/S19/S27 (Amazon-Book, merge_t10)",
    "V2-mergedt10": "Table 8; Table 9; Figure 2; Online Resource 1 Tables S13/S13b/S17/S17b/S19/S27 (Amazon-Book, merge_t10)",
    "V3-warm20": "Table 8; Table 9; Figure 2; Online Resource 1 Tables S13/S13b/S17/S17b/S19/S20/S27 (Amazon-Book V3: independent run of the main configuration; see Section S23)",
    "V0-nowarm": "Online Resource 1 Table S20 (warm-start removal, random direction)",
    "V3-nowarm": "Online Resource 1 Table S20 (warm-start removal, taxonomy direction)",
}
NOTE = {
    "V3-warm20": "NO (independent run of the main configuration on other hardware; test metrics within 0.26% of the main checkpoint)",
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    fields = list(rows[0].keys())
    ref = next(r for r in rows if r["dataset"] == "amazon-book" and r["variant"] == "TaxPro-CL-main" and r["seed"] == "42")
    have = {(r["dataset"], r["variant"], r["seed"]) for r in rows}
    appended = 0
    new_rows = []
    for dirname, label in RUNS.items():
        for seed in SEEDS:
            run_dir = BASE / dirname / f"seed{seed}"
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            if manifest.get("status") != "completed":
                raise SystemExit(f"run not completed: {run_dir}")
            if ("amazon-book", label, str(seed)) in have:
                continue
            config = manifest["configuration"]
            ckpt = run_dir / "best_validation_model.pt"
            row = {f: "" for f in fields}
            row.update({
                "run_id": f"TaxPro-CL|amazon-book|{label}|seed{seed}",
                "model": "TaxPro-CL", "dataset": "amazon-book", "variant": label, "seed": str(seed),
                "status": manifest.get("status", ""),
                "run_dir": str(run_dir.relative_to(ROOT)).replace("/", "\\"),
                "git_commit": manifest.get("environment", {}).get("git_commit", manifest.get("git_commit", "")),
                "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                "train_split_sha256": ref["train_split_sha256"],
                "validation_split_sha256": ref["validation_split_sha256"],
                "test_split_sha256": ref["test_split_sha256"],
                "split_hash_source": ref["split_hash_source"],
                "taxonomy_hash": ref["taxonomy_hash"],
                "evaluation_protocol_hash": ref["evaluation_protocol_hash"],
                "checkpoint_sha256_best_validation_model": sha256_of(ckpt) if ckpt.exists() else "",
                "v3_is_same_checkpoint_as_taxprocl_main": NOTE.get(label, "N/A"),
                "selection_metric": "validation Recall@20 Overall",
                "test_policy": "once after loading best validation checkpoint",
                "completed_epochs": str(manifest.get("completed_epochs", "")),
                "duration_seconds": str(manifest.get("duration_seconds", "")),
                "temperature": str(config.get("temperature", "")),
                "temperature_user": str(config.get("temperature_user", "")),
                "epsilon_max": str(config.get("epsilon_max", "")),
                "warm_start_epochs": str(config.get("warm_start_epochs", "")),
                "augmentation_direction": str(config.get("augmentation_direction", "")),
                "use_adaptive_epsilon": str(config.get("use_adaptive_epsilon", "")),
                "taxonomy_policy": str(config.get("taxonomy_policy", "")),
                "used_in_tables": USED_IN[label],
            })
            new_rows.append(row)
            appended += 1
    rows.extend(new_rows)

    flagged = 0
    for r in rows:
        if r["dataset"] == "amazon-book" and r["variant"] in ("V0", "V1", "V2", "V3"):
            if "NOT REPORTED" not in r["used_in_tables"]:
                r["used_in_tables"] = ("NOT REPORTED (taxonomy_policy=no_merge, not the main configuration's policy; "
                                       "superseded by the merge_t10 runs) -- " + r["used_in_tables"])
                flagged += 1
    with open(MANIFEST, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    # per-seed metrics
    mrows = list(csv.DictReader(open(METRICS, encoding="utf-8")))
    mfields = list(mrows[0].keys())
    mhave = {(r["run_id"], r["group"]) for r in mrows}
    madded = 0
    for r in new_rows:
        dirname = Path(r["run_dir"].replace("\\", "/")).parent.name
        seed = r["seed"]
        gm = json.loads((ROOT / r["run_dir"].replace("\\", "/") / "final_test_group_metrics.json").read_text(encoding="utf-8"))
        for group in ("overall", "near_cold", "long_tail", "warm"):
            if (r["run_id"], group) in mhave or group not in gm:
                continue
            g = gm[group]
            mrows.append({
                "run_id": r["run_id"], "model": "TaxPro-CL", "dataset": "amazon-book", "variant": r["variant"],
                "seed": seed, "group": group,
                "eligible_users": g.get("eligible_users", ""), "relevant_items": g.get("relevant_items", ""),
                "recall_at_10": g["recall"]["10"], "recall_at_20": g["recall"]["20"],
                "ndcg_at_10": g["ndcg"]["10"], "ndcg_at_20": g["ndcg"]["20"],
            })
            madded += 1
    with open(METRICS, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=mfields, lineterminator="\n")
        w.writeheader()
        w.writerows(mrows)
    print(f"results_manifest.csv: +{appended} rows, {flagged} old Amazon-Book V0-V3 rows flagged; metrics_seed.csv: +{madded} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
