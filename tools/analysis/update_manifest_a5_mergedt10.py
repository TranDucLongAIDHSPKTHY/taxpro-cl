"""Add results_manifest.csv / metrics_seed.csv rows for the A5 parent-granularity
runs of Amazon-Book trained under taxonomy_policy=merge_t10
(log/p0/taxprocl/amazon-book/ablation-A5-parent-mergedt10, launched by
tools/experiments/run_a5_parent_mergedt10.ps1), the policy-matched comparator
of TaxPro-CL-main (prototype_mode=leaf).

Idempotent: rows already present (same dataset/variant/seed, or same
run_id/group in metrics_seed.csv) are not duplicated.

Usage:
    python -m tools.analysis.update_manifest_a5_mergedt10
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "results" / "results_manifest.csv"
METRICS = ROOT / "results" / "metrics_seed.csv"
RUN_DIR = ROOT / "log" / "p0" / "taxprocl" / "amazon-book" / "ablation-A5-parent-mergedt10"
LABEL = "A5-parent-mergedt10"
SEEDS = (42, 0, 1)
USED_IN = ("Online Resource 1 Table S7; Section 5.3 (A5); RQ4 (prototype_mode=parent vs. TaxPro-CL-main "
           "prototype_mode=leaf, both taxonomy_policy=merge_t10)")


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
    new_rows = []
    for seed in SEEDS:
        run_dir = RUN_DIR / f"seed{seed}"
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            raise SystemExit(f"run not completed: {run_dir}")
        if ("amazon-book", LABEL, str(seed)) in have:
            continue
        config = manifest["configuration"]
        if config.get("prototype_mode") != "parent" or config.get("taxonomy_policy") != "merge_t10":
            raise SystemExit(f"unexpected configuration in {run_dir}")
        ckpt = run_dir / "best_validation_model.pt"
        row = {f: "" for f in fields}
        row.update({
            "run_id": f"TaxPro-CL|amazon-book|{LABEL}|seed{seed}",
            "model": "TaxPro-CL", "dataset": "amazon-book", "variant": LABEL, "seed": str(seed),
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
            "v3_is_same_checkpoint_as_taxprocl_main": "N/A",
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
            "used_in_tables": USED_IN,
        })
        new_rows.append(row)
    rows.extend(new_rows)
    with open(MANIFEST, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    mrows = list(csv.DictReader(open(METRICS, encoding="utf-8")))
    mfields = list(mrows[0].keys())
    mhave = {(r["run_id"], r["group"]) for r in mrows}
    madded = 0
    for r in new_rows:
        gm = json.loads((ROOT / r["run_dir"].replace("\\", "/") / "final_test_group_metrics.json").read_text(encoding="utf-8"))
        for group in ("overall", "near_cold", "long_tail", "warm"):
            if (r["run_id"], group) in mhave or group not in gm:
                continue
            g = gm[group]
            mrows.append({
                "run_id": r["run_id"], "model": "TaxPro-CL", "dataset": "amazon-book", "variant": LABEL,
                "seed": r["seed"], "group": group,
                "eligible_users": g.get("eligible_users", ""), "relevant_items": g.get("relevant_items", ""),
                "recall_at_10": g["recall"]["10"], "recall_at_20": g["recall"]["20"],
                "ndcg_at_10": g["ndcg"]["10"], "ndcg_at_20": g["ndcg"]["20"],
            })
            madded += 1
    with open(METRICS, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=mfields, lineterminator="\n")
        w.writeheader()
        w.writerows(mrows)
    print(f"results_manifest.csv: +{len(new_rows)} rows; metrics_seed.csv: +{madded} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
