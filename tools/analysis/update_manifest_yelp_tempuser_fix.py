"""Add results_manifest.csv rows for the 12 corrected Yelp2018 factorial
checkpoints (temperature_user=0.15 fix), and mark the original (confound-
bearing) V0-V3 Yelp2018 rows as superseded in their used_in_tables field.
"""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "results" / "results_manifest.csv"

FIELDS = ["run_id", "model", "dataset", "variant", "seed", "status", "run_dir", "git_commit",
          "config_sha256", "train_split_sha256", "validation_split_sha256", "test_split_sha256",
          "split_hash_source", "taxonomy_hash", "evaluation_protocol_hash",
          "checkpoint_sha256_best_validation_model", "v3_is_same_checkpoint_as_taxprocl_main",
          "selection_metric", "test_policy", "completed_epochs", "duration_seconds",
          "temperature", "temperature_user", "epsilon_max", "warm_start_epochs",
          "augmentation_direction", "use_adaptive_epsilon", "taxonomy_policy", "used_in_tables"]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_row(variant_label, seed, run_dir):
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    config = manifest["configuration"]
    ckpt = run_dir / "best_validation_model.pt"
    row = {f: "" for f in FIELDS}
    row.update({
        "run_id": f"TaxPro-CL|yelp2018|{variant_label}|seed{seed}",
        "model": "TaxPro-CL",
        "dataset": "yelp2018",
        "variant": variant_label,
        "seed": str(seed),
        "status": manifest.get("status", ""),
        "run_dir": str(run_dir.relative_to(ROOT)).replace("/", "\\"),
        "git_commit": manifest.get("git_commit", ""),
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "checkpoint_sha256_best_validation_model": sha256_of(ckpt) if ckpt.exists() else "",
        "v3_is_same_checkpoint_as_taxprocl_main": "N/A (superseded fix, see ESM S26)" if variant_label != "V3-tempuser0.15" else "NO (separately trained, but temperature_user now matches main config)",
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
        "used_in_tables": "Table 8 (factorial); Table 9; Figure 3; Online Resource 1 Tables S16/S20/S22 (Yelp2018 rows, tempuser0.15-corrected, supersedes original A2-V0..V3)",
    })
    return row


def main():
    variants = {
        "A2-V0-tempuser0.15": "V0-tempuser0.15",
        "A2-V1-tempuser0.15": "V1-tempuser0.15",
        "A2-V2-tempuser0.15": "V2-tempuser0.15",
        "A2-V3-tempuser0.15": "V3-tempuser0.15",
    }
    new_rows = []
    for dir_name, variant_label in variants.items():
        for seed in (42, 0, 1):
            run_dir = ROOT / "log" / "p0" / "taxprocl" / "yelp2018" / dir_name / f"seed{seed}"
            new_rows.append(build_row(variant_label, seed, run_dir))

    existing_rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    existing_keys = {(r["dataset"], r["variant"], r["seed"]) for r in existing_rows}

    appended = 0
    for r in new_rows:
        key = (r["dataset"], r["variant"], r["seed"])
        if key not in existing_keys:
            existing_rows.append(r)
            appended += 1

    # Mark original Yelp2018 V0-V3 rows as superseded.
    superseded = 0
    for r in existing_rows:
        if r["dataset"] == "yelp2018" and r["variant"] in ("V0", "V1", "V2", "V3"):
            if "SUPERSEDED" not in r["used_in_tables"]:
                r["used_in_tables"] = (
                    "SUPERSEDED by V{0-3}-tempuser0.15 (temperature_user confound, see Online Resource 1 "
                    "Section S26) -- " + r["used_in_tables"]
                )
                superseded += 1

    with open(MANIFEST, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(existing_rows)

    print(f"Appended {appended} new rows; marked {superseded} original Yelp2018 V0-V3 rows as superseded.")


if __name__ == "__main__":
    main()
