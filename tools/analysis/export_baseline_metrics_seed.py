"""A4 fix (JIIS V58 review): append LightGCN/SGL-ED/XSimGCL/NCL per-seed
Recall@10/20, NDCG@10/20 rows (near_cold, long_tail, overall, warm) to
results/metrics_seed.csv, so the file's Data-availability declaration
("the runs underlying Table 7 (deltas)") actually covers Table 7's A1
(LightGCN) block, not only its A2 (SimGCL) block.

Pure export/re-analysis: no retraining, no re-inference. Sources:
- results/week6/mid_tail_degree6_10.json for near_cold/long_tail
  (already the official checkpoint-resolved per-seed source used by
  tools/analysis/main_results_table.py for Table S1/Table 7).
- each of those same checkpoints' own final_test_group_metrics.json for
  overall/warm (same convention main_results_table.py itself documents).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MID_TAIL_FILE = ROOT / "results" / "week6" / "mid_tail_degree6_10.json"
METRICS_SEED_CSV = ROOT / "results" / "metrics_seed.csv"

DATASETS = ["amazon-book", "yelp2018", "musical-instruments", "arts-crafts-and-sewing"]
BASELINES_TO_ADD = ["LightGCN", "SGL-ED", "XSimGCL", "NCL"]
VARIANT_NAME = {
    "LightGCN": "LightGCN-main",
    "SGL-ED": "SGL-ED-main",
    "XSimGCL": "XSimGCL-main",
    "NCL": "NCL-main",
}
# mid_tail_degree6_10.json's method-name spelling may differ from the
# manuscript's; verified identical during this export ("SGL-ED" and
# "XSimGCL" match exactly in that file's own METHODS list).

CSV_FIELDS = ["run_id", "model", "dataset", "variant", "seed", "group",
              "eligible_users", "relevant_items", "recall_at_10", "recall_at_20",
              "ndcg_at_10", "ndcg_at_20"]


def load_existing_keys():
    keys = set()
    with METRICS_SEED_CSV.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            keys.add((row["model"], row["dataset"], row["variant"], row["seed"], row["group"]))
    return keys


def build_rows(mid_tail_doc):
    rows = []
    skipped = []
    for dataset in DATASETS:
        for method in BASELINES_TO_ADD:
            entry = mid_tail_doc[dataset].get(method)
            if entry is None:
                skipped.append((dataset, method, "missing from mid_tail_degree6_10.json"))
                continue
            checkpoint_dirs = entry["checkpoint_dirs"]
            variant = VARIANT_NAME[method]
            for seed_str, per_seed in entry["per_seed"].items():
                run_dir = ROOT / checkpoint_dirs[seed_str]
                final_metrics_path = run_dir / "final_test_group_metrics.json"
                if not final_metrics_path.exists():
                    skipped.append((dataset, method, seed_str, "no final_test_group_metrics.json"))
                    continue
                final_metrics = json.loads(final_metrics_path.read_text(encoding="utf-8"))
                run_id = f"{method}|{dataset}|{variant}|seed{seed_str}"
                for group in ("near_cold", "long_tail"):
                    g = per_seed[group]
                    rows.append({
                        "run_id": run_id, "model": method, "dataset": dataset,
                        "variant": variant, "seed": seed_str, "group": group,
                        "eligible_users": g["eligible_users"],
                        "relevant_items": g["relevant_items"],
                        "recall_at_10": g["recall"]["10"], "recall_at_20": g["recall"]["20"],
                        "ndcg_at_10": g["ndcg"]["10"], "ndcg_at_20": g["ndcg"]["20"],
                    })
                for group in ("overall", "warm"):
                    g = final_metrics[group]
                    rows.append({
                        "run_id": run_id, "model": method, "dataset": dataset,
                        "variant": variant, "seed": seed_str, "group": group,
                        "eligible_users": g.get("eligible_users", ""),
                        "relevant_items": g.get("relevant_items", ""),
                        "recall_at_10": g["recall"]["10"], "recall_at_20": g["recall"]["20"],
                        "ndcg_at_10": g["ndcg"]["10"], "ndcg_at_20": g["ndcg"]["20"],
                    })
    return rows, skipped


def main():
    mid_tail_doc = json.loads(MID_TAIL_FILE.read_text(encoding="utf-8"))
    existing = load_existing_keys()
    rows, skipped = build_rows(mid_tail_doc)

    new_rows = [r for r in rows if (r["model"], r["dataset"], r["variant"], r["seed"], r["group"]) not in existing]
    duplicate_count = len(rows) - len(new_rows)

    with METRICS_SEED_CSV.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
        for row in new_rows:
            writer.writerow(row)

    print(f"Appended {len(new_rows)} new rows ({duplicate_count} already present, skipped).")
    if skipped:
        print(f"Skipped {len(skipped)} entries (missing data):")
        for s in skipped:
            print("  ", s)
    return new_rows, skipped


if __name__ == "__main__":
    main()
