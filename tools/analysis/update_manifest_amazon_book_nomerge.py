"""Set the used_in_tables field of the 12 Amazon-Book no_merge V0-V3 rows
(log/p0/taxprocl/amazon-book/A2-V{0,1,2,3}) in results/results_manifest.csv
to the Online Resource 1 tables that report them (Tables S13c-S13e). Only that
field of those rows is changed; no rows are added or removed.

Idempotent: running it again on an updated manifest is a no-op.

Usage:
    python -m tools.analysis.update_manifest_amazon_book_nomerge"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "results" / "results_manifest.csv"

OLD_VALUE = "NOT REPORTED (taxonomy_policy=no_merge, not the main configuration's policy; superseded by the merge_t10 runs)"
NEW_VALUE = (
    "Online Resource 1 Tables S13c/S13d (Amazon-Book, no_merge policy-sensitivity "
    "check for the direction/epsilon-adaptivity factorial); Table S13e (Holm-Bonferroni "
    "correction references the merge_t10 family, not these no_merge rows directly)"
)
TARGET_VARIANTS = {"V0", "V1", "V2", "V3"}


def main() -> int:
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8", newline="")))
    fieldnames = list(rows[0].keys())
    updated = 0
    for row in rows:
        if (
            row["dataset"] == "amazon-book"
            and row["variant"] in TARGET_VARIANTS
            and row["used_in_tables"] == OLD_VALUE
        ):
            row["used_in_tables"] = NEW_VALUE
            updated += 1
    if updated == 0:
        print("No matching rows found (already updated, or manifest changed).")
        return 0
    with open(MANIFEST, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Updated {updated} rows in {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
