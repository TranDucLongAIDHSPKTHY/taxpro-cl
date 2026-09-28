"""GVHD V6 review, point A1: the Amazon-Book no_merge V0-V3 factorial
(log/p0/taxprocl/amazon-book/A2-V{0,1,2,3}) was marked "NOT REPORTED ...
superseded by the merge_t10 runs" in results_manifest.csv when the
manuscript body dropped it. It is restored in V63 (Online Resource 1,
Tables S13c-S13e), so this script updates those 12 rows' used_in_tables
field in place -- no rows added or removed, no other field touched.

Idempotent: running it again on an already-updated manifest is a no-op.

Usage:
    python -m tools.analysis.update_manifest_a1_nomerge_restored
"""
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
