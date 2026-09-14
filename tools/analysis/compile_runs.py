"""Compile standalone TaxProCL run manifests without inventing missing metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-root", type=Path, default=ROOT / "log" / "p0" / "taxprocl"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "taxprocl_runs.csv"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows = []
    for manifest_path in sorted(args.log_root.rglob("run_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        final_path = manifest_path.parent / "final_test_metrics.json"
        row = {
            "dataset": manifest.get("dataset", ""),
            "model": manifest.get("model", ""),
            "seed": manifest.get("training_seed", ""),
            "status": manifest.get("status", ""),
            "taxonomy_policy": manifest.get("configuration", {}).get(
                "taxonomy_policy", ""
            ),
            "augmentation_direction": manifest.get("configuration", {}).get(
                "augmentation_direction", ""
            ),
            "artifact_path": manifest_path.parent.relative_to(ROOT).as_posix(),
            "final_result_present": final_path.is_file(),
        }
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "dataset",
                "model",
                "seed",
                "status",
                "taxonomy_policy",
                "augmentation_direction",
                "artifact_path",
                "final_result_present",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print("Compiled {} run manifests to {}".format(len(rows), args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
