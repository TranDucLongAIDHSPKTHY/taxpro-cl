"""Compile only complete Week-6 artifacts; never synthesize missing results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GROUPS = ("overall", "near_cold", "long_tail", "warm")
METRICS = ("precision", "recall", "ndcg")
KS = ("10", "20")
SUPPORT_FIELDS = ("eligible_users", "relevant_items", "positive_interactions")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-root", type=Path, default=ROOT / "log" / "p0" / "taxprocl"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "week6")
    return parser.parse_args(argv)


def infer_study(config_id):
    lowered = str(config_id).lower()
    if lowered.startswith("a3-"):
        return "A3"
    if lowered.startswith("a4-"):
        return "A4"
    if lowered.startswith("a5-"):
        return "A5"
    if lowered.startswith("main-"):
        return "main"
    return "unclassified"


def validate_group_payload(payload):
    protocol = payload.get("protocol", {})
    if (
        protocol.get("candidate_catalog") != "full"
        or protocol.get("target_split") != "test"
    ):
        raise ValueError("missing_or_invalid_protocol")
    for group in GROUPS:
        if group not in payload:
            raise ValueError("missing_group:{}".format(group))
        for metric in METRICS:
            for k in KS:
                if k not in payload[group].get(metric, {}):
                    raise ValueError(
                        "missing_metric:{}:{}@{}".format(group, metric, k)
                    )
                value = payload[group][metric][k]
                if (
                    not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or not 0.0 <= float(value) <= 1.0
                ):
                    raise ValueError(
                        "invalid_metric:{}:{}@{}".format(group, metric, k)
                    )
        if group != "overall":
            for field in SUPPORT_FIELDS:
                value = payload[group].get(field)
                if not isinstance(value, int) or value < 0:
                    raise ValueError(
                        "missing_support:{}:{}".format(group, field)
                    )


def collect(log_root):
    rows, support_rows, incomplete = [], [], []
    for manifest_path in sorted(log_root.rglob("run_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        group_path = manifest_path.parent / "final_test_group_metrics.json"
        relative_parts = manifest_path.parent.parts
        config_id = relative_parts[-2] if len(relative_parts) >= 2 else ""
        reason = None
        if manifest.get("status") != "completed":
            reason = "manifest_status_{}".format(manifest.get("status"))
        elif not group_path.is_file():
            reason = "missing_group_metrics"
        if reason:
            incomplete.append(
                {
                    "artifact_path": manifest_path.parent.as_posix(),
                    "reason": reason,
                }
            )
            continue
        payload = json.loads(group_path.read_text(encoding="utf-8"))
        try:
            validate_group_payload(payload)
        except ValueError as error:
            incomplete.append(
                {
                    "artifact_path": manifest_path.parent.as_posix(),
                    "reason": str(error),
                }
            )
            continue
        config = manifest.get("configuration", {})
        base = {
            "study": infer_study(config_id),
            "config_id": config_id,
            "dataset": manifest.get("dataset"),
            "seed": int(manifest.get("training_seed")),
            "taxonomy_policy": config.get("taxonomy_policy"),
            "taxonomy_granularity": config.get("taxonomy_granularity", "leaf"),
            "epsilon_max": float(config.get("epsilon_max", 0.1)),
            "temperature": float(config.get("temperature", 0.2)),
            "augmentation_direction": config.get("augmentation_direction"),
            "run_type": manifest.get("run_type", "NEW"),
            "source_run": manifest.get("source_run"),
            "artifact_path": manifest_path.parent.as_posix(),
        }
        for group in GROUPS:
            for metric in METRICS:
                for k in KS:
                    rows.append(
                        {
                            **base,
                            "group": group,
                            "metric": metric,
                            "k": int(k),
                            "value": float(payload[group][metric][k]),
                        }
                    )
            if group != "overall":
                support_rows.append(
                    {
                        **base,
                        "group": group,
                        **{
                            field: int(payload[group][field])
                            for field in SUPPORT_FIELDS
                        },
                    }
                )
    return rows, support_rows, incomplete


def aggregate(rows):
    buckets = {}
    keys = (
        "study",
        "config_id",
        "dataset",
        "taxonomy_policy",
        "taxonomy_granularity",
        "epsilon_max",
        "temperature",
        "augmentation_direction",
        "run_type",
        "group",
        "metric",
        "k",
    )
    for row in rows:
        key = tuple(row[field] for field in keys)
        buckets.setdefault(key, []).append(row["value"])
    output = []
    for key, values in sorted(buckets.items()):
        output.append(
            {
                **dict(zip(keys, key)),
                "n_seeds": len(values),
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "complete_three_seeds": len(values) == 3,
            }
        )
    return output


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    args = parse_args(argv)
    raw, supports, incomplete = collect(args.log_root)
    summary = aggregate(raw)
    raw_fields = (
        "study", "config_id", "dataset", "seed", "taxonomy_policy",
        "taxonomy_granularity", "epsilon_max", "temperature",
        "augmentation_direction", "run_type", "source_run", "group", "metric",
        "k", "value", "artifact_path",
    )
    summary_fields = (
        "study", "config_id", "dataset", "taxonomy_policy",
        "taxonomy_granularity", "epsilon_max", "temperature",
        "augmentation_direction", "run_type", "group", "metric", "k", "n_seeds",
        "mean", "std", "complete_three_seeds",
    )
    write_csv(args.output_dir / "week6_raw_results.csv", raw, raw_fields)
    write_csv(args.output_dir / "week6_summary_results.csv", summary, summary_fields)
    support_fields = (
        "study", "config_id", "dataset", "seed", "taxonomy_policy",
        "taxonomy_granularity", "epsilon_max", "temperature",
        "augmentation_direction", "run_type", "source_run", "group",
        *SUPPORT_FIELDS, "artifact_path",
    )
    write_csv(
        args.output_dir / "week6_support_counts.csv",
        supports,
        support_fields,
    )
    write_csv(
        args.output_dir / "week6_incomplete_runs.csv",
        incomplete,
        ("artifact_path", "reason"),
    )
    print(
        "Compiled {} metric rows; {} incomplete runs; output={}".format(
            len(raw), len(incomplete), args.output_dir
        )
    )
    return 0 if raw and not incomplete else 2


if __name__ == "__main__":
    raise SystemExit(main())
