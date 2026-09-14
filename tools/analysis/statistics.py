"""Paired seed-level Week-6 tests with Holm correction and effect sizes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from scipy.stats import rankdata, wilcoxon


ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "results" / "week6" / "week6_raw_results.csv",
    )
    parser.add_argument("--reference-config", required=True)
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "week6" / "statistical_tests.csv",
    )
    return parser.parse_args(argv)


def rank_biserial(differences):
    nonzero = [value for value in differences if value != 0]
    if not nonzero:
        return 0.0
    ranks = rankdata([abs(value) for value in nonzero], method="average")
    positive = sum(rank for rank, value in zip(ranks, nonzero) if value > 0)
    negative = sum(rank for rank, value in zip(ranks, nonzero) if value < 0)
    return float((positive - negative) / (positive + negative))


def holm(rows):
    ordered = sorted(enumerate(rows), key=lambda pair: pair[1]["p_value"])
    adjusted = [1.0] * len(rows)
    running = 0.0
    total = len(rows)
    for rank, (index, row) in enumerate(ordered):
        value = min(1.0, (total - rank) * row["p_value"])
        running = max(running, value)
        adjusted[index] = running
    for row, value in zip(rows, adjusted):
        row["p_holm"] = value
        row["significant_0_05"] = value < 0.05
    return rows


def main(argv=None):
    args = parse_args(argv)
    with args.input.open("r", encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    indexed = {}
    for row in records:
        key = (row["dataset"], row["group"], row["metric"], row["k"], row["seed"])
        indexed[(row["config_id"], key)] = float(row["value"])
    metric_keys = sorted(
        key for config, key in indexed if config == args.reference_config
    )
    rows = []
    for key_without_seed in sorted({key[:-1] for key in metric_keys}):
        pairs = []
        for seed in ("42", "0", "1"):
            key = (*key_without_seed, seed)
            reference = indexed.get((args.reference_config, key))
            candidate = indexed.get((args.candidate_config, key))
            if reference is not None and candidate is not None:
                pairs.append((reference, candidate))
        if len(pairs) != 3:
            raise ValueError(
                "Paired analysis requires exactly seeds 42,0,1 for {}".format(
                    key_without_seed
                )
            )
        differences = [candidate - reference for reference, candidate in pairs]
        if all(value == 0 for value in differences):
            statistic, p_value = 0.0, 1.0
        else:
            tested = wilcoxon(
                [pair[1] for pair in pairs],
                [pair[0] for pair in pairs],
                alternative="two-sided",
                method="exact",
            )
            statistic, p_value = float(tested.statistic), float(tested.pvalue)
        rows.append(
            {
                "reference_config": args.reference_config,
                "candidate_config": args.candidate_config,
                "dataset": key_without_seed[0],
                "group": key_without_seed[1],
                "metric": key_without_seed[2],
                "k": key_without_seed[3],
                "n_pairs": 3,
                "mean_paired_delta": sum(differences) / 3.0,
                "rank_biserial_sign": rank_biserial(differences),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
            }
        )
    holm(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "reference_config", "candidate_config", "dataset", "group", "metric",
        "k", "n_pairs", "mean_paired_delta", "rank_biserial_sign",
        "wilcoxon_statistic", "p_value", "p_holm", "significant_0_05",
    )
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote {} paired tests to {}".format(len(rows), args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
