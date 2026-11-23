"""Run-level stability of the complete-configuration comparisons (main paper Table 6).

One row per dataset, comparison and group (Near-Cold, Long-Tail, Overall Recall@20): the number of runs, the
numbers of positive and negative run effects, their mean and sample standard deviation, the range of the split
means on re-drawn splits, the classification of protocol Section 7 (stable positive, stable negative, unstable),
and the exact two-sided sign test (descriptive). A run effect is the complete configuration's test mean minus
the comparator's on the same seed and split.

The SimGCL comparisons and the Amazon-Book direction contrasts are read from the locked analysis
(results/extended/extended_analysis.json, "stability"; Online Resource 1, Tables S46-S47). The comparison with
LightGCN on the re-drawn Office splits (Table S46b) is computed from the locked per-run table
results/extended/results_extended.csv with the same rule. Nothing is re-estimated.

Writes results/extended/run_level_summary.csv.

Usage:
    python -m tools.analysis.run_level_summary
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analysis.extended_analysis import classify, sign_test  # noqa: E402

RESULTS = ROOT / "results" / "extended"
GROUPS = ("near_cold", "long_tail", "overall")
# (dataset, contrast in extended_analysis.json, label of the comparison)
FROM_ANALYSIS = [(dataset, contrast, label)
                 for dataset in ("amazon-book", "yelp2018", "musical-instruments", "arts-crafts-and-sewing",
                                 "office-products")
                 for contrast, label in (("complete_vs_default_simgcl", "V3-SimGCL(default)"),
                                         ("complete_vs_tuned_simgcl", "V3-SimGCL(tuned)"))]
FROM_ANALYSIS += [("amazon-book", "direction_fixed_magnitude", "V1-V0"),
                  ("amazon-book", "direction_adaptive_magnitude", "V3-V2")]
SPLITS = (101, 102, 103)
SEEDS = (42, 0, 1)
FIELDS = ("dataset", "comparison", "group", "runs", "positive", "negative", "mean", "std", "split_mean_min",
          "split_mean_max", "classification", "sign_test_p")


def row(dataset, comparison, group, per_split):
    values = [v for split in per_split for v in split]
    means = [statistics.fmean(split) for split in per_split]
    redrawn = len(per_split) > 1
    return {"dataset": dataset, "comparison": comparison, "group": group, "runs": len(values),
            "positive": sum(v > 0 for v in values), "negative": sum(v < 0 for v in values),
            "mean": statistics.fmean(values), "std": statistics.stdev(values),
            "split_mean_min": min(means) if redrawn else "", "split_mean_max": max(means) if redrawn else "",
            "classification": classify(per_split), "sign_test_p": sign_test(values)}


def analysis_rows():
    stability = json.loads((RESULTS / "extended_analysis.json").read_text(encoding="utf-8"))["stability"]
    rows = []
    for dataset, contrast, label in FROM_ANALYSIS:
        cells = {(c["group"], c["metric"]): c for c in stability[dataset][contrast]}
        for group in GROUPS:
            cell = cells.get((group, "recall"))
            if cell is None:
                continue
            per_split = [[run["value"] for run in split["runs"]] for split in cell["per_split"]]
            built = row(dataset, label, group, per_split)
            # The locked analysis already holds these statistics; they must agree.
            assert abs(built["mean"] - cell["mean"]) < 1e-15 and built["classification"] == cell["classification"], \
                (dataset, contrast, group)
            assert built["positive"] == cell["positive_runs"] and abs(built["sign_test_p"] - cell["sign_test_p"]) < 1e-12
            rows.append(built)
    return rows


def lightgcn_rows():
    """V3 minus LightGCN on the re-drawn Office splits, from the locked per-run table."""
    table = {}
    with (RESULTS / "results_extended.csv").open(newline="", encoding="utf-8") as stream:
        for entry in csv.DictReader(stream):
            table[entry["dataset"], entry["variant"], int(entry["seed"]), entry["group"]] = float(entry["recall_at_20"])
    rows = []
    for group in GROUPS:
        per_split = [[table[f"office-products-split{split}", "V3", seed, group]
                      - table[f"office-products-split{split}", "LightGCN", seed, group] for seed in SEEDS]
                     for split in SPLITS]
        rows.append(row("office-products", "V3-LightGCN", group, per_split))
    return rows


def main():
    rows = analysis_rows() + lightgcn_rows()
    target = RESULTS / "run_level_summary.csv"
    with target.open("w", newline="\n", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows; wrote {target.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
