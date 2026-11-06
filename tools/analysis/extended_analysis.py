"""All analyses of docs/extended_evaluation_protocol.md, in one place.

Reads test results that were already opened and never opens a sealed run. New runs carry
final_test_per_user.npz from their single opening. For reference runs, whose test split
was read when they were published, per-user values are recomputed once from the locked
checkpoint, checked against the stored group metrics and cached under
log/p0/extended/per_user/. A package that was not run is reported as not run.

Usage:
    python -m tools.analysis.extended_analysis [--device cuda] [--lock]
Output: results/extended/extended_analysis.json, results/extended/results_extended.csv;
--lock also records the SHA-256 of the CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analysis.factorial_multiplicity import (  # noqa: E402
    N_PERM_PRIMARY,
    bootstrap_ci,
    holm_bonferroni,
    sign_flip_pvalue,
)
from tools.experiments.evaluate_sealed_test import (  # noqa: E402
    check_against_group_metrics,
    per_user_arrays,
)
from tools.experiments.run_extended import (  # noqa: E402
    EXPECTED_RUNS,
    FAMILIES,
    LOCKED_SPLIT,
    OFFICE,
    OUTPUT,
    RESULTS,
    SEEDS,
    NEW_SEEDS,
    SELECTION_GRID,
    grid_family,
    planned,
    sha256_path,
)
from tools.ranking.inference import load_model  # noqa: E402
from utility.utility_data.taxonomy import REDRAWN_SPLIT_BASES, REDRAWN_SPLIT_SEEDS  # noqa: E402

GROUPS = ("near_cold", "long_tail", "warm", "overall")
PRIMARY_GROUPS = ("near_cold", "long_tail")
CONFIGURATION_GROUPS = ("near_cold", "long_tail", "overall")
METRICS = ("recall", "ndcg")
BOOT_CONFIGURATION = 2000
BOOT_CONTRAST = 5000
CONTROL_CONTRASTS = {
    "true_vs_shuffled": ("V3", "V3-shuffled"),
    "full_vs_detached": ("V3", "V3-detached"),
    "detached_vs_random": ("V3-detached", "V2"),
    "detached_true_vs_shuffled": ("V3-detached", "V3-shuffled-detached"),
}
LABEL_CONTRASTS = {
    "clean": ("V3", "V2"),
    "noise50": ("V3-noise50", "V2"),
    "noise100": ("V3-shuffled", "V2"),
    "missing50": ("V3-missing50", "V2-missing50"),
    "mask_effect": ("V2-missing50", "V2"),
}
FACTORIAL_CONTRASTS = {
    "direction_fixed_magnitude": ("V1", "V0"),
    "direction_adaptive_magnitude": ("V3", "V2"),
    "magnitude_random_direction": ("V2", "V0"),
    "magnitude_prototype_direction": ("V3", "V1"),
}
COMPARATOR_METHODS = ("KGCL", "GALORE")
SELECTION_COMPARATORS = RESULTS / "selection_comparators.json"
SIMGCL_CONTRASTS = {
    "complete_vs_default_simgcl": ("V3", "SimGCL-default"),
    "complete_vs_tuned_simgcl": ("V3", "SimGCL-tuned"),
}
CSV_FIELDS = ("run_id", "model", "dataset", "variant", "seed", "group", "eligible_users", "relevant_items",
              "recall_at_10", "recall_at_20", "ndcg_at_10", "ndcg_at_20")


def new_run_index():
    """(base dataset, split, arm, seed) -> run directory, for every planned new run."""
    index = {}
    for package in EXPECTED_RUNS:
        if package == "smoke" or package.startswith(("grid", "pretrain")):
            continue
        for item in planned(package):
            index[item.base_dataset, item.split, item.arm, item.seed] = ROOT / item.run_dir
    return index


def run_of(index, dataset, split, arm, seed):
    key = (dataset, split, arm, seed)
    if key in index:
        return index[key]
    if split == LOCKED_SPLIT and seed in SEEDS and arm in FAMILIES[dataset]:
        return ROOT / FAMILIES[dataset][arm] / f"seed{seed}"
    return None


def is_open(run_dir):
    return run_dir is not None and (run_dir / "final_test_group_metrics.json").exists()


def group_metrics(run_dir):
    return json.loads((run_dir / "final_test_group_metrics.json").read_text(encoding="utf-8"))


def per_user_values(run_dir, device):
    """Per-user arrays of an opened run; reference runs go through the checked cache."""
    own = run_dir / "final_test_per_user.npz"
    if own.exists():
        return dict(np.load(own))
    relative = run_dir.relative_to(ROOT).as_posix().replace("/", "__")
    cache = ROOT / OUTPUT / "per_user" / f"{relative}.npz"
    if cache.exists():
        return dict(np.load(cache))
    model, dataset, config, _ = load_model(run_dir, device)
    arrays = per_user_arrays(model, dataset, device, str(config["dataset"]))
    check_against_group_metrics(arrays, group_metrics(run_dir))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **arrays)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return arrays


def paired_differences(first_runs, second_runs, metric, group, device):
    """One value per user: the mean over the matched runs of the arm difference."""
    per_run = []
    for first, second in zip(first_runs, second_runs):
        a, b = per_user_values(first, device), per_user_values(second, device)
        if not np.array_equal(a[f"{group}_users"], b[f"{group}_users"]):
            raise ValueError(f"paired runs do not share their {group} users: {first}, {second}")
        per_run.append(a[f"{group}_{metric}"] - b[f"{group}_{metric}"])
    return np.mean(per_run, axis=0)


def contrast_cell(dataset, group, metric, first_runs, second_runs, n_boot, rng, device, test=True):
    differences = paired_differences(first_runs, second_runs, metric, group, device)
    cell = bootstrap_ci(differences, n_boot, rng)
    reference = statistics.fmean(float(group_metrics(r)[group][metric]["20"]) for r in second_runs)
    cell.update(dataset=dataset, group=group, metric=metric, reference_mean=reference,
                percent=100.0 * cell["mean_diff"] / reference if reference else None,
                runs=len(first_runs))
    if test:
        cell["p"] = sign_flip_pvalue(differences, N_PERM_PRIMARY, rng)
    return cell


def wording(cell):
    """The three permitted phrases: Holm, then the uncorrected interval, then neither."""
    if cell.get("holm_significant"):
        return "evidence"
    return "nominal signal" if cell["excludes_zero"] else "not detected"


def runs_for(index, dataset, split, arm, seeds):
    return [run_of(index, dataset, split, arm, seed) for seed in seeds]


def analyse_controls(index, rng, device):
    result = {}
    for name, (first, second) in CONTROL_CONTRASTS.items():
        result[name] = {"not_run": [], "cells": []}
        for metric in METRICS:
            family, context = [], []
            for dataset in FAMILIES:
                first_runs = runs_for(index, dataset, LOCKED_SPLIT, first, SEEDS)
                second_runs = runs_for(index, dataset, LOCKED_SPLIT, second, SEEDS)
                if not all(map(is_open, first_runs + second_runs)):
                    if dataset not in result[name]["not_run"]:
                        result[name]["not_run"].append(dataset)
                    continue
                for group in GROUPS:
                    cell = contrast_cell(dataset, group, metric, first_runs, second_runs, BOOT_CONTRAST, rng, device)
                    (family if group in PRIMARY_GROUPS else context).append(cell)
            if family:
                holm_bonferroni(family, "p", "holm_significant")
            for cell in family + context:
                cell["wording"] = wording(cell)
            result[name]["cells"] += family + context
    return result


def label_coverage(run_dir):
    labels = run_dir / "taxonomy_labels.npy"
    return float((np.load(labels) >= 0).mean()) * 100.0 if labels.exists() else None


def analyse_label_quality(index, rng, device):
    result = {}
    for dataset in ("musical-instruments", OFFICE):
        block = {"coverage_percent": {}, "contrasts": {}, "not_run": []}
        for name, (first, second) in LABEL_CONTRASTS.items():
            first_runs = runs_for(index, dataset, LOCKED_SPLIT, first, SEEDS)
            second_runs = runs_for(index, dataset, LOCKED_SPLIT, second, SEEDS)
            if not all(map(is_open, first_runs + second_runs)):
                block["not_run"].append(name)
                continue
            block["contrasts"][name] = [
                contrast_cell(dataset, group, metric, first_runs, second_runs, BOOT_CONTRAST, rng, device, test=False)
                for metric in METRICS for group in GROUPS]
            coverage = label_coverage(first_runs[0])
            if coverage is not None:
                block["coverage_percent"][name] = coverage
        result[dataset] = block
    return result


def classify(values_by_split):
    """Stable sign: mean of that sign, at most one run against it, and every split mean
    of that sign. With three splits of three runs this is 8 of 9; with six runs, 5 of 6."""
    values = [value for split in values_by_split for value in split]
    n = len(values)
    mean = statistics.fmean(values)
    split_means = [statistics.fmean(split) for split in values_by_split]
    if mean > 0 and sum(v > 0 for v in values) >= n - 1 and all(m > 0 for m in split_means):
        return "stable positive"
    if mean < 0 and sum(v < 0 for v in values) >= n - 1 and all(m < 0 for m in split_means):
        return "stable negative"
    return "unstable"


def sign_test(values):
    """Exact two-sided sign test; descriptive only."""
    positive = sum(v > 0 for v in values)
    negative = sum(v < 0 for v in values)
    n = positive + negative
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(max(positive, negative), n + 1))
    return min(1.0, 2.0 * tail / 2 ** n)


def stability_design(dataset):
    """(split, seeds) of the runs that enter the classification."""
    if dataset in REDRAWN_SPLIT_BASES:
        return [(split, SEEDS) for split in REDRAWN_SPLIT_SEEDS]
    return [(LOCKED_SPLIT, SEEDS + NEW_SEEDS)]


def analyse_stability(index, rng, device):
    result = {}
    contrasts = {**FACTORIAL_CONTRASTS, **SIMGCL_CONTRASTS}
    for dataset in FAMILIES:
        block = {}
        for name, (first, second) in contrasts.items():
            groups = PRIMARY_GROUPS if name in FACTORIAL_CONTRASTS else CONFIGURATION_GROUPS
            n_boot = BOOT_CONTRAST if name in FACTORIAL_CONTRASTS else BOOT_CONFIGURATION
            rows = []
            for metric in METRICS:
                for group in groups:
                    per_split = []
                    for split, seeds in stability_design(dataset):
                        pairs = [(run_of(index, dataset, split, first, seed), run_of(index, dataset, split, second, seed),
                                  seed) for seed in seeds]
                        values = [{"seed": seed,
                                   "value": float(group_metrics(a)[group][metric]["20"])
                                   - float(group_metrics(b)[group][metric]["20"])}
                                  for a, b, seed in pairs if is_open(a) and is_open(b)]
                        if values:
                            per_split.append({"split": split, "runs": values})
                    if not per_split:
                        rows.append({"metric": metric, "group": group, "status": "not run"})
                        continue
                    values_by_split = [[row["value"] for row in split["runs"]] for split in per_split]
                    flat = [value for split in values_by_split for value in split]
                    row = {"metric": metric, "group": group, "status": "run", "per_split": per_split,
                           "mean": statistics.fmean(flat), "std": statistics.stdev(flat) if len(flat) > 1 else 0.0,
                           "positive_runs": sum(v > 0 for v in flat), "runs": len(flat),
                           "split_means": [statistics.fmean(v) for v in values_by_split],
                           "classification": classify(values_by_split), "sign_test_p": sign_test(flat)}
                    if dataset in REDRAWN_SPLIT_BASES:
                        row["split_intervals"] = []
                        for split in per_split:
                            seeds = [entry["seed"] for entry in split["runs"]]
                            row["split_intervals"].append({"split": split["split"], **contrast_cell(
                                dataset, group, metric,
                                runs_for(index, dataset, split["split"], first, seeds),
                                runs_for(index, dataset, split["split"], second, seeds),
                                n_boot, rng, device, test=False)})
                    rows.append(row)
            block[name] = rows
        result[dataset] = block
    return result


def at_upper_edge(cells):
    """True when a selected cell sits on the highest temperature searched for its method."""
    complete = cells["complete_configuration"]
    return {"complete_configuration": complete.startswith("temp0.25-") or complete.endswith("tempuser0.4"),
            "simgcl": cells["simgcl"].startswith("temp0.4-")}


def analyse_grid(rng, device):
    path = ROOT / SELECTION_GRID
    if not path.exists():
        return {"status": "not selected"}
    selection = json.loads(path.read_text(encoding="utf-8"))
    cells = {side: selection[side]["rule_R"] for side in ("complete_configuration", "simgcl")}
    runs = {side: [ROOT / grid_family(side, cell) / f"seed{seed}" for seed in SEEDS] for side, cell in cells.items()}
    document = {"selected": cells, "rule_Ov": {side: selection[side]["rule_Ov"] for side in cells},
                "selected_at_upper_edge": at_upper_edge(cells)}
    if not all(map(is_open, runs["complete_configuration"] + runs["simgcl"])):
        document["status"] = "test not opened"
        return document
    document["status"] = "complete"
    document["cells"] = [contrast_cell(OFFICE, group, metric, runs["complete_configuration"], runs["simgcl"],
                                       BOOT_CONFIGURATION, rng, device)
                         for metric in METRICS for group in CONFIGURATION_GROUPS]
    return document


def analyse_comparators(index, rng, device):
    """The rule-R cell of each added baseline against the complete configuration."""
    path = ROOT / SELECTION_COMPARATORS
    if not path.exists():
        return {"status": "not selected"}
    selection = json.loads(path.read_text(encoding="utf-8"))
    complete = runs_for(index, OFFICE, LOCKED_SPLIT, "V3", SEEDS)
    document = {}
    for method in COMPARATOR_METHODS:
        if method not in selection:
            document[method] = {"status": "not run"}
            continue
        cell = selection[method]["rule_R"]
        runs = [ROOT / OUTPUT / OFFICE / method / f"grid-{cell}" / f"seed{seed}" for seed in SEEDS]
        block = {"selected": cell, "rule_Ov": selection[method]["rule_Ov"]}
        if not all(map(is_open, runs)):
            document[method] = {**block, "status": "test not opened"}
            continue
        cells = []
        for metric in METRICS:
            family = [contrast_cell(OFFICE, group, metric, complete, runs, BOOT_CONFIGURATION, rng, device)
                      for group in PRIMARY_GROUPS]
            holm_bonferroni(family, "p", "holm_significant")
            context = [contrast_cell(OFFICE, group, metric, complete, runs, BOOT_CONFIGURATION, rng, device)
                       for group in GROUPS if group not in PRIMARY_GROUPS]
            for item in family + context:
                item["wording"] = wording(item)
            cells += family + context
        document[method] = {**block, "status": "complete", "cells": cells}
    return document


def mean_std(values):
    """Mean and sample standard deviation (n - 1), as Table S1."""
    return {"mean": statistics.fmean(values), "std": statistics.stdev(values) if len(values) > 1 else None,
            "n": len(values)}


def result_rows(index):
    rows = []
    for (dataset, split, arm, seed), run_dir in sorted(index.items(), key=str):
        if not is_open(run_dir):
            continue
        metrics = group_metrics(run_dir)
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        data_id = dataset if split == LOCKED_SPLIT else f"{dataset}-split{split}"
        for group in GROUPS:
            entry = metrics[group]
            rows.append({"run_id": f"{manifest['model']}|{data_id}|{arm}|seed{seed}", "model": manifest["model"],
                         "dataset": data_id, "variant": arm, "seed": seed, "group": group,
                         "eligible_users": entry.get("eligible_users", ""),
                         "relevant_items": entry.get("relevant_items", ""),
                         "recall_at_10": entry["recall"]["10"], "recall_at_20": entry["recall"]["20"],
                         "ndcg_at_10": entry["ndcg"]["10"], "ndcg_at_20": entry["ndcg"]["20"]})
    return rows


def summary_table(rows):
    """Mean and std over seeds, per dataset, arm, group and metric."""
    collected = {}
    for row in rows:
        for metric in ("recall_at_20", "ndcg_at_20"):
            key = (row["dataset"], row["variant"], row["group"], metric)
            collected.setdefault(key, []).append(float(row[metric]))
    return [{"dataset": d, "variant": v, "group": g, "metric": m, **mean_std(values)}
            for (d, v, g, m), values in sorted(collected.items())]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lock", action="store_true")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    rng = np.random.default_rng(42)
    index = new_run_index()
    rows = result_rows(index)
    document = {"protocol": "docs/extended_evaluation_protocol.md",
                "controls": analyse_controls(index, rng, device),
                "label_quality": analyse_label_quality(index, rng, device),
                "stability": analyse_stability(index, rng, device),
                "grid": analyse_grid(rng, device),
                "comparators": analyse_comparators(index, rng, device),
                "test_mean_std": summary_table(rows)}
    output = ROOT / RESULTS
    output.mkdir(parents=True, exist_ok=True)
    (output / "extended_analysis.json").write_bytes((json.dumps(document, indent=2) + "\n").encode("utf-8"))
    table = output / "results_extended.csv"
    with table.open("w", newline="\n", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    if args.lock:
        table.with_suffix(".sha256").write_bytes(f"{sha256_path(table)}  {table.name}\n".encode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
