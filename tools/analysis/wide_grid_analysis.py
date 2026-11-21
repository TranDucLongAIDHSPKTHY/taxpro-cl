"""Deviation 2 of docs/extended_evaluation_protocol.md: the widened KGCL and NCL grids on Office-Products.

For each method: validation means of every cell of the union, the rule-R and rule-Ov cells before and after
widening, whether the selected cell lies at an edge of the widened grid, and, when the selection changed and
its runs were opened, the paired test differences of the complete configuration from the newly selected cell
(bootstrap 2,000, sign-flip, Holm over NC and LT within each metric, as Section 9). The random stream is its
own (seed 2026), so the locked results/extended/extended_analysis.json is neither read nor written.

Usage:
    python -m tools.analysis.wide_grid_analysis [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analysis.extended_analysis import (  # noqa: E402
    BOOT_CONFIGURATION,
    GROUPS,
    METRICS,
    PRIMARY_GROUPS,
    contrast_cell,
    group_metrics,
    is_open,
    mean_std,
    new_run_index,
    runs_for,
    wording,
)
from tools.analysis.factorial_multiplicity import holm_bonferroni  # noqa: E402
from tools.experiments.run_extended import LOCKED_SPLIT, OFFICE, RESULTS, SEEDS  # noqa: E402
from tools.experiments.run_wide_grids import SEARCHES  # noqa: E402

OUTPUT_FILE = RESULTS / "wide_grids.json"
RANDOM_SEED = 2026


def parameters(name, cell):
    first, second = cell.removeprefix("temp").split("-ssl" if name == "kgcl" else "-k")
    return float(first), float(second)


def edges(name, cell):
    """Parameters of the cell at the lowest or highest value searched in the widened grid."""
    existing, new = SEARCHES[name]["cells"]()
    grid = [parameters(name, c) for c in existing + new]
    labels = ("temperature", "ssl_weight") if name == "kgcl" else ("temperature", "clusters")
    value = parameters(name, cell)
    return [label for i, label in enumerate(labels)
            if value[i] in (min(g[i] for g in grid), max(g[i] for g in grid))]


def analyse(name, complete, rng, device):
    search = SEARCHES[name]
    path = ROOT / search["selection"]
    if not path.exists():
        return {"status": "not selected"}
    key = search["original"][1]
    block = json.loads(path.read_text(encoding="utf-8"))[key]
    document = {"rule_R": block["rule_R"], "rule_R_before": block["rule_R_before"], "rule_Ov": block["rule_Ov"],
                "selection_changed": block["selection_changed"], "selected_cell_is_new": block["selected_cell_is_new"],
                "selected_at_edge_in": edges(name, block["rule_R"]), "cells": block["cells"]}
    if not block["selection_changed"]:
        return {**document, "status": "selection unchanged; no test split opened"}
    runs = [ROOT / search["family"](block["rule_R"]) / f"seed{seed}" for seed in SEEDS]
    if not all(map(is_open, runs)):
        return {**document, "status": "test not opened"}
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
    means = {metric: {group: mean_std([float(group_metrics(run)[group][metric]["20"]) for run in runs])
                      for group in GROUPS} for metric in METRICS}
    return {**document, "status": "complete", "test_mean_std": means, "cells": cells}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    rng = np.random.default_rng(RANDOM_SEED)
    complete = runs_for(new_run_index(), OFFICE, LOCKED_SPLIT, "V3", SEEDS)
    document = {"protocol": "docs/extended_evaluation_protocol.md", "deviation": "Deviation 2 (Section 14)",
                "random_seed": RANDOM_SEED, "device": str(device),
                **{name: analyse(name, complete, rng, device) for name in SEARCHES}}
    target = ROOT / OUTPUT_FILE
    target.write_bytes((json.dumps(document, indent=2) + "\n").encode("utf-8"))
    print(f"wrote {OUTPUT_FILE.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
