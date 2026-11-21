"""Wider KGCL and NCL grids on Office-Products: Deviation 2 of docs/extended_evaluation_protocol.md.

Exploratory runs after the data lock. The selected KGCL temperature is the highest searched and the selected
NCL cell has the lowest temperature and the fewest clusters searched, so each grid is widened past that edge.
The runs keep the run rules of the protocol (Sections 3, 9 and 10) and write their own registry,
results/extended/run_registry_wide.csv, and their own selection files, so no locked file of Section 13
changes. Rule R selects on the union of the existing and the new cells, existing cells first in tie order,
and rule Ov is recorded. The test split of the selected cell is opened once, and only if that cell differs
from the cell selected before; the cell selected before keeps its reported result.

Usage:
    python -m tools.experiments.run_wide_grids --package comparator-kgcl-wide grid-ncl-wide [--dry-run]
    python -m tools.experiments.run_wide_grids --select kgcl
    python -m tools.experiments.run_wide_grids --open-test kgcl [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.experiments import run_comparators as comparators  # noqa: E402
from tools.experiments import run_extended as extended  # noqa: E402
from tools.experiments.run_confirmatory import rule_ov, rule_r, validation_means  # noqa: E402
from tools.experiments.run_extended import (  # noqa: E402
    LOCKED_SPLIT,
    NCL_CLUSTERS,
    NCL_TEMPS,
    OFFICE,
    OUTPUT,
    RESULTS,
    SEEDS,
    Job,
    committed_selection,
    grid_family,
    log,
    require_clean_code,
    require_frozen_protocol,
    run_status,
    sha256_path,
    update_registry,
)

KGCL_PACKAGE = "comparator-kgcl-wide"
NCL_PACKAGE = "grid-ncl-wide"
KGCL_WIDE_TEMPERATURES = (0.25, 0.30)          # beyond the selected 0.20, the highest of {0.05, ..., 0.20}
NCL_WIDE_TEMPERATURES = (0.02, 0.03)           # below the selected 0.05, the lowest of {0.05, ..., 0.20}
NCL_WIDE_CLUSTERS = (500, 1000)                # 500 below the selected 1000, the fewest of {1000, 2000, 3000}
EXPECTED_RUNS = {KGCL_PACKAGE: 18, NCL_PACKAGE: 12}
DEVIATION = "Deviation 2 (Section 14)"


def kgcl_cell(temperature, weight):
    return f"temp{temperature}-ssl{weight}"


def kgcl_job(package, temperature, weight, seed):
    """A KGCL run in the layout of run_comparators.plan."""
    cell = kgcl_cell(temperature, weight)
    config = tuple(sorted((key, str(value)) for key, value in
                          {"temperature": temperature, "ssl_weight": weight}.items()))
    return Job(package, f"KGCL-{cell}", "KGCL", OFFICE, OFFICE, LOCKED_SPLIT, f"grid-{cell}", seed, config)


def ncl_job(package, temperature, clusters, seed):
    """An NCL run in the layout of run_extended.plan_grid_ncl."""
    return extended.job(package, OFFICE, "NCL-default", "grid", seed, {"temperature": temperature, "k": clusters},
                        arm=f"NCL-temp{temperature}-k{clusters}")


def plan(package):
    if package == KGCL_PACKAGE:
        jobs = [kgcl_job(package, t, w, seed) for t in KGCL_WIDE_TEMPERATURES for w in comparators.KGCL_WEIGHTS
                for seed in SEEDS]
    elif package == NCL_PACKAGE:
        jobs = [ncl_job(package, t, k, seed) for t in NCL_WIDE_TEMPERATURES for k in NCL_WIDE_CLUSTERS
                for seed in SEEDS]
    else:
        raise ValueError(package)
    if len(jobs) != EXPECTED_RUNS[package]:
        raise ValueError(f"{package}: {len(jobs)} runs planned, {EXPECTED_RUNS[package]} expected")
    if len({j.run_dir for j in jobs}) != len(jobs) or len({j.run_id for j in jobs}) != len(jobs):
        raise ValueError(f"{package}: two runs share a directory or an identifier")
    return jobs


# Per method: package of the new cells, the selection file of the existing grid and its key, the selection file
# of the union, the cells of the union (existing first, the order in which ties are broken), and a cell's family.
SEARCHES = {
    "kgcl": {
        "package": KGCL_PACKAGE,
        "original": (comparators.selection_file("KGCL"), "KGCL"),
        "selection": RESULTS / "selection_kgcl_wide.json",
        "cells": lambda: ([kgcl_cell(t, w) for t in comparators.KGCL_TEMPERATURES for w in comparators.KGCL_WEIGHTS],
                          [kgcl_cell(t, w) for t in KGCL_WIDE_TEMPERATURES for w in comparators.KGCL_WEIGHTS]),
        "family": lambda cell: f"{OUTPUT.as_posix()}/{OFFICE}/KGCL/grid-{cell}",
    },
    "ncl": {
        "package": NCL_PACKAGE,
        "original": (extended.SELECTION_NCL, "ncl"),
        "selection": RESULTS / "selection_ncl_wide.json",
        "cells": lambda: ([f"temp{t}-k{k}" for t in NCL_TEMPS for k in NCL_CLUSTERS],
                          [f"temp{t}-k{k}" for t in NCL_WIDE_TEMPERATURES for k in NCL_WIDE_CLUSTERS]),
        "family": lambda cell: grid_family("ncl", cell),
    },
}


def run_packages(packages, device, dry_run):
    plans = {package: plan(package) for package in packages}
    for package, jobs in plans.items():
        remaining = [j for j in jobs if run_status(j.run_dir) != "completed"]
        print(f"{package}: {len(jobs)} runs planned, {len(remaining)} remaining")
        if dry_run:
            for item in remaining[:3]:
                print("  would run", item.run_id, item.run_dir.as_posix())
    if dry_run:
        return 0
    require_clean_code()
    require_frozen_protocol()
    for package, jobs in plans.items():
        update_registry(jobs)
        for item in jobs:
            if item.model == "KGCL":
                comparators.execute(item, device)
            else:
                extended.execute(item, device)
        log(f"PACKAGE COMPLETE {package}; test split still sealed", package)
    return 0


def union_cells(name):
    """[(cell, validation means)] of the union, existing cells first; every run must be complete."""
    search = SEARCHES[name]
    existing, new = search["cells"]()
    unfinished = [j.run_dir for j in plan(search["package"]) if run_status(j.run_dir) != "completed"]
    if unfinished:
        raise SystemExit(f"{name}: {len(unfinished)} runs are not complete; nothing is selected")
    cells = []
    for cell in existing + new:
        dirs = [ROOT / search["family"](cell) / f"seed{seed}" for seed in SEEDS]
        missing = [d for d in dirs if run_status(d.relative_to(ROOT)) != "completed"]
        if missing:
            raise SystemExit(f"{name}: cell {cell} has no completed runs at {missing[0]}")
        cells.append((cell, validation_means(dirs)))
    return cells, set(new)


def selection_document(name):
    search = SEARCHES[name]
    original_path, key = search["original"]
    original = json.loads((ROOT / original_path).read_text(encoding="utf-8"))[key]["rule_R"]
    cells, new = union_cells(name)
    selected = rule_r(cells)
    return {"protocol": extended.PROTOCOL.as_posix(), "deviation": DEVIATION,
            "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            key: {"rule_R": selected, "rule_Ov": rule_ov(cells), "rule_R_before": original,
                  "selection_changed": selected != original, "selected_cell_is_new": selected in new,
                  "new_cells": sorted(new, key=[c for c, _ in cells].index),
                  "cells": {cell: value["mean"] for cell, value in cells}}}


def freeze_selection(name):
    path = ROOT / SEARCHES[name]["selection"]
    if path.exists():
        raise SystemExit(f"{path} is frozen already")
    document = selection_document(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(document, indent=2) + "\n").encode("utf-8"))
    digest = sha256_path(path)
    path.with_suffix(".sha256").write_bytes(f"{digest}  {path.name}\n".encode())
    block = document[SEARCHES[name]["original"][1]]
    log(f"{name} wide selection written sha256={digest}; commit it before opening the test split")
    print(name, "rule R:", block["rule_R"], "| before:", block["rule_R_before"], "| rule Ov:", block["rule_Ov"])
    return 0


def runs_to_open(name, selection):
    """The three runs of the rule-R cell, if it is not the cell selected before and is still sealed."""
    search = SEARCHES[name]
    block = selection[search["original"][1]]
    if block["rule_R"] == block["rule_R_before"]:
        return []
    runs = [ROOT / search["family"](block["rule_R"]) / f"seed{seed}" for seed in SEEDS]
    return [run for run in runs if not (run / "final_test_metrics.json").exists()]


def opened_jobs(name, cell):
    """Registry rows of the opened cell, written to the wide registry; an existing cell keeps its locked row."""
    package = SEARCHES[name]["package"]
    if name == "kgcl":
        temperature, weight = cell.removeprefix("temp").split("-ssl")
        return [kgcl_job(package, float(temperature), float(weight), seed) for seed in SEEDS]
    temperature, clusters = cell.removeprefix("temp").split("-k")
    return [ncl_job(package, float(temperature), int(clusters), seed) for seed in SEEDS]


def open_test(name, device, dry_run=False):
    """With dry_run, list the runs that would be evaluated on test and evaluate none."""
    search = SEARCHES[name]
    selection = committed_selection(search["selection"])
    runs = runs_to_open(name, selection)
    if dry_run:
        print(f"{name}: {len(runs)} runs would be evaluated on test")
        for run in runs:
            print(" ", run.relative_to(ROOT).as_posix())
        return 0
    log(f"OPEN TEST {name} (wide): {len(runs)} runs")
    if not runs:
        return 0
    code = subprocess.run([sys.executable, "-m", "tools.experiments.evaluate_sealed_test",
                           "--device", device, "--runs", *map(str, runs)], cwd=ROOT).returncode
    update_registry(opened_jobs(name, selection[search["original"][1]]["rule_R"]))
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--package", nargs="+", choices=sorted(EXPECTED_RUNS))
    action.add_argument("--select", choices=sorted(SEARCHES))
    action.add_argument("--open-test", choices=sorted(SEARCHES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.package:
        return run_packages(args.package, args.device, args.dry_run)
    if args.select:
        return freeze_selection(args.select)
    return open_test(args.open_test, args.device, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
