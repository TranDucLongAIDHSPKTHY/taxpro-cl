"""Finish the remaining SimGCL / Amazon-Book cells of the A7 grid.

Runs only the not-yet-completed (model=SimGCL, dataset=amazon-book) cells that
tools/experiments/run_a7_full_grid.py would otherwise reach, in the same order
and with the same idempotent skip/resume logic and progress log, without
starting the XSimGCL or NCL grids.

Usage:
    python -m tools.experiments.run_a7_simgcl_amazonbook_remaining
    python -m tools.experiments.run_a7_simgcl_amazonbook_remaining --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.experiments import run_a7_full_grid as grid  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    cells = [c for c in grid.iter_simxsim_cells()
             if c[0] == "SimGCL" and c[1] == "amazon-book"
             and grid.manifest_status(grid.cell_dir(*c)) != "completed"]
    grid.log(f"A7 SimGCL amazon-book remaining: {len(cells)} cells")
    for model, dataset, p1, p2, seed in cells:
        status = grid.manifest_status(grid.cell_dir(model, dataset, p1, p2, seed))
        if args.dry_run:
            print(f"{model} {dataset} temp={p1} eps={p2} seed={seed} status={status}")
        else:
            grid.run_one(model, dataset, p1, p2, seed, args.device)
    if not args.dry_run:
        grid.log("A7 SimGCL amazon-book remaining: all cells processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
