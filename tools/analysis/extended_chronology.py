"""Chronology of the extended evaluation (Online Resource 1, Tables S50 and S50b).

Table S50 lists the records in time order: the protocol freeze and its publication, each selection commit and
its publication, the first test opening of each selected search, the data lock, and the changes of Section 14
of docs/extended_evaluation_protocol.md. Table S50b lists, per package, its runs and its test openings.

Times are UTC. Run and test-opening times come from results/extended/run_registry.csv and, for Deviation 2,
run_registry_wide.csv; commit times are the author times recorded by git; publication times are the pushes in
the public activity log of the GitHub repository (PUSHES, retrieved on 2026-11-23; --check-github compares
them with the live log). Two events are read from run records that are not shipped (EVENTS).

Writes results/extended/chronology.csv and chronology_packages.csv.

Usage:
    python -m tools.analysis.extended_chronology [--check-github]
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "extended"
REGISTRIES = (RESULTS / "run_registry.csv", RESULTS / "run_registry_wide.csv")
WIDE_RUNS = 30   # Deviation 2: 18 KGCL and 12 NCL runs (tools/experiments/run_wide_grids.py, EXPECTED_RUNS)
REPOSITORY = "TranDucLongAIDHSPKTHY/taxpro-cl"
BRANCH = "extended-evaluation"

# Pushes of the branch, from https://github.com/TranDucLongAIDHSPKTHY/taxpro-cl/activity (retrieved 2026-11-23).
PUSHES = (
    ("2026-11-08T01:50:33Z", None, "324e2be"),
    ("2026-11-11T11:05:50Z", "324e2be", "fd25c99"),
    ("2026-11-18T16:29:07Z", "fd25c99", "543c93e"),
    ("2026-11-19T02:21:01Z", "543c93e", "fcae810"),
    ("2026-11-21T15:13:44Z", "fcae810", "37e9aaf"),
    ("2026-11-23T00:52:46Z", "37e9aaf", "34db024"),
    ("2026-11-23T08:36:12Z", "34db024", "9ad26b0"),
)
# Files whose first commit is a record of Table S50, with the event it records and the test data seen then.
RECORDS = (
    ("docs/extended_evaluation_protocol.sha256", "Protocol frozen (SHA-256 490907ce...)",
     "Office test split viewed on 2026-09-28 (Section S36)"),
    ("results/extended/selection_grid.json", "Wider Office search: selection committed",
     "Validation only"),
    ("results/extended/selection_kgcl.json", "KGCL selection committed", "Validation only"),
    ("results/extended/selection_galore.json", "GALORE selection committed",
     "Validation only"),
    ("results/extended/selection_ncl.json", "Tuned-NCL selection committed", "Validation only"),
    ("results/extended/results_extended.sha256", "Data locked",
     "All openings of protocol Sections 4-10"),
    ("docs/extended_evaluation_protocol_deviations.sha256", "Deviations 1-3 recorded in protocol Section 14",
     "None of Deviation 2"),
    ("results/extended/selection_kgcl_wide.json", "Widened KGCL grid: selection committed",
     "Validation only"),
    ("results/extended/selection_ncl_wide.json", "Widened NCL grid: selection committed",
     "Validation only"),
)
# Events kept in run records that are not shipped (log/p0/extended/...), recorded in protocol Section 14.
EVENTS = (
    ("2026-11-08T18:31:25Z", "Per-user test values of one opened run completed after a tie-order fix (commit e4dc6a8); "
     "they reproduce its stored test metrics",
     "stamp of the run", "That run, opened at 18:17"),
    ("2026-11-16T03:56:58Z", "NCL run (0.2, 1000 clusters, seed 42) resumed at epoch 57 (Deviation 3)", "training log", "None"),
    ("2026-11-16T19:10:49Z", "Default SimGCL, ACS split 101, seed 1, resumed at epoch 32 (Deviation 3)", "training log", "None"),
)
# Package: (description, label of its results).
PACKAGES = {
    "grid-simgcl": ("Wider search, SimGCL", "Exploratory"),
    "grid-complete": ("Wider search, TaxPro-CL", "Exploratory"),
    "controls-office-products": ("Controls, Office", "Exploratory"),
    "controls-musical-instruments": ("Controls, MI", "Exploratory"),
    "label-quality-office-products": ("Label quality, Office", "Exploratory"),
    "label-quality-musical-instruments": ("Label quality, MI", "Exploratory"),
    "controls-amazon-book": ("Controls, AB", "Exploratory"),
    "comparator-kgcl": ("KGCL grid, Office", "Exploratory, Deviation 1"),
    "splits-office-products": ("Re-drawn Office splits", "Prospectively specified"),
    "seeds-amazon-book-factorial": ("Seeds 2-4, AB, V0-V3", "Exploratory"),
    "seeds-amazon-book-simgcl": ("Seeds 2-4, AB, SimGCL", "Exploratory"),
    "pretrain-lightgcn": ("GALORE's starting LightGCN", "Exploratory, Deviation 1"),
    "comparator-galore": ("GALORE grid, Office", "Exploratory, Deviation 1"),
    "splits-musical-instruments": ("Re-drawn MI splits", "Prospectively specified"),
    "grid-ncl": ("NCL grid, Office", "Exploratory, Deviation 1"),
    "controls-arts-crafts-and-sewing": ("Controls, ACS", "Exploratory"),
    "splits-arts-crafts-and-sewing": ("Re-drawn ACS splits", "Prospectively specified"),
    "seeds-yelp2018": ("Seeds 2-4, Yelp", "Exploratory"),
    "controls-yelp2018": ("Controls, Yelp", "Exploratory"),
    "splits-office-products-lightgcn": ("LightGCN, re-drawn Office", "Prospectively specified"),
    "comparator-kgcl-wide": ("Widened KGCL grid, Office", "Exploratory, Deviation 2"),
    "grid-ncl-wide": ("Widened NCL grid, Office", "Exploratory, Deviation 2"),
}


def utc(text):
    """ISO time with any offset -> aware UTC datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def stamp(moment):
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def git(*arguments):
    return subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def first_commit(path):
    """(short hash, author time) of the commit that added the file, or None."""
    lines = git("log", "--diff-filter=A", "--format=%h %aI", "--", path).splitlines()
    return lines[-1].split() if lines else None


def published(commit):
    """The first push whose new head contains the commit."""
    for moment, _, head in PUSHES:
        if subprocess.run(["git", "merge-base", "--is-ancestor", commit, head], cwd=ROOT).returncode == 0:
            return moment, head
    return None


def registry_rows():
    rows = []
    for path in REGISTRIES:
        if path.exists():
            with path.open(newline="", encoding="utf-8") as stream:
                rows += [row for row in csv.DictReader(stream) if row["exp"] != "reference"]
    return rows


def package_rows(rows):
    by_package = defaultdict(list)
    for row in rows:
        by_package[row["exp"]].append(row)
    table = []
    for package, members in by_package.items():
        if package not in PACKAGES:
            raise SystemExit(f"package without a description: {package}")
        starts = [utc(r["start_utc"]) for r in members]
        ends = [utc(r["end_utc"]) for r in members if r["end_utc"]]
        opened = [utc(r["test_eval_utc"]) for r in members if r["test_eval_utc"]]
        description, label = PACKAGES[package]
        table.append({"package": package, "description": description, "label": label, "runs": len(members),
                      "completed": sum(r["status"] == "completed" for r in members),
                      "first_start_utc": stamp(min(starts)), "last_end_utc": stamp(max(ends)) if ends else "",
                      "test_opened": len(opened), "first_opening_utc": stamp(min(opened)) if opened else "",
                      "last_opening_utc": stamp(max(opened)) if opened else ""})
    return sorted(table, key=lambda row: row["first_start_utc"])


def event_rows(packages):
    events, record_commits = [], []
    for path, event, seen in RECORDS:
        commit = first_commit(path)
        if commit is None:
            continue
        short, moment = commit
        record_commits.append(short)
        events.append({"time_utc": stamp(utc(moment)), "event": event, "record": f"{Path(path).name}, commit {short}",
                       "test_data_seen": seen})
    publication = {commit: published(commit) for commit in dict.fromkeys(record_commits)}
    for moment, _, head in PUSHES:
        commits = [commit for commit, push in publication.items() if push and push[1] == head]
        if commits:
            events.append({"time_utc": moment, "event": "Published on GitHub (push)",
                           "record": f"commit{'s' if len(commits) > 1 else ''} {', '.join(commits)}",
                           "test_data_seen": ""})
    for commit, push in publication.items():
        if push is None:
            events.append({"time_utc": "", "event": "Not yet published", "record": f"commit {commit}",
                           "test_data_seen": ""})
    openings = {row["package"]: row for row in packages}
    spans = {}   # opening event -> (first, last) opening time
    # A search's opening covers its selected cells of every method it selects, and GALORE's covers its start.
    for members, event in ((("grid-complete", "grid-simgcl"), "Wider Office search: test opened"),
                           (("comparator-kgcl",), "KGCL: test opened"),
                           (("comparator-galore", "pretrain-lightgcn"), "GALORE and its start: test opened"),
                           (("grid-ncl",), "Tuned NCL: test opened"),
                           (("comparator-kgcl-wide",), "Widened KGCL grid: test opened"),
                           (("grid-ncl-wide",), "Widened NCL grid: test opened")):
        rows = [openings[p] for p in members if p in openings and openings[p]["first_opening_utc"]]
        if rows:
            spans[event] = (min(r["first_opening_utc"] for r in rows), max(r["last_opening_utc"] for r in rows))
            events.append({"time_utc": spans[event][0], "event": event, "record": "run registry",
                           "test_data_seen": f"Its {sum(r['test_opened'] for r in rows)} runs, once each"})
    first = min(packages, key=lambda row: row["first_start_utc"])
    events.append({"time_utc": first["first_start_utc"], "event": f"First run of the protocol ({first['description']})",
                   "record": "run registry", "test_data_seen": "None"})
    for package, event in (("comparator-kgcl-wide", "First run of Deviation 2"),):
        if package in openings:
            events.append({"time_utc": openings[package]["first_start_utc"], "event": event, "record": "run registry",
                           "test_data_seen": "None"})
    wide = [row for row in packages if row["package"].endswith("-wide")]
    if sum(row["completed"] for row in wide) == WIDE_RUNS:
        events.append({"time_utc": max(row["last_end_utc"] for row in wide), "event": "Last run of Deviation 2 ended",
                       "record": "run registry", "test_data_seen": "None"})
    for moment, event, record, seen in EVENTS:
        events.append({"time_utc": moment, "event": event, "record": record, "test_data_seen": seen})
    # An opening event carries the time of its first opening. When another record falls before its last opening
    # (the tie-order fix during the wider search's openings), the last opening time is given, so the order holds.
    for row in events:
        if row["event"] in spans:
            first, last = spans[row["event"]]
            if any(first < other["time_utc"] < last for other in events if other is not row):
                row["test_data_seen"] += f", the last at {last[11:16]}"
    return sorted(events, key=lambda row: (row["time_utc"] or "9", row["event"]))


def check_github():
    request = urllib.request.Request(f"https://api.github.com/repos/{REPOSITORY}/activity?ref={BRANCH}&per_page=100",
                                     headers={"User-Agent": "taxpro-chronology/1.0",
                                              "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        live = {(item["timestamp"], item["after"][:7]) for item in json.load(response)}
    missing = [(moment, head) for moment, _, head in PUSHES if (moment, head) not in live]
    print("pushes match the GitHub activity log" if not missing else f"not in the live log: {missing}")
    return 1 if missing else 0


def write(path, rows):
    with path.open("w", newline="\n", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-github", action="store_true")
    args = parser.parse_args(argv)
    if args.check_github:
        return check_github()
    packages = package_rows(registry_rows())
    write(RESULTS / "chronology_packages.csv", packages)
    write(RESULTS / "chronology.csv", event_rows(packages))
    print(f"{len(packages)} packages; wrote chronology.csv and chronology_packages.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
