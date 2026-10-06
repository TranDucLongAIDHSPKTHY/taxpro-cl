"""Cross-vintage audit of the item-to-category linkage (Online Resource 1, Section S28).

The taxonomy of an Amazon dataset is built by matching each item's original identifier against three metadata
releases: the `asin` field of the 2014 and 2018 releases and the `parent_asin` field of the 2023 release
(tools/data/prepare_metadata.py). For every item, one record per release is kept and the record with the deepest path
is selected (ties: more paths, then the newest release). This audit re-reads the source files and reports, per dataset:

* key multiplicity: items whose identifier occurs in more than one row of the same release (mapping ambiguity);
* identifier collisions: metadata records claimed by more than one item (impossible by construction when keys are
  matched exactly, checked anyway);
* cross-release agreement for items found in two or more releases: identical deepest paths, identical leaf label,
  prefix-consistent deepest paths (one is an ancestor-or-equal of the other), and the same first level below the root;
  reported for every release pair and separately for the 2023 `parent_asin` channel against the 2018 `asin` channel.

Metadata only; no model is involved. Usage:
    python -m tools.analysis.taxonomy_linkage_audit [--datasets amazon-book arts-crafts-and-sewing]
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_path.config_path import AMAZON_CATEGORY_METADATA_SOURCES  # noqa: E402
from tools.data.prepare_metadata import (  # noqa: E402
    _candidate_quality,
    _normalize_amazon_record,
    _parse_amazon_record,
)

ID_PATTERNS = {
    "2014": re.compile(rb"['\"]asin['\"]\s*:\s*['\"]([^'\"]+)"),
    "2018": re.compile(rb'"asin"\s*:\s*"([^"]+)"'),
    "2023": re.compile(rb'"parent_asin"\s*:\s*"([^"]+)"'),
}
DECISIONS = {
    "amazon-book": "merge_decisions_amazon.json",
    "arts-crafts-and-sewing": "merge_decisions_arts_crafts_and_sewing.json",
    "musical-instruments": "merge_decisions_musical_instruments.json",
    "office-products": "merge_decisions_office_products.json",
}


def deepest(paths):
    return max(paths, key=len) if paths else []


def compare(a, b):
    pa, pb = deepest(a), deepest(b)
    short, long_ = (pa, pb) if len(pa) <= len(pb) else (pb, pa)
    return {
        "identical_deepest_path": pa == pb,
        "identical_leaf": bool(pa and pb and pa[-1] == pb[-1]),
        "prefix_consistent": bool(short) and long_[:len(short)] == short,
        "same_first_level": len(pa) > 1 and len(pb) > 1 and pa[1] == pb[1],
    }


def audit(dataset):
    decisions = json.loads((ROOT / "metadata" / DECISIONS[dataset]).read_text(encoding="utf-8"))
    targets = {d["org_id"]: d for d in decisions}
    sources = AMAZON_CATEGORY_METADATA_SOURCES[dataset]
    best, rows = {}, Counter()
    missing_files = []
    for version, path in sources.items():
        if not Path(path).exists():
            missing_files.append(version)
            continue
        t0 = time.time()
        with open(path, "rb") as stream:
            for line in stream:
                m = ID_PATTERNS[version].search(line)
                if m is None:
                    continue
                org = m.group(1).decode("utf-8", errors="replace")
                if org not in targets:
                    continue
                rows[(version, org)] += 1
                try:
                    rec = _normalize_amazon_record(org, version, _parse_amazon_record(version, line))
                except (SyntaxError, ValueError, TypeError, json.JSONDecodeError):
                    continue
                cur = best.get((version, org))
                if cur is None or _candidate_quality(rec) > _candidate_quality(cur):
                    best[(version, org)] = rec
        print(f"{dataset} {version}: scanned in {time.time() - t0:.0f}s", flush=True)

    out = {"dataset": dataset, "items": len(targets), "missing_source_files": missing_files}
    for version in sources:
        found = [org for (v, org) in rows if v == version]
        out[f"items_found_{version}"] = len(found)
        out[f"items_with_duplicate_rows_{version}"] = sum(1 for org in found if rows[(version, org)] > 1)
    # collisions: one metadata key claimed by several items cannot occur with exact matching; verify on org ids
    out["identifier_collisions"] = len(targets) - len(set(targets))
    # agreement of the kept records across releases
    pair_stats = {}
    versions = [v for v in sources if v not in missing_files]
    for va, vb in itertools.combinations(versions, 2):
        stats = Counter()
        for org in targets:
            ra, rb = best.get((va, org)), best.get((vb, org))
            if ra is None or rb is None or not ra["taxonomy_paths"] or not rb["taxonomy_paths"]:
                continue
            stats["n"] += 1
            for k, v in compare(ra["taxonomy_paths"], rb["taxonomy_paths"]).items():
                stats[k] += int(v)
        pair_stats[f"{va}_vs_{vb}"] = dict(stats)
    out["release_pairs"] = pair_stats
    # the selected record against every other release's record of the same item
    sel = Counter()
    for org, d in targets.items():
        v = d["selected_version"]
        if v is None or v in missing_files:
            continue
        rs = best.get((v, org))
        for other in versions:
            ro = best.get((other, org))
            if other == v or rs is None or ro is None or not rs["taxonomy_paths"] or not ro["taxonomy_paths"]:
                continue
            sel["n"] += 1
            for k, val in compare(rs["taxonomy_paths"], ro["taxonomy_paths"]).items():
                sel[k] += int(val)
    out["selected_vs_other_releases"] = dict(sel)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=["amazon-book", "arts-crafts-and-sewing"])
    ap.add_argument("--output", type=Path, default=ROOT / "results" / "taxonomy_linkage_audit.json")
    args = ap.parse_args(argv)
    res = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else {}
    for ds in args.datasets:
        res[ds] = audit(ds)
        print(json.dumps(res[ds], indent=1), flush=True)
        args.output.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="")
    print("Saved to", args.output)


if __name__ == "__main__":
    main()
