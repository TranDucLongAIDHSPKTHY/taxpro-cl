"""Main results table (GVHD review B1, manuscript Section 5.1, Tables 7-10):
Recall@20 mean+-std over 3 seeds, all six methods x four datasets x five
evaluation groups (Near-Cold, Mid-Tail, Long-Tail, Overall, Warm).

Reuses two already-computed, cross-checked sources instead of re-running
inference:

- tools/analysis/mid_tail_degree6_10.py's output (near_cold/mid_tail/
  long_tail, cross-checked against each run's own saved metrics) for the
  official checkpoint resolution (checkpoint_selection.select_checkpoint,
  the same one used everywhere else in the paper) and its three groups.
- each of those same checkpoints' own final_test_group_metrics.json for
  overall/warm (not stored by the mid_tail script, since it only recomputes
  the disjoint mid_tail slice).

Run tools/analysis/mid_tail_degree6_10.py first (or with --resume, if some
entries are already up to date) if results/week6/mid_tail_degree6_10.json
does not exist or is stale relative to the current checkpoint_selection
resolution -- this script does not re-verify that itself.

Usage:
    python -m tools.analysis.main_results_table
    python -m tools.analysis.main_results_table --latex-only
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from config_path.config_path import RESULT_DIR, PROJECT_ROOT

MID_TAIL_FILE = RESULT_DIR / "week6" / "mid_tail_degree6_10.json"
OUTPUT_JSON = RESULT_DIR / "main_results_table.json"
OUTPUT_TEX = RESULT_DIR / "main_results_table.tex"

DATASETS = ["amazon-book", "yelp2018", "musical-instruments", "arts-crafts-and-sewing"]
METHODS = ["LightGCN", "SGL-ED", "SimGCL", "XSimGCL", "NCL", "TaxPro-CL"]
GROUPS = ["near_cold", "mid_tail", "long_tail", "overall", "warm"]

DATASET_LABEL = {
    "amazon-book": "Amazon-Book",
    "yelp2018": "Yelp2018",
    "musical-instruments": "Musical-Instruments",
    "arts-crafts-and-sewing": "Arts-Crafts-and-Sewing",
}
DATASET_SHORT = {
    "amazon-book": "ab", "yelp2018": "yelp",
    "musical-instruments": "mi", "arts-crafts-and-sewing": "acs",
}
GROUP_LABEL = {
    "near_cold": "Near-Cold", "mid_tail": "Mid-Tail", "long_tail": "Long-Tail",
    "overall": "Overall", "warm": "Warm",
}


def mean_std(values):
    if not values:
        return None, None
    m = statistics.fmean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return m, s


def build_table(mid_tail_doc):
    """{dataset: {method: {group: {recall20_mean, recall20_std, ndcg20_mean,
    ndcg20_std, n_seeds}}}}, reusing mid_tail_doc's checkpoint_dirs for
    overall/warm so every group in one row comes from the same checkpoints."""
    result = {}
    for ds in DATASETS:
        result[ds] = {}
        for method in METHODS:
            entry = mid_tail_doc[ds][method]
            ckpt_dirs = entry["checkpoint_dirs"]  # {seed: relpath}
            row = {}
            for g in ("near_cold", "mid_tail", "long_tail"):
                r_vals = [entry["per_seed"][s][g]["recall"]["20"] for s in entry["per_seed"]]
                n_vals = [entry["per_seed"][s][g]["ndcg"]["20"] for s in entry["per_seed"]]
                r_mean, r_std = mean_std(r_vals)
                n_mean, n_std = mean_std(n_vals)
                row[g] = {
                    "recall20_mean": r_mean, "recall20_std": r_std,
                    "ndcg20_mean": n_mean, "ndcg20_std": n_std,
                    "n_seeds": len(r_vals),
                }
            for g in ("overall", "warm"):
                r_vals, n_vals = [], []
                for _seed, rel in ckpt_dirs.items():
                    d = json.loads((PROJECT_ROOT / rel / "final_test_group_metrics.json").read_text(encoding="utf-8"))
                    r_vals.append(d[g]["recall"]["20"])
                    n_vals.append(d[g]["ndcg"]["20"])
                r_mean, r_std = mean_std(r_vals)
                n_mean, n_std = mean_std(n_vals)
                row[g] = {
                    "recall20_mean": r_mean, "recall20_std": r_std,
                    "ndcg20_mean": n_mean, "ndcg20_std": n_std,
                    "n_seeds": len(r_vals),
                }
            result[ds][method] = row
    return result


def render_latex(table):
    """One \\begin{table} per dataset, Recall@20 (x1e-3) mean+-std, methods
    as rows, groups as columns, best in \\textbf{}, 2nd-best \\underline{}."""
    lines = []
    for ds in DATASETS:
        rank = {}
        for g in GROUPS:
            vals = sorted(((m, table[ds][m][g]["recall20_mean"]) for m in METHODS), key=lambda x: -x[1])
            rank[g] = {vals[0][0]: "best", vals[1][0]: "second"}

        lines.append(f"\n% ---- {ds} ----")
        lines.append(r"\begin{table}[h]")
        lines.append(r"\centering\scriptsize")
        lines.append(r"\setlength{\tabcolsep}{3pt}")
        lines.append(
            r"\caption{Recall@20 ($\times 10^{-3}$), mean$\pm$std over 3 seeds, "
            + DATASET_LABEL[ds]
            + r". \textbf{Bold} = best, \underline{underline} = 2nd best, per column.}"
            + rf"\label{{tab:b1-{DATASET_SHORT[ds]}}}"
        )
        lines.append(r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l" + "c" * len(GROUPS) + r"@{}}")
        lines.append(r"\toprule")
        lines.append("Method & " + " & ".join(GROUP_LABEL[g] for g in GROUPS) + r" \\")
        lines.append(r"\midrule")
        for meth in METHODS:
            cells = [meth]
            for g in GROUPS:
                m = table[ds][meth][g]["recall20_mean"] * 1000.0
                s = table[ds][meth][g]["recall20_std"] * 1000.0
                txt = f"{m:.3f}$\\pm${s:.3f}"
                status = rank[g].get(meth)
                if status == "best":
                    txt = r"\textbf{" + txt + "}"
                elif status == "second":
                    txt = r"\underline{" + txt + "}"
                cells.append(txt)
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular*}")
        lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--latex-only", action="store_true",
        help="Skip rebuilding the JSON; render LaTeX from an existing main_results_table.json.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    if args.latex_only:
        table = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
    else:
        if not MID_TAIL_FILE.is_file():
            raise SystemExit(
                f"{MID_TAIL_FILE} not found -- run "
                "`python -m tools.analysis.mid_tail_degree6_10` first."
            )
        mid_tail_doc = json.loads(MID_TAIL_FILE.read_text(encoding="utf-8"))
        table = build_table(mid_tail_doc)
        OUTPUT_JSON.write_text(json.dumps(table, indent=2), encoding="utf-8")
        print("Saved", OUTPUT_JSON)

    tex = render_latex(table)
    OUTPUT_TEX.write_text(tex, encoding="utf-8")
    print("Saved", OUTPUT_TEX)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
