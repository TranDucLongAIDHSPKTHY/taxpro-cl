"""Descriptive summaries derived only from results already on disk.

No model is trained or scored here; the script reads two stored files:
  results/metrics_seed.csv                    per-seed group metrics of every run
  results/factorial_multiplicity.json    factorial bootstrap contrasts

(1) Decomposition of the SimGCL -> TaxPro-CL gap through the factorial's
    random-direction, fixed-epsilon control V0 (Recall@20, three-seed means):
        step 1  SimGCL (default)  -> V0        (every difference of Table 2 except
                                                the direction generator and
                                                epsilon-adaptivity, bundled)
        step 2  V0 -> complete configuration   (direction + epsilon-adaptivity)
(2) The factorial's Recall@20 contrasts re-expressed as a percentage of the
    control arm's mean, with the per-user bootstrap 95% CI printed in Tables
    S13/S17 (current sets; read from the JSON files behind them) or S13c/S13d
    (no_merge set; factorial_multiplicity.json) rescaled by the same
    control mean.

Output: results/factorial_decomposition_summary.json
Run   : python -m tools.analysis.factorial_decomposition_summary [--print-latex]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "results" / "metrics_seed.csv"
FACTORIAL = ROOT / "results" / "factorial_multiplicity.json"
OUT = ROOT / "results" / "factorial_decomposition_summary.json"

SEEDS = ["42", "0", "1"]
GROUPS = ["near_cold", "long_tail", "overall", "warm"]
HEADLINE = ["near_cold", "long_tail"]
MIN_GAP_FRACTION = 0.01  # a share of the gap is reported only if the gap is >= 1% of SimGCL

# metrics_seed.csv variant names of the factorial arms, mirroring
# CHECKPOINT_SETS in tools/analysis/factorial_multiplicity.py.
CURRENT = {
    "amazon-book": {"V0": "V0-mergedt10", "V1": "V1-mergedt10", "V2": "V2-mergedt10", "V3": "V3-warm20"},
    "yelp2018": {"V0": "V0-tempuser0.15", "V1": "V1-tempuser0.15", "V2": "V2-tempuser0.15", "V3": "V3-tempuser0.15"},
    "musical-instruments": {"V0": "V0", "V1": "V1", "V2": "V2", "V3": "V3"},
    "arts-crafts-and-sewing": {"V0": "V0", "V1": "V1", "V2": "V2", "V3": "V3"},
}
NOMERGE = {"amazon-book": {"V0": "V0", "V1": "V1", "V2": "V2", "V3": "V3"}}
CONTROL_OF = {
    "direction_fixed_eps__V1-V0": "V0",
    "direction_adaptive_eps__V3-V2": "V2",
    "eps_random_dir__V2-V0": "V0",
    "eps_taxonomy_dir__V3-V1": "V1",
}
LABEL = {"amazon-book": "AB", "yelp2018": "Yelp", "musical-instruments": "MI", "arts-crafts-and-sewing": "ACS"}
GROUP_LABEL = {"near_cold": "NC", "long_tail": "LT", "overall": "Overall", "warm": "Wm"}


def load_metrics():
    table = {}
    with open(METRICS, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            table[(row["dataset"], row["variant"], row["seed"], row["group"])] = float(row["recall_at_20"])
    return table


def seed_values(table, dataset, variant, group):
    return [table[(dataset, variant, s, group)] for s in SEEDS]


def mean_of(table, dataset, variant, group):
    return st.mean(seed_values(table, dataset, variant, group))


def decomposition(table):
    out = {}
    for dataset, arms in CURRENT.items():
        out[dataset] = {}
        for group in GROUPS:
            simgcl = mean_of(table, dataset, "SimGCL-main", group)
            v0 = mean_of(table, dataset, arms["V0"], group)
            main = mean_of(table, dataset, "TaxPro-CL-main", group)
            gap = main - simgcl
            share = (v0 - simgcl) / gap * 100.0 if abs(gap) >= MIN_GAP_FRACTION * simgcl else None
            per_seed = []
            for s in SEEDS:
                s_simgcl = table[(dataset, "SimGCL-main", s, group)]
                s_v0 = table[(dataset, arms["V0"], s, group)]
                s_main = table[(dataset, "TaxPro-CL-main", s, group)]
                s_gap = s_main - s_simgcl
                per_seed.append({
                    "seed": s,
                    "v0_vs_simgcl_pct": (s_v0 - s_simgcl) / s_simgcl * 100.0,
                    "main_vs_simgcl_pct": s_gap / s_simgcl * 100.0,
                    "share_pct": (s_v0 - s_simgcl) / s_gap * 100.0 if abs(s_gap) >= MIN_GAP_FRACTION * s_simgcl else None,
                })
            out[dataset][group] = {
                "simgcl": simgcl,
                "v0": v0,
                "main": main,
                "v0_vs_simgcl_pct": (v0 - simgcl) / simgcl * 100.0,
                "main_vs_v0_pct": (main - v0) / v0 * 100.0,
                "main_vs_simgcl_pct": gap / simgcl * 100.0,
                "share_of_gap_in_v0_pct": share,
                "per_seed": per_seed,
            }
    return out


RESULTS = ROOT / "results"
# Files behind the published Tables S13/S17 (Recall@20 intervals): Amazon-Book merge_t10,
# Yelp2018 (temperature_user=0.15), and Musical-Instruments / Arts-Crafts-and-Sewing.
PUBLISHED_AB = RESULTS / "factorial_amazon_book.json"
PUBLISHED_YELP = RESULTS / "factorial_direction_bootstrap_yelp.json"
PUBLISHED_OTHER = RESULTS / "factorial_direction_bootstrap.json"
PAIR_KEY = {
    "direction_fixed_eps__V1-V0": "V1_vs_V0",
    "direction_adaptive_eps__V3-V2": "V3_vs_V2",
    "eps_random_dir__V2-V0": "V2_vs_V0",
    "eps_taxonomy_dir__V3-V1": "V3_vs_V1",
}


def published_interval(dataset, comparison, group):
    """Interval exactly as printed in Tables S13/S17 (Recall@20, per-user bootstrap, 5000 resamples)."""
    if dataset == "amazon-book":
        cell = json.loads(PUBLISHED_AB.read_text(encoding="utf-8"))["comparisons"][comparison]["recall"][group]
    elif dataset == "yelp2018":
        cell = json.loads(PUBLISHED_YELP.read_text(encoding="utf-8"))["yelp2018"][PAIR_KEY[comparison]][group]
    else:
        cell = json.loads(PUBLISHED_OTHER.read_text(encoding="utf-8"))[dataset][PAIR_KEY[comparison]][group]
    return cell


def compatible_ranges(table, factorial):
    out = {}
    for set_name, sets in (("current", CURRENT), ("nomerge", NOMERGE)):
        out[set_name] = {}
        for dataset, arms in sets.items():
            cells = factorial["cells"][f"{dataset}|{set_name}"]
            out[set_name][dataset] = {}
            for comparison, control in CONTROL_OF.items():
                out[set_name][dataset][comparison] = {}
                for group in HEADLINE:
                    # p-values: sign-flip test of Table S13e; intervals: the ones printed in
                    # Tables S13/S17 (current sets) or S13c/S13d (no_merge, from the same JSON as p).
                    p_cell = cells[f"{comparison}|recall|{group}"]
                    cell = published_interval(dataset, comparison, group) if set_name == "current" else p_cell
                    control_mean = mean_of(table, dataset, arms[control], group)
                    out[set_name][dataset][comparison][group] = {
                        "control_arm": control,
                        "control_mean": control_mean,
                        "mean_diff_pct": cell["mean_diff"] / control_mean * 100.0,
                        "ci95_lo_pct": cell["ci95_lo"] / control_mean * 100.0,
                        "ci95_hi_pct": cell["ci95_hi"] / control_mean * 100.0,
                        "excludes_zero": bool(cell["excludes_zero"]),
                        "p_signflip_1e5_seed42": p_cell["p_signflip_1e5_seed42"],
                    }
    return out


def tex_num(x, digits=2, plus=True):
    text = f"{abs(x):.{digits}f}"
    if float(text) == 0.0:
        return text
    if x < 0:
        return "$-$" + text
    return ("+" if plus else "") + text


def print_latex(dec, rng):
    print("% ---- Table S32 rows: Dataset & Group & SimGCL & V0 & Complete & V0 vs SimGCL & Complete vs V0 & Complete vs SimGCL & Share")
    for dataset in CURRENT:
        for group in GROUPS:
            d = dec[dataset][group]
            share = "--" if d["share_of_gap_in_v0_pct"] is None else f"{d['share_of_gap_in_v0_pct']:.1f}\\%"
            print(
                f"{LABEL[dataset]} & {GROUP_LABEL[group]} & {d['simgcl']*1e3:.3f} & {d['v0']*1e3:.3f} & {d['main']*1e3:.3f} & "
                f"{tex_num(d['v0_vs_simgcl_pct'])}\\% & {tex_num(d['main_vs_v0_pct'])}\\% & {tex_num(d['main_vs_simgcl_pct'])}\\% & {share} \\\\"
            )
    print("% ---- Table S33 rows: Dataset & Contrast & NC & LT")
    names = {
        "direction_fixed_eps__V1-V0": "V1$-$V0",
        "direction_adaptive_eps__V3-V2": "V3$-$V2",
        "eps_random_dir__V2-V0": "V2$-$V0",
        "eps_taxonomy_dir__V3-V1": "V3$-$V1",
    }
    for set_name, label_suffix in (("current", ""), ("nomerge", " (no\\_merge)")):
        for dataset, comps in rng[set_name].items():
            for comparison, per_group in comps.items():
                cols = []
                for group in HEADLINE:
                    c = per_group[group]
                    cols.append(f"{tex_num(c['mean_diff_pct'])} [{tex_num(c['ci95_lo_pct'])}, {tex_num(c['ci95_hi_pct'])}]")
                print(f"{LABEL[dataset]}{label_suffix} & {names[comparison]} & {cols[0]} & {cols[1]} \\\\")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-latex", action="store_true")
    args = parser.parse_args()
    table = load_metrics()
    factorial = json.loads(FACTORIAL.read_text(encoding="utf-8"))
    dec = decomposition(table)
    rng = compatible_ranges(table, factorial)
    payload = {
        "description": (
            "Recall@20 three-seed means. step 1 = V0 vs SimGCL (all Table-2 differences except direction and "
            "epsilon-adaptivity, bundled); step 2 = complete configuration vs V0. compatible_ranges = stored "
            "bootstrap contrasts as percent of the control arm's mean; intervals condition on the trained "
            "checkpoints (user resampling only)."
        ),
        "min_gap_fraction_for_share": MIN_GAP_FRACTION,
        "decomposition": dec,
        "compatible_ranges": rng,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", OUT)
    if args.print_latex:
        print_latex(dec, rng)


if __name__ == "__main__":
    main()
