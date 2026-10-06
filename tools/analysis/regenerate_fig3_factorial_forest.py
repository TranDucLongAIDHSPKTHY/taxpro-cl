"""Regenerate the factorial forest plot (main paper Figure 2): per-user bootstrap
95% CIs for both RQ5 conditional-effect comparisons, Near-Cold and Long-Tail, all
four datasets. Reads results/factorial_direction_bootstrap.json (the Amazon-Book
block holds the merge_t10 factorial; see tools/analysis/merge_factorial_results.py)
and takes the Yelp2018 block from results/factorial_direction_bootstrap_yelp.json
(V0-V3 trained with temperature_user=0.15, the main configuration's value).

Usage:
    python -m tools.analysis.regenerate_fig3_factorial_forest
Output: results/Fig3.pdf
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
ROOT_RESULTS = ROOT / "results"
DATASETS = [("amazon-book", "AB"), ("yelp2018", "Yelp"),
            ("musical-instruments", "MI"), ("arts-crafts-and-sewing", "ACS")]


def load_results():
    with open(ROOT_RESULTS / "factorial_direction_bootstrap.json") as f:
        base = json.load(f)
    with open(ROOT_RESULTS / "factorial_direction_bootstrap_yelp.json") as f:
        base["yelp2018"] = json.load(f)["yelp2018"]
    return base


def rows_for(base, panel_keys, panel_label_fn):
    rows = []
    for ds, ds_label in DATASETS:
        for key in panel_keys:
            for g, g_label in [("near_cold", "NC"), ("long_tail", "LT")]:
                s = base[ds][key][g]
                rows.append({
                    "label": f"{ds_label}  {panel_label_fn(key)}  {g_label}",
                    "mean": s["mean_diff"] * 1000,
                    "lo": s["ci95_lo"] * 1000,
                    "hi": s["ci95_hi"] * 1000,
                    "excl0": s["excludes_zero"],
                    "ds": ds,
                })
    return rows


def main():
    # Drawn at the printed width (31 pc = 5.15 in, the journal's text width), so the font sizes below are the printed
    # sizes. Filled versus open markers (not colour) mark whether an interval excludes zero, so the figure also reads in
    # greyscale; differences are plotted in Recall@20 x 10^3, the scale stated in the caption.
    base = load_results()
    panel_a = rows_for(base, ["V1_vs_V0", "V3_vs_V2"], lambda k: "V1−V0" if k == "V1_vs_V0" else "V3−V2")
    panel_b = rows_for(base, ["V2_vs_V0", "V3_vs_V1"], lambda k: "V2−V0" if k == "V2_vs_V0" else "V3−V1")

    fig, axes = plt.subplots(1, 2, figsize=(5.15, 2.85))

    for ax, rows, title in [
        (axes[0], panel_a, "(a) Direction\n(taxonomy − random)"),
        (axes[1], panel_b, "(b) Epsilon-adaptivity\n(adaptive − fixed $\\epsilon$)"),
    ]:
        n = len(rows)
        ys = list(range(n, 0, -1))
        for y, r in zip(ys, rows):
            ax.plot([r["lo"], r["hi"]], [y, y], color="black", linewidth=0.8, zorder=1)
            facecolor = "black" if r["excl0"] else "white"
            ax.scatter([r["mean"]], [y], marker="o", s=12, facecolor=facecolor,
                       edgecolor="black", linewidth=0.7, zorder=2)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.6, zorder=0)
        ax.set_yticks(ys)
        ax.set_yticklabels([r["label"] for r in rows], fontsize=6)
        ax.set_ylim(0.3, n + 0.7)
        ax.tick_params(axis="both", length=2, pad=1.5, labelsize=6)
        ax.set_xlabel("Recall@20 difference (×10$^{3}$)", fontsize=6.5, labelpad=2)
        ax.set_title(title, fontsize=7, fontweight="bold", pad=3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        # dataset separators
        prev_ds = None
        for y, r in zip(ys, rows):
            if prev_ds is not None and r["ds"] != prev_ds:
                ax.axhline(y + 0.5, color="gray", linewidth=0.5, linestyle=":")
            prev_ds = r["ds"]

    legend_elems = [
        plt.Line2D([0], [0], marker="o", color="black", markerfacecolor="black",
                   markersize=3.5, markeredgewidth=0.7, linestyle="", label="95% CI excludes 0"),
        plt.Line2D([0], [0], marker="o", color="black", markerfacecolor="white",
                   markersize=3.5, markeredgewidth=0.7, linestyle="", label="95% CI includes 0"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False, fontsize=6.5,
               handletextpad=0.3, columnspacing=1.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=1.0)

    out_path = ROOT_RESULTS / "Fig3.pdf"
    fig.savefig(out_path)
    print("Saved", out_path)


if __name__ == "__main__":
    main()
