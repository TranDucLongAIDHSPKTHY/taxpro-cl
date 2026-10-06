"""Regenerate the two result figures that summarize the main comparison.

Figure 1 of the main paper (file Fig2.pdf: Near-Cold / Long-Tail Recall@20 bars,
three-seed mean +- 1 sample SD) reads results/main_results_table.json.

Figure S2 of Online Resource 1 (file Fig4.pdf: Pareto view of Overall vs.
Long-Tail Recall@20) reads results/beyond_accuracy.json, i.e. one representative checkpoint per
method and dataset, the same basis as the coverage / share diagnostics next to
it (ESM Section S10).

Usage:
    python -m tools.analysis.plot_main_figures
    python -m tools.analysis.plot_main_figures --out-dir <figure directory>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MAIN_TABLE = ROOT / "results" / "main_results_table.json"
BEYOND = ROOT / "results" / "beyond_accuracy.json"

DATASETS = ["amazon-book", "yelp2018", "musical-instruments", "arts-crafts-and-sewing"]
TITLES = ["(a) Amazon-Book", "(b) Yelp2018", "(c) Musical-Instruments", "(d) Arts-Crafts-and-Sewing"]
METHODS = ["LightGCN", "SGL-ED", "SimGCL", "XSimGCL", "NCL", "TaxPro-CL"]
# beyond_accuracy.json names the SGL baseline "SGL".
BEYOND_NAME = {"SGL-ED": "SGL"}
SHADES = ["#c8c8c8", "#a0a0a0", "#808080", "#606060", "#404040", "#000000"]
HATCHES = ["/", "\\", "x", ".", "o", ""]
MARKERS = ["o", "s", "^", "D", "v", "*"]
SCALE = 1e3


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def _value_label(v):
    return f"{v:.1f}" if v >= 10 else f"{v:.2f}"


def figure_bars(out_path):
    # Drawn at the printed width (31 pc = 5.15 in, the journal's text width), so the font sizes below are the printed
    # sizes. Methods are identified by shade, hatch, and the legend; each bar carries its mean, and every panel keeps its
    # own vertical scale (stated in the caption).
    table = json.loads(MAIN_TABLE.read_text(encoding="utf-8"))
    titles = ["(a) Amazon-Book", "(b) Yelp2018", "(c) Musical-\nInstruments", "(d) Arts-Crafts-\nand-Sewing"]
    fig, axes = plt.subplots(2, 4, figsize=(5.15, 3.0))
    for row, (group, label) in enumerate((("near_cold", "Near-Cold"), ("long_tail", "Long-Tail"))):
        for col, dataset in enumerate(DATASETS):
            ax = axes[row][col]
            means = [table[dataset][m][group]["recall20_mean"] * SCALE for m in METHODS]
            stds = [table[dataset][m][group]["recall20_std"] * SCALE for m in METHODS]
            ax.bar(range(len(METHODS)), means, yerr=stds, color=SHADES, edgecolor="black", width=0.78,
                   linewidth=0.5, error_kw={"elinewidth": 0.6, "capsize": 1.2, "capthick": 0.6})
            for patch, hatch in zip(ax.patches, HATCHES):
                patch.set_hatch(hatch)
            top = max(m + s for m, s in zip(means, stds))
            for i, (m, s) in enumerate(zip(means, stds)):
                ax.text(i, m + s + 0.02 * top, _value_label(m), rotation=90, ha="center", va="bottom", fontsize=5.5)
            ax.set_ylim(0, top * 1.42 if top > 0 else 1)
            ax.set_xticks([])
            ax.tick_params(axis="y", labelsize=6.5, length=2, pad=1.5)
            _style(ax)
            if row == 0:
                ax.set_title(titles[col], fontsize=7.5, fontweight="bold", pad=3)
            if col == 0:
                ax.set_ylabel(label + "\nRecall@20 (×10$^{3}$)", fontsize=7)
    handles = [Patch(facecolor=c, edgecolor="black", hatch=h, label=m, linewidth=0.5)
               for c, h, m in zip(SHADES, HATCHES, METHODS)]
    fig.legend(handles=handles, loc="lower center", ncol=len(METHODS), frameon=False, fontsize=6.5,
               handlelength=1.6, columnspacing=0.9, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=0.6, w_pad=0.5)
    fig.savefig(out_path)
    plt.close(fig)


def figure_pareto(out_path):
    rows = json.loads(BEYOND.read_text(encoding="utf-8"))
    index = {(r["dataset"], r["model"]): r for r in rows}
    fig, axes = plt.subplots(2, 2, figsize=(6.92, 6.42))
    for ax, dataset, title in zip(axes.ravel(), DATASETS, TITLES):
        for method, marker, shade in zip(METHODS, MARKERS, SHADES):
            row = index[(dataset, BEYOND_NAME.get(method, method))]
            star = marker == "*"
            ax.scatter(row["overall_recall20"] * SCALE, row["long_tail_recall20"] * SCALE,
                       marker=marker, s=120 if star else 55, c="black" if star else "#505050",
                       edgecolors="black", linewidths=0.6, zorder=3)
        ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
        ax.set_xlabel("Overall Recall@20 (×10$^{-3}$)", fontsize=9)
        ax.set_ylabel("Long-Tail Recall@20 (×10$^{-3}$)", fontsize=9)
        ax.tick_params(labelsize=8)
        _style(ax)
        ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.7)
    handles = [Line2D([], [], marker=m, linestyle="", markersize=8 if m == "*" else 6,
                      markerfacecolor="black" if m == "*" else "#505050", markeredgecolor="black", label=n)
               for m, n in zip(MARKERS, METHODS)]
    fig.legend(handles=handles, loc="lower center", ncol=len(METHODS), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figure_bars(args.out_dir / "Fig2.pdf")
    figure_pareto(args.out_dir / "Fig4.pdf")
    print("Wrote", args.out_dir / "Fig2.pdf", "and", args.out_dir / "Fig4.pdf")


if __name__ == "__main__":
    main()
