"""Figure 1 of the main paper: conditional effects of the direction and magnitude
factors on Recall@20 for Near-Cold and Long-Tail items, as per-user bootstrap 95% CIs
(uncorrected) of the seed-averaged V0-V3 contrasts. Forty contrasts: the 32 development
contrasts (four datasets x two contrasts x Near-Cold/Long-Tail per factor; Online
Resource 1, Tables S13 and S17) and the 8 Office-Products contrasts (Table S36c).

Reads the existing analysis outputs and draws them; nothing is re-estimated:
- results/factorial_direction_bootstrap.json (Amazon-Book block: merge_t10; see
  tools/analysis/merge_factorial_results.py), with the Yelp2018 block taken from
  results/factorial_direction_bootstrap_yelp.json (V0-V3 trained with
  temperature_user=0.15, the main configuration's value);
- results/confirmatory/confirmatory_analysis_office-products.json ("factorial").

Panel (a): prototype minus random direction (V1-V0, V3-V2). Panel (b): degree-adaptive
minus degree-independent magnitude (V2-V0, V3-V1). Both panels share one x scale, in
Recall@20 x 10^3. Circles mark Near-Cold and squares Long-Tail; filled markers mark
intervals that exclude zero, and gray markers contrasts that are zero for every
evaluated user (interval [0, 0]). Each dataset has a header row; the Office-Products
rows are shaded.

Usage:
    python -m tools.analysis.plot_figure1_factorial_forest
Output: results/Fig1.pdf and results/figure1_factorial_contrasts.csv (the 40 plotted values)
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
ROOT_RESULTS = ROOT / "results"
DEV_DATASETS = [("amazon-book", "AB"), ("yelp2018", "Yelp"),
                ("musical-instruments", "MI"), ("arts-crafts-and-sewing", "ACS")]
GROUPS = [("near_cold", "NC"), ("long_tail", "LT")]
FACTORS = [
    ("direction", ["V1-V0", "V3-V2"], "a", "Table S13"),
    ("magnitude", ["V2-V0", "V3-V1"], "b", "Table S17"),
]
OFFICE_KEYS = {"direction": "direction", "magnitude": "epsilon_adaptivity"}
FIELDS = ["dataset", "factor", "contrast", "group", "users", "mean", "ci_low", "ci_high", "source_table"]


def load_rows():
    with open(ROOT_RESULTS / "factorial_direction_bootstrap.json", encoding="utf-8") as f:
        dev = json.load(f)
    with open(ROOT_RESULTS / "factorial_direction_bootstrap_yelp.json", encoding="utf-8") as f:
        dev["yelp2018"] = json.load(f)["yelp2018"]
    with open(ROOT_RESULTS / "confirmatory" / "confirmatory_analysis_office-products.json", encoding="utf-8") as f:
        office = json.load(f)["factorial"]

    rows = []
    for factor, contrasts, _, source in FACTORS:
        for ds, ds_label in DEV_DATASETS:
            for contrast in contrasts:
                key = contrast.replace("-", "_vs_")
                for group, g_label in GROUPS:
                    s = dev[ds][key][group]
                    rows.append({"dataset": ds_label, "factor": factor, "contrast": contrast, "group": g_label,
                                 "users": s["n_users"], "mean": s["mean_diff"], "ci_low": s["ci95_lo"],
                                 "ci_high": s["ci95_hi"], "source_table": source})
        by_group = {cell["group"]: cell for cell in office[OFFICE_KEYS[factor]]}
        for contrast in contrasts:
            for group, g_label in GROUPS:
                s = by_group[group]["contrasts"][contrast]
                rows.append({"dataset": "Office", "factor": factor, "contrast": contrast, "group": g_label,
                             "users": s["n"], "mean": s["mean_diff"], "ci_low": s["ci95"][0],
                             "ci_high": s["ci95"][1], "source_table": "Table S36c"})
    if len(rows) != 40:
        raise ValueError(f"expected 40 contrasts, got {len(rows)}")
    for r in rows:
        if not r["ci_low"] <= r["mean"] <= r["ci_high"]:
            raise ValueError(f"point estimate outside its interval: {r}")
    return rows


DATASET_ORDER = ["AB", "Yelp", "MI", "ACS", "Office"]
PANELS = [("direction", ["V1-V0", "V3-V2"], "(a) Direction\nprototype − random"),
          ("magnitude", ["V2-V0", "V3-V1"], "(b) Magnitude\ndegree-adaptive − independent")]
INK = "#1f3b5c"
ZERO = "#8a96a3"      # contrasts that are zero for every evaluated user
SHADE = "#e9edf2"     # Office-Products rows


def draw(rows, stem, formats=("pdf",)):
    """Draws the 40 contrasts of load_rows() into <stem>.<format> for each format; returns the paths."""
    scaled = [{**r, **{k: 1000 * r[k] for k in ("mean", "ci_low", "ci_high")}} for r in rows]
    low = min(r["ci_low"] for r in scaled)
    high = max(r["ci_high"] for r in scaled)
    pad = 0.06 * (high - low)
    xlim = (low - pad, high + pad)
    style = {"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 7.6}
    with plt.rc_context(style):
        # The height keeps the figure and its caption within one top float of the journal class.
        fig, axes = plt.subplots(1, 2, figsize=(5.0, 5.45), sharex=True)
        for ax, (factor, contrasts, title) in zip(axes, PANELS):
            y, ticks, labels = 0, [], []
            for dataset in DATASET_ORDER:
                top = y
                # Header in the label column, right-aligned like the tick labels, clear of the zero line.
                ax.annotate(dataset, xy=(0, y), xycoords=("axes fraction", "data"), xytext=(-2, 0),
                            textcoords="offset points", fontsize=7.6, fontweight="bold", va="center", ha="right",
                            annotation_clip=False)
                y += 1
                for contrast in contrasts:
                    for group, marker in (("NC", "o"), ("LT", "s")):
                        r = next(r for r in scaled if r["dataset"] == dataset and r["factor"] == factor
                                 and r["contrast"] == contrast and r["group"] == group)
                        excludes = r["ci_low"] > 0 or r["ci_high"] < 0
                        zero = r["ci_low"] == r["ci_high"] == 0
                        if zero and (excludes or r["mean"] != 0):
                            raise ValueError(f"inconsistent zero contrast: {r}")
                        if not zero:
                            ax.plot([r["ci_low"], r["ci_high"]], [y, y], color=INK, linewidth=1.0,
                                    solid_capstyle="butt")
                        ax.plot(r["mean"], y, marker=marker, markersize=4.0, markeredgewidth=0.9,
                                color=ZERO if zero else INK, markerfacecolor=INK if excludes else "white",
                                linestyle="none", zorder=3)
                        ticks.append(y)
                        labels.append(f"{contrast.replace('-', chr(8722))}  {group}")
                        y += 1
                if dataset == "Office":
                    ax.axhspan(top - 0.5, y - 0.5, color=SHADE, zorder=0, linewidth=0)
                y += 0.35
            ax.axvline(0, color="#555555", linewidth=0.7, linestyle="--", zorder=1)
            ax.set_yticks(ticks, labels, fontsize=7.2)
            ax.set_ylim(y - 0.6, -0.8)
            ax.set_xlim(*xlim)
            ax.set_title(title, fontsize=7.8, loc="left", pad=4)
            ax.tick_params(axis="y", length=0, pad=2)
            ax.tick_params(axis="x", labelsize=7.2, length=2.5)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.grid(axis="x", color="#dddddd", linewidth=0.5)
            ax.set_axisbelow(True)
            ax.set_xlabel("Recall@20 difference (×10$^{3}$)", fontsize=7.4, labelpad=2)
        legend = [Line2D([], [], marker="o", color=INK, markerfacecolor="white", linestyle="none", markersize=4,
                         label="NC"),
                  Line2D([], [], marker="s", color=INK, markerfacecolor="white", linestyle="none", markersize=4,
                         label="LT"),
                  Line2D([], [], marker="o", color=INK, markerfacecolor=INK, linestyle="none", markersize=4,
                         label="95% interval excludes 0"),
                  Line2D([], [], marker="o", color=ZERO, markerfacecolor="white", linestyle="none", markersize=4,
                         markeredgewidth=0.9, label="Zero for every user")]
        fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, fontsize=7.2,
                   bbox_to_anchor=(0.55, 0.0), columnspacing=1.2, handletextpad=0.3)
        fig.tight_layout(rect=(0, 0.035, 1, 1), w_pad=1.2)
        # Every title, label and the legend must lie inside the figure; savefig does not crop to them.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for artist in [*axes, *fig.legends]:
            box = artist.get_tightbbox(renderer)
            if box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.width or box.y1 > fig.bbox.height:
                raise ValueError(f"{artist} extends beyond the figure: {box}")
        stem.parent.mkdir(parents=True, exist_ok=True)
        paths = []
        for ext in formats:
            path = stem.with_suffix("." + ext)
            fig.savefig(path)
            paths.append(path)
        plt.close(fig)
    return paths


def main():
    rows = load_rows()
    csv_path = ROOT_RESULTS / "figure1_factorial_contrasts.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({**r, **{k: f"{r[k]:.10g}" for k in ("mean", "ci_low", "ci_high")}})

    out_path, = draw(rows, ROOT_RESULTS / "Fig1")
    print("Saved", out_path, "and", csv_path)


if __name__ == "__main__":
    main()
