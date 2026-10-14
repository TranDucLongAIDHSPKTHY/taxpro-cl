"""Figure 1 of the main paper: conditional effects of the direction and magnitude
factors on sparse-item Recall@20, as per-user bootstrap 95% CIs (uncorrected) of
the seed-averaged V0-V3 contrasts. Forty contrasts: the 32 development contrasts
(four datasets x two contrasts x Near-Cold/Long-Tail per factor; Online Resource 1,
Tables S13 and S17) and the 8 Office-Products contrasts (Table S36c).

Reads the existing analysis outputs and draws them; nothing is re-estimated:
- results/factorial_direction_bootstrap.json (Amazon-Book block: merge_t10; see
  tools/analysis/merge_factorial_results.py), with the Yelp2018 block taken from
  results/factorial_direction_bootstrap_yelp.json (V0-V3 trained with
  temperature_user=0.15, the main configuration's value);
- results/confirmatory/confirmatory_analysis_office-products.json ("factorial").

Panel (a): prototype minus random direction (V1-V0, V3-V2). Panel (b): degree-adaptive
minus degree-independent magnitude (V2-V0, V3-V1). Differences in Recall@20 x 10^3;
squares mark the Office-Products rows, below the dotted separator. Each panel is
drawn as its own plot and the two vector pages are placed side by side (PyMuPDF).

Usage:
    python -m tools.analysis.plot_figure1_factorial_forest
Output: results/Fig1.pdf and results/figure1_factorial_contrasts.csv (the 40 plotted values)
"""
import csv
import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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


def draw_panel(rows, letter, path):
    fig, ax = plt.subplots(figsize=(2.65, 5.2))
    for y, r in enumerate(rows):
        value, low, high = (1000 * r[k] for k in ("mean", "ci_low", "ci_high"))
        ax.errorbar(value, y, xerr=[[value - low], [high - value]],
                    fmt="s" if r["dataset"] == "Office" else "o",
                    markersize=3.3, capsize=2, linewidth=0.9)
    ax.axvline(0, linestyle="--", linewidth=0.7)
    ax.axhline(15.5, linestyle=":", linewidth=0.8)
    ax.set_yticks(range(len(rows)), [f'{r["dataset"]} {r["contrast"]} {r["group"]}' for r in rows], fontsize=8.4)
    ax.invert_yaxis()
    ax.set_xlabel(r"Recall@20 difference ($\times10^3$)", fontsize=8.5)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.tick_params(axis="x", labelsize=8.4)
    ax.text(0.02, 1.02, f"({letter})", transform=ax.transAxes, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.4)
    # The x-axis label is centred under the axes; in a panel with short tick labels it can run past the right edge of
    # the page. Shift it left by the overflow so that the whole label stays inside the panel.
    fig.canvas.draw()
    label = ax.xaxis.label
    box = label.get_window_extent()
    margin = 2.0
    overflow = box.x1 - (fig.bbox.width - margin)
    if overflow > 0:
        x_disp, y_disp = (box.x0 + box.x1) / 2 - overflow, (box.y0 + box.y1) / 2
        x_ax, y_ax = ax.transAxes.inverted().transform((x_disp, y_disp))
        ax.xaxis.set_label_coords(x_ax, y_ax)
        label.set_verticalalignment("center")
    fig.savefig(path)
    plt.close(fig)


def main():
    import pymupdf

    rows = load_rows()
    csv_path = ROOT_RESULTS / "figure1_factorial_contrasts.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({**r, **{k: f"{r[k]:.10g}" for k in ("mean", "ci_low", "ci_high")}})

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.family"] = "DejaVu Sans"
    order = {"AB": 0, "Yelp": 1, "MI": 2, "ACS": 3, "Office": 4}
    with tempfile.TemporaryDirectory() as tmp:
        panel_paths = []
        for factor, contrasts, letter, _ in FACTORS:
            panel = sorted((r for r in rows if r["factor"] == factor),
                           key=lambda r: (order[r["dataset"]], contrasts.index(r["contrast"]), r["group"] != "NC"))
            path = Path(tmp) / f"panel_{letter}.pdf"
            draw_panel(panel, letter, path)
            panel_paths.append(path)
        out = pymupdf.open()
        panels = [pymupdf.open(p) for p in panel_paths]
        page = out.new_page(width=sum(p[0].rect.width for p in panels),
                            height=max(p[0].rect.height for p in panels))
        left = 0.0
        for panel in panels:
            width, height = panel[0].rect.width, panel[0].rect.height
            page.show_pdf_page(pymupdf.Rect(left, 0, left + width, height), panel, 0)
            left += width
        out_path = ROOT_RESULTS / "Fig1.pdf"
        out.save(out_path)
        out.close()
        for panel in panels:
            panel.close()
    print("Saved", out_path, "and", csv_path)


if __name__ == "__main__":
    main()
