"""Regenerate the paper's factorial-forest figure (per-user bootstrap 95% CI for
both RQ5 marginal comparisons). The Amazon-Book block of
results/factorial_direction_bootstrap_full_FIXED.json holds the merge_t10 factorial
(see tools/analysis/merge_a2_ab_into_result_jsons.py).

Originally written after the Yelp2018 checkpoint fix
(temperature_user 0.2 -> 0.15) changed one cell's significance status
(epsilon-adaptivity, V2-V0, Long-Tail: was significant-negative, now
non-significant). All other 31 cells are unchanged, sourced from
results/factorial_direction_bootstrap_full_FIXED.json exactly as the
original figure was (values independently verified to match the published
Table S16/S20 to the last digit before this regeneration).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
ROOT_RESULTS = ROOT / "results"

with open(ROOT_RESULTS / "factorial_direction_bootstrap_full_FIXED.json") as f:
    base = json.load(f)
with open(ROOT_RESULTS / "factorial_direction_bootstrap_yelp_tempuser_fix.json") as f:
    yelp_fixed = json.load(f)["yelp2018"]

# Patch in the corrected Yelp2018 cells (direction unaffected in headline sign,
# but epsilon-adaptivity's V2-V0 Long-Tail cell flips from significant to not).
base["yelp2018"] = yelp_fixed

DATASETS = [("amazon-book", "AB"), ("yelp2018", "Yelp"),
            ("musical-instruments", "MI"), ("arts-crafts-and-sewing", "ACS")]

def rows_for(panel_keys, panel_label_fn):
    rows = []
    for ds, ds_label in DATASETS:
        for key in panel_keys:
            for g, g_label in [("near_cold", "NC"), ("long_tail", "LT")]:
                s = base[ds][key][g]
                rows.append({
                    "label": f"{ds_label}, {panel_label_fn(key)}, {g_label}",
                    "mean": s["mean_diff"] * 1000,
                    "lo": s["ci95_lo"] * 1000,
                    "hi": s["ci95_hi"] * 1000,
                    "excl0": s["excludes_zero"],
                    "ds": ds,
                })
    return rows

panel_a = rows_for(["V1_vs_V0", "V3_vs_V2"], lambda k: "V1-V0" if k == "V1_vs_V0" else "V3-V2")
panel_b = rows_for(["V2_vs_V0", "V3_vs_V1"], lambda k: "V2-V0" if k == "V2_vs_V0" else "V3-V1")

fig, axes = plt.subplots(1, 2, figsize=(13, 8))

for ax, rows, title in [
    (axes[0], panel_a, "(a) Direction conditional effect\n(taxonomy $-$ random, V1$-$V0 and V3$-$V2)"),
    (axes[1], panel_b, "(b) Epsilon-adaptivity conditional effect\n(adaptive $-$ fixed $\\epsilon$, V2$-$V0 and V3$-$V1)"),
]:
    n = len(rows)
    ys = list(range(n, 0, -1))
    for y, r in zip(ys, rows):
        ax.plot([r["lo"], r["hi"]], [y, y], color="black", linewidth=1.2, zorder=1)
        marker = "o"
        facecolor = "black" if r["excl0"] else "white"
        ax.scatter([r["mean"]], [y], marker=marker, s=55, facecolor=facecolor,
                   edgecolor="black", linewidth=1.2, zorder=2)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, zorder=0)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=8.5)
    ax.set_ylim(0.3, n + 0.7)
    ax.set_xlabel("Recall@20 difference ($\\times 10^{-3}$)")
    ax.set_title(title, fontsize=10.5)
    # dataset separators
    prev_ds = None
    for y, r in zip(ys, rows):
        if prev_ds is not None and r["ds"] != prev_ds:
            ax.axhline(y + 0.5, color="gray", linewidth=0.5, linestyle=":")
        prev_ds = r["ds"]

legend_elems = [
    plt.Line2D([0], [0], marker="o", color="black", markerfacecolor="black",
               markersize=8, linestyle="", label="95% CI excludes 0"),
    plt.Line2D([0], [0], marker="o", color="black", markerfacecolor="white",
               markersize=8, linestyle="", label="95% CI includes 0"),
]
fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.03, 1, 1])

out_path = ROOT_RESULTS / "Fig3.pdf"
plt.savefig(out_path, bbox_inches="tight")
print("Saved", out_path)

# Report exactly what changed vs. the original figure for the audit trail.
print("\nCells where excludes_zero status differs from the pre-fix data (should be exactly 1: Yelp V2-V0 Long-Tail):")
old_yelp_v2v0_lt = {"mean_diff": -7.0e-05, "excludes_zero": True}  # published pre-fix value
new_yelp_v2v0_lt = yelp_fixed["V2_vs_V0"]["long_tail"]
print(f"  Yelp V2-V0 Long-Tail: OLD mean={old_yelp_v2v0_lt['mean_diff']:.6f} excl0={old_yelp_v2v0_lt['excludes_zero']}  "
      f"NEW mean={new_yelp_v2v0_lt['mean_diff']:.6f} excl0={new_yelp_v2v0_lt['excludes_zero']}")
