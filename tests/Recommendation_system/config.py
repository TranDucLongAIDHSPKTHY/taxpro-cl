"""User-editable selection knobs for the taxonomy rank-impact pipeline.

Edit the lists below to change what a bare `run_comparison.py` invocation
does, or override any of them from the command line (see run_comparison.py
--help). TAXONOMY_MODEL is fixed: TaxPro-CL is always the "with taxonomy"
side of every comparison.
"""

from config_path.config_path import P0_BASELINE_OUTPUT_DIR, P0_TAXPROCL_OUTPUT_DIR

DATASETS = [
    "amazon-book",
    "yelp2018",
    "arts-crafts-and-sewing",
    "musical-instruments",
]

WITHOUT_TAXONOMY_MODELS = [
    "LightGCN",
    "SimGCL",
    "SGL",
    "NCL",
]

TAXONOMY_MODEL = "TaxPro-CL"

K_VALUES = [10, 20]

# Preferred seed when a (model, dataset) has several equally-good candidate
# checkpoints (see checkpoint_selection.select_checkpoint).
DEFAULT_SEED_PREFERENCE = 42

# Explicit escape hatch: {(model_name, dataset_name): Path(run_dir)}.
# Overrides auto-selection entirely for that pair. Needed whenever the
# project's actual "best" run is not the one with the highest plain Overall
# validation score -- auto-selection (checkpoint_selection.py) only looks at
# Overall recall@selection_K, but TaxPro-CL is deliberately tuned to trade a
# little Overall for much better near_cold/long_tail (see project memory
# project_taxprocl_v15_breakthrough.md). Confirmed for amazon-book: the
# auto-picked "ablation-A4-temperature0.2" scores higher on plain Overall
# (0.1254 vs v15's 0.1247) but is far worse on near_cold (0.00085 vs
# 0.00213) and long_tail (0.00937 vs 0.01259) -- v15 is the project's
# actually-locked result, so it is pinned here explicitly.
CHECKPOINT_OVERRIDES = {
    ("TaxPro-CL", "amazon-book"): (
        P0_TAXPROCL_OUTPUT_DIR
        / "amazon-book"
        / "taxpro-cl-v15-prototype-leaf-lambda0.5-same_leaf_weight0-user_ssl-warmstart20-noblend-temp0.1-DONE-overall-2.30pct-BEST-overall-nearcold-longtail-positive"
        / "seed0"
    ),
    # v7 (temperature=0.15, gamma_cold=5.0) matches project memory
    # ("Musical-Instruments THẮNG 8/8"); confirmed it and auto-picked v5
    # (higher plain Overall, 0.1637 vs 0.1634) are near-identical on
    # near_cold (0.00779 both) but v7 is the campaign's declared winner
    # against baselines, not just the highest-Overall sweep entry.
    ("TaxPro-CL", "musical-instruments"): (
        P0_TAXPROCL_OUTPUT_DIR
        / "musical-instruments"
        / "taxpro-cl-FINAL-no_merge-temp0.15-gammacold5.0"
        / "seed42"
    ),
    # v12 (temperature_user=0.15) matches project memory ("v12 sau đó
    # +19,21%"). Confirmed: auto-selection would otherwise pick the
    # v3/v5/v6/v8/v10/v11/v14 cluster (Overall sel=0.1139, near_cold=0.00093,
    # long_tail=0.00457) -- v12/seed1 trades a little Overall (sel=0.1119)
    # for 2x near_cold (0.00186) and 1.5x long_tail (0.00690), the same
    # Overall-vs-group tradeoff pattern found for amazon-book's v15.
    ("TaxPro-CL", "yelp2018"): (
        P0_TAXPROCL_OUTPUT_DIR / "yelp2018" / "taxpro-cl-FINAL-no_merge-temp0.125-tempuser0.15-gammacold1.5" / "seed1"
    ),
    # Pin SimGCL/yelp2018 to seed1 to MATCH TaxPro-CL's pinned seed above.
    # Auto-selection would otherwise pick seed42 (highest validation
    # selection_value among seed0/seed1/seed42 for the single
    # simgcl-20c63ca8d16c family -- there is no competing family here, so
    # this override only changes WHICH of the 3 already-averaged seeds is
    # treated as "the" single representative checkpoint for single-pair
    # tools like the rank-level comparison; it does not affect main_results,
    # which averages all 3 seeds regardless of this override). Fixed
    # 2026-09-05 after review flagged this as the only dataset where the
    # rank-level Table 8/9 compared TaxPro-CL and SimGCL under different
    # training seeds.
    ("SimGCL", "yelp2018"): (
        P0_BASELINE_OUTPUT_DIR / "SimGCL" / "yelp2018" / "simgcl-20c63ca8d16c" / "seed1"
    ),
    # v2 (same_leaf_weight=0.0) is the paper's locked main config as of
    # 2026-09-05 -- beta=0 (exact standard InfoNCE) is now uniform across all
    # 4 datasets. v6 (same_leaf_weight=0.4, the PREVIOUS main config) moved to
    # ablation A6_same_leaf_weight_ACS. v2 originally had only seed42; seed0
    # and seed1 were trained fresh 2026-09-05 to complete the 3-seed set
    # (identical config otherwise to v6: temperature=0.125, gamma_cold=1.5,
    # no_merge -- verified via config_resolved.json diff, only
    # same_leaf_weight differs). Also supersedes an earlier note about v5
    # (same_leaf_weight=0.5, single-seed-only, never reported) being wrongly
    # auto-selected -- that risk is now moot since v2 is pinned explicitly.
    ("TaxPro-CL", "arts-crafts-and-sewing"): (
        P0_TAXPROCL_OUTPUT_DIR / "arts-crafts-and-sewing" / "taxpro-cl-FINAL-no_merge-temp0.125-gammacold1.5-sameleaf0" / "seed42"
    ),
}

DEFAULT_BATCH_SIZE = 256
