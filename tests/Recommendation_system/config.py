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

# Run-family directories whose name starts with one of these prefixes hold
# exploratory / budget-matching sweeps (the A7 comparable-budget grid, which is
# reported as supplementary context only). They are never candidates for the
# canonical checkpoint that represents a (model, dataset) pair in the paper's
# tables, so re-running the sweeps inside the same output tree cannot change a
# published number (see checkpoint_selection.find_candidate_runs).
EXPLORATORY_FAMILY_PREFIXES = ("A7-",)

# Pin the run FAMILY (the directory holding seed0/seed1/seed42) when a baseline
# has more than one non-exploratory family. The seed inside the family is still
# chosen by checkpoint_selection (highest validation Overall Recall@20).
#   SimGCL / musical-instruments: simgcl-bb2de4fab275 is an earlier run with the
#   framework's 50-epoch default cap (training_epochs=50), superseded by the
#   200-epoch family used for every reported SimGCL number.
CHECKPOINT_FAMILY_OVERRIDES = {
    ("SimGCL", "musical-instruments"): P0_BASELINE_OUTPUT_DIR / "SimGCL" / "musical-instruments" / "simgcl-f263d2bff497",
}

# Explicit escape hatch: {(model_name, dataset_name): Path(run_dir)}.
# Overrides auto-selection entirely for that pair. Needed whenever the
# reported "best" run is not the one with the highest plain Overall
# validation score -- auto-selection (checkpoint_selection.py) only looks at
# Overall recall@selection_K, but TaxPro-CL is deliberately tuned to trade a
# little Overall for much better near_cold/long_tail. Confirmed for
# amazon-book: the auto-picked "ablation-A4-temperature0.2" scores higher on
# plain Overall (0.1254 vs the pinned run's 0.1247) but is far worse on
# near_cold (0.00085 vs 0.00213) and long_tail (0.00937 vs 0.01259) -- the
# pinned run is the paper's actually-reported result, so it is pinned here
# explicitly.
CHECKPOINT_OVERRIDES = {
    ("TaxPro-CL", "amazon-book"): (
        P0_TAXPROCL_OUTPUT_DIR
        / "amazon-book"
        / "taxpro-cl-v15-prototype-leaf-lambda0.5-same_leaf_weight0-user_ssl-warmstart20-noblend-temp0.1-DONE-overall-2.30pct-BEST-overall-nearcold-longtail-positive"
        / "seed0"
    ),
    # This configuration (temperature=0.15, gamma_cold=5.0) and the
    # auto-picked alternative (higher plain Overall, 0.1637 vs 0.1634) are
    # near-identical on near_cold (0.00779 both), but this is the
    # configuration reported against baselines, not just the
    # highest-Overall sweep entry.
    ("TaxPro-CL", "musical-instruments"): (
        P0_TAXPROCL_OUTPUT_DIR
        / "musical-instruments"
        / "taxpro-cl-FINAL-no_merge-temp0.15-gammacold5.0"
        / "seed42"
    ),
    # temperature_user=0.15. Auto-selection would otherwise pick a
    # higher-plain-Overall cluster (Overall sel=0.1139, near_cold=0.00093,
    # long_tail=0.00457) -- this pinned run trades a little Overall
    # (sel=0.1119) for 2x near_cold (0.00186) and 1.5x long_tail (0.00690),
    # the same Overall-vs-group tradeoff pattern found for amazon-book.
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
    # which averages all 3 seeds regardless of this override). This keeps
    # the rank-level Table 8/9 comparing TaxPro-CL and SimGCL under the
    # same training seed for this dataset.
    ("SimGCL", "yelp2018"): (
        P0_BASELINE_OUTPUT_DIR / "SimGCL" / "yelp2018" / "simgcl-20c63ca8d16c" / "seed1"
    ),
    # same_leaf_weight=0.0 is the paper's locked main config -- beta=0
    # (exact standard InfoNCE) is uniform across all 4 datasets. The
    # same_leaf_weight=0.4 configuration (the previous main config) moved to
    # ablation A6_same_leaf_weight_ACS. This configuration otherwise matches
    # that one (temperature=0.125, gamma_cold=1.5, no_merge -- verified via
    # config_resolved.json diff, only same_leaf_weight differs).
    ("TaxPro-CL", "arts-crafts-and-sewing"): (
        P0_TAXPROCL_OUTPUT_DIR / "arts-crafts-and-sewing" / "taxpro-cl-FINAL-no_merge-temp0.125-gammacold1.5-sameleaf0" / "seed42"
    ),
}

DEFAULT_BATCH_SIZE = 256
