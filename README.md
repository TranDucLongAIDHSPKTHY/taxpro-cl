# TaxPro-CL

> Taxonomy-guided perturbation for sparse-item recommendation in graph contrastive learning.

TaxPro-CL is a research codebase for studying taxonomy-aware item
perturbations in graph collaborative filtering. It keeps SimGCL's LightGCN
encoder and two-view InfoNCE training and changes only the item-side
perturbation: each item is pushed toward the prototype of its product-taxonomy
leaf, with a magnitude scaled by its training degree. Every run uses
validation-based checkpoint selection and a single full-catalog test
evaluation.

## Overview

Long-tail recommendation is difficult because low-degree items have limited
collaborative evidence. TaxPro-CL takes each item's leaf category from external
product metadata; leaf merging and the validity mask use training interactions
only. Items without a valid leaf remain in the BPR objective and the evaluation
catalog, but receive no taxonomy direction and are excluded from the item-side
InfoNCE term.

This implementation provides:

- A TaxPro-CL model built on a LightGCN backbone, using the same 2-view
  self-supervised structure as SimGCL (one clean forward pass for BPR, two
  independent perturbed passes for InfoNCE) but replacing SimGCL's random, sign-aligned
  perturbation direction with a taxonomy-prototype-guided direction and a
  degree-adaptive magnitude.
- Train-only taxonomy policies: `no_merge`, `merge_t5`, `merge_t10`, and
  `merge_t15`.
- Baselines runnable through this same entry point: LightGCN, SGL (edge-dropout
  variant, referred to as "SGL-ED" in the paper), SimGCL, NCL, and XSimGCL. The
  paper's main comparison table (6 models) uses LightGCN, SGL-ED, SimGCL,
  XSimGCL, NCL, and TaxPro-CL, each with 3 seeds (42, 0, 1) on all 4 datasets,
  completed 2026-09-05. LightGCL appears in the paper only as a literature
  comparison (Related Work, Table 1) and is not implemented in this repository.
- Overall and groupwise evaluation for Strict-Cold, Near-Cold, Mid-Tail,
  Long-Tail, and Warm items.
- Checkpoint selection and early stopping on validation Overall Recall@20,
  followed by a single full-catalog test evaluation. The test split plays no
  role in epoch or checkpoint selection. As the paper discloses (Section 4.3,
  Limitation 10), TaxPro-CL's configuration and taxonomy policies were chosen
  during development with test-split metrics in view.
- Experiment runners for policy screening, main multi-seed runs, the A1-A7
  ablation and sensitivity checks, the V0-V3 factorial, and the prospectively
  specified held-out evaluation.

## Installation

Requires Python 3.10. The paper's runs used two local environments (identified per run
from its `training.log` and summarized in Online Resource 1, Table S23): most
runs Python 3.10.0 / PyTorch 2.1.0+cu121 / Windows 10 / RTX 4060 Ti (the
versions pinned in `requirements.txt`), and a subset of the Amazon-Book runs
Python 3.10.20 / PyTorch 2.6.0+cu124 / Linux / RTX 3090 Ti; part of the
CDs-and-Vinyl comparison (Online Resource 1, Section S30) ran on a cloud notebook. From a fresh clone:

```powershell
python -m venv .venv
.venv\Scripts\activate                  # Linux/macOS: source .venv/bin/activate
# GPU runs (as in the paper): install a CUDA build of PyTorch first. On Windows
# the default PyPI wheel of torch is CPU-only.
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt         # torch==2.1.0 is already satisfied
# Optional, only needed to run NCL without a CUDA-compatible FAISS-GPU build:
pip install -r requirements-ncl-cpu.txt
```

CPU-only machines can skip the first `pip install` line: every command below
also accepts `--device cpu`, which is enough for the tests and the smoke run
but far too slow for the full experiments.

## Quick Start (no GPU needed)

The locked splits (`dataset_verify/`), evaluation-protocol artifacts
(`preprocessed/`) and taxonomy artifacts (`metadata/taxonomy_variants/`) ship
with the repository, so nothing has to be downloaded to run the checks below.

```powershell
# 1. Unit tests (CPU).
python -m pytest -q

# 2. Check that models import and every shipped dataset artifact is present.
#    NCL additionally needs FAISS: run `pip install -r requirements-ncl-cpu.txt`
#    first and add NCL to --models, otherwise preflight reports it as unavailable.
python -m tools.repository.preflight --models TaxPro-CL LightGCN SGL SimGCL XSimGCL --datasets amazon-book yelp2018 musical-instruments arts-crafts-and-sewing

# 3. A 2-epoch TaxPro-CL smoke run on the smallest dataset (about a minute on CPU).
python -m tools.experiments.run_taxpro --dataset musical-instruments --seeds 42 --smoke --epochs 2 --warm-start-epochs 1 --config-id smoke_check --device cpu
```

Step 2 must print `"status": "PASS"` (checked from a fresh clone and virtual
environment). Step 3 writes a complete run directory to
`log/p0/taxprocl/musical-instruments/smoke_check/seed42/` (see Expected Run
Outputs); delete it afterwards, it is not a paper result.

## Scope

This repository implements the six-model main comparison, the A1-A6
ablation and sensitivity checks, and the RQ5 direction-by-magnitude factorial
reported in the paper, the A7 comparable-budget grid for SimGCL and XSimGCL
(complete on all four datasets; the NCL part was not completed and is not
reported), and the prospectively specified held-out evaluation on Office-Products
(`docs/confirmatory_protocol.md`; see Prospectively Specified Held-Out Evaluation below).
This repository does **not** implement sibling loss, gating, or multi-level
prototype memory; the taxonomy-guided direction and degree-adaptive magnitude
described in the paper's Method section are the full extent of the mechanism
evaluated here.

## Datasets And Evaluation Protocol

The main experiments use four public datasets from two source platforms; a
held-out fifth dataset and an excluded candidate are listed below them:

| Dataset                    | Platform | Role                                                       |
| -------------------------- | -------- | ---------------------------------------------------------- |
| `amazon-book`            | Amazon   | Product recommendation with hierarchical book taxonomy     |
| `yelp2018`               | Yelp     | Business recommendation with multi-label category metadata |
| `musical-instruments`    | Amazon   | Product recommendation, Musical Instruments category       |
| `arts-crafts-and-sewing` | Amazon   | Product recommendation, Arts/Crafts/Sewing category        |
| `office-products`        | Amazon   | Prospectively specified held-out evaluation (Table 12)     |
| `cds-and-vinyl`          | Amazon   | Excluded candidate, reported in Online Resource 1, S30     |

`amazon-book` and `yelp2018` reuse the canonical splits established by
NGCF/LightGCN/SGL/SimGCL/NCL/XSimGCL: iterative 5-core filtering applied to the
original train file only, before extracting a validation slice; the original
test file is copied byte-for-byte, unfiltered. `musical-instruments` and
`arts-crafts-and-sewing` have no pre-existing train/validation/test split, so
the pipeline instead runs iterative 5-core filtering ONCE over the whole
interaction pool, then cuts the filtered pool 70/10/20 (train/validation/test)
per user; test here is therefore the remainder of that single filtering pass
and is not guaranteed to be individually 5-core. Protocol A is built by
`tools.data.build_splits` and Protocol B by the per-dataset builders
`tools.data.build_<dataset>` (see Data Bootstrap below); both share the same
downstream group/evaluation logic. Every dataset uses split seed 42.

The immutable model-ready splits are stored in `dataset_verify/`. Item groups
are defined exclusively from training degree:

| Group            | Definition                         |
| ---------------- | ---------------------------------- |
| Strict-Cold (SC) | 0 training interactions            |
| Near-Cold (NC)   | 1-5 training interactions          |
| Mid-Tail (MT)    | 6-10 training interactions         |
| Long-Tail (LT)   | 1-10 training interactions (NC + MT) |
| Warm (Wm)        | More than 10 training interactions |
| Overall          | SC + LT + Wm                       |

Near-Cold is a subset of Long-Tail. Overall covers every item with at least
one test interaction, including Strict-Cold items, which the item-side
mechanism cannot steer (zero direction). All group metrics rank the complete
item catalog and restrict only which test positives count for the group.
The code never uses test metrics to choose an epoch or a checkpoint.

## Models

| Category        | Models                                                                |
| --------------- | --------------------------------------------------------------------- |
| Baselines       | LightGCN, SGL (`aug_type=ed`, i.e. "SGL-ED"), SimGCL, NCL, XSimGCL. |
| Proposed method | TaxPro-CL                                                             |

NCL requires FAISS. The other models depend on the packages listed in
`requirements.txt`.

**Epoch cap:** the upstream ID-GRec configuration files default SGL, SimGCL,
and XSimGCL to `training_epochs = 50`, which is too few for early stopping to
end training on the larger datasets. In this repository `configure/SGL.txt`,
`configure/SimGCL.txt`, and `configure/XSimGCL.txt` are set to `200`
(early-stopping patience 20, 20, and 30 respectively), while
`configure/LightGCN.txt` and `configure/NCL.txt` keep the upstream `1000` and
`500`. Every baseline command below still passes `--training_epochs 200`
explicitly for SGL, SimGCL, and XSimGCL so the cap is visible in the command
history. A run that used the 50-epoch upstream default for those three models
should be treated as undertrained and re-run, not reused.

## Run Commands

All commands use the shared `--seeds 42 0 1` sequential multi-seed flag (same
process, one seed after another; use `--seed <N>` for a single seed). The paper
uses exactly these three training seeds, which are also the default of
`--seeds`; the commands below pass them explicitly for clarity. Output
directories are computed automatically from a hash of the effective
configuration, so these commands cannot collide with or silently overwrite an
existing run. Replace `<ds>` with any of `amazon-book`, `yelp2018`,
`musical-instruments`, `arts-crafts-and-sewing`.

### Baselines (4 datasets each)

```powershell
python main.py --model LightGCN --dataset <ds> --seeds 42 0 1
python main.py --model SGL --dataset <ds> --training_epochs 200 --seeds 42 0 1
python main.py --model SimGCL --dataset <ds> --training_epochs 200 --seeds 42 0 1
python main.py --model NCL --dataset <ds> --seeds 42 0 1
python main.py --model XSimGCL --dataset <ds> --training_epochs 200 --seeds 42 0 1
```

### TaxPro-CL main results (4 datasets, per-dataset hyperparameters)

Every flag below that differs from `configure/TaxPro-CL.txt` (the
`amazon-book` defaults) is spelled out explicitly; everything not overridden
keeps the file's default.

```powershell
python main.py --model TaxPro-CL --dataset amazon-book --seeds 42 0 1

python main.py --model TaxPro-CL --dataset yelp2018 --taxonomy_policy no_merge --temperature 0.125 --temperature_user 0.15 --seeds 42 0 1

python main.py --model TaxPro-CL --dataset musical-instruments --taxonomy_policy no_merge --temperature 0.15 --gamma_cold 5.0 --seeds 42 0 1

python main.py --model TaxPro-CL --dataset arts-crafts-and-sewing --taxonomy_policy no_merge --temperature 0.125 --seeds 42 0 1

# same_leaf_weight=0.4 (A6 ablation, NOT the main config) -- only run this
# to reproduce the A6 sensitivity check, not the paper's headline numbers:
python main.py --model TaxPro-CL --dataset arts-crafts-and-sewing --taxonomy_policy no_merge --temperature 0.125 --same_leaf_weight 0.4 --seeds 42 0 1
```

Each command is a single line, so it runs unchanged in PowerShell, cmd, or bash.

### Ablations A3-A7

A3 sweeps `epsilon_max` (0.20 is the main-result value above, already
covered); A4 sweeps `temperature` (0.10 is the main-result value, already
covered); A5 compares the default leaf-level taxonomy prototype against a
parent-level one. A3-A5 run on Amazon-Book only. A6 (same-leaf soft-positive
weight, Arts-Crafts-and-Sewing) is listed above under TaxPro-CL main results,
since it is a one-flag variant of that dataset's main command. A7
(comparable-budget SimGCL/XSimGCL grid on the four datasets, temperature
{0.05, 0.10, 0.15, 0.20} x epsilon {0.05, 0.10, 0.20}, 3 seeds; the NCL grid was
not completed and is not reported) is a long-running orchestrator rather than a
single command: `python -m tools.experiments.run_a7_full_grid` runs the whole
grid (idempotent; completed cells are skipped and interrupted cells resume from
`last_model.pt`), and `python -m tools.experiments.run_a7_simgcl_amazonbook_remaining`
finishes only the remaining SimGCL/Amazon-Book cells. The RQ5 2x2 factorial
(V0-V3) is documented in `tools/README.md`.

```powershell
# A3: epsilon_max sensitivity (0.01-0.20 plus the 0.40/0.80 extension; the
# full 6-value table is Table S4 of the paper's Online Resource 1)
python main.py --model TaxPro-CL --dataset amazon-book --epsilon_max 0.01 --seeds 42 0 1
python main.py --model TaxPro-CL --dataset amazon-book --epsilon_max 0.05 --seeds 42 0 1
python main.py --model TaxPro-CL --dataset amazon-book --epsilon_max 0.10 --seeds 42 0 1
python main.py --model TaxPro-CL --dataset amazon-book --epsilon_max 0.40 --seeds 42 0 1
python main.py --model TaxPro-CL --dataset amazon-book --epsilon_max 0.80 --seeds 42 0 1

# A4: temperature sensitivity
python main.py --model TaxPro-CL --dataset amazon-book --temperature 0.05 --seeds 42 0 1
python main.py --model TaxPro-CL --dataset amazon-book --temperature 0.20 --seeds 42 0 1

# A5: leaf (default, see main results above) vs. parent prototype granularity.
# Parent granularity under merge_t10, the Amazon-Book main policy, so the two
# differ in prototype_mode only (Online Resource 1, Tables S7/S7b).
# tools/experiments/run_a5_parent_mergedt10.ps1 runs the same command for the
# three seeds with an explicit output directory; the comparison is computed by
# python -m tools.analysis.a5_leaf_vs_parent_mergedt10
python main.py --model TaxPro-CL --dataset amazon-book --prototype_mode parent --seeds 42 0 1
```

### Prospectively Specified Held-Out Evaluation (Office-Products)

The protocol `docs/confirmatory_protocol.md` was frozen before the data of the
new dataset were built; its SHA-256 and those of its two logged deviations are in
`docs/confirmatory_protocol*.sha256`. These files were first published, with the
frozen selections, in commit 1379271 (2026-09-28), after training and before any
test evaluation; no third-party registration was made. Every run of this
evaluation is trained with `TAXPRO_DEFER_TEST=1`, so no test metric is computed
during training; each run's test metrics are computed once, after the
validation-only selections are frozen in `results/confirmatory/confirmatory_selection.json`
(timeline: Online Resource 1, Table S36e).

```powershell
# 1. Screening (data statistics only, no model) and data build (Protocol B)
python -m tools.data.screen_confirmatory_candidates Office_Products_5.json.gz --out screening_office_products.json
python -m tools.data.build_office_products --reviews-file <path-to-decompressed-Office_Products_5.json>
python -m tools.data.download_office_products_metadata
python -m tools.data.build_office_products_metadata
python -m tools.data.build_office_products_taxonomy

# 2. Validation-only selection (policy, TaxPro-CL grid, SimGCL grid), factorial
#    controls, and default baselines; writes confirmatory_selection.json + SHA-256
python -m tools.experiments.run_confirmatory --dataset office-products

# 3. Open the test split once for the selected runs
python -m tools.experiments.evaluate_sealed_test --selection results/confirmatory/confirmatory_selection.json

# 4. Analyses (Table 12; Online Resource 1, Section S36)
python -m tools.analysis.confirmatory_analysis --dataset office-products
python -m tools.analysis.confirmatory_six_methods --dataset office-products
python -m tools.analysis.confirmatory_peruser --dataset office-products
python -m tools.analysis.update_manifest_confirmatory
```

## Main Results (Recall@20, Mean±Std over 3 seeds: 42, 0, 1)

Current comparison: 6 models x 4 datasets (baselines LightGCN, SGL-ED, SimGCL,
XSimGCL, NCL, plus TaxPro-CL). Near-Cold and Long-Tail are the paper's primary
target groups; Overall and Warm are reported for transparency, not the
optimization target. XSimGCL completed 3 seeds x 4 datasets on 2026-09-05 and
is included below with the same protocol as the other baselines (default
hyperparameters + --training_epochs 200, no per-dataset tuning sweep).

#### Near-Cold (NC) and Long-Tail (LT)

| Model | Amazon-Book (NC) | Yelp2018 (NC) | Musical-Instruments (NC) | Arts-Crafts-and-Sewing (NC) | Amazon-Book (LT) | Yelp2018 (LT) | Musical-Instruments (LT) | Arts-Crafts-and-Sewing (LT) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **LightGCN** | 0.0015±0.0002 | 0.0000±0.0000 | 0.0062±0.0003 | 0.0105±0.0007 | 0.0025±0.0000 | 0.0003±0.0001 | 0.0131±0.0008 | **0.0230±0.0010** |
| **SGL-ED** | 0.0024±0.0003 | 0.0000±0.0000 | 0.0074±0.0002 | 0.0111±0.0006 | 0.0030±0.0000 | 0.0002±0.0001 | 0.0154±0.0005 | 0.0225±0.0005 |
| **SimGCL** | 0.0026±0.0008 | 0.0002±0.0001 | 0.0084±0.0007 | 0.0107±0.0010 | 0.0050±0.0004 | 0.0028±0.0002 | 0.0149±0.0008 | 0.0201±0.0005 |
| **XSimGCL** | 0.0031±0.0000 | 0.0002±0.0001 | 0.0038±0.0003 | 0.0063±0.0004 | 0.0053±0.0002 | 0.0017±0.0003 | 0.0100±0.0004 | 0.0152±0.0006 |
| **NCL** | 0.0009±0.0002 | 0.0000±0.0000 | 0.0013±0.0004 | 0.0082±0.0004 | 0.0029±0.0002 | 0.0009±0.0001 | 0.0057±0.0011 | 0.0170±0.0011 |
| **TaxPro-CL** | **0.0037±0.0001** | **0.0004±0.0001** | **0.0085±0.0002** | **0.0113±0.0007** | **0.0059±0.0001** | **0.0033±0.0001** | **0.0165±0.0004** | 0.0221±0.0005 |

#### Warm (Wm) and Overall

| Model | Amazon-Book (Wm) | Yelp2018 (Wm) | Musical-Instruments (Wm) | Arts-Crafts-and-Sewing (Wm) | Amazon-Book (Overall) | Yelp2018 (Overall) | Musical-Instruments (Overall) | Arts-Crafts-and-Sewing (Overall) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **LightGCN** | 0.0529±0.0002 | 0.0726±0.0002 | 0.2448±0.0005 | 0.2381±0.0014 | 0.0402±0.0002 | 0.0620±0.0002 | 0.1860±0.0004 | 0.1776±0.0008 |
| **SGL-ED** | 0.0574±0.0007 | 0.0778±0.0002 | 0.2458±0.0017 | **0.2554±0.0010** | 0.0438±0.0006 | 0.0662±0.0002 | 0.1867±0.0014 | **0.1898±0.0007** |
| **SimGCL** | **0.0608±0.0007** | 0.0826±0.0006 | 0.2378±0.0006 | 0.2356±0.0010 | **0.0472±0.0005** | 0.0708±0.0005 | 0.1811±0.0005 | 0.1750±0.0009 |
| **XSimGCL** | 0.0606±0.0005 | **0.0831±0.0006** | **0.2604±0.0010** | 0.2518±0.0006 | 0.0472±0.0003 | **0.0711±0.0004** | **0.1957±0.0005** | 0.1848±0.0004 |
| **NCL** | 0.0539±0.0002 | 0.0770±0.0005 | 0.2523±0.0007 | 0.2348±0.0022 | 0.0414±0.0002 | 0.0660±0.0004 | 0.1888±0.0004 | 0.1737±0.0012 |
| **TaxPro-CL** | 0.0587±0.0004 | 0.0781±0.0004 | 0.2301±0.0018 | 0.2246±0.0015 | 0.0457±0.0003 | 0.0670±0.0004 | 0.1761±0.0012 | 0.1683±0.0011 |

Source: `log/p0/baseline/<model>/<dataset>/` and `log/p0/taxprocl/<dataset>/`
(`final_test_group_metrics.json` per seed). Run
`python -m tools.analysis.mid_tail_degree6_10` then
`python -m tools.analysis.main_results_table` to regenerate this table
(mean±std, all five groups, bold/underline ranking) exactly as reported in
the manuscript, including the Mid-Tail column omitted above.

## Reproducing The Paper's Tables And Statistics

Trained checkpoints are large and are not shipped, so the tables are
reproduced in three steps: (1) prepare the data (the shipped splits can be used
as they are; see Data Bootstrap to rebuild them), (2) train with the Run
Commands above, (3) run the analysis scripts, which read the checkpoints from
`log/`. The per-seed metrics behind every table are also shipped in
`results/metrics_seed.csv`, and `results/claim_evidence.csv` maps every
quantitative claim of the manuscript to its table, run, and script. The main
entry points are:

| Paper item | Script (`python -m tools.analysis.<name>`) |
| --- | --- |
| Main results (Tables 7, S1; Figure 1), NDCG@20 table (Table S18) | `mid_tail_degree6_10`, then `main_results_table`; figure: `plot_main_figures` |
| Table 11 (per-user bootstrap, TaxPro-CL vs. SimGCL), Table S6 | `seed_matched_bootstrap` |
| Multiplicity-corrected inference for Table 11 (Table S26) | `a2_multiplicity_correction` |
| Wilcoxon tests with Holm correction (Table S5) | `compile_ablation_sweep`, then `statistics` |
| Taxonomy-policy sweep, validation and test (Table S3; CDs-and-Vinyl Table S30d) | `policy_sweep_splits` |
| RQ5 factorial: direction and epsilon-adaptivity (Tables 8, 9, S13, S17, S19), interaction (Table S27), warm-start removal (Table S20); Figure 2 | `factorial_direction_bootstrap` (all datasets; Yelp2018 with `--datasets yelp2018`), `factorial_amazon_book` (Amazon-Book, `merge_t10`), `factorial_interaction_bootstrap`, `warmstart_removal_bootstrap`, `merge_factorial_results`; figure: `regenerate_fig3_factorial_forest` |
| RQ5 NDCG@20 companions (Tables S13b, S17b) | `factorial_ndcg_bootstrap`, `epsilon_ndcg_bootstrap` |
| Amazon-Book `no_merge` factorial and Holm-Bonferroni families (Tables S13c-S13e) | `factorial_multiplicity`, `factorial_holm_table` |
| Distinctness of the factorial checkpoints (Table S13f) | `results_manifest.csv` (checkpoint hashes) and each run's `final_test_group_metrics.json` |
| V3 replaced by the Environment-A main checkpoint (Table S23b) | `factorial_v3main_substitution` |
| Gap decomposition through V0 and contrasts as % of the control (Tables S32, S33) | `factorial_decomposition_summary` |
| Prototype-construction variants (Tables 10, S14, S14b, S14c) | `leaf_variants_bootstrap`, `rescue_vs_variants_bootstrap`, `prototype_variants_overlap`, `prototype_variants_ndcg` |
| A5 leaf vs. parent, policy-matched (Tables S7, S7b) | `a5_leaf_vs_parent_mergedt10` |
| Leaf-size distribution (Table S15), prototype-to-embedding distance (Table S16), degree transition (Table S21) | `leaf_size_distribution`, `prototype_distance_by_space`, `degree_transition` |
| Re-check under both degree definitions (Table S25) | `intrinsic_sparsity_bootstrap` |
| Beyond-accuracy diagnostics (Table S10, Figure S2) | `beyond_accuracy`; figure: `plot_main_figures` |
| Item-level rank audit (Tables S11, S12) | `rank_audit_seed_matched` |
| View cosine (Table S22), realized displacement (Table S29) | `view_cosine_by_degree`, `realized_perturbation_norm` |
| Strict-Cold audit (Table S34) | `strict_cold_audit` |
| CDs-and-Vinyl comparison (Tables S30b, S30c) | `cds_and_vinyl_summary`, `update_manifest_cds_and_vinyl` |
| Full p-value vector of the factorial (Table S13e) | `export_factorial_pvalues` |
| Yelp2018 per-user hit decomposition (Table 8 note a; Tables S13g, S13h) | `yelp_factorial_hits` |
| A7 comparable-budget grid (Section 5.3; Tables S24-S24e) | `a7_tuned_comparison` |
| Validation-only re-selection of the configuration (Tables S35-S35c) | `validation_reselection_audit`, then `s35_export` |
| Prospectively specified held-out evaluation (Table 12; Tables S36-S36e) | `confirmatory_analysis`, `confirmatory_six_methods`, `confirmatory_peruser` |

`results/README.md` lists every shipped result file with the script that
writes it and the manuscript item it supports.

Approximate cost of the training runs on one RTX 4060 Ti: a TaxPro-CL run on
Amazon-Book takes about 3.5 hours per seed; the smaller datasets and the
baselines take considerably less (a SimGCL run takes about 12 minutes on
Musical-Instruments and Arts-Crafts-and-Sewing and about an hour on Yelp2018).
The paper's runs were split across the two environments described under
Installation.

## Repository Structure

```text
TaxPro-CL/
├── main.py                    Entry point for every model
├── Parser.py                  Command-line parsing shared by all models
├── config_path/               Resolves configuration, dataset, and log paths
├── configure/                 One default hyperparameter file per model
├── models/                    LightGCN, SGL, SimGCL, XSimGCL, NCL, TaxPro-CL
├── utility/                   Data loading, losses, training loop, evaluation
├── dataset_verify/            Locked train/validation/test splits (4 datasets)
├── preprocessed/              Evaluation-protocol artifacts (groups, targets)
├── metadata/taxonomy_variants/ Taxonomy assignments for the four policies
├── tools/                     Research command packages (see tools/README.md)
│   ├── data/                  Download, split building, metadata preparation
│   ├── protocol/              Protocol builders and gate validation
│   ├── experiments/           Baseline, TaxPro-CL, ablation, A7 runners
│   ├── analysis/              Bootstrap statistics, tables, figures, evidence
│   ├── ranking/               Checkpoint loading, full-catalog ranking, rank comparison
│   └── repository/            Preflight check
├── tests/                     Unit tests (python -m pytest)
├── results/                   Shipped audit trail: run/claim manifests and the
│                              analysis outputs cited by the paper (results/README.md)
├── docs/                      Data, model, and reproducibility notes
├── requirements.txt           Python dependencies
├── CITATION.cff, LICENSE
└── log/                       Created by training runs (not shipped)
```

## Reproducibility Artifacts

Shipped with this repository (no training required to inspect them):

- `results/results_manifest.csv`: run-level provenance for the runs behind the
  main comparison (all six methods, four datasets, three seeds), the V0-V3
  factorial and the A5 runs (dataset, variant, seed, config/split/checkpoint
  hashes, selection metric, which table each run backs). Sensitivity sweeps
  (A3/A4, A6, isotropic blend, prototype variants) and the supplementary A7
  grid keep their evidence in their own run directories and result JSONs.
- `results/metrics_seed.csv`: per-seed absolute Recall/NDCG by group for
  every reported run.
- `results/claim_evidence.csv`: a claim-by-claim map from the manuscript's
  quantitative statements to their source dataset, configuration, metric,
  and script/run.

Local checkpoints and dataset artifacts (not shipped -- `.gitignore`d, since
they are large and regenerable) once training/data-preparation are run:

- `preprocessed/evaluation_protocol/`: group masks, targets, and protocol
  manifests.
- `metadata/taxonomy_variants/`: taxonomy assignments, masks, statistics, and
  policy manifests (this one *is* shipped; see Repository Structure above).
- `log/p0/baseline/<model>/<dataset>/<config_id>/seed<seed>/`: canonical
  baseline artifacts.
- `log/p0/taxprocl/<dataset>/<config_id>/seed<seed>/`: canonical TaxProCL
  artifacts. Each run directory contains logs, checkpoints, manifests, and
  final metrics.

Direct `main.py` TaxProCL runs are isolated by dataset, taxonomy policy, seed,
and effective configuration hash. This prevents one policy or hyperparameter
setting from overwriting another run.

See [`tools/README.md`](tools/README.md) for the complete command catalog.

## CPU And GPU Execution

The same entry point supports both environments:

```powershell
python main.py --model TaxPro-CL --device auto
python main.py --model TaxPro-CL --device cpu --seed 42
python main.py --model TaxPro-CL --device cuda --gpu_id 0 --seed 42
```

`auto` selects CUDA when available and otherwise uses CPU. An explicit `cuda`
request fails clearly when CUDA is unavailable; it never silently starts a
long paper experiment on CPU. CPU is intended for development, tests, data
preparation, report compilation, and smoke runs. Full multi-seed paper runs
should use the validated GPU environment described in `docs/reproducibility.md`.

## Data Bootstrap

```powershell
python -m tools.data.download
python -m tools.data.build_splits
python -m tools.data.prepare_metadata
python -m tools.data.metadata_manifest
python -m tools.protocol.build --build
python -m tools.protocol.validate_determinism
python -m tools.protocol.validate_gate
```

Re-downloading the `amazon-book` and `yelp2018` interaction files from the
public sources and running `tools.data.build_splits` reproduces the shipped
`dataset_verify/` splits byte for byte (the SHA256 checksums of the downloaded
files match Online Resource 1, Section S28); this was verified for the release.

Downloaded `.gz` and `.zip` archives are deleted after successful extraction
and validation. Pass `--keep-archives` only when an archive must be retained
temporarily. Raw metadata remains local and is excluded from Git; the source
URLs and extracted-file inventory are written to `manifests/metadata_sources.json`
by `tools.data.metadata_manifest` (generated locally, not shipped).

`build_splits` implements two protocols depending on the dataset (see Datasets
And Evaluation Protocol above):

- **`amazon-book`, `yelp2018`** (pre-existing train/test split): iterative
  user/item 5-core filtering is applied to raw train only, then each user's
  sorted train interactions are shuffled with one seeded RNG. Users with 2-9
  interactions contribute one validation item; users with at least 10
  contribute `floor(0.1 * N)`; singleton users remain entirely in train. Raw
  test is copied byte-for-byte, unfiltered.
- **`musical-instruments`, `arts-crafts-and-sewing`** (no pre-existing split):
  iterative 5-core filtering is applied once to the entire raw interaction
  pool, then the filtered pool is cut 70/10/20 (train/validation/test) per
  user. Test is therefore the remainder of that single filtering pass, not a
  separately-filtered slice.

Seeds 42, 123, and 2026 are supported for the split RNG, with 42 locked for
the main paper split (independent of the {42, 0, 1} training seeds used to
repeat every reported experiment 3 times). Default output is staged under
`results/`. Promoting with `--promote --force` is an explicit protocol-changing
action and must be followed by a complete protocol/taxonomy rebuild.

### Dataset And Metadata Sources

| Source | URL | Auto-fetched by `tools.data.download`? |
| --- | --- | --- |
| `amazon-book` / `yelp2018` pre-split interactions | `https://raw.githubusercontent.com/gusye1234/LightGCN-PyTorch/master/data/{dataset}/{train,test}.txt` | Yes |
| Amazon category metadata, 2014 vintage (Books) | `https://mcauleylab.ucsd.edu/public_datasets/data/amazon/categoryFiles/meta_Books.json.gz` | Yes |
| Amazon category metadata, 2018 vintage (Books) | `https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Books.json.gz` | Yes |
| Amazon category metadata, 2023 vintage (Books) | `https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Books.jsonl.gz` | Yes |
| Yelp category metadata, 2018/2021/2022 (third-party re-host) | `https://zenodo.org/records/10998102/files/Yelp{2018,2021,2022}.zip?download=1` | Yes |
| `musical-instruments` / `arts-crafts-and-sewing` raw 5-core reviews | `https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/{Musical_Instruments,Arts_Crafts_and_Sewing}_5.json.gz` | No -- manual, see below |
| Amazon category metadata for Musical-Instruments / Arts-Crafts-and-Sewing (same 3 vintages as Books above, category name substituted) | Same three URL templates as the Books rows, with `Musical_Instruments` / `Arts_Crafts_and_Sewing` in place of `Books` (`config_path.AMAZON_CATEGORY_METADATA_URLS`) | No -- `tools.data.download` only fetches the Books category; download manually to `metadata/amazon/{year}/meta_{Category}.json(l)` (`.jsonl` for 2023) |

All three Amazon-based datasets (`amazon-book`, `musical-instruments`,
`arts-crafts-and-sewing`) share the same 3-vintage (2014/2018/2023) category
metadata scheme and the same ID-matching/merge logic
(`tools/data/prepare_metadata.py`); only the category name and, for
`musical-instruments`/`arts-crafts-and-sewing`, the local file paths differ.
SHA256 checksums of every original source file, and of every derived
train/validation/test file, are recorded in each dataset's
`dataset_verify/<dataset>/split_manifest.json` (`source`/`source_hashes` and
`output_hashes` fields respectively); the paper's Online Resource 1, Section
S28 reports the same checksums alongside dataset-provenance narrative.

### `musical-instruments` And `arts-crafts-and-sewing`: Additional Manual Step

These two datasets have no pre-existing train/validation/test split (unlike
`amazon-book`/`yelp2018`, which reuse LightGCN's canonical split), so they are
not produced by `tools.data.build_splits` and need one extra, currently-manual
download step before the rest of the pipeline can run:

1. Manually download the decompressed 5-core review file for the category you
   want (`Musical_Instruments_5.json.gz` or `Arts_Crafts_and_Sewing_5.json.gz`)
   from `https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/`
   and decompress it to a `.json` file. This step is manual because this
   repository automates the resulting file's checksum but not its retrieval.
2. Build the dataset-specific split (reads the file from step 1, verifies
   5-core still holds after this project's own ID remap -- see Dataset And
   Metadata Sources above -- then does one deterministic 70/10/20 per-user
   split):
   ```powershell
   python -m tools.data.build_musical_instruments --reviews-file <path-to-decompressed-json>
   # or, for the other dataset:
   python -m tools.data.build_arts_crafts_and_sewing --reviews-file <path-to-decompressed-json>
   ```
3. Manually download that same category's metadata for all three vintages
   (2014/2018/2023, URLs in Dataset And Metadata Sources above) to
   `metadata/amazon/{year}/meta_{Musical_Instruments,Arts_Crafts_and_Sewing}.json(l)`,
   then merge it (dataset-specific companions to `prepare_metadata.py` that
   never touch `amazon-book`'s locked metadata files):
   ```powershell
   python -m tools.data.build_musical_instruments_metadata
   python -m tools.data.build_musical_instruments_taxonomy
   # or, for the other dataset:
   python -m tools.data.build_arts_crafts_and_sewing_metadata
   python -m tools.data.build_arts_crafts_and_sewing_taxonomy
   ```

After this, both datasets have the same `dataset_verify/<dataset>/` and
`metadata/taxonomy_variants/<dataset>/` artifacts as `amazon-book`/`yelp2018`,
and the Run Commands above work identically.

CDs-and-Vinyl, used only in Online Resource 1, Section S30, follows the same
steps with `CDs_and_Vinyl_5.json`: `python -m tools.data.build_cds_and_vinyl
--reviews-file <path>`, `python -m tools.data.download_cds_and_vinyl_metadata`,
`python -m tools.data.build_cds_and_vinyl_metadata`, and
`python -m tools.data.build_cds_and_vinyl_taxonomy`.

## Expected Run Outputs

A completed training run contains:

- `training.log`
- `run_manifest.json` (`status` is `completed` once the run has finished)
- `config_resolved.json` and `environment.json` (the exact configuration and
  Python/PyTorch/CUDA/GPU/commit used)
- `validation_metrics.json` (per-epoch validation metrics)
- `best_validation_model.pt`
- `last_model.pt`
- `final_test_metrics.json`
- `final_test_group_metrics.json`
- `prototype_init.json` (TaxPro-CL runs only)

The group metrics include Precision, Recall, and NDCG at K=10 and K=20 for
Overall, Near-cold, Long-tail, and Warm slices. Non-overall groups also record
eligible users, relevant items, and positive interactions.

## Acknowledgements And Upstream Code

The baseline implementations (LightGCN, SGL, SimGCL, NCL, XSimGCL) and the
training/evaluation pipeline used here (command-line parser, the
`configure/*.txt` per-model configuration layout, the training loop, and the
evaluation and logging conventions) are adapted from
[ID-GRec](https://github.com/BlueGhostYi/ID-GRec), a graph recommendation
framework by Yi Zhang and colleagues. The per-model training settings (epoch
cap, early-stopping patience, batch size) follow ID-GRec's configuration files,
with one deliberate change: the epoch cap of SGL, SimGCL, and XSimGCL is raised
from ID-GRec's default of 50 to 200 (see "Epoch cap" above).
Many thanks to the ID-GRec authors. If you use these baselines or this
pipeline, please also cite the ID-GRec associated work:

```bibtex
@article{zhang2025simplify,
  title   = {Simplify to the Limit! Embedding-less Graph Collaborative
             Filtering for Recommender Systems},
  author  = {Zhang, Yi and Zhang, Yiwen and Sang, Lei and Sheng, Victor S.},
  journal = {ACM Transactions on Information Systems},
  volume  = {43},
  number  = {1},
  articleno = {22},
  pages   = {22:1--22:30},
  year    = {2025},
  doi     = {10.1145/3701230}
}
```

ID-GRec itself acknowledges LightGCN (training process and evaluation
metrics), NGCF (sparsity testing), SELFRec (framework architecture), and
SSLRec (output logging); the original method papers for each baseline are cited
in the TaxPro-CL manuscript.

## Citation

If you use this repository, please cite the associated TaxPro-CL manuscript.
The entry below uses placeholder author names; the full reference (and
`CITATION.cff`) will be updated once the paper is published.

```bibtex
@unpublished{taxprocl2026,
  title  = {TaxPro-CL: Taxonomy-Guided Perturbation for Sparse-Item
            Recommendation in Graph Contrastive Learning -- A Controlled
            Ablation Study},
  author = {Author 1 and Author 2 and Author 3 and Author 4},
  year   = {2026},
  note   = {Manuscript; code and results: release v1.1-jiis}
}
```

Machine-readable citation metadata for this repository (as software, separate
from the paper above) is provided in `CITATION.cff`.

## License And Data

This repository distributes code and experiment metadata. Dataset access and use
remain subject to the respective source licenses and terms of service.

The MIT License in `LICENSE` covers this project's own contributions (the
TaxPro-CL model, taxonomy construction, and experiment tooling). Code adapted
from ID-GRec (see "Acknowledgements And Upstream Code" above) remains subject
to its authors' rights: the ID-GRec repository declared no license when checked
on 2026-09-20. The adapted code is credited in the source headers and above, and
the associated paper (Zhang et al., ACM TOIS 43(1), Article 22, 2025) is cited in
the manuscript and in `CITATION.cff`.
