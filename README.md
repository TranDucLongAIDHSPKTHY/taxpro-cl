# TaxPro-CL

> Taxonomy-guided prototype contrastive learning for long-tail graph recommendation.

TaxPro-CL is a research codebase for studying taxonomy-aware item
perturbations in graph collaborative filtering. The method augments a LightGCN
encoder with train-locked taxonomy guidance and an item-level InfoNCE objective
to improve representation learning while preserving a strict validation-first
evaluation protocol.

## Overview

Long-tail recommendation is difficult because low-degree items have limited
collaborative evidence. TaxPro-CL uses item taxonomy only from training data to
construct structured contrastive views. Invalid or unknown taxonomy items remain
in the recommender objective and evaluation catalog, but are excluded from the
taxonomy contrastive term.

This implementation provides:

- A TaxPro-CL model built on a LightGCN backbone, using the same 2-view
  self-supervised structure as SimGCL (one clean forward pass for BPR, two
  independent perturbed passes for InfoNCE) but replacing SimGCL's isotropic
  random perturbation with a taxonomy-prototype-guided direction and a
  degree-adaptive magnitude.
- Train-only taxonomy policies: `no_merge`, `merge_t5`, `merge_t10`, and
  `merge_t15`.
- Baselines runnable through this same entry point: LightGCN, SGL (edge-dropout
  variant, referred to as "SGL-ED" in the paper), SimGCL, NCL, and XSimGCL. The
  paper's main comparison table (6 models) uses LightGCN, SGL-ED, SimGCL,
  XSimGCL, NCL, and TaxPro-CL, each with 3 seeds (42, 0, 1) on all 4 datasets,
  completed 2026-09-05. LightGCL is discussed in the paper's Related Work as a
  literature comparison only (Table 1) and was never part of this repository's
  experimental scope.
- Overall and groupwise evaluation for Near-cold, Long-tail, and Warm items.
- Checkpoint selection on validation Recall@20 Overall, followed by a single
  full-catalog test evaluation (test data is never used to pick a taxonomy
  policy, hyperparameter, epoch, or checkpoint).
- Experiment runners for policy screening, main multi-seed runs, and the
  A1-A7 ablation and sensitivity checks.

## Installation

Requires Python 3.10 (tested on 3.10.0). From a fresh clone:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
# Optional, only needed to run NCL without a CUDA-compatible FAISS-GPU build:
pip install -r requirements-ncl-cpu.txt
```

Verify the install and module wiring without training anything:

```powershell
python -m pytest --collect-only -q
```

## Scope

This repository implements the six-model main comparison plus the A1-A7
ablation and sensitivity checks reported in the paper (A7, the supplementary
comparable-budget SimGCL grid, was still running for part of the paper's
scope at submission -- see `tools/experiments/run_a7_full_grid.py`). This
repository does **not** implement
sibling loss, gating, or multi-level prototype memory; the taxonomy-guided
direction and degree-adaptive magnitude described in the paper's Method
section are the full extent of the mechanism evaluated here.

## Datasets And Evaluation Protocol

The experiments target four public datasets, from two source platforms:

| Dataset                    | Platform | Role                                                       |
| -------------------------- | -------- | ---------------------------------------------------------- |
| `amazon-book`            | Amazon   | Product recommendation with hierarchical book taxonomy     |
| `yelp2018`               | Yelp     | Business recommendation with multi-label category metadata |
| `musical-instruments`    | Amazon   | Product recommendation, Musical Instruments category       |
| `arts-crafts-and-sewing` | Amazon   | Product recommendation, Arts/Crafts/Sewing category        |

`amazon-book` and `yelp2018` reuse the canonical splits established by
NGCF/LightGCN/SGL/SimGCL/NCL/XSimGCL: iterative 5-core filtering applied to the
original train file only, before extracting a validation slice; the original
test file is copied byte-for-byte, unfiltered. `musical-instruments` and
`arts-crafts-and-sewing` have no pre-existing train/validation/test split, so
the pipeline instead runs iterative 5-core filtering ONCE over the whole
interaction pool, then cuts the filtered pool 70/10/20 (train/validation/test)
per user; test here is therefore the remainder of that single filtering pass
and is not guaranteed to be individually 5-core. Both protocols are driven by
`tools.data.build_splits` (see Data Bootstrap below) and share the same
downstream group/evaluation logic. All 4 datasets use split seed 42.

The immutable model-ready splits are stored in `dataset_verify/`. Item groups
are defined exclusively from training degree:

| Group     | Definition                         |
| --------- | ---------------------------------- |
| Near-cold | 1-5 training interactions          |
| Long-tail | 1-10 training interactions         |
| Warm      | More than 10 training interactions |

Near-cold is a subset of Long-tail; Long-tail and Warm are disjoint and
together cover every item with at least one test interaction (their union is
what the paper reports as "Overall"). All group metrics retain the complete
item catalog and restrict only the relevant target positives for the group.
Test data must not be used to select taxonomy policy, hyperparameters, epochs,
or checkpoints.

## Models

| Category        | Models                                                                |
| --------------- | --------------------------------------------------------------------- |
| Baselines       | LightGCN, SGL (`aug_type=ed`, i.e. "SGL-ED"), SimGCL, NCL, XSimGCL. |
| Proposed method | TaxPro-CL                                                             |

NCL requires FAISS. The other models depend on the packages listed in
`requirements.txt`.

**Known epoch-cap pitfall:** `configure/SGL.txt` and `configure/SimGCL.txt`
default `training_epochs` to `50` with no early stopping strong enough to
reliably converge within that cap on the larger datasets (`configure/XSimGCL.txt`
already defaults to `200`; `configure/LightGCN.txt` and `configure/NCL.txt`
default to `1000` and `500` respectively -- none of these three need an
override). **Always pass `--training_epochs 200` explicitly for SGL and
SimGCL runs** -- every command below already does this (the flag is also
passed for XSimGCL commands for clarity, though it is redundant with that
file's own default). A run that silently used the 50-epoch default for SGL or
SimGCL should be treated as undertrained and re-run, not reused.

## Run Commands

All commands use the shared `--seeds 42 0 1` sequential multi-seed flag (same
process, one seed after another; use `--seed <N>` for a single seed). Output
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

python main.py --model TaxPro-CL --dataset yelp2018 ^
  --taxonomy_policy no_merge --temperature 0.125 --temperature_user 0.15 ^
  --seeds 42 0 1

python main.py --model TaxPro-CL --dataset musical-instruments ^
  --taxonomy_policy no_merge --temperature 0.15 --gamma_cold 5.0 ^
  --seeds 42 0 1

python main.py --model TaxPro-CL --dataset arts-crafts-and-sewing ^
  --taxonomy_policy no_merge --temperature 0.125 ^
  --seeds 42 0 1

# same_leaf_weight=0.4 (A6 ablation, NOT the main config) -- only run this
# to reproduce the A6 sensitivity check, not the paper's headline numbers:
python main.py --model TaxPro-CL --dataset arts-crafts-and-sewing ^
  --taxonomy_policy no_merge --temperature 0.125 --same_leaf_weight 0.4 ^
  --seeds 42 0 1
```

(`^` is the PowerShell/cmd line-continuation character; on a single line, drop
the `^` and join the arguments with spaces.)

### Ablations A3-A7

A3 sweeps `epsilon_max` (0.20 is the main-result value above, already
covered); A4 sweeps `temperature` (0.10 is the main-result value, already
covered); A5 compares the default leaf-level taxonomy prototype against a
parent-level one. A3-A5 run on Amazon-Book only. A6 (same-leaf soft-positive
weight, Arts-Crafts-and-Sewing) is listed above under TaxPro-CL main results,
since it is a one-flag variant of that dataset's main command. A7
(comparable-budget SimGCL/XSimGCL/NCL grid, all four datasets, supplementary
to the paper's default-hyperparameter scope) is a long-running orchestrator,
not a single reproducible command; see
`tools/experiments/run_a7_full_grid.py` and
`tools/experiments/run_a7_matched_budget.py`.

```powershell
# A3: epsilon_max sensitivity (0.01-0.20 plus the 0.40/0.80 extension, completed
# 2026-09-12; full 6-value table in Document/paper_P0/V57/ESM_1.pdf, Table S4)
python main.py --model TaxPro-CL --dataset amazon-book --epsilon_max 0.01 --seeds 42 0 1
python main.py --model TaxPro-CL --dataset amazon-book --epsilon_max 0.05 --seeds 42 0 1
python main.py --model TaxPro-CL --dataset amazon-book --epsilon_max 0.10 --seeds 42 0 1
python main.py --model TaxPro-CL --dataset amazon-book --epsilon_max 0.40 --seeds 42 0 1
python main.py --model TaxPro-CL --dataset amazon-book --epsilon_max 0.80 --seeds 42 0 1

# A4: temperature sensitivity
python main.py --model TaxPro-CL --dataset amazon-book --temperature 0.05 --seeds 42 0 1
python main.py --model TaxPro-CL --dataset amazon-book --temperature 0.20 --seeds 42 0 1

# A5: leaf (default, see main results above) vs. parent prototype granularity
python main.py --model TaxPro-CL --dataset amazon-book ^
  --prototype_mode parent --taxonomy_policy no_merge --seeds 42 0 1
```

## Main Results (Recall@20, Mean±Std over 3 seeds: 42, 0, 1)

Current comparison: 6 models x 4 datasets (baselines LightGCN, SGL-ED, SimGCL,
XSimGCL, NCL, plus TaxPro-CL). Near-Cold and Long-Tail are the paper's primary
target groups; Overall and Warm are reported for transparency, not the
optimization target. XSimGCL completed 3 seeds x 4 datasets on 2026-09-05 and
is included below with the same protocol as the other baselines (default
hyperparameters + --training_epochs 200, no per-dataset tuning sweep).

#### Near-Cold (NC) and Long-Tail (LT)

| Model               | Amazon-Book (NC) | Yelp2018 (NC) | Musical-Instruments (NC) | Arts-Crafts-and-Sewing (NC) | Amazon-Book (LT) | Yelp2018 (LT) | Musical-Instruments (LT) | Arts-Crafts-and-Sewing (LT) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **LightGCN** | 0.0015±0.0002 | 0.0000±0.0000 | 0.0062±0.0003 | 0.0105±0.0007 | 0.0025±0.0000 | 0.0003±0.0001 | 0.0131±0.0008 | **0.0230±0.0010** |
| **SGL-ED** | 0.0024±0.0003 | 0.0000±0.0000 | 0.0074±0.0002 | 0.0111±0.0006 | 0.0030±0.0000 | 0.0002±0.0001 | 0.0154±0.0005 | 0.0225±0.0005 |
| **SimGCL** | 0.0026±0.0008 | 0.0002±0.0001 | 0.0084±0.0007 | 0.0107±0.0010 | 0.0050±0.0004 | 0.0028±0.0002 | 0.0149±0.0008 | 0.0201±0.0005 |
| **XSimGCL** | 0.0031±0.0000 | 0.0002±0.0001 | 0.0038±0.0003 | 0.0063±0.0004 | 0.0053±0.0002 | 0.0017±0.0003 | 0.0100±0.0004 | 0.0152±0.0006 |
| **NCL** | 0.0009±0.0002 | 0.0000±0.0000 | 0.0013±0.0004 | 0.0082±0.0004 | 0.0029±0.0002 | 0.0009±0.0001 | 0.0057±0.0011 | 0.0170±0.0011 |
| **TaxPro-CL** | **0.0037±0.0001** | **0.0004±0.0001** | **0.0085±0.0002** | **0.0113±0.0007** | **0.0059±0.0001** | **0.0033±0.0001** | **0.0165±0.0004** | 0.0221±0.0005 |

#### Overall (Ov) and Warm (Wm)

| Model               | Amazon-Book (Ov) | Yelp2018 (Ov) | Musical-Instruments (Ov) | Arts-Crafts-and-Sewing (Ov) | Amazon-Book (Wm) | Yelp2018 (Wm) | Musical-Instruments (Wm) | Arts-Crafts-and-Sewing (Wm) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **LightGCN** | 0.0402±0.0002 | 0.0620±0.0002 | 0.1860±0.0004 | 0.1776±0.0008 | 0.0529±0.0002 | 0.0726±0.0002 | 0.2448±0.0005 | 0.2381±0.0014 |
| **SGL-ED** | 0.0438±0.0006 | 0.0662±0.0002 | 0.1867±0.0014 | **0.1898±0.0007** | 0.0574±0.0007 | 0.0778±0.0002 | 0.2458±0.0017 | **0.2554±0.0010** |
| **SimGCL** | **0.0472±0.0005** | 0.0708±0.0005 | 0.1811±0.0005 | 0.1750±0.0009 | **0.0608±0.0007** | 0.0826±0.0006 | 0.2378±0.0006 | 0.2356±0.0010 |
| **XSimGCL** | 0.0472±0.0003 | **0.0711±0.0004** | **0.1957±0.0005** | 0.1848±0.0004 | 0.0606±0.0005 | **0.0831±0.0006** | **0.2604±0.0010** | 0.2518±0.0006 |
| **NCL** | 0.0414±0.0002 | 0.0660±0.0004 | 0.1888±0.0004 | 0.1737±0.0012 | 0.0539±0.0002 | 0.0770±0.0005 | 0.2523±0.0007 | 0.2348±0.0022 |
| **TaxPro-CL** | 0.0457±0.0003 | 0.0670±0.0004 | 0.1761±0.0012 | 0.1683±0.0011 | 0.0587±0.0004 | 0.0781±0.0004 | 0.2301±0.0018 | 0.2246±0.0015 |

Source: `log/p0/baseline/<model>/<dataset>/` and `log/p0/taxprocl/<dataset>/`
(`final_test_group_metrics.json` per seed). Run
`python -m tools.analysis.mid_tail_degree6_10` then
`python -m tools.analysis.main_results_table` to regenerate this table
(mean±std, all five groups, bold/underline ranking) exactly as reported in
the manuscript, including the Mid-Tail column omitted above.

## Repository Structure

```text
TaxPro-CL/
├── configure/                 Model hyperparameter files
├── dataset_verify/            Locked train, validation, and test splits
├── metadata/taxonomy_variants/ Generated TaxProCL taxonomy artifacts
├── models/                    Baseline and TaxProCL implementations
├── preprocessed/              Evaluation protocol artifacts
├── tools/                     Reproducible research command packages
│   ├── data/                  Download, metadata preparation, and audits
│   ├── protocol/              Protocol builders and gate validation
│   ├── experiments/           Baseline, TaxProCL, and ablation runners
│   ├── analysis/              Result compilation, statistics, and evidence
│   └── repository/            Preflight and repository manifests
├── utility/                   Data, loss, training, and evaluation components
├── main.py                    Model entry point
└── requirements.txt           Python dependencies
```

## Reproducibility Artifacts

Shipped with this repository (no training required to inspect them):

- `results/results_manifest.csv`: run-level provenance for every reported
  checkpoint (dataset, variant, seed, config/split/checkpoint hashes,
  selection metric, which table each run backs).
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

See [`tools/README.md`](tools/README.md) for the complete command catalog and
the migration table from the former flat tool layout.

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

Downloaded `.gz` and `.zip` archives are deleted after successful extraction
and validation. Pass `--keep-archives` only when an archive must be retained
temporarily. Raw metadata remains local and is excluded from Git; the source
URLs and extracted-file inventory are stored in `manifests/metadata_sources.json`.

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

## Expected Run Outputs

A completed training run contains:

- `training.log`
- `run_manifest.json`
- `best_validation_model.pt`
- `last_model.pt`
- `final_test_metrics.json`
- `final_test_group_metrics.json`

The group metrics include Precision, Recall, and NDCG at K=10 and K=20 for
Overall, Near-cold, Long-tail, and Warm slices. Non-overall groups also record
eligible users, relevant items, and positive interactions.

## Citation

If you use this repository, please cite the associated TaxPro-CL manuscript.
The entry below uses placeholder author names pending the final accepted
version; replace it (and `CITATION.cff`) once the paper is published.

```bibtex
@article{taxprocl2026,
  title   = {TaxPro-CL: Taxonomy-Guided Perturbation for Sparse-Item
             Recommendation in Graph Contrastive Learning -- A Controlled
             Ablation Study},
  author  = {Author 1 and Author 2 and Author 3 and Author 4},
  journal = {Journal of Intelligent Information Systems},
  year    = {2026},
  note    = {Manuscript under review; update year/volume/DOI once published}
}
```

Machine-readable citation metadata for this repository (as software, separate
from the paper above) is provided in `CITATION.cff`.

## License And Data

This repository distributes code and experiment metadata. Dataset access and use
remain subject to the respective source licenses and terms of service.
