# TaxPro-CL Tools

This directory contains the reproducible command-line workflows used to build
data artifacts, validate the research protocol, run experiments, compile
results, and export evidence.

Run commands from the repository root with Python module syntax:

```powershell
python -m tools.<package>.<command> [options]
```

Module syntax keeps imports independent of the operating system and avoids
relying on the current script directory.

## Layout

```text
tools/
├── data/          Dataset download, metadata preparation, and data audits
├── protocol/      Protocol construction and deterministic gate validation
├── experiments/   Baseline, TaxProCL, screening, and ablation runners
├── analysis/      Result compilation, statistics, and evidence export
├── repository/    Preflight checks and repository manifests
└── _shared/       Private implementation helpers; not a public CLI surface
```

## Data Commands

| Command | Purpose | Primary output |
| --- | --- | --- |
| `tools.data.download` | Download datasets and raw metadata | `dataset/`, `metadata/` |
| `tools.data.build_splits` | Apply iterative 5-core filtering and build deterministic train/validation/test splits | staged splits and `split_manifest.json` |
| `tools.data.prepare_metadata` | Normalize and merge metadata | merged metadata and item-taxonomy mappings |
| `tools.data.metadata_manifest` | Inventory extracted metadata | `manifests/metadata_sources.json` |
| `tools.data.audit` | Audit a prepared dataset and taxonomy artifacts | `results/data_audit/` |

Examples:

```powershell
python -m tools.data.download --force
python -m tools.data.build_splits
python -m tools.data.prepare_metadata --domain all
python -m tools.data.metadata_manifest
python -m tools.data.audit --dataset amazon-book
```

Downloaded `.gz`, `.zip`, and partial files are removed after successful
extraction unless `--keep-archives` is explicitly supplied.

`tools.data.build_splits` stages outputs under
`results/data_splits/split_seed_42/` by default. Review the manifest and hashes
before running `python -m tools.data.build_splits --promote --force`. Promotion
changes the paper protocol and therefore requires rebuilding all protocol and
taxonomy artifacts before any model run.

The main paper split uses seed 42. Seeds 123 and 2026 are available for split
sensitivity checks through `--split-seed`; they must be staged separately and
must not replace the main split during ordinary model runs.

## Protocol Commands

| Command | Purpose |
| --- | --- |
| `tools.protocol.build` | Build evaluation groups, taxonomy variants, audits, and the experiment protocol |
| `tools.protocol.validate_determinism` | Rebuild artifacts and compare their hashes |
| `tools.protocol.validate_gate` | Validate the complete G1/G1.5 handoff |

```powershell
python -m tools.protocol.build --build
python -m tools.protocol.validate_determinism
python -m tools.protocol.validate_gate
```

These commands must complete before official training. They do not select a
configuration from test metrics.

## Experiment Commands

| Command | Purpose |
| --- | --- |
| `tools.experiments.run_baselines` | Run the six inherited baselines |
| `tools.experiments.run_taxpro` | Run protocol-scoped TaxProCL configurations |
| `tools.experiments.run_ablation` | Run screening and controlled ablation studies (A3/A4/A5) |
| `tools.experiments.run_stability_smoke` | Run engineering-only stability checks |

Examples:

```powershell
python -m tools.experiments.run_baselines --dataset amazon-book --seeds 42 --device cpu --smoke
python -m tools.experiments.run_taxpro --dataset amazon-book --seeds 42 --taxonomy-policy no_merge --device cpu --smoke
python -m tools.experiments.run_ablation --study epsilon --datasets amazon-book --seeds 42 --dry-run
```

Use `--device cuda` for official GPU runs. CPU smoke and stability outputs are
engineering diagnostics and are not paper-eligible results.

### Reproducing the RQ5 2x2 factorial (V0-V3; paper Section 5.4, Tables 6/11, Online Resource 1 Tables S16/S20/S22)

The factorial holds temperature, `temperature_user`, `gamma_cold`, and
`warm_start_epochs` fixed at each dataset's own main-configuration value
(same values as the "TaxPro-CL main results" commands above) and varies only
two factors: `--augmentation-direction` (`taxonomy` vs. `random`) and
`--use-adaptive-epsilon` (`true` vs. `false`). The four combinations are:

| Variant | Direction | Adaptive epsilon | Meaning |
| --- | --- | --- | --- |
| V0 | random | false | both factors at their non-taxonomy/non-adaptive control value |
| V1 | taxonomy | false | direction only |
| V2 | random | true | epsilon-adaptivity only |
| V3 | taxonomy | true | both factors at the main-configuration value |

Run all four, 3 seeds each, per dataset (fixed factors shown per dataset;
omit a flag to keep its `configure/TaxPro-CL.txt` default). **The `--config-id`
values below are not arbitrary** -- they must match exactly what
`tools.analysis.factorial_direction_bootstrap.DATASET_DIRS` expects, since
that is how the bootstrap script locates each run's checkpoints:

```powershell
# Amazon-Book (no extra fixed-factor overrides needed -- file defaults apply)
python -m tools.experiments.run_taxpro --dataset amazon-book --seeds 42 0 1 --config-id A2-V0 --augmentation-direction random --use-adaptive-epsilon false
python -m tools.experiments.run_taxpro --dataset amazon-book --seeds 42 0 1 --config-id A2-V1 --augmentation-direction taxonomy --use-adaptive-epsilon false
python -m tools.experiments.run_taxpro --dataset amazon-book --seeds 42 0 1 --config-id A2-V2 --augmentation-direction random --use-adaptive-epsilon true
python -m tools.experiments.run_taxpro --dataset amazon-book --seeds 42 0 1 --config-id A2-V3 --augmentation-direction taxonomy --use-adaptive-epsilon true

# Yelp2018 (fixed factor: temperature=0.125)
python -m tools.experiments.run_taxpro --dataset yelp2018 --seeds 42 0 1 --config-id A2-V0 --temperature 0.125 --augmentation-direction random --use-adaptive-epsilon false
python -m tools.experiments.run_taxpro --dataset yelp2018 --seeds 42 0 1 --config-id A2-V1 --temperature 0.125 --augmentation-direction taxonomy --use-adaptive-epsilon false
python -m tools.experiments.run_taxpro --dataset yelp2018 --seeds 42 0 1 --config-id A2-V2 --temperature 0.125 --augmentation-direction random --use-adaptive-epsilon true
python -m tools.experiments.run_taxpro --dataset yelp2018 --seeds 42 0 1 --config-id A2-V3 --temperature 0.125 --augmentation-direction taxonomy --use-adaptive-epsilon true

# Musical-Instruments (fixed factors: temperature=0.15, gamma-cold=5.0).
# V3 is intentionally NOT run here: it reuses the "TaxPro-CL main results"
# run below bit-for-bit (same configuration), so run that once and point
# the bootstrap script at it instead of training a fourth variant.
python -m tools.experiments.run_taxpro --dataset musical-instruments --seeds 42 0 1 --config-id gvhd-taxctrl-V0 --temperature 0.15 --gamma-cold 5.0 --augmentation-direction random --use-adaptive-epsilon false
python -m tools.experiments.run_taxpro --dataset musical-instruments --seeds 42 0 1 --config-id gvhd-taxctrl-V1 --temperature 0.15 --gamma-cold 5.0 --augmentation-direction taxonomy --use-adaptive-epsilon false
python -m tools.experiments.run_taxpro --dataset musical-instruments --seeds 42 0 1 --config-id gvhd-taxctrl-V2 --temperature 0.15 --gamma-cold 5.0 --augmentation-direction random --use-adaptive-epsilon true
python -m tools.experiments.run_taxpro --dataset musical-instruments --seeds 42 0 1 --config-id taxpro-cl-FINAL-no_merge-temp0.15-gammacold5.0 --temperature 0.15 --gamma-cold 5.0 --augmentation-direction taxonomy --use-adaptive-epsilon true

# Arts-Crafts-and-Sewing (fixed factor: temperature=0.125). Same V3 reuse note as above.
python -m tools.experiments.run_taxpro --dataset arts-crafts-and-sewing --seeds 42 0 1 --config-id gvhd-taxctrl-V0 --temperature 0.125 --augmentation-direction random --use-adaptive-epsilon false
python -m tools.experiments.run_taxpro --dataset arts-crafts-and-sewing --seeds 42 0 1 --config-id gvhd-taxctrl-V1 --temperature 0.125 --augmentation-direction taxonomy --use-adaptive-epsilon false
python -m tools.experiments.run_taxpro --dataset arts-crafts-and-sewing --seeds 42 0 1 --config-id gvhd-taxctrl-V2 --temperature 0.125 --augmentation-direction random --use-adaptive-epsilon true
python -m tools.experiments.run_taxpro --dataset arts-crafts-and-sewing --seeds 42 0 1 --config-id taxpro-cl-FINAL-no_merge-temp0.125-gammacold1.5-sameleaf0 --temperature 0.125 --augmentation-direction taxonomy --use-adaptive-epsilon true
```

V3 is, by construction, the same configuration as that dataset's main result
(Section 5.1) but is **not guaranteed to be the identical trained checkpoint**
-- see `results/results_manifest.csv` (`v3_is_same_checkpoint_as_taxprocl_main`
column) for which datasets reuse the main checkpoint (Musical-Instruments,
Arts-Crafts-and-Sewing, hence no separate V3 command above) vs. train V3
separately (Amazon-Book, Yelp2018, hence a distinct `A2-V3` command above).
Counting runs: 4 datasets x 4 variants x 3 seeds = 48 cells, minus 2 datasets'
V3 (Musical-Instruments, Arts-Crafts-and-Sewing) x 3 seeds = 6 cells reused
from each dataset's own "TaxPro-CL main results" run (README.md) instead of
being trained separately -- 42 new training runs total, plus the 4 datasets'
main-results runs (needed anyway, and already the V3 checkpoint for 2 of
them). Once these exist, feed them to `tools.analysis.factorial_direction_bootstrap`
(Table 6/11/S16) and the same script's Overall/Warm extension (Table S22).

The companion warm-start-removal check (ESM Table S23) uses the V3 command
above with `--warm-start-epochs 0` added, Amazon-Book only, both directions.

## Analysis Commands

| Command | Purpose |
| --- | --- |
| `tools.analysis.compile_runs` | Inventory TaxProCL run manifests |
| `tools.analysis.compile_ablation_sweep` | Compile complete main/policy-sweep/ablation metrics and support counts |
| `tools.analysis.statistics` | Run paired seed-level tests and Holm correction |
| `tools.analysis.export_evidence` | Export a small checksummed evidence bundle |
| `tools.analysis.bootstrap_ci` | Primary TaxPro-CL-vs-baseline bootstrap CI (paper's main statistical evidence, e.g. Table 12) |
| `tools.analysis.seed_matched_bootstrap` | Seed-matched (42-42/0-0/1-1) bootstrap CI, one baseline vs. TaxPro-CL, one dataset |
| `tools.analysis.factorial_direction_bootstrap` | V0-V3 direction factorial bootstrap, all 4 datasets (paper Section 5.4, Table 11) |
| `tools.analysis.mid_tail_degree6_10` | Disjoint degree-6-10 Mid-Tail slice, all 6 methods x 4 datasets, cross-checked |
| `tools.analysis.main_results_table` | Full mean±std results table + LaTeX, all 6 methods x 4 datasets x 5 groups (paper Tables 7-10) |
| `tools.analysis.rank_audit_seed_matched` | Seed-matched item-rank audit, one baseline vs. TaxPro-CL |
| `tools.analysis.beyond_accuracy` | Coverage@20, Long-Tail share@20, avg. recommended popularity, all 6 methods x 4 datasets |
| `tools.analysis.view_cosine_by_degree` | Cosine similarity between the two perturbed views, by degree group (paper Section 5.4, ESM Table S25) |
| `tools.analysis.rescue_vs_variants_bootstrap` | Bootstrap CI, a rescue/ablation variant vs. an A2 factorial variant (Amazon-Book) |
| `tools.analysis.leaf_variants_bootstrap` | Bootstrap CI, prototype-construction variants (`leaf_uniform`/`leave_one_out`) vs. V3, Amazon-Book (ESM Table S17) |
| `tools.analysis.warmstart_removal_bootstrap` | Bootstrap CI, `warm_start_epochs=0` vs. main config, both direction controls, Amazon-Book (ESM Table S23) |
| `tools.analysis.build_results_manifest` | GVHD-A1 audit trail: `results/results_manifest.csv` (run-level provenance: config/split/taxonomy/checkpoint SHA256) and `results/metrics_seed.csv` (per-seed absolute metrics), from `run_manifest.json`/`final_test_group_metrics.json` only -- no invented values |
| `tools.analysis.build_claim_evidence` | GVHD-A6 audit trail: `results/claim_evidence.csv`, mapping every major quantitative claim in the manuscript to its dataset/configuration/metric/target table/source script |

```powershell
python -m tools.analysis.compile_runs
python -m tools.analysis.compile_ablation_sweep
python -m tools.analysis.statistics --reference-config <id> --candidate-config <id>
python -m tools.analysis.export_evidence --run-dir <completed-run> --output-dir <directory>
python -m tools.analysis.mid_tail_degree6_10
python -m tools.analysis.main_results_table
python -m tools.analysis.factorial_direction_bootstrap
python -m tools.analysis.seed_matched_bootstrap --dataset <ds> --baseline-name <model>
python -m tools.analysis.rank_audit_seed_matched --baselines SimGCL
python -m tools.analysis.beyond_accuracy
python -m tools.analysis.view_cosine_by_degree
python -m tools.analysis.leaf_variants_bootstrap
python -m tools.analysis.warmstart_removal_bootstrap
python -m tools.analysis.build_results_manifest
python -m tools.analysis.build_claim_evidence
```

Compilers never synthesize missing metrics. Evidence export accepts only a
completed run and excludes large checkpoints. The bootstrap/audit/coverage
scripts are inference-only (reuse trained checkpoints via
`checkpoint_selection.select_checkpoint`, no retraining) and require the
relevant run directories to already exist -- see each script's module
docstring for exactly which ones and which manuscript table it reproduces.

## Repository Commands

| Command | Purpose |
| --- | --- |
| `tools.repository.preflight` | Validate dependencies, data, protocol artifacts, and model imports |
| `tools.repository.build_manifest` | Build migration and source inventories |
| `tools.repository.validate_manifest` | Validate the migration manifest against the current repository |

```powershell
python -m tools.repository.build_manifest
python -m tools.repository.validate_manifest
python -m tools.repository.preflight --require-cuda
```

## Conventions

- Public commands live in one responsibility-based package.
- Public command modules expose `main` or a module entry point and return a
  nonzero process status on validation failure.
- Shared helpers belong in `_shared/` and must not become user-facing commands.
- All paths resolve from the repository root, not from the caller's shell.
- Machine-readable artifacts use English field names and UTF-8 encoding.
- New output directories must be isolated by dataset, configuration, and seed.
- A command that consumes test metrics must not select hyperparameters or
  checkpoints.
- Tests import command modules through their full package path.

Run `python -m tools.<package>.<command> --help` for command-specific options.

## Migration From The Legacy Layout

The tools restructure intentionally removes the former flat scripts. Use these
module replacements in automation and notebooks:

| Legacy path | Current module |
| --- | --- |
| `tools/download_data.py` | `tools.data.download` |
| `tools/prepare_metadata.py` | `tools.data.prepare_metadata` |
| `tools/build_metadata_manifest.py` | `tools.data.metadata_manifest` |
| `tools/audit_taxpro_data.py` | `tools.data.audit` |
| `tools/week3_protocol.py` | `tools.protocol.build` |
| `tools/validate_week3_determinism.py` | `tools.protocol.validate_determinism` |
| `tools/validate_gate_g1_5.py` | `tools.protocol.validate_gate` |
| `tools/running/*.py` | `tools.experiments.*` |
| `tools/compile_results.py` | `tools.analysis.compile_runs` |
| `tools/compile_week6_results.py` | `tools.analysis.compile_ablation_sweep` |
| `tools/statistical_analysis.py` | `tools.analysis.statistics` |
| `tools/export_evidence.py` | `tools.analysis.export_evidence` |
| `tools/preflight.py` | `tools.repository.preflight` |
| `tools/build_migration_manifest.py` | `tools.repository.build_manifest` |
| `tools/validate_migration.py` | `tools.repository.validate_manifest` |
