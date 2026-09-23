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

### Reproducing the RQ5 2x2 factorial (V0-V3; paper Section 5.4, Tables 8-9, Figure 2, Online Resource 1 Tables S13/S17/S19/S27)

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
# Amazon-Book (taxonomy_policy=merge_t10, the main configuration's policy and
# the configure/TaxPro-CL.txt default; passed explicitly here for clarity)
python -m tools.experiments.run_taxpro --dataset amazon-book --seeds 42 0 1 --taxonomy-policy merge_t10 --config-id A2-V0-mergedt10 --augmentation-direction random --use-adaptive-epsilon false
python -m tools.experiments.run_taxpro --dataset amazon-book --seeds 42 0 1 --taxonomy-policy merge_t10 --config-id A2-V1-mergedt10 --augmentation-direction taxonomy --use-adaptive-epsilon false
python -m tools.experiments.run_taxpro --dataset amazon-book --seeds 42 0 1 --taxonomy-policy merge_t10 --config-id A2-V2-mergedt10 --augmentation-direction random --use-adaptive-epsilon true
python -m tools.experiments.run_taxpro --dataset amazon-book --seeds 42 0 1 --taxonomy-policy merge_t10 --config-id A2-V3-warm20 --augmentation-direction taxonomy --use-adaptive-epsilon true

# Yelp2018 (fixed factors: temperature=0.125, temperature_user=0.15)
python -m tools.experiments.run_taxpro --dataset yelp2018 --seeds 42 0 1 --config-id A2-V0-tempuser0.15 --temperature 0.125 --temperature-user 0.15 --augmentation-direction random --use-adaptive-epsilon false
python -m tools.experiments.run_taxpro --dataset yelp2018 --seeds 42 0 1 --config-id A2-V1-tempuser0.15 --temperature 0.125 --temperature-user 0.15 --augmentation-direction taxonomy --use-adaptive-epsilon false
python -m tools.experiments.run_taxpro --dataset yelp2018 --seeds 42 0 1 --config-id A2-V2-tempuser0.15 --temperature 0.125 --temperature-user 0.15 --augmentation-direction random --use-adaptive-epsilon true
python -m tools.experiments.run_taxpro --dataset yelp2018 --seeds 42 0 1 --config-id A2-V3-tempuser0.15 --temperature 0.125 --temperature-user 0.15 --augmentation-direction taxonomy --use-adaptive-epsilon true

# Musical-Instruments (fixed factors: temperature=0.15, gamma-cold=5.0).
# V3 is intentionally NOT run here: it reuses the "TaxPro-CL main results"
# run below bit-for-bit (same configuration), so run that once and point
# the bootstrap script at it instead of training a fourth variant.
python -m tools.experiments.run_taxpro --dataset musical-instruments --seeds 42 0 1 --config-id taxctrl-V0 --temperature 0.15 --gamma-cold 5.0 --augmentation-direction random --use-adaptive-epsilon false
python -m tools.experiments.run_taxpro --dataset musical-instruments --seeds 42 0 1 --config-id taxctrl-V1 --temperature 0.15 --gamma-cold 5.0 --augmentation-direction taxonomy --use-adaptive-epsilon false
python -m tools.experiments.run_taxpro --dataset musical-instruments --seeds 42 0 1 --config-id taxctrl-V2 --temperature 0.15 --gamma-cold 5.0 --augmentation-direction random --use-adaptive-epsilon true
python -m tools.experiments.run_taxpro --dataset musical-instruments --seeds 42 0 1 --config-id taxpro-cl-FINAL-no_merge-temp0.15-gammacold5.0 --temperature 0.15 --gamma-cold 5.0 --augmentation-direction taxonomy --use-adaptive-epsilon true

# Arts-Crafts-and-Sewing (fixed factor: temperature=0.125). Same V3 reuse note as above.
python -m tools.experiments.run_taxpro --dataset arts-crafts-and-sewing --seeds 42 0 1 --config-id taxctrl-V0 --temperature 0.125 --augmentation-direction random --use-adaptive-epsilon false
python -m tools.experiments.run_taxpro --dataset arts-crafts-and-sewing --seeds 42 0 1 --config-id taxctrl-V1 --temperature 0.125 --augmentation-direction taxonomy --use-adaptive-epsilon false
python -m tools.experiments.run_taxpro --dataset arts-crafts-and-sewing --seeds 42 0 1 --config-id taxctrl-V2 --temperature 0.125 --augmentation-direction random --use-adaptive-epsilon true
python -m tools.experiments.run_taxpro --dataset arts-crafts-and-sewing --seeds 42 0 1 --config-id taxpro-cl-FINAL-no_merge-temp0.125-gammacold1.5-sameleaf0 --temperature 0.125 --augmentation-direction taxonomy --use-adaptive-epsilon true
```

V3 is, by construction, the same configuration as that dataset's main result
(Section 5.1) but is **not guaranteed to be the identical trained checkpoint**
-- see `results/results_manifest.csv` (`v3_is_same_checkpoint_as_taxprocl_main`
column) for which datasets reuse the main checkpoint (Musical-Instruments,
Arts-Crafts-and-Sewing, hence no separate V3 command above) vs. train V3
separately (Amazon-Book as `A2-V3-warm20`, Yelp2018 as `A2-V3-tempuser0.15`).
The Amazon-Book V3 run was trained on a second machine (Online Resource 1,
Table S23) and its test metrics are within 0.26% of the main checkpoint's;
`python -m tools.analysis.a2_mergedt10_factorial_recompute --v3-dir <main
checkpoint dir> --output <file>` repeats the Amazon-Book analysis with the main
checkpoint as V3.

PowerShell launchers for the reported reruns (idempotent, three seeds, explicit
output directories): `tools/experiments/rerun_amazonbook_factorial_mergedt10.ps1`
(Amazon-Book V0-V2 under `merge_t10`), `tools/experiments/rerun_yelp_factorial_tempuser_fix.ps1`
(Yelp2018 V0-V3 with `temperature_user=0.15`), and
`tools/experiments/run_a5_parent_mergedt10.ps1` (A5 parent granularity under
`merge_t10`). Musical-Instruments and Arts-Crafts-and-Sewing V0-V2 use the
`run_taxpro` commands above with each dataset's main-configuration flags.
Counting runs: 4 datasets x 4 variants x 3 seeds = 48 cells, minus 2 datasets'
V3 (Musical-Instruments, Arts-Crafts-and-Sewing) x 3 seeds = 6 cells reused
from each dataset's own "TaxPro-CL main results" run (README.md) instead of
being trained separately -- 42 new training runs total, plus the 4 datasets'
main-results runs (needed anyway, and already the V3 checkpoint for 2 of
them). Once these exist, feed them to `tools.analysis.factorial_direction_bootstrap`
(ESM Tables S13/S17/S19; direction and epsilon-adaptivity conditional effects, Near-Cold/Long-Tail and Overall/Warm) and `tools.analysis.factorial_interaction_bootstrap` (ESM Table S27).

The companion warm-start-removal check (ESM Table S20) trains `A2-V0-nowarm`
and `A2-V3-nowarm` with the corresponding Amazon-Book command above plus
`--warm-start-epochs 0` and compares them with `A2-V0-mergedt10` and
`A2-V3-warm20` (same taxonomy policy on both sides), Amazon-Book only.

## Analysis Commands

| Command | Purpose |
| --- | --- |
| `tools.analysis.compile_runs` | Inventory TaxProCL run manifests |
| `tools.analysis.compile_ablation_sweep` | Compile complete main/policy-sweep/ablation metrics and support counts |
| `tools.analysis.statistics` | Run paired seed-level tests and Holm correction |
| `tools.analysis.export_evidence` | Export a small checksummed evidence bundle |
| `tools.analysis.bootstrap_ci` | Primary TaxPro-CL-vs-baseline bootstrap CI (paper's main statistical evidence, Table 11) |
| `tools.analysis.seed_matched_bootstrap` | Seed-matched (42-42/0-0/1-1) bootstrap CI, one baseline vs. TaxPro-CL, one dataset |
| `tools.analysis.factorial_direction_bootstrap` | V0-V3 direction factorial bootstrap, all 4 datasets (paper Section 5.4, ESM Tables S13/S17/S19) |
| `tools.analysis.factorial_interaction_bootstrap` | Direction x epsilon-adaptivity interaction, all 4 datasets (ESM Table S27) |
| `tools.analysis.a2_mergedt10_factorial_recompute` | Amazon-Book factorial under `merge_t10`: Recall@20 and NDCG@20, interaction, and the warm-start-removal pair in one pass (ESM Tables S13/S13b/S17/S17b/S19/S20/S27) |
| `tools.analysis.a1a2_factorial_multiplicity` | Amazon-Book `no_merge` policy-sensitivity check (both metrics, all groups) plus sign-flip/Holm-Bonferroni multiplicity correction of the direction and epsilon-adaptivity 8-cell families (GPU-vectorized bootstrap/sign-flip; V63, ESM Tables S13c-S13e) |
| `tools.analysis.update_manifest_a1_nomerge_restored` | Restore the `used_in_tables` field for the Amazon-Book `no_merge` V0-V3 rows in `results/results_manifest.csv` (V63) |
| `tools.analysis.regenerate_fig3_factorial_forest` | Figure 2 (factorial forest plot) from the bootstrap JSONs |
| `tools.analysis.plot_main_figures` | Figure 1 (`Fig2.pdf`, Near-Cold/Long-Tail bars, from `main_results_table.json`) and Online Resource 1 Figure S2 (`Fig4.pdf`, Overall vs. Long-Tail Pareto view, from `beyond_accuracy.json`) |
| `tools.analysis.update_manifest_a2_mergedt10` | Append the Amazon-Book `merge_t10` factorial runs to `results/results_manifest.csv` and `results/metrics_seed.csv` |
| `tools.analysis.update_manifest_baselines_main` | Record the LightGCN/SGL-ED/XSimGCL/NCL main-comparison runs in `results/results_manifest.csv` and re-export their per-seed metrics (refuses exploratory A7 runs) |
| `tools.analysis.update_manifest_a5_mergedt10` | Append the Amazon-Book A5 parent-granularity runs (`merge_t10`) to the two manifests |
| `tools.analysis.update_manifest_yelp_tempuser_fix` | Append the Yelp2018 V0-V3 runs trained with `temperature_user=0.15` to the two manifests |
| `tools.analysis.a5_leaf_vs_parent_mergedt10` | A5 leaf vs. parent prototype granularity, policy-matched: Recall/NDCG bootstrap and top-20 overlap (Tables S7/S7b) |
| `tools.analysis.leaf_size_distribution` | Leaf-size distribution of the selected taxonomy policy (Table S15) |
| `tools.analysis.prototype_distance_by_leaf_size` | Prototype-to-embedding relative distance by leaf size (Table S16) |
| `tools.analysis.degree_transition` | Pool-to-train degree-bucket transition of the items (Table S21) |
| `tools.analysis.export_baseline_metrics_seed` | Export per-seed metrics of the baseline runs into `results/metrics_seed.csv` |
| `tools.analysis.mid_tail_degree6_10` | Disjoint degree-6-10 Mid-Tail slice, all 6 methods x 4 datasets, cross-checked |
| `tools.analysis.main_results_table` | Full mean±std results table + LaTeX, all 6 methods x 4 datasets x 5 groups (paper Tables 7-10) |
| `tools.analysis.rank_audit_seed_matched` | Seed-matched item-rank audit, one baseline vs. TaxPro-CL |
| `tools.analysis.beyond_accuracy` | Coverage@20, Long-Tail share@20, avg. recommended popularity, all 6 methods x 4 datasets |
| `tools.analysis.view_cosine_by_degree` | Cosine similarity between the two perturbed views, by degree group (paper Section 5.4, ESM Table S22) |
| `tools.analysis.rescue_vs_variants_bootstrap` | Bootstrap CI, a rescue/ablation variant vs. an A2 factorial variant (Amazon-Book) |
| `tools.analysis.leaf_variants_bootstrap` | Bootstrap CI, prototype-construction variants (`leaf_uniform`/`leave_one_out`) vs. TaxPro-CL-main, Amazon-Book (ESM Table S14) |
| `tools.analysis.warmstart_removal_bootstrap` | Bootstrap CI, `warm_start_epochs=0` vs. main config, both direction controls, Amazon-Book (ESM Table S20) |
| `tools.analysis.build_results_manifest` | Audit trail: `results/results_manifest.csv` (run-level provenance: config/split/taxonomy/checkpoint SHA256) and `results/metrics_seed.csv` (per-seed absolute metrics), from `run_manifest.json`/`final_test_group_metrics.json` only -- no invented values |
| `tools.analysis.build_claim_evidence` | Audit trail: `results/claim_evidence.csv`, mapping every major quantitative claim in the manuscript to its dataset/configuration/metric/target table/source script |

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

Reproduction means re-running the training and evaluation commands from the
data, configuration and seeds; trained checkpoints are outputs of those
commands, not inputs. Point `TAXPRO_OUTPUT_ROOT` at an empty directory for a
clean re-run so the analysis stage only sees the run families those commands
created. Inside a populated output tree, `checkpoint_selection` still resolves
the canonical run family of each (model, dataset): it ignores exploratory
families (names starting with `A7-`), and it stops with
`AmbiguousCheckpointFamilyError` when several canonical families remain and
none is pinned in `tests/Recommendation_system/config.py`
(`CHECKPOINT_FAMILY_OVERRIDES` / `CHECKPOINT_OVERRIDES`).

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

```powershell
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
