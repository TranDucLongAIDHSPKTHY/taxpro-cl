# Shipped results

Small analysis outputs behind the manuscript's tables and figures. Every file is
written by the script listed next to it (`python -m tools.analysis.<script>`);
`claim_evidence.csv` maps each quantitative statement of the paper to these
files and to the runs in `results_manifest.csv`. Trained checkpoints are not
shipped: the inference-only scripts re-read them from `log/` after the training
commands of the top-level README have been run. Table numbers refer to the main
paper (Tables 1-12, Figures 1-2) and to Online Resource 1 (Tables S1-S36d).

## Audit trail

| File | Content | Written by |
| --- | --- | --- |
| `results_manifest.csv` | One row per run: configuration, split, taxonomy, and checkpoint SHA256; selection metric; tables the run backs | `build_results_manifest`, `update_manifest_*` |
| `metrics_seed.csv` | Per-seed Recall/NDCG@10/20 by group for every reported run | `build_results_manifest`, `export_baseline_metrics_seed`, `update_manifest_*` |
| `claim_evidence.csv` | Quantitative claim -> table, dataset, metric, run, script | `build_claim_evidence` |

## Main comparison

| File | Manuscript item | Written by |
| --- | --- | --- |
| `mid_tail_degree6_10.json` | Mid-Tail slice and checkpoint resolution for Table S1 | `mid_tail_degree6_10` |
| `main_results_table.json` | Tables 7, S1, S18 (Recall@20 and NDCG@20, mean and std); input of Figure 1 | `main_results_table` |
| `{dataset}_seed_matched_bootstrap.json` (4 files) | Table 11 and Table S6 (per-user bootstrap, TaxPro-CL vs. SimGCL) | `seed_matched_bootstrap` |
| `a2_multiplicity_correction_signflip_100k.json` | Table S26 (sign-flip p, centred-bootstrap p, earlier tail p; Holm) | `a2_multiplicity_correction` |
| `a2_multiplicity_correction_signflip_1M_seed123.json` | Table S26 precision re-run (10^6 permutations, seed 123) | `a2_multiplicity_correction` |
| `a2_multiplicity_peruser_diffs.npz` | Seed-averaged per-user differences behind Table S26 | `a2_multiplicity_correction --save-diffs` |
| `statistical_tests_A1.csv`, `statistical_tests_A2_full.csv`, `statistical_tests_A2_xsimgcl.csv` | Table S5 (Wilcoxon + Holm; A1 LightGCN, A2 SimGCL, A2' XSimGCL) | `compile_ablation_sweep`, then `statistics` |
| `intrinsic_sparsity_bootstrap.json` | Table S25 (both degree definitions); its `published_table_s24_rate` key holds the Table S21 reclassification rate (the key name predates the current table numbering) | `intrinsic_sparsity_bootstrap` |
| `beyond_accuracy.json` | Tables S10-S10d, Figure S2 | `beyond_accuracy` |
| `harmonic_mean_3seed.json` | "H (3-seed mean)" column of Tables S10-S10d | computed from Table S1 values |
| `rank_audit_seed_matched.json` | Tables S11, S12 | `rank_audit_seed_matched` |
| `strict_cold_audit.json` | Table S34, Limitation 3 | `strict_cold_audit` |

## RQ5 factorial and prototype analyses

| File | Manuscript item | Written by |
| --- | --- | --- |
| `factorial_direction_bootstrap.json` | Tables 8, S13, S17, S19 (all datasets; Amazon-Book block: `merge_t10`) | `factorial_direction_bootstrap`, `merge_factorial_results` |
| `factorial_direction_bootstrap_yelp.json` | Yelp2018 rows (V0-V3 trained with `temperature_user=0.15`) | `factorial_direction_bootstrap --datasets yelp2018` |
| `factorial_amazon_book.json` | Amazon-Book `merge_t10` factorial, both metrics, interaction, warm-start removal (Tables S13, S13b, S17, S17b, S19, S20, S27) | `factorial_amazon_book` |
| `factorial_amazon_book_v3main.json` | Section S23: V3 replaced by the main checkpoint | `factorial_amazon_book --v3-dir <main run>` |
| `factorial_ndcg_bootstrap.json`, `factorial_ndcg_bootstrap_yelp.json` | Table S13b | `factorial_ndcg_bootstrap` |
| `epsilon_ndcg_bootstrap.json` | Table S17b | `epsilon_ndcg_bootstrap` |
| `factorial_interaction_bootstrap.json` | Table S27 | `factorial_interaction_bootstrap` |
| `warmstart_removal_bootstrap.json` | Table S20 | `warmstart_removal_bootstrap` |
| `factorial_multiplicity.json` | Tables S13c-S13e (Amazon-Book `no_merge` check; Holm families) | `factorial_multiplicity` |
| `factorial_holm_table.json` | Table S13e (component p, cell p, per-family and pooled-32 adjustment) | `factorial_holm_table` |
| `factorial_v3main_substitution.json` | Table S23b | `factorial_v3main_substitution` |
| `factorial_pvalues.csv` | Table S13e and Section 5.4 (full p-value vector: both contrasts' p and signs, cell p, Holm rank/threshold/decision/adjusted p per 8-cell family, adjusted p in the pooled 32-cell family) | `export_factorial_pvalues` |
| `factorial_decomposition_summary.json` | Tables S32, S33 | `factorial_decomposition_summary` |
| `Fig3.pdf` | Figure 2 (factorial forest plot) | `regenerate_fig3_factorial_forest` |
| `leaf_variants_bootstrap.json`, `rescue_vs_variants_bootstrap.json` | Tables 10, S14 | `leaf_variants_bootstrap`, `rescue_vs_variants_bootstrap` |
| `prototype_variants_overlap.json` | Table S14b | `prototype_variants_overlap` |
| `prototype_variants_ndcg.json` | Table S14c | `prototype_variants_ndcg` |
| `a5_leaf_vs_parent_mergedt10.json` | Tables S7, S7b (A5) | `a5_leaf_vs_parent_mergedt10` |
| `leaf_size_distribution.json` | Table S15 | `leaf_size_distribution` |
| `prototype_distance_by_space.json` | Table S16 | `prototype_distance_by_space` |
| `prototype_distance_by_leaf_size.json` | Table S16, "Layer 0 (lookup)" rows | `prototype_distance_by_leaf_size` |
| `view_cosine_seed{42,0,1}.json`, `view_cosine_by_degree_allseeds.json` | Table S22 (per seed, and mean over seeds) | `view_cosine_by_degree` |
| `realized_perturbation_norm.json` | Table S29 | `realized_perturbation_norm` |
| `expected_perturbation_by_degree.json` | Section S29, closed-form expected magnitude by degree group | analytic, from Eq. (2) |
| `valid_mask_count.json` | Items excluded by the validity mask (degree 0 or no valid leaf), per dataset | taxonomy variants and train degrees |

## Datasets, policies, and supplementary grids

| File | Manuscript item | Written by |
| --- | --- | --- |
| `degree_transition.json` | Table S21 | `degree_transition` |
| `policy_sweep_splits.json` | Table S3 and, for CDs-and-Vinyl, Table S30d | `policy_sweep_splits` |
| `cds_and_vinyl_summary.json` | Tables S30b, S30c | `cds_and_vinyl_summary` |
| `a7_grid_status.json` | Partial A7 grid summary at the 2026-09-15 data cut-off (not used for any reported table; see `a7_tuned_comparison.json`) | `a7_grid_status` |
| `a7_tuned_comparison.json` | Section 5.3 (A7), Tables S24-S24e (A7 grid: selected cells by rule Ov and rule R, statistics vs. TaxPro-CL) | `a7_tuned_comparison` |
| `validation_reselection_audit.json` | Table S35 (retrospective validation-only re-selection of the configuration) | `validation_reselection_audit` |
| `s35_rules.csv`, `s35_candidates.csv`, `s35_overall_rule_cells.csv` | Tables S35b-S35c (rule definitions and tie-breaks, candidate set, the four AB/Yelp cells under the Overall rule per seed, unrounded); re-runs and checks every selection | `s35_export` |
| `s35_results.csv` | Table S35 (per rule and dataset: selected configuration, test change vs. SimGCL, rank of six) | `s35_export` |
| `yelp_factorial_hits.csv`, `yelp_factorial_hits_summary.json` | Table S13g and Table 8 note a (Yelp2018 per-user Near-Cold/Mid-Tail hits of V0-V3, checked against the evaluator) | `yelp_factorial_hits` |
| `yelp_lt_decomposition.csv` | Tables S13g-S13h (per variant and seed: H, J, S, N, C_NC, C_MT; checkpoint, configuration, and split hashes) | `yelp_factorial_hits` |

## Pre-registered held-out evaluation (`confirmatory/`)

Protocol: `docs/confirmatory_protocol.md` (hashes in `docs/confirmatory_protocol*.sha256`).

| File | Manuscript item | Written by |
| --- | --- | --- |
| `screening_*.json` | Section S35 (screening of Digital_Music, Prime_Pantry, Office_Products) | `tools.data.screen_confirmatory_candidates`; taxonomy files from the metadata/taxonomy build |
| `confirmatory_selection.json`, `.sha256` | Table S36 (validation-only selection, frozen before the test split was opened) | `tools.experiments.run_confirmatory` |
| `confirmatory_analysis_office-products.json` | Table 12, Tables S36b-S36c | `tools.analysis.confirmatory_analysis` |
| `six_methods_office-products.json` | Table S36b, top block (six methods, rank of six; descriptive) | `tools.analysis.confirmatory_six_methods` |
| `peruser_office-products.csv.gz` | Per-user test Recall@20/NDCG@20 of every Near-Cold and Long-Tail user behind Table 12 and Table S36c | `tools.analysis.confirmatory_peruser` |
