# -*- coding: utf-8 -*-
"""Builds claim_evidence.csv -- maps every major quantitative claim in
main.tex/ESM_1.tex to its location, dataset, configuration, metric, group,
target table, and evidence source (run_id pattern into results_manifest.csv/
metrics_seed.csv where applicable, or the specific analysis script/results
json for bootstrap/derived statistics). Covers every "significantly worse",
"monotonic", "improves", "stable" claim and every headline percentage;
does not re-derive numbers, only records where each one comes from.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ROWS = [
    # (location, claim_summary, dataset, config_comparison, metric, group, target_table, evidence_type, source)
    ("Abstract; §5.1", "TaxPro-CL highest mean Recall@20 on Near-Cold, all 4 datasets", "all 4", "TaxPro-CL-main vs. 5 baselines", "Recall@20", "near_cold", "Table S1", "point estimate, 3-seed mean", "metrics_seed.csv (variant=TaxPro-CL-main)"),
    ("Abstract; §5.6", "Bootstrap interval excludes zero only on Amazon-Book (Near-Cold)", "amazon-book", "TaxPro-CL vs. SimGCL", "Recall@20 diff.", "near_cold", "Table 11 (bootstrap)", "per-user bootstrap CI, 2000 resamples", "tools/analysis/seed_matched_bootstrap.py -> results/ (Table 11 source)"),
    ("Abstract; §5.1", "+10.03% to +18.98% on Long-Tail vs. SimGCL, all 4 datasets", "all 4", "TaxPro-CL-main vs. SimGCL-main", "Recall@20 % change", "long_tail", "Table 7 (deltas)", "point estimate, 3-seed mean", "metrics_seed.csv (variant=TaxPro-CL-main, SimGCL-main)"),
    ("§5.1", "NDCG@20 agrees with Recall@20 ranking in 6 of 8 cells, both exceptions on ACS", "all 4", "6 methods", "NDCG@20 vs Recall@20 rank", "near_cold, long_tail", "Table S21", "point estimate, 3-seed mean, rank comparison", "results/main_results_table.json; tools/analysis/main_results_table.py"),
    ("§5.1", "Yelp2018/Near-Cold near-zero at K=20, exactly zero at K=10, all 6 methods", "yelp2018", "6 methods", "Recall@10, Recall@20", "near_cold", "Table S1, Table S2", "point estimate, 3-seed mean", "metrics_seed.csv; results/week... K=10 rescoring script"),
    ("§5.1", "ACS/Long-Tail: TaxPro-CL 3rd of 6, within 1 SD of LightGCN/SGL-ED", "arts-crafts-and-sewing", "6 methods", "Recall@20", "long_tail", "Table S1", "point estimate + std, 3-seed", "metrics_seed.csv (variant=TaxPro-CL-main); Table S1 std column"),
    ("§5.1", "A6 (β=0.4) recovers 1st place on ACS/Long-Tail", "arts-crafts-and-sewing", "β=0.4 vs. β=0", "Recall@20", "long_tail", "Table S9", "point estimate, 3-seed mean", "log/p0/taxprocl/arts-crafts-and-sewing/ablation-A6-same_leaf_weight0.4"),
    ("§5.1", "K=10 rescoring does not change main finding's direction", "all 4", "TaxPro-CL vs. SimGCL", "Recall@10, NDCG@10 % change", "near_cold, long_tail", "Table S2", "point estimate, 3-seed mean", "same checkpoints as Table 7, cutoff changed only"),
    ("§5.1", "TaxPro-CL trails SimGCL/XSimGCL on Overall/Warm, all 4 datasets; last of 6 on MI/ACS", "all 4", "6 methods", "Recall@20", "overall, warm", "Table S1", "point estimate, 3-seed mean", "metrics_seed.csv (variant=TaxPro-CL-main)"),
    ("§5.1", "Baseline rank and SimGCL-relative drop magnitude do not track together (Yelp largest drop but ranks 3rd; MI smallest drop but ranks last)", "yelp2018, musical-instruments", "TaxPro-CL vs. SimGCL", "Recall@20 % change, rank", "overall, warm", "Table 7 (deltas), Table S1", "point estimate, 3-seed mean", "metrics_seed.csv"),
    ("§5.1", "TaxPro-CL highest harmonic mean on 3 of 4 datasets, 3rd on ACS -- NOTE: computed from a single representative checkpoint per method/dataset (validation-selected), not the 3-seed mean Recall@20 in Table S1, so it will not exactly reconcile via H=2*LT*Overall/(LT+Overall) applied to Table S1's means", "all 4", "6 methods", "harmonic mean(LT, Overall)", "n/a (derived)", "Table S10-S13", "point estimate, single selected checkpoint (see beyond_accuracy.py:select_checkpoint)", "tools/analysis/beyond_accuracy.py -> results/beyond_accuracy.json"),
    ("§5.1", "NCL higher LT-share/coverage but lower LT-Recall@20 on ACS (17.12%/46.47% vs. 14.46%/29.80%; Recall 17.025 vs 22.112e-3)", "arts-crafts-and-sewing", "NCL vs. TaxPro-CL", "coverage@20, LT-share@20, Recall@20", "long_tail", "Table S13, Table S1", "point estimate", "results/beyond_accuracy.json; metrics_seed.csv"),
    ("§5.2", "Rank-audit: 41.89-57.81% of users have improved TaxPro-CL rank across 8 cells", "all 4", "TaxPro-CL vs. SimGCL, seed-matched", "rank improvement %", "near_cold, long_tail", "Table S14", "seed-matched aggregate, 3 seed pairs", "tools/analysis/rank_audit_seed_matched.py -> results/"),
    ("§5.2", "Yelp2018 audit mean negative despite higher 3-seed Recall@20 (sign mismatch, expected)", "yelp2018", "TaxPro-CL vs. SimGCL", "mean rank diff. vs Recall@20", "near_cold, long_tail", "Table S14, Table S1", "seed-matched aggregate", "Table S14 mean diff column vs Table S1 Recall@20"),
    ("§5.2", "MI/Near-Cold only cell with more top-20 exits than entries", "musical-instruments", "TaxPro-CL vs. SimGCL", "entered/exited top-20 counts", "near_cold", "Table S14", "seed-matched aggregate", "Table S14 Entered/Exited columns"),
    ("§5.3 (A1)", "TaxPro-CL beats LightGCN on NC/LT, 3 of 4 datasets; ACS/LT exception -3.96%", "all 4", "TaxPro-CL vs. LightGCN", "Recall@20 % change", "near_cold, long_tail", "Table 7 (deltas)", "point estimate, 3-seed mean", "metrics_seed.csv"),
    ("§5.3 (A2)", "TaxPro-CL beats SimGCL on NC/LT all 4 datasets, trails on Overall/Warm all 4", "all 4", "TaxPro-CL vs. SimGCL", "Recall@20 % change", "all 4 groups", "Table 7 (deltas)", "point estimate, 3-seed mean", "metrics_seed.csv"),
    ("§5.3 (A3)", "epsilon_max sweep 0.01-0.80: Long-Tail/Overall/Warm flat, Near-Cold non-monotonic (dips at 0.40, recovers at 0.80)", "amazon-book", "epsilon_max in {0.01,...,0.80}", "Recall@20", "all 4 groups", "Table S4", "point estimate, 3-seed mean", "log/p0/taxprocl/amazon-book/... epsilon_max sweep runs"),
    ("§5.3 (A4)", "Temperature sweep: Near-Cold peaks at tau=0.1, ~3x higher than at either end", "amazon-book", "temperature in {0.05,0.10,0.20}", "Recall@20", "near_cold", "Table S4", "point estimate, 3-seed mean", "log/p0/taxprocl/amazon-book/... temperature sweep runs"),
    ("§5.3 (A4)", "Temperature-matched (tau=0.2) reverses ranking: TaxPro-CL trails SimGCL on all 4 groups", "amazon-book", "TaxPro-CL(tau=0.2) vs. SimGCL(tau=0.2)", "Recall@20", "all 4 groups", "main.tex prose only (§5.3)", "point estimate, 3-seed mean", "log/p0/taxprocl/amazon-book/... temperature-matched run"),
    ("§5.3 (A5)", "Leaf vs. Parent granularity: <1.5% Recall@20 diff, 98.74% top-20 overlap", "amazon-book", "prototype_mode=leaf vs. parent", "Recall@20, top-20 overlap", "all 4 groups", "Table S7", "point estimate, 3-seed mean", "tools/analysis/ (A5 comparison scripts)"),
    ("§5.3 (A6)", "beta=0.4 improves all 4 groups vs. beta=0 on ACS, +2.46% to +5.24%", "arts-crafts-and-sewing", "same_leaf_weight=0.4 vs. 0.0", "Recall@20 % change", "all 4 groups", "Table S9", "point estimate, 3-seed mean", "log/p0/taxprocl/arts-crafts-and-sewing/ablation-A6-same_leaf_weight0.4"),
    ("§5.4", "Direction factorial (V1-V0, V3-V2): 2 of 16 NC/LT cells exclude zero, both AB/Near-Cold, both against taxonomy", "all 4", "V1-V0, V3-V2", "Recall@20 diff., bootstrap CI", "near_cold, long_tail", "Table S16", "per-user bootstrap CI, pooled 3 seeds", "tools/analysis/factorial_direction_bootstrap.py -> results/factorial_direction_bootstrap_full_FIXED.json"),
    ("§5.4", "Epsilon-adaptivity factorial (V2-V0, V3-V1): 1 of 16 NC/LT cells excludes zero (Yelp/LT, V2-V0), against adaptive", "all 4", "V2-V0, V3-V1", "Recall@20 diff., bootstrap CI", "near_cold, long_tail", "Table S20", "per-user bootstrap CI, pooled 3 seeds", "same source as S16 (shared script/output)"),
    ("§5.4", "Overall/Warm direction factorial: 11 of 16 cells exclude zero, all against taxonomy", "all 4", "V1-V0, V3-V2", "Recall@20 diff., bootstrap CI", "overall, warm", "Table S22", "per-user bootstrap CI, pooled 3 seeds", "same source as S16, Overall/Warm groups"),
    ("§5.4", "Warm-start removal raises Long-Tail Recall@20 under both direction controls (+18.90% random, +21.38% taxonomy)", "amazon-book", "warm_start_epochs=0 vs. 20", "Recall@20 % change, bootstrap CI", "long_tail", "Table S23", "per-user bootstrap CI, pooled 3 seeds", "tools/analysis/warmstart_removal_bootstrap.py -> results/warmstart_removal_bootstrap.json"),
    ("§5.4", "Warm-start removal: Overall/Warm opposite signs between random (+) and taxonomy (-) direction controls", "amazon-book", "warm_start_epochs=0 vs. 20", "Recall@20 diff., bootstrap CI", "overall, warm", "Table S23", "per-user bootstrap CI, pooled 3 seeds", "same source as above"),
    ("§5.4", "View cosine similarity: TaxPro-CL 0.99992-0.99999, SimGCL 0.99991-0.99996, ratio 1.2-3.0x", "amazon-book", "TaxPro-CL-main vs. SimGCL-main", "cos(z_i^a, z_i^b)", "near_cold, mid_tail, warm", "Table S25", "per-item cosine, averaged per group, 3 seeds", "tools/analysis/view_cosine_by_degree.py -> results/a4_cosine_seed{42,0,1}.json"),
    ("§3.2; ESM S8", "isotropic_blend: AB degrades monotonically 0.25->0.75; Yelp flat", "amazon-book, yelp2018", "isotropic_blend in {0.25,0.5,0.75}", "Recall@20", "all 4 groups", "Table S8", "point estimate, 3-seed mean", "log/p0/taxprocl/{amazon-book,yelp2018}/gvhd-isoblend{0.25,0.5,0.75}"),
    ("§4.1", "Taxonomy-policy sweep: margin small on every dataset (within 1 SD Yelp/MI/ACS, comparable on AB)", "all 4", "no_merge/merge_t5/t10/t15", "Recall@20 Overall (validation)", "overall", "Table S3", "point estimate + std, 3-seed", "Table S3 mean+-std columns"),
    ("§4.1", "Leaf-size skew: 49.03%/50.34% of leaves are size 1-2 on MI/ACS, vs. 1.60%/32.52% on AB/Yelp", "all 4", "selected taxonomy policy", "leaf count, item share", "n/a (catalog stat)", "Table S18", "computed directly from taxonomy build", "dataset_verify/<dataset>/... taxonomy build output"),
    ("§4.3.1", "Near-Cold split-carve artifact: 48.79% (AB), 52.97% (Yelp), 56.12% (MI), 55.09% (ACS)", "all 4", "train-degree 1-5 items, pool vs. train degree", "item count, %", "near_cold", "Table S24", "computed directly from saved split files", "dataset_verify/<dataset>/{train,validation,test}.txt (this session's script)"),
    ("§5.5", "Prototype-to-embedding distance: 0.78-0.87 (AB/Yelp), 0.50-0.52 (MI/ACS) even at leaf size 1", "all 4", "leaf size 1 (singleton leaves)", "||p_l - e_i|| / ||e_i||", "n/a (catalog stat)", "Table S19", "computed from trained checkpoint embeddings", "tools/analysis/ (A5/prototype-distance script)"),
    ("§5.5", "Prototype-construction variants (leaf_uniform, leave_one_out, rescue) vs. TaxPro-CL-main: leave_one_out bit-for-bit identical on all 4 groups; leaf_uniform excludes zero only on Long-Tail (negative); rescue excludes zero on no group", "amazon-book", "leaf_uniform/leave_one_out/rescue vs. TaxPro-CL-main", "Recall@20 diff., bootstrap CI", "all 4 groups", "Table S17", "per-user bootstrap CI, pooled 3 seeds", "tools/analysis/leaf_variants_bootstrap.py, rescue_vs_variants_bootstrap.py"),
    ("§5.6", "5 of 8 dataset x group cells exclude zero (main bootstrap, TaxPro-CL vs. SimGCL)", "all 4", "TaxPro-CL-main vs. SimGCL-main", "Recall@20 diff., bootstrap CI", "near_cold, long_tail", "Table 11 (bootstrap)", "per-user bootstrap CI, 2000 resamples, seed-averaged", "tools/analysis/seed_matched_bootstrap.py"),
    ("§5.6", "Per-seed-pair: Long-Tail unanimous positive sign on all 4 datasets (3 of 3 pairs); Near-Cold disagreement on MI (1 of 3 negative), ACS (2 of 3 negative)", "all 4", "TaxPro-CL vs. SimGCL, same-seed pairs", "Recall@20 diff. sign, CI-excludes-0 count", "near_cold, long_tail", "Table S6", "per-seed-pair bootstrap", "same source as Table 11, disaggregated by seed pair"),
    ("§5.6", "Wilcoxon: minimum raw p=0.25 at n=3 seeds; Holm-corrected p capped at 1.0 for all comparisons", "all 4", "A1/A2/A2' vs. TaxPro-CL", "Wilcoxon signed-rank p, rank-biserial", "near_cold, long_tail", "Table S5", "exact two-sided Wilcoxon, Holm-Bonferroni corrected", "tools/analysis/ (Wilcoxon script)"),
]

FIELDS = ["location", "claim_summary", "dataset", "config_or_comparison", "metric", "group", "target_table", "evidence_type", "source"]

out_path = ROOT / "results" / "claim_evidence.csv"
with out_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(FIELDS)
    w.writerows(ROWS)

print(f"Wrote {len(ROWS)} rows to {out_path}")
