# Confirmatory evaluation protocol (frozen before any data of the new dataset is downloaded)

Status: FROZEN on 2026-09-27. The SHA-256 of this file is recorded in
`docs/confirmatory_protocol.sha256`. Any change after that date goes to the
Deviations log at the end, never into the sections above it.

## 1. Purpose

Every result in the V64 manuscript was obtained on datasets, configurations and
taxonomy policies chosen during development with test metrics in view
(Limitation 10). This protocol evaluates TaxPro-CL once on a dataset that played
no role in development, with every choice fixed in advance and the test split
opened exactly once, after all selections are frozen.

## 2. Frozen code

Git HEAD `8cf16953b58ea90e4b6e9101d64c6f961fdc5db0` plus the working-tree files
below, identified by SHA-256 at freeze time:

| File | SHA-256 |
|---|---|
| models/TaxPro-CL.py | 82bf02e4227d987ec5687c477a2275e86650eb64f6e682eb5847728f54b41cf3 |
| models/SimGCL.py | 80d4357f7d4ffb2edce468825cb864b69bf07aded55941f878d1f6ead4535348 |
| models/XSimGCL.py | 6e7e02dd45a5486514a0de58a985d7c6b6ec6f6b2526157d99f3bf0084ce567c |
| models/LightGCN.py | 8c902657c30f7b86513c1dc983a7577f7831de0b88c9ce52bc8deb59bb344e62 |
| models/NCL.py | 0b8d832a5a1dccecd883e3138117d318d5daaef3ac638d0c4690caba095ecdbc |
| models/SGL.py | f1f83a405d23cf70c20a01878fcccf4f8e9504506417ca9b62d17d50a21e82d7 |
| utility/utility_train/batch_test.py | afc92d893abfddd80751b22905407e1ad72d45b1a4c3db6b1c6942ff3f3f4893 |
| configure/TaxPro-CL.txt | 5a9c05b50258031d734b154b03dfe2ebbcc481289d04d5f62ca2aca7ec14dbfd |
| configure/SimGCL.txt | e957ccd8faa41f82d7a32d3148506827279ff00276cad6449aa2aad44e342ccc |

The only code added after freeze is (a) an environment switch
`TAXPRO_DEFER_TEST=1` in `main.py` that skips the end-of-run test evaluation, and
(b) `tools/experiments/evaluate_sealed_test.py`, which later evaluates the saved
best-validation checkpoints on the test split with the unchanged
`batch_test.final_test`. Neither changes training, validation, or scoring. The
new Protocol B build script for the selected category is a copy of
`tools/data/build_arts_crafts_and_sewing.py` with only the category name, file
names and algorithm id changed.

## 3. Dataset selection rule

Candidate pool: the per-category 5-core files of Amazon Review Data 2018
(McAuley Lab, `categoryFilesSmall/{Category}_5.json.gz`), the same source and
vintage as Musical-Instruments (MI) and Arts-Crafts-and-Sewing (ACS).

Excluded because they were considered during development: Books (Amazon-Book),
Musical_Instruments, Arts_Crafts_and_Sewing, CDs_and_Vinyl, Video_Games,
Industrial_and_Scientific.

Pre-screen on the provider's published 5-core review counts (public page, read
before this protocol was written; no processed statistic of any candidate is
known): keep categories with 100,000–600,000 published 5-core reviews. Protocol B
keeps 79–89% of the published count on MI and ACS, so categories outside this
range cannot meet criterion S1 below. This leaves, in alphabetical order:
Digital_Music (169,781), Prime_Pantry (137,788).

Screening criteria, computed from data statistics only (no model is trained on
any candidate before one is selected), with the values of the retained Protocol B
datasets for reference:

| Id | Criterion | MI | ACS | Threshold |
|---|---|---|---|---|
| S1 | Interactions after Protocol B preprocessing | 206,241 | 392,935 | 100,000–500,000 |
| S2 | Near-Cold-eligible test users | 4,728 | 9,765 | ≥ 3,000 |
| S3 | Long-Tail-eligible test users | 8,816 | 17,815 | ≥ 5,000 |
| S4 | Mean taxonomy depth under `no_merge` | 4.59 (selected policy) | 4.22 (selected policy) | ≥ 3.0 |
| S5 | Taxonomy coverage under `no_merge` | 98.92% | 97.49% | ≥ 95% |

Rationale. S1: the upper bound keeps the full protocol (Section 6) within about
three GPU-days on Environment A (an ACS-sized run of all six methods takes about
5.4 GPU-hours per seed set, Online Resource 1, Table S31); the lower bound avoids
datasets too small for stable group metrics. S2: Industrial_and_Scientific was
excluded at 2,268 Near-Cold users and the smallest retained dataset has 4,728; the
threshold lies between them. S3: the primary endpoint is Long-Tail, so its user
count is bounded separately. S4: the lowest retained Amazon dataset has mean depth
3.38; Yelp2018 (0.9998) shows that a nearly flat taxonomy leaves the direction
little to act on. S5: every retained dataset has at least 96.5%.

Selection: evaluate candidates in alphabetical order and select the FIRST one
that meets S1–S5. S1–S3 are computed from the 5-core file alone; S4–S5 need the
metadata and are computed only for a candidate that meets S1–S3. Exactly one
dataset is selected and all its results are reported whatever they are. If no
candidate meets S1–S5, that outcome is reported and the thresholds are not
relaxed.

## 4. Preprocessing and taxonomy

- Protocol B, identical to MI/ACS: deduplicate (user, item) pairs, remap IDs,
  iterative 5-core re-verification over the whole pool, per-user 70/10/20 split,
  split seed 42. The SHA-256 of the source file and of each split file is
  recorded when the split is created, before any training.
- Taxonomy: the same metadata merge as MI/ACS (vintages 2014/2018/2023, deepest
  path, ties by more paths then newest vintage), leaf policies `no_merge`,
  `merge_t5`, `merge_t10`, `merge_t15`.

## 5. Configurations

Base configuration: the final ACS configuration of manuscript Table 6 (item
temperature 0.125, user temperature 0.2, epsilon_max 0.2, gamma_cold/gamma_warm
1.5/1.0, warm_start_epochs 20, lambda_ssl = lambda_ssl,user = 0.5, mu 0.9,
epsilon_user 0.05, same_leaf_weight 0), with the training settings of manuscript
Section 4.4. Seeds 42, 0, 1 for every run.

Step 1 — taxonomy policy: train the base configuration under the four policies
and apply the validation-only rule of manuscript Section 4.1 (keep the nominal
top scorer on validation Overall Recall@20, substituting `no_merge` when its
difference from the top scorer is smaller than the top scorer's seed-to-seed
standard deviation).

Step 2 — TaxPro-CL configuration: under the selected policy, train the grid item
temperature ∈ {0.10, 0.125, 0.15} × gamma_cold ∈ {1.5, 5.0} (the base
configuration is one cell) and select with rule R below.

Step 3 — baselines: LightGCN, SGL-ED, SimGCL, XSimGCL, NCL at the framework
defaults with the training budgets of manuscript Table 5. In addition, SimGCL is
tuned with a comparable budget: temperature ∈ {0.05, 0.10, 0.15, 0.20} × epsilon ∈
{0.05, 0.10, 0.20} (12 cells, the default is one of them), selected with the same
rule R.

Rule R (applied identically to TaxPro-CL and to tuned SimGCL): among the cells,
compute three-seed means of validation Recall@20 at each run's best-validation
checkpoint; keep the cells whose Overall is within 2% of the best Overall; select
the one with the highest Long-Tail. Ties go to the cell listed first above.

Step 4 — factorial: V0, V1, V2 at the selected TaxPro-CL configuration and policy
(V3 is the selected configuration itself), as in manuscript Section 5.4.

All selections are written to `confirmatory_selection.json`, whose SHA-256 is
recorded before the test split is opened.

## 6. Test sealing

All runs set `TAXPRO_DEFER_TEST=1`, so no test metric is computed during
training or selection. After Step 4 and the selection file are frozen,
`evaluate_sealed_test.py` evaluates on the test split, once, exactly the runs
needed for Section 7: the selected TaxPro-CL cell, V0–V2, the five default
baselines and the selected tuned-SimGCL cell. No other cell's test metric is
computed.

## 7. Endpoints and analysis

Per-user differences are averaged over the three same-seed pairs before
resampling or sign-flipping, as in manuscript Table 11. Paired sign-flip test:
two-sided, 10^5 random sign assignments, generator seed 42, Monte Carlo p =
(b+1)/(B+1). Bootstrap: 2,000 resamples, percentile 95% interval.

- Primary endpoint P1: test Long-Tail Recall@20, selected TaxPro-CL minus tuned
  SimGCL. Tested alone at alpha = 0.05.
- Secondary family (Holm, alpha = 0.05): Long-Tail vs default SimGCL; Near-Cold vs
  tuned SimGCL; Near-Cold vs default SimGCL.
- Factorial (Recall@20): direction family {Near-Cold, Long-Tail} and
  epsilon-adaptivity family {Near-Cold, Long-Tail}; cell p = the larger of the
  cell's two conditional-contrast p-values (intersection–union test); Holm within
  each 2-cell family.
- Descriptive only: NDCG@20, Overall and Warm, ranks among the six methods.

## 8. Interpretation criteria (fixed now)

- P1 p < 0.05 and difference > 0: "Long-Tail improvement over a comparably tuned
  SimGCL confirmed on a held-out dataset".
- P1 p < 0.05 and difference < 0: "reversed on the held-out dataset".
- Otherwise: "not confirmed".
- Direction effect supported on Long-Tail only if the direction Long-Tail cell is
  Holm-significant and both contrasts (V1−V0, V3−V2) are positive; likewise for
  Near-Cold and for epsilon-adaptivity.

Every outcome is reported in the manuscript, including "not confirmed" and
"reversed". No second dataset, extra seed, grid change, or alternative endpoint is
added in response to the results; anything that has to change is recorded below.

## Deviations log

### Deviation 1 (2026-09-27, before any file of the new candidates was downloaded)

Outcome of Section 3 as frozen: no candidate met S1–S5. Digital_Music met S1–S3
but 8,129 of its 8,132 metadata-matched items have the root-only path "Digital
Music" (fails S4/S5); Prime_Pantry has 1,796 Near-Cold and 3,888 Long-Tail test
users (fails S2/S3). Files: `results/confirmatory/`. This outcome is reported in
the manuscript as it stands.

Contrary to the last sentence of Section 3, the candidate pool is widened once, on
the only criterion whose rationale is computational (S1's upper bound), because
the frozen pool contained no eligible dataset. The decision uses only the data
statistics above; no model was trained on any candidate and no processed statistic
of the new candidates is known. Changes:

- Pre-screen: 100,000–950,000 published 5-core reviews (was 100,000–600,000).
  Excluding the categories already screened, this adds, in alphabetical order:
  Office_Products (800,357) and Patio_Lawn_and_Garden (798,415); the next larger
  categories (Cell_Phones_and_Accessories 1,128,437; Grocery_and_Gourmet_Food
  1,143,860) stay outside. Counts re-read from the provider's public page on
  2026-09-27.
- S1 upper bound: 750,000 interactions after Protocol B (was 500,000). Expected
  cost of the full protocol at this size is about 2.3 GPU-days on Environment A
  (was about three GPU-days at the old bound, a conservative estimate); S2–S5,
  every configuration, rule R, the endpoints and the interpretation criteria are
  unchanged.
- Run order (no effect on any selection): Steps 1, 2, the tuned-SimGCL grid of
  Step 3 and Step 4 run first; the four remaining default baselines (LightGCN,
  SGL-ED, XSimGCL, NCL) run last. Each run's test split is still evaluated at most
  once, after `confirmatory_selection.json` is frozen.

The SHA-256 of this file after Deviation 1 is recorded in
`docs/confirmatory_protocol_deviation1.sha256`.

Screening under Deviation 1: Office_Products meets S1–S5 (677,247 interactions;
11,203 Near-Cold and 22,202 Long-Tail test users; mean depth 4.99; `no_merge`
coverage 98.13%; `results/confirmatory/screening_office_products*.json`) and is
the selected dataset. Patio_Lawn_and_Garden is not screened.

### Deviation 2 (2026-09-27, after the dataset was selected, before any model run on it)

Additions only; nothing above is changed. No model has been trained on
Office_Products.

- Selection rule. Rule R stays the primary rule for TaxPro-CL and tuned SimGCL.
  Rationale: the primary endpoint is Long-Tail, so the baseline is tuned toward
  that endpoint under the same Overall guard as TaxPro-CL; on the development
  datasets an Overall-only rule selected SimGCL cells with lower Long-Tail Recall@20
  than rule R on three of four datasets and the same cell on the fourth (Online
  Resource 1, Table S24b), i.e. a weaker baseline for this endpoint. Pre-specified sensitivity analysis: the highest-validation-Overall cell
  of each method (rule Ov) is also recorded in `confirmatory_selection.json`; if it
  differs from the rule-R cell, its test split is evaluated once as well, and P1 is
  recomputed with rule-Ov cells for both methods. This result is reported as a
  sensitivity analysis and does not change the interpretation of Section 8.
- Smallest effect size of interest (SESOI) for P1: 5% of the tuned SimGCL test
  Long-Tail Recall@20 mean (relative difference = mean per-user difference divided
  by that mean). Basis, from the development datasets (Table S24b; Table 12):
  seed-to-seed SD of Long-Tail Recall@20 has median 3.3% of the mean; the smallest
  development-stage Long-Tail gain over default SimGCL was +10%; the expected 95%
  interval half-width at this dataset's size is about 4.6% (ACS, 17,815 users:
  5.1%). The minimum detectable effect at 80% power is therefore about 6–7%; an
  effect near 5% may not be resolved, and this is stated with the result.
- Interpretation (refines Section 8; each P1 outcome keeps its Section-8 label):
  confirmed with p < 0.05 and a relative difference ≥ +5% → "practically
  meaningful"; confirmed with a relative difference < +5% → "below the smallest
  effect of interest"; not confirmed and the 90% interval of the relative
  difference inside (−5%, +5%) → "equivalent within ±5%"; otherwise "inconclusive".
  Reversed results are reported with their relative difference.
- Seeds: unchanged (42, 0, 1).
- Budget reporting: for each method, the number of configurations and runs, total
  GPU-hours, mean training time and mean epochs to early stopping (TaxPro-CL: 4
  policies + 5 further grid cells = 9 configurations; tuned SimGCL: 12).

The SHA-256 of this file after Deviation 2 is recorded in
`docs/confirmatory_protocol_deviation2.sha256`.
