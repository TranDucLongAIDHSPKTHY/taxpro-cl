# Extended evaluation protocol

Status: specified before the first run of Sections 4-10. The SHA-256 of this file is
recorded in `docs/extended_evaluation_protocol.sha256`, together with those of the
Vietnamese working copy of this protocol and of its PDF, which the authors keep. This file
is the authoritative text; the working copy specifies the same analyses. Any later change
goes to Section 14 before it is carried out, never into the sections above it.

## 1. Purpose and evidential status

This protocol specifies additional analyses of the taxonomy-guided perturbation study.
They address label dependence, reproducibility, mechanism, selection range and comparator
coverage. It does not modify `docs/confirmatory_protocol.md`, its deviations, or
`confirmatory_selection.json`. The prospectively specified Office-Products result stays
as reported, and nothing here replaces it.

Every new result carries exactly one label, in the paper and in the accompanying
material:

- **Prospectively specified.** The rule is fixed here and the test set of that dataset
  and split has never been viewed for any method (re-drawn splits, Section 7). This label
  also applies to the added baselines on Office-Products (Section 9) and to tuned NCL on
  Office-Products (Section 10): their selection rule is fixed here and their test results are
  evaluated once. They are secondary comparisons.
- **Exploratory.** The test set of that split was viewed for another method, or the
  analysis is not specified here. This covers Sections 5, 6 and 8, the new seeds of
  Section 7 and the development datasets of Section 10.
- **Diagnostic.** Measurements on checkpoints that use no test set (Section 4).

Re-drawn splits share interactions with split seed 42, and their configurations were
selected on split seed 42. They are re-drawn splits, not independent data. Every run is
reported whatever its direction, including failed and stopped runs.

## 2. Fixed elements

| Item | Value |
|---|---|
| Base commit | `e04d16ceea0edab89510e719d28fcc5bcee40816` of branch `main` (tag `v1.6` is its ancestor) |
| `docs/confirmatory_protocol.md` | SHA-256 `d9c583bb72e451f69bbef38248d4e145dde781c1eefb1597e00d705070a0f5ea` |
| Office `confirmatory_selection.json` | SHA-256 `29eba07961148709cf2bcf4dd648e6b5f7acf85f9484e75a8b29a631cd5b150c` |
| Training seeds | Existing 42, 0, 1. New 2, 3, 4 |
| Split seeds | Existing 42. New 101, 102, 103 |
| Checkpoint selection | Validation Overall Recall@20, epoch cap and patience as in the main paper |

Complete configuration, as (item temperature, user temperature, gamma_cold, policy), with
epsilon_max 0.2, gamma_warm 1, epsilon_user 0.05, mu 0.9 and a 20-epoch warm start:

| Dataset | Values |
|---|---|
| Amazon-Book | 0.100, 0.200, 1.5, `merge_t10` |
| Yelp2018 | 0.125, 0.150, 1.5, `no_merge` |
| Musical-Instruments | 0.150, 0.200, 5.0, `no_merge` |
| Arts-Crafts-and-Sewing | 0.125, 0.200, 1.5, `no_merge` |
| Office-Products | 0.150, 0.200, 1.5, `no_merge` |

Default SimGCL uses temperature 0.2 and epsilon 0.05. Tuned SimGCL uses (0.15, 0.1) on
Amazon-Book, (0.2, 0.1) on Office-Products and (0.2, 0.2) on the other three datasets. On
those three datasets both values are the largest of the searched grid, temperature in
{0.05, 0.10, 0.15, 0.20} x epsilon in {0.05, 0.10, 0.20}.

Reference run families are those of `FAMILIES` in `tools/experiments/run_extended.py`.
They are the main-comparison families of `results/results_manifest.csv`, including default
NCL on Office-Products, the factorial families V0, V1, V2 and the tuned families. All of
them ran on Environment A, and the registry lists every reference run with its environment
(Section 13).

## 3. Common rules

- Configuration inheritance. The configuration of a new run equals the `run_manifest.json`
  configuration of its reference family plus the overrides stated here. The runner asserts
  this and stops on any other difference. Configurations are not tuned again outside
  Sections 8 and 9.
- Execution. Environment A only: one RTX 4060 Ti 16 GB, Python 3.10.0, torch 2.1.0+cu121.
  Both arms of every new contrast run on it. On Amazon-Book, V0, V1, V2 and V3 of seeds 42,
  0 and 1 all ran on Environment A, and every new Amazon-Book contrast uses the V3 family of
  `FAMILIES` for all seeds. The published factorial used an Environment B V3 on Amazon-Book;
  Online Resource 1, Table S23b, shows that its Holm decisions do not change when the
  Environment A V3 replaces it. Runs go one at a time, with the exception of Section 11.
  Deterministic mode stays off, as in the original runs.
- Provenance. Code is committed before each package starts, so `environment.json` records
  the code used. The runner refuses to start if training code is not committed or if this
  file does not match its recorded SHA-256. Short pilot runs check the code and measure
  time; they are stored apart and never enter a result.
- Sealing. New runs set `TAXPRO_DEFER_TEST=1`. Each run's test split is evaluated once. In
  the search packages this happens after the selection file is committed, and only for the
  selected cell. In the other packages it happens after the run is registered. The
  evaluation script stamps the time and refuses a second evaluation. The same opening stores
  per-user Recall@20 and NDCG@20, so paired analyses do not read the test split again.
- Reference runs. Their test results already exist. Per-user values are recomputed once
  from the saved checkpoint and their group means are checked against the recorded metrics;
  no new metric is derived from them.
- Integrity. A SHA-256 snapshot of locked splits, taxonomy variants, evaluation protocols
  and frozen selections is compared after the last run.
- Outputs. Run artifacts go to `log/p0/extended/<dataset>/<Model>/<step>-<parameters>/seed<seed>`.
  Analysis outputs go to `results/extended/`.
- Paired effect. For arms A and B and group g, the mean over users of g of the difference
  in their metric, after averaging over paired seeds. Groups are Near-Cold (NC, degree 1-5),
  Long-Tail (LT, degree 1-10), Warm (degree above 10) and Overall, computed on each split's
  own training degrees. NC and LT are primary; Warm and Overall are context.
- Metrics. Recall@20 is primary and NDCG@20 secondary.
- Inference. Percentile user bootstrap with 2,000 resamples for configuration comparisons
  and Office-Products, 5,000 for factorial contrasts. Two-sided paired sign-flip test with
  10^5 assignments. Alpha 0.05. A Holm family is one contrast on one metric; its cells are
  the dataset and group pairs (NC, LT) of that analysis.
- Reporting. Absolute differences accompany percentages, which are relative to arm B. All
  test tables give mean and sample standard deviation (n - 1) over seeds.
- Wording. "Evidence" means significant after Holm. "Nominal signal" means the unadjusted
  95% interval excludes 0 without surviving Holm. "No detectable difference" is used
  otherwise and always comes with the interval bounds. The words "no effect", "equivalent"
  and "taxonomy is useless" are not used: this design does not test equivalence.

## 4. Checkpoint diagnostics

Diagnostic. Inputs: the selected checkpoints of V0, V1, V2 and V3, seeds 42, 0, 1, split 42,
all five datasets. No training and no test split. Fifty fixed minibatches (generator seed 7,
training batch size) run through the training forward pass with two views.

Measured for NC, Mid-Tail (degree 6-10) and Warm items, on valid items:

1. Realized perturbation norm `||epsilon omega||`, by propagation layer.
2. Ratio `||epsilon omega|| / ||E_i^(k,v)||`, by propagation layer.
3. Cosine between the two final views of an item.
4. Norm of the gradient of the item-side InfoNCE alone, unweighted, with respect to the
   layer-0 item table, and each group's share of the summed norm.
5. Embedding shift between arms, `||e^A - e^B|| / ||e^B||`, and their cosine, on clean
   embeddings, for (V2, V0) and (V3, V1), and also (V1, V0) and (V3, V2).

Validity check, required before any of this is used: the ratio of measure 1 between the
adaptive and the fixed arm must lie within 2% of gamma_cold for NC and of 1 for Mid-Tail and
Warm, per layer, for (V2, V0) and (V3, V1). On failure the script exits with an error, and
the fault is found and any affected run repeated first.

Also reported, labelled exploratory, on the validation split: Recall@50, Recall@100, NDCG@100,
the mean percentile rank of held-out positives, and the ranking change ratio `r`. Let `T_A(u, s)`
be the top-20 list of arm A at seed s for user u, ranked over the full catalog minus u's training
items, and `J` the Jaccard index `|T ∩ T'| / |T ∪ T'|`. For group g, whose users are those with at
least one validation positive in g,

    r_g(A, B) = mean_s mean_u [1 - J(T_A(u, s), T_B(u, s))]
                / mean_(arm, s < s') mean_u [1 - J(T_arm(u, s), T_arm(u, s'))]

The numerator averages over the three seeds; the denominator over the four arms and their three
seed pairs. `r` near 1 means that changing the arm moves the top-20 lists about as much as changing
the seed, and `r` near 0 much less. The other measures are given as the arm difference per seed and
its mean over the three seeds, without interval or test. All four factorial contrasts (V1 - V0,
V3 - V2, V2 - V0, V3 - V1) are reported for NC and LT, with Warm and Overall as context, on all five
datasets, in one table, without selection.

Reporting. If the check holds, the views stay almost identical (cosine above 0.999) and the NC
gradient changes little: "Degree adaptation enlarged the realized NC perturbation as specified, but the two
views remained nearly identical and the item-side contrastive gradient for NC items changed
by the reported percentage, consistent with the null Recall@20 contrasts." If the gradient or
embeddings of NC items change clearly: "Degree adaptation changed the training signal for NC
items without a detectable change in Recall@20." These measures describe a final checkpoint
and do not establish why a method behaves as it does.

## 5. Label and gradient-path controls

Exploratory. New arms at the V3 configuration of each dataset:

| Arm | Labels | Direction |
|---|---|---|
| V3-shuffled | Permuted | As V3 |
| V3-detached | True | Detached |
| V3-shuffled-detached | Permuted | Detached |

Label permutation. After the taxonomy policy is applied, labels are permuted among all items
with a valid label, with generator seed 1000 + training seed. The set of valid items, the
validity mask, coverage and category sizes are unchanged; only membership is broken. The
label file and its SHA-256 are stored. Prototypes are initialized and updated on the permuted
groups by the usual procedure.

Detached direction. `omega_i = (sg(p_c) - sg(E_i^(k,v))) / (||.|| + delta)`. The value of the
perturbation in the forward pass is unchanged and no gradient flows through `omega`.

Runs: 3 arms x 3 seeds (42, 0, 1) on each dataset, paired with V3 and V2 of the same seed.

Contrasts: primary, V3 minus V3-shuffled (label content) and V3 minus V3-detached (gradient
path); secondary, V3-detached minus V2 (geometry without the gradient path) and V3-detached
minus V3-shuffled-detached (label content without the gradient path). Bootstrap 5,000,
sign-flip, Holm per contrast and metric.

Reporting.

- No Holm-significant cell in the primary contrasts: "Neither replacing the true leaf labels
  with size-preserving shuffled labels nor detaching the direction changed NC or LT Recall@20
  beyond the reported intervals; the evidence does not indicate an effect specific to category
  content or to the direction's gradient path."
- Label contrast positive after Holm: "With the true labels, Recall@20 exceeded the
  shuffled-label control on the reported cells after Holm correction. Because the original
  factorial found no benefit over the random-direction control, this label-specific difference
  does not by itself establish a recommendation benefit."
- Label contrast negative after Holm: "The shuffled-label control exceeded the true-label arm
  on the reported cells after Holm correction."
- Gradient contrast non-zero after Holm: "Detaching the direction changed Recall@20 on the
  reported cells, indicating that the gradient path through the prototype direction affects
  training in this configuration." This is not attributed to semantic information.
- Signs differ across datasets: each cell is reported, with "The direction of these differences
  varied across datasets."

## 6. Label quality

Exploratory. Musical-Instruments and Office-Products, split 42, seeds 42, 0, 1, with generator
seed 2000 + training seed:

- Noise 50%: half of the valid items, chosen at random, have their labels permuted among
  themselves. Mask and coverage are unchanged. Noise 100% is V3-shuffled of Section 5.
- Missing 50%: half of the valid items lose their label. Coverage falls by about half and the
  mask changes.

New arms: V3-noise50, V3-missing50 and V2-missing50. The last two use one label file, so the
V2 arm removes the effect of the changed mask, which alters both the perturbed items and the
InfoNCE anchors.

Reported: coverage of each condition; Recall@20 for NC, LT, Warm and Overall; V3 minus V2 within
each condition (clean, noise 50, noise 100, missing 50); the mask effect V2-missing50 minus V2.
Bootstrap 5,000, unadjusted intervals, no hypothesis test.

Reporting. Little change across noise levels: "The prototype-direction results changed little
when 50% or 100% of the labels were corrupted, consistent with the absence of a label-specific
effect in the factorial." Differences falling with noise: "NC/LT differences from the
random-direction control decreased as label corruption increased, suggesting that the arm uses
label content; this analysis is exploratory." A clear mask effect: "Reducing category coverage
changed the random-direction control by the reported amount, an effect of the validity mask
rather than of label content." Robustness to noise is not presented as an advantage.

## 7. Seeds and re-drawn splits

Design:

- Amazon-Book, Yelp2018: split 42, new seeds 2, 3, 4, six seeds in all. Exploratory. These
  datasets are not re-split: their test partition is the benchmark file distributed with
  LightGCN, so a new partition would change the benchmark.
- Musical-Instruments, Arts-Crafts-and-Sewing, Office-Products: split seeds 101, 102, 103 x
  training seeds 42, 0, 1. Prospectively specified, for a descriptive rule. The original split
  is reported separately.

Methods: V0, V1, V2, V3, default SimGCL and tuned SimGCL. LightGCN on Office-Products only.
Configurations stay fixed from Section 2, including if Section 8 changes a selection, because
the question is the stability of the selected configurations.

Re-drawn splits. The locked 5-core pool is partitioned per user 70/10/20 with the published
splitter, which must first reproduce the locked split under seed 42. Each re-drawn split has its
own dataset identifier, taxonomy variant and evaluation protocol. Locked splits are never
written. Before any run, users and training interactions must equal the locked split, observed
items must lie within 0.1%, and the NC and LT item shares within 1 percentage point.

Unit of analysis: a run r = (split, seed), with effect `D_r`, the mean user difference between
two arms of the same split and seed.

Reported: every `D_r`; mean, standard deviation, numbers of positive and negative runs, mean per
split; for each new split, a user bootstrap interval pooling its three seeds.

Classification. Stable positive: on Amazon-Book and Yelp2018, the mean of six runs is above 0 and
at least 5 of 6 are positive; on the other datasets, the mean of nine runs is above 0, at least 8
of 9 are positive and all three split means are above 0. Stable negative is the mirror image.
Otherwise unstable. The two-sided exact sign test is descriptive: 6/6 gives 0.031, 5/6 gives
0.219, 9/9 gives 0.0039, 8/9 gives 0.039, 7/9 gives 0.18. With so few runs it has low power and
is not used to claim the absence of an effect. Bootstrapping over splits is not used because
three splits give an unstable cluster bootstrap.

Contrasts: V1 - V0 and V3 - V2 (direction); V2 - V0 and V3 - V1 (magnitude); complete configuration
minus default SimGCL and minus tuned SimGCL, on NC, LT and Overall.

Reporting. No stable positive direction or magnitude cell: "Across the additional runs, no direction
or magnitude contrast had a consistently positive sign." A stable positive cell: "The direction
contrast was positive in k of n runs; this run-level consistency is descriptive and was not tested
with multiplicity control." A sign change across splits against SimGCL: "The long-tail difference
from SimGCL changed sign across splits, so the single-split results should not be read as stable."

## 8. Extended temperature grid on Office-Products

Exploratory, as a sensitivity analysis. The locked grids stop at their upper temperature edge:
SimGCL selected 0.20 in 0.05-0.20, the complete configuration selected an item temperature of
0.15 in 0.10-0.15, and its user temperature of 0.20 was not searched on this dataset.

New cells, three seeds each, equal budgets:

- SimGCL: temperature in {0.25, 0.30, 0.40} x epsilon in {0.05, 0.10, 0.20}. 27 runs.
- Complete configuration: user temperature in {0.20, 0.30, 0.40} x item temperature in {0.15, 0.20,
  0.25}, `no_merge`, gamma_cold 1.5. The cell (0.15, 0.20) is the locked selected cell and is not run
  again: 8 new cells, 24 runs, one cell fewer than SimGCL. The row at user temperature 0.20 changes the
  item temperature alone, so a changed selection can be attributed to one temperature or the other.

Rule R of the confirmatory protocol, on the union of locked and new cells, separately per method (21
cells for SimGCL, 14 for the complete configuration): the highest three-seed mean validation LT
Recall@20 among cells within 2% of the best validation Overall Recall@20. Rule Ov is recorded. The
selection is committed to `results/extended/selection_grid.json` before any test evaluation. If it
changes, only the newly selected cell is evaluated, once. The other cells are never evaluated on test.

Analysis: LT, NC and Overall differences between the two selected configurations, bootstrap 2,000,
sign-flip. The report states whether a selected cell lies inside the extended grid or at its edge.
The primary Office-Products result is not replaced.

Reporting.

- Neither selection changes: "Extending the temperature grid did not change either validation-selected
  configuration; the reported Office comparison is unchanged."
- A selection changes and LT stays positive with an interval excluding 0: "With the extended grid, the
  validation rule selected the reported cell; the LT difference became the reported value."
- LT narrows, its interval contains 0, or it reverses: "The Office long-tail advantage is sensitive to
  the temperature range searched: with the extended grid, the difference was the reported value."
- A selected cell at the new edge: "The selected value remains at the edge of the extended grid, so the
  optimum is not bracketed."

The limitations section of the paper states that tuned SimGCL on Yelp2018, Musical-Instruments and
Arts-Crafts-and-Sewing selected temperature 0.20 and epsilon 0.20, the upper corner of its grid, so its
optimum on those datasets is not bracketed. No further cells are run for them.

## 9. Added baselines

Prospectively specified, secondary. Office-Products, split 42.

| Method | Role | Public code | Commit | License |
|---|---|---|---|---|
| KGCL adapted to an item-category graph | Metadata-aware | `yuh-yang/KGCL-SIGIR22` | `89c8d062e1fd100132bee7745131fc67b808467d` | MIT |
| GALORE | Long-tail item recommendation | `sichunluo/GALORE` | `ff7e3e79bf99f004778fc9379fb91d972717dc75` | None |

Both run from their authors' code, unmodified, in a directory outside this repository, in a Python
environment that adds the packages they need to the torch of Environment A. Every adaptation is
applied at run time by `tools/comparators/upstream.py`. GALORE's code is not redistributed.

Data. The comparators receive the same training split. The slot their code reads as a test file
receives the validation split, so they never see the test split.

Evaluation. Final tables of the best validation state are scored by this repository's evaluator, on
the same candidates and groups as every other method, through the same sigmoid. Users and items with
no training interaction keep a zero vector.

KGCL. Its knowledge-side learning reads item-headed triples only, so each item is linked to the nodes of
its category path, one relation per path level, with the leaf at level 0 and listed first, since the
authors' code keeps ten triples per item. Path tokens containing HTML markup are removed. Items without a
valid label have no triples. The graph holds category knowledge only. Linking each item to every ancestor
gives this baseline more category information than the evaluated configuration, which uses the leaf only;
this favours the baseline, and the paper states it. The settings of the authors' Amazon dataset apply,
with embedding size 64, three layers and batch size 2048. Grid: contrastive temperature `kgc_temp` in
{0.05, 0.10, 0.15, 0.20} x contrastive weight `ssl_reg` in {0.02, 0.06, 0.10}. Early stopping on
validation Overall Recall@20, as in the authors' code.

GALORE. It keeps the authors' configuration file, with two layers. The paper trains in two stages
(Sections 4.1 and 4.5 of Luo et al., 2023): a standard graph-based recommender on the original graph,
with LightGCN as the backbone, then the same model on the augmented graph. The public code implements the
second stage; it loads the first-stage checkpoint (`lightgcn_best_<size>.pt`) but does not include the
step that produces it. Each run therefore starts from a first stage trained here on the same seed, one
LightGCN run per seed, three in all, with two layers and pooled over the propagated layers, as GALORE's
encoder pools. The paper's readout (its Eq. 4) also includes layer 0; the runs follow the public code.
These LightGCN runs are GALORE's starting point. Their test split is evaluated once, in the same opening
as GALORE's selected cell, after GALORE's selection is committed. The public code trains for a fixed 60
epochs on a BPR loss with sampling mixup; its configured contrastive options are not read by that
training loop. Validation runs every second epoch from epoch 7, and the checkpoint is selected on
validation Overall Recall@20 instead of the authors' four-metric vote at rank 50. Grid: `edge_num` in {1,
3, 5, 10}, the number of structural neighbors that edge addition draws from, x `drop_rate` in {0.1, 0.3,
0.5}, the ranges the authors searched (their Section 5.3). The authors' command line does not reach
`edge_num`, so the wrapper passes it to the function. In a validation pilot at the code's default cell
(2, 0.1), Overall Recall@20 was highest before this training started and fell from then on. The
evaluation schedule stays the authors', so a selected state comes from their training, and the validation
value of the starting LightGCN is reported beside it. GALORE's code has no license and is not
redistributed; the code availability statement of the paper points to the authors' repository and the
commit above.

Budget and selection: 12 cells x 3 seeds for each method, the SimGCL budget on this dataset. Rule R as in
Section 8. Each method's selection is committed to its own file, `results/extended/selection_kgcl.json`
or `selection_galore.json`, before its test evaluation, and the selected cell is evaluated once.

Reported: Recall@20 and NDCG@20 for NC, LT, Warm and Overall. Paired contrasts against the complete
configuration, bootstrap 2,000, Holm over the two cells NC and LT per baseline.

Reporting. Lower LT and higher Overall: "The baseline had lower LT but higher Overall Recall@20 than the
complete configuration." Higher LT: "The baseline exceeded the complete configuration on LT Recall@20,"
and every statement about ranks is updated. Not significant after Holm: "No detectable LT difference
from the baseline," with its interval.

GALORE against its starting point. For NC, LT, Warm and Overall, GALORE minus the LightGCN it started
from, paired by seed, bootstrap 2,000, no test. A mean difference at or below 0: "GALORE did not improve
over its LightGCN starting point." Above 0 with an interval containing 0: "No detectable improvement of
GALORE over its LightGCN starting point." An interval above 0: "GALORE improved over its LightGCN starting
point by the reported amount." The selected epoch of each seed is reported.

## 10. Tuned NCL

Tuned NCL enters the paper whatever its result, and every statement that relied on default NCL is
updated. Grid: temperature in {0.05, 0.10, 0.15, 0.20} x number of clusters in {1000, 2000, 3000}, 12
cells x 3 seeds, the default (0.05, 2000) being one cell.

Development datasets. Exploratory. No new run. Primary selection on validation Overall Recall@20:
Amazon-Book (0.1, 1000); Yelp2018, Musical-Instruments, Arts-Crafts-and-Sewing (0.05, 1000). Rule R as a
sensitivity check: rule R of Section 8 also selects a cell from the same 12 cells and three seeds; that
cell is reported beside the primary one with its test result, and a cell selected by both rules is stated
as such. Conclusions rest on the primary cell. Test results of these cells already exist. Two earlier
Amazon-Book cells at temperatures 0.03 and 0.08 lie outside the grid and do not enter the selection.

Office-Products. Prospectively specified, secondary, as the added baselines of Section 9. The default
cell is the published reference family, so 11 new cells x 3 seeds, 33 runs. Rule R, as for every method
on this dataset, with rule Ov recorded. The selection is committed to `results/extended/selection_ncl.json`
before test evaluation. A newly selected cell is evaluated once; if the default cell is selected, its
published results stand. Analysis and wording as in Section 9. If this package is omitted (Section 11),
the paper states that NCL was not tuned on Office-Products and makes no ranking statement on NCL there.

## 11. Order, priorities and cuts

Order: diagnostics; the two grids of Section 8; controls and label quality on Office-Products and
Musical-Instruments; controls on Amazon-Book; KGCL; re-drawn splits of Office-Products; Amazon-Book seeds
for V0-V3; Amazon-Book seeds for SimGCL, GALORE with its LightGCN, re-drawn splits of Musical-Instruments,
tuned NCL on Office-Products; controls on Arts-Crafts-and-Sewing, its re-drawn splits, Yelp2018 seeds,
controls on Yelp2018, LightGCN on re-drawn splits.

Priorities. Required: diagnostics, Section 6, Section 8, KGCL, controls on Amazon-Book, Musical-Instruments
and Office-Products, re-drawn splits of Office-Products, Amazon-Book seeds for V0-V3. Second: Amazon-Book
seeds for SimGCL, GALORE, re-drawn splits of Musical-Instruments, tuned NCL on Office-Products. Third: the
rest.

Progress checks. On 2026-11-12 the diagnostics must be complete and valid and at least a quarter of the
required compute done; on 2026-11-16 at least 65%; from 2026-11-19 no run outside the required
packages starts; the data are locked on 2026-11-21. If the first check is missed, short jobs (runs on
Musical-Instruments, Arts-Crafts-and-Sewing and Office-Products) share the GPU first, at most two at a
time. Omissions start only if, at that rate, the required and second-priority packages would not finish
by 2026-11-19. Times of runs that shared the GPU are not used as run-time measurements.

If compute runs out, omissions follow this order, and each is recorded in Section 14 and stated with the
results: LightGCN on re-drawn splits; controls on Yelp2018; Yelp2018 seeds; re-drawn splits of
Arts-Crafts-and-Sewing; controls on Arts-Crafts-and-Sewing; re-drawn splits of Musical-Instruments; tuned
NCL on Office-Products; GALORE; Amazon-Book seeds for SimGCL; then one re-drawn split of Office-Products and
one Amazon-Book seed for V0-V3. With fewer runs the classification of Section 7 needs at most one run
against the sign and, where splits apply, all split means of that sign. Environment B is not used to catch
up.

Failed runs are repeated once with the same seed and configuration. A second failure is recorded as failed,
without changing the seed. A completed run is never removed because of its result.

## 12. Checks before the first run

| Check | Criterion |
|---|---|
| Label permutation | Category sizes and the valid set unchanged; retained share near `sum (n_c/N)^2`; SHA-256 stored |
| Noise 50% | Exactly half of the valid items chosen; mask unchanged |
| Missing 50% | Coverage near half; the V3 and V2 arms have the same SHA-256 |
| Detached direction | Same seed gives `max |delta| < 1e-6` in the forward pass; no gradient through `omega` |
| Re-drawn splits | The thresholds of Section 7 |
| Diagnostics | The validity check of Section 4 |
| Pilot runs | Each new kind of run completes and fills every registry column |
| Test sealing | Evaluation is refused without a committed selection, and refused for an evaluated run |

## 13. Registry and data lock

`results/extended/run_registry.csv` has one row per run: `run_id`, `exp`, `dataset`, `arm`, `split_seed`,
`train_seed`, `config_sha256`, `label_file_sha256`, `env`, `start_utc`, `end_utc`, `status`,
`best_epoch`, `val_overall_r20`, `val_lt_r20`, `test_eval_utc`, `checkpoint_sha256`, `git_commit`. Run
identifiers read `<kind>_<dataset>_<arm>_s<split>_r<seed>`. Reference runs have rows too, with `exp` =
`reference`, `env` read from `environment.json` and checked against the path style of `training.log`, and
an empty `test_eval_utc`, since their test results predate this protocol. Label files, split manifests
and selections are stored with their SHA-256 under `results/extended/`.
`results/extended/results_extended.csv` has the format of `results/metrics_seed.csv`, and its SHA-256 is
recorded when the data are locked. No run starts after the lock.

## 14. Changes after freezing

None.
