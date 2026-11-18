# Added baselines

KGCL and GALORE (`docs/extended_evaluation_protocol.md`, Section 9) run from their authors' code,
which is not part of this repository. This folder holds only the adapter `upstream.py`, which
imports the authors' code at run time and never edits it.

| Method | Role | Public code | Commit | License |
|---|---|---|---|---|
| KGCL, on an item-category graph | Metadata-aware | [yuh-yang/KGCL-SIGIR22](https://github.com/yuh-yang/KGCL-SIGIR22) | `89c8d062e1fd100132bee7745131fc67b808467d` | MIT |
| GALORE | Long-tail item recommendation | [sichunluo/GALORE](https://github.com/sichunluo/GALORE) | `ff7e3e79bf99f004778fc9379fb91d972717dc75` | None declared |

GALORE's public repository carries no license, so its code is not redistributed here; please cite
the authors' papers when you use either method.

## Download

Clone both repositories at the pinned commits, next to this repository:

```bash
mkdir ../TaxPro-CL_comparators
cd ../TaxPro-CL_comparators
git clone https://github.com/sichunluo/GALORE.git GALORE
git -C GALORE checkout ff7e3e79bf99f004778fc9379fb91d972717dc75
git clone https://github.com/yuh-yang/KGCL-SIGIR22.git KGCL
git -C KGCL checkout 89c8d062e1fd100132bee7745131fc67b808467d
```

Another location works if the environment variable `TAXPRO_COMPARATORS` points to it.

## Environment

The comparators use the Python and torch of the main environment (Python 3.10, torch 2.1.0 with
CUDA 12.1) plus the packages their code imports. In `../TaxPro-CL_comparators`:

```bash
python -m venv --system-site-packages .venv
.venv/Scripts/python -m pip install pandas==1.5.3 numba==0.68.1
```

The runner calls `.venv/Scripts/python.exe` in that folder (Windows layout).

## Settings

Office-Products, split 42, seeds 42, 0 and 1, 12 cells per method (36 runs each).

| | KGCL | GALORE |
|---|---|---|
| Fixed settings | Settings of the authors' Amazon dataset; embedding size 64, three layers, batch size 2048; early stopping on validation Overall Recall@20, as in the authors' code | The authors' configuration file, two layers; a fixed 60 epochs; validation every second epoch from epoch 7 |
| Grid | Contrastive temperature `kgc_temp` in {0.05, 0.10, 0.15, 0.20} x contrastive weight `ssl_reg` in {0.02, 0.06, 0.10} | `edge_num` in {1, 3, 5, 10} x `drop_rate` in {0.1, 0.3, 0.5}, the ranges the authors searched |
| Input built here | Item-headed triples: each item is linked to the nodes of its category path, one relation per level, leaf first; path tokens containing HTML markup are removed; items without a valid label have no triples | A two-layer LightGCN per seed (`pretrain-lightgcn`), pooled over the propagated layers as GALORE's encoder pools: the first training stage, which the public code loads but does not produce |
| Applied at run time by `upstream.py` | The prepared data is served under the name of the authors' Amazon dataset; packages imported but unused on the executed path are replaced by empty stand-ins, `degree` by its definition; data loaders run in-process | `edge_num` is passed to `graph_add_edge`, which the authors' command line does not reach; the checkpoint is selected on validation Overall Recall@20 instead of the authors' four-metric vote at rank 50 |

For both methods, the embedding tables of the best validation state are exported and scored by this
repository's evaluator, on the same candidates and user groups as every other method and through the
same sigmoid. Users and items without training interactions keep a zero vector.

## Run

From the root of this repository:

```bash
python -m tools.experiments.run_comparators --prepare
python -m tools.experiments.run_extended --package pretrain-lightgcn
python -m tools.experiments.run_comparators --package comparator-kgcl comparator-galore
python -m tools.experiments.run_comparators --select kgcl
python -m tools.experiments.run_comparators --open-test kgcl
python -m tools.experiments.run_comparators --select galore
python -m tools.experiments.run_comparators --open-test galore
```

- `--prepare` writes the data in the authors' formats outside this repository. Their test slots
  receive the validation split, so their code never sees the test split.
- `pretrain-lightgcn` trains the LightGCN runs from which GALORE starts. Their test split stays
  sealed until GALORE's opening.
- `--package` trains the grid; a finished run is skipped, so the command can be repeated.
- `--select` applies rules R and Ov (protocol Section 8) to the validation results of the 36 runs
  and writes `results/extended/selection_kgcl.json` or `selection_galore.json` with its SHA-256.
  Rule R gives the selected cell. The selection file must be committed before `--open-test`.
- `--open-test` evaluates the test split of the selected cell only, once per run. GALORE's opening
  also evaluates its three LightGCN runs once, so GALORE is compared with its starting point.
  `--open-test <method> --dry-run` lists the runs that would be evaluated and evaluates none.

Each run is written to `log/p0/extended/office-products/<KGCL|GALORE>/grid-<cell>/seed<seed>/`,
with the same files as the other runs of the extended evaluation (`run_manifest.json`,
`environment.json`, `validation_metrics.json`, `training.log`, and after the opening
`final_test_metrics.json`, `final_test_group_metrics.json`, `final_test_per_user.npz`).
