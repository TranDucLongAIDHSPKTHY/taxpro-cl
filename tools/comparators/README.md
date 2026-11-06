# Added baselines

KGCL and GALORE (`docs/extended_evaluation_protocol.md`) run from their authors' code, which
is not part of this repository. GALORE's public repository carries no license, so its code is
not redistributed here. This folder holds only the adapter `upstream.py`, which imports the
authors' code at run time and never edits it.

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

The comparators use the torch of the main environment plus the packages their code imports:

```bash
python -m venv --system-site-packages .venv
.venv/Scripts/python -m pip install pandas==1.5.3 numba
```

## Run

From the root of this repository:

```bash
python -m tools.experiments.run_comparators --prepare
python -m tools.experiments.run_extended --package pretrain-lightgcn
python -m tools.experiments.run_comparators --package comparator-kgcl comparator-galore
python -m tools.experiments.run_comparators --select
python -m tools.experiments.run_comparators --open-test
```

`--prepare` writes the data in the authors' formats outside this repository. Their test slots
receive the validation split, so their code never sees the test split. `pretrain-lightgcn` trains
the two-layer LightGCN from which GALORE starts, since its public code has no such stage; the test
split of those runs is never evaluated. Selections must be committed before `--open-test`.
