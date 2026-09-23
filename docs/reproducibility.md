# Reproducibility

## Environment roles

A machine without a GPU is enough for code changes, CPU unit tests, metadata
preparation, validation, and result compilation. The paper's training runs used
two environments, recorded per run in `environment.json` and tabulated in the
paper's Online Resource 1 (Table S23):

- Environment A (most runs): Windows 10, Intel Core i5 12th generation CPU,
  32 GB RAM, NVIDIA RTX 4060 Ti 16 GB, Python 3.10.0, PyTorch 2.1.0+cu121,
  CUDA 12.1 (the versions pinned in `requirements.txt`).
- Environment B (a subset of the Amazon-Book runs): Linux, NVIDIA RTX 3090 Ti,
  Python 3.10.20, PyTorch 2.6.0+cu124, CUDA 12.4.

Hyperparameters, seeds, data splits, and evaluation code are identical across
the two; results agree to floating-point differences (Online Resource 1,
Section S23).

## Determinism

All runs seed Python, NumPy, and PyTorch (CPU and CUDA) from `--seed`, and the
data splits, taxonomy artifacts, and evaluation protocol are deterministic
(`python -m tools.protocol.validate_determinism`, which rebuilds them from the raw
downloads described in the README, Data Bootstrap section, and compares every
output hash). Bitwise-identical training
across different GPUs, drivers, or PyTorch versions is not guaranteed (sparse
GPU kernels are not forced to be deterministic): an independent run of the same
configuration on the second environment above differs from the original by at
most 0.26% on any reported group metric (Online Resource 1, Section S23).

## Device contract

- `--device auto`: use CUDA when available, otherwise CPU.
- `--device cpu`: force CPU.
- `--device cuda --gpu_id N`: require CUDA device N and fail if unavailable.
- The legacy `--cuda true|false` option is accepted for compatibility.

Every run manifest records the requested/resolved device, Python, PyTorch,
CUDA runtime, operating platform, hardware name, and Git commit.

## Reproducible run

1. Commit all code and configuration changes.
2. Clone or pull that exact commit on the training workstation.
3. Create the Python environment from `requirements.txt`.
4. Run `python -m tools.repository.preflight`.
5. Run CPU smoke tests before starting the GPU matrix.
6. Run training with explicit seeds and `--device cuda`.
7. Compile results without editing metrics by hand.
8. Export the completed run with `python -m tools.analysis.export_evidence`.

Checkpoint files are portable because all loaders use `map_location=device`.
NCL full training additionally requires a compatible FAISS installation.

## Resume an interrupted run

Every model saves `last_model.pt` at each validation interval. Resume with the
same model, dataset, seed, and hyperparameters, and point `resume_checkpoint`
to that file. For example:

```powershell
python main.py --model LightGCN --dataset amazon-book --seed 42 --device cuda --gpu_id 0 --resume_checkpoint .\log\p0\baseline\LightGCN\amazon-book\<config_id>\seed42\last_model.pt
```

The model, optimizer, random-number generators, validation history, current
epoch, and early-stopping state are restored. `best_validation_model.pt` is
retained for final test evaluation; it is not the normal resume checkpoint.
Baseline runs created through the experiment runner can instead use `--resume`.

## Determinism and output isolation

Seeds fix the Python, NumPy and PyTorch random number generators. PyTorch's
deterministic-algorithm mode and the cuDNN deterministic flag are not enabled,
so runs are not expected to be bitwise reproducible across GPUs, operating
systems or library versions; the reported means and spreads come from three
seeds on two documented environments. Set `TAXPRO_OUTPUT_ROOT` to an empty
directory to keep a reproduction run apart from any existing `log/` tree; the
checkpoint resolver additionally ignores exploratory `A7-*` families and fails
on an unpinned choice between several canonical families.

## Result policy

`log/`, checkpoints, and runtime figures are generated artifacts and are not
committed. The audit trail under `results/` is: `results_manifest.csv` (run-level
provenance), `metrics_seed.csv` (per-seed metrics), `claim_evidence.csv` (claim to
table, run, and script), and the small bootstrap/analysis outputs cited by the paper.
A paper claim must point to an evidence bundle containing the run manifest,
resolved configuration, final metrics, relevant logs, and checksums.
