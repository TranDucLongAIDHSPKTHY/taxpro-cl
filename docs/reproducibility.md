# Reproducibility

## Environment roles

The development machine may have no GPU. It is used for code changes, CPU unit
tests, metadata preparation, validation, result compilation, and reporting.
Official training is performed on the workstation with an Intel Core i5 12th
generation CPU, 32 GB RAM, and an NVIDIA RTX 4060 Ti 16 GB GPU.

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
9. Import the bundle into `TaxPro-CL-Research` and generate the weekly report.

Checkpoint files are portable because all loaders use `map_location=device`.
NCL full training additionally requires a compatible FAISS installation.

## Resume an interrupted run

Every model saves `last_model.pt` at each validation interval. Resume with the
same model, dataset, seed, and hyperparameters, and point `resume_checkpoint`
to that file. For example:

```powershell
python main.py --model LightGCN --dataset amazon-book --seed 42 --device cuda --gpu_id 0 --resume_checkpoint .\log\LightGCN\amazon-book\seed42\last_model.pt
```

The model, optimizer, random-number generators, validation history, current
epoch, and early-stopping state are restored. `best_validation_model.pt` is
retained for final test evaluation; it is not the normal resume checkpoint.
Baseline runs created through the experiment runner can instead use `--resume`.

## Result policy

`log/`, `results/`, checkpoints, and runtime figures are generated artifacts
and are not committed by default. A paper claim must point to an immutable
evidence bundle containing the run manifest, resolved configuration, final
metrics, relevant logs, and checksums.
