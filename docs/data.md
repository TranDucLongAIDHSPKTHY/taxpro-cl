# Data And Metadata

`dataset_verify/` contains the locked model-ready splits. Training and
evaluation use these files and the artifacts under `preprocessed/`. Both, and
`metadata/taxonomy_variants/`, are verified by SHA256, so `.gitattributes` stores
them byte for byte (no line-ending conversion) on every platform.

`python -m tools.data.build_splits` reconstructs these inputs from raw
`dataset/<dataset>/train.txt` and `test.txt`. It repeatedly removes users and
items below degree 5 until stable, then creates a per-user validation holdout
by shuffling sorted user interactions with a seeded RNG. Users with 2-9
interactions contribute one item, users with at least 10 contribute
`floor(0.1 * N)`, and singleton users remain in train. Seeds 42, 123, and 2026
are accepted; 42 is the main paper split. Raw test is copied without rewriting,
and every source/output hash is stored in `split_manifest.json`.
Staging manifests also compare their hashes with the currently locked split.

The default destination is `results/data_splits/split_seed_42/`, so an audit
cannot overwrite the locked paper data. Use `--promote --force` only when
intentionally locking a new campaign; rebuild protocol and taxonomy artifacts
immediately afterward.

Raw metadata is downloaded from the URLs declared in
`config_path/config_path.py`. After extraction, the downloaded `.gz` or `.zip`
archive is removed. Only the extracted source is retained locally. Run
`python -m tools.data.metadata_manifest` to refresh the source inventory; add
`--hash-content` when full content hashes are required.

Large raw metadata is ignored by Git. Generated taxonomy variants and compact
protocol manifests are rebuilt with `python -m tools.protocol.build --build`.
Run `python -m tools.protocol.validate_gate` before training.
