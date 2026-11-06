"""Grids of the two added baselines on Office-Products (docs/extended_evaluation_protocol.md).

The authors' code runs in the comparator environment through tools/comparators/upstream.py.
Its final tables come back here and are evaluated with the repository's own evaluator, so
each run has the usual files and the test split opens through evaluate_sealed_test, once,
for the rule-R cell only, after the selection is committed.

Usage:
    python -m tools.experiments.run_comparators --prepare
    python -m tools.experiments.run_comparators --package comparator-kgcl comparator-galore [--dry-run]
    python -m tools.experiments.run_comparators --select
    python -m tools.experiments.run_comparators --open-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as training_entry  # noqa: E402
from config_path.config_path import evaluation_protocol_dir  # noqa: E402
from models.ExternalEmbeddings import ExternalEmbeddings  # noqa: E402
from tools.data.build_redrawn_splits import read_pairs  # noqa: E402
from tools.experiments.run_confirmatory import rule_ov, rule_r, validation_means  # noqa: E402
from tools.experiments.run_extended import (  # noqa: E402
    FAMILIES,
    LOCKED_SPLIT,
    OFFICE,
    OUTPUT,
    PRETRAINING_STEP,
    RESULTS,
    SEEDS,
    Job,
    committed_selection,
    log,
    read_manifest,
    require_clean_code,
    require_frozen_protocol,
    run_status,
    sha256_path,
    update_registry,
)
from utility.utility_data.data_loader import Data  # noqa: E402
from utility.utility_train.group_evaluator import evaluate_model_full_catalog, load_targets  # noqa: E402

COMPARATORS = Path(os.environ.get("TAXPRO_COMPARATORS", ROOT.parent / "TaxPro-CL_comparators"))
PYTHON = COMPARATORS / ".venv" / "Scripts" / "python.exe"
PREPARED = COMPARATORS / "prepared" / OFFICE
SELECTION = RESULTS / "selection_comparators.json"
UPSTREAM = ROOT / "tools" / "comparators" / "upstream.py"
KGCL_TEMPERATURES = (0.05, 0.10, 0.15, 0.20)
KGCL_WEIGHTS = (0.02, 0.06, 0.10)
GALORE_EDGES = (1, 2, 5, 10)
GALORE_DROPS = (0.05, 0.10, 0.20)
PACKAGES = {"comparator-kgcl": "KGCL", "comparator-galore": "GALORE"}
EXPECTED_RUNS = 36


def plan(package):
    method = PACKAGES[package]
    if method == "KGCL":
        cells = [(f"temp{t}-ssl{w}", {"temperature": t, "ssl_weight": w})
                 for t in KGCL_TEMPERATURES for w in KGCL_WEIGHTS]
    else:
        cells = [(f"edges{e}-drop{d}", {"edge_num": e, "drop_rate": d}) for e in GALORE_EDGES for d in GALORE_DROPS]
    jobs = [Job(package, f"{method}-{cell}", method, OFFICE, OFFICE, LOCKED_SPLIT, f"grid-{cell}", seed,
                tuple(sorted((key, str(value)) for key, value in values.items())))
            for cell, values in cells for seed in SEEDS]
    if len(jobs) != EXPECTED_RUNS:
        raise ValueError(f"{package}: {len(jobs)} runs planned, {EXPECTED_RUNS} expected")
    return jobs


def category_triples(rows):
    """(item, level, node) for each category on an item's path, leaf at level 0. KGCL reads
    item-headed triples only, so the path is attached to the item, one relation per level.
    Tokens holding HTML are not categories and are left out."""
    nodes, triples = {}, []
    for row in rows:
        if row["final_leaf"] is None:
            continue
        tokens = [token.strip() for token in row["final_leaf"].split(">")]
        tokens = [token for token in tokens if token and "<" not in token]
        for depth in range(len(tokens)):
            node = " > ".join(tokens[:depth + 1])
            triples.append((row["item_id"], len(tokens) - 1 - depth, nodes.setdefault(node, len(nodes))))
    return triples


def by_user(pairs):
    grouped = {}
    for user, item in sorted(pairs):
        grouped.setdefault(user, []).append(item)
    return grouped


def write_user_lines(path, pairs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="\n", encoding="utf-8") as stream:
        for user, items in by_user(pairs).items():
            stream.write(f"{user} {' '.join(map(str, items))}\n")


def write_triples(path, pairs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="\n", encoding="utf-8") as stream:
        for user, item in sorted(pairs):
            stream.write(f"{user} {item} 1\n")


def prepare():
    """The authors' formats, written outside the repository. Validation fills every test slot."""
    source = ROOT / "dataset_verify" / OFFICE
    train, validation = read_pairs(source / "train.txt"), read_pairs(source / "validation.txt")
    kgcl = PREPARED / "kgcl" / "amazon-book"
    write_user_lines(kgcl / "train.txt", train)
    write_user_lines(kgcl / "test.txt", validation)
    provenance = ROOT / "metadata" / "taxonomy_variants" / OFFICE / "no_merge" / "mapping_provenance.jsonl"
    with provenance.open("r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    triples = category_triples(rows)
    with (kgcl / "kg.txt").open("w", newline="\n", encoding="utf-8") as stream:
        stream.writelines(f"{head} {relation} {tail}\n" for head, relation, tail in triples)
    targets = load_targets(evaluation_protocol_dir(OFFICE), "validation")["overall"]
    PREPARED.mkdir(parents=True, exist_ok=True)
    (PREPARED / "validation_overall.json").write_text(
        json.dumps({str(user): sorted(items) for user, items in targets.items()}), encoding="utf-8")
    galore = COMPARATORS / "GALORE" / "dataset" / OFFICE
    write_triples(galore / "train.txt", train)
    write_triples(galore / "train_raw.txt", train)
    for name in ("valid.txt", "test.txt", "test1.txt", "test2.txt"):
        write_triples(galore / name, validation)
    subprocess.run([str(PYTHON), str(UPSTREAM), "--method", "prepare-galore", "--comparators", str(COMPARATORS)],
                   check=True)
    print(f"triples={len(triples)} entities={len({t for _, _, t in triples})} "
          f"relations={len({r for _, r, _ in triples})}")


def pretrained_tables(seed):
    """The two-layer LightGCN of the same seed, from which GALORE starts."""
    path = PREPARED / f"lightgcn_2layers_seed{seed}.npz"
    if not path.exists():
        source = ROOT / OUTPUT / OFFICE / "LightGCN" / PRETRAINING_STEP / f"seed{seed}" / "best_validation_model.pt"
        if not source.exists():
            raise SystemExit(f"run the package pretrain-lightgcn first: {source} is missing")
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        state = checkpoint["model_state_dict"]
        np.savez(path, users=state["user_embedding.weight"].numpy(), items=state["item_embedding.weight"].numpy())
    return path


def upstream_command(item):
    config = dict(item.config)
    command = [str(PYTHON), str(UPSTREAM), "--comparators", str(COMPARATORS), "--prepared", str(PREPARED),
               "--output", str(ROOT / item.run_dir), "--seed", str(item.seed)]
    if item.model == "KGCL":
        return command + ["--method", "kgcl", "--temperature", config["temperature"],
                          "--ssl-weight", config["ssl_weight"]]
    return command + ["--method", "galore", "--edge-num", config["edge_num"], "--drop-rate", config["drop_rate"],
                      "--pretrained", str(pretrained_tables(item.seed))]


def evaluation_config():
    """Evaluation settings of the LightGCN family on this dataset."""
    config = dict(read_manifest(f"{FAMILIES[OFFICE]['LightGCN']}/seed42")["configuration"])
    config.pop("resume_checkpoint", None)
    return config


def pad(table, rows):
    """Items or users with no training interaction keep a zero vector, so a zero score."""
    padded = np.zeros((rows, table.shape[1]), dtype=np.float32)
    padded[:table.shape[0]] = table
    return torch.as_tensor(padded)


def import_tables(item, device, started_at, duration):
    run_dir = ROOT / item.run_dir
    config = evaluation_config()
    logger = logging.getLogger("comparators")
    dataset = Data(str(ROOT / str(config.get("dataset_path", "./dataset_verify/")) / OFFICE), config, logger=logger)
    arrays = np.load(run_dir / "embeddings.npz")
    model = ExternalEmbeddings(config, dataset, device)
    model.load_tables(pad(arrays["users"], dataset.num_users).to(device), pad(arrays["items"], dataset.num_items).to(device))
    result = evaluate_model_full_catalog(dataset, model, device, evaluation_protocol_dir(OFFICE),
                                         split="validation", ks=(10, 20))
    overall = float(result["overall"]["recall"]["20"])
    epoch = int(arrays["best_epoch"])
    arrays.close()
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "validation_metrics": {"groupwise": result},
                "selection_metric": "validation Recall@20 Overall", "selection_value": overall,
                "training_seed": item.seed, "configuration": config}, run_dir / "best_validation_model.pt")
    (run_dir / "validation_metrics.json").write_text(
        json.dumps([{"epoch": epoch, "primary_value": overall, "metrics": {"groupwise": result}}], indent=2) + "\n",
        encoding="utf-8")
    environment = training_entry.runtime_environment(device, str(device))
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {"status": "completed", "started_at_utc": started_at, "model": "ExternalEmbeddings",
                "comparator": item.model, "dataset": OFFICE, "training_seed": str(item.seed),
                "environment": environment, "selection_metric": "validation Recall@20 Overall",
                "test_policy": "once after loading best validation checkpoint", "model_metadata": None,
                "comparator_parameters": dict(item.config), "configuration": config,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(), "duration_seconds": round(duration, 6),
                "completed_epochs": epoch}
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (run_dir / "embeddings.npz").unlink()


def execute(item, device):
    if run_status(item.run_dir) == "completed":
        return
    run_dir = ROOT / item.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log(f"START {item.run_id} {item.run_dir}", item.package)
    started_at, started = datetime.now(timezone.utc).isoformat(), time.time()
    with (run_dir / "training.log").open("a", encoding="utf-8") as stream:
        for attempt in (1, 2):
            code = subprocess.run(upstream_command(item), stdout=stream, stderr=subprocess.STDOUT,
                                  env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONHASHSEED="0")).returncode
            if code == 0 and (run_dir / "embeddings.npz").exists():
                break
            log(f"FAILED attempt {attempt} (exit {code}) {item.run_id}", item.package)
        else:
            raise RuntimeError(f"run failed twice: {item.run_dir}")
    import_tables(item, torch.device(device), started_at, time.time() - started)
    update_registry([item])
    log(f"DONE ({time.time() - started:.0f}s) {item.run_id}", item.package)


def run_packages(packages, device, dry_run):
    plans = {package: plan(package) for package in packages}
    for package, jobs in plans.items():
        remaining = [j for j in jobs if run_status(j.run_dir) != "completed"]
        print(f"{package}: {len(jobs)} runs planned, {len(remaining)} remaining")
        if dry_run:
            for item in remaining[:2]:
                print("  would run", item.run_id, item.run_dir.as_posix())
    if dry_run:
        return 0
    require_clean_code()
    require_frozen_protocol()
    for package, jobs in plans.items():
        for item in jobs:
            execute(item, device)
        log(f"PACKAGE COMPLETE {package}; test split still sealed", package)
    return 0


def cell_of(item):
    return item.cell.removeprefix("grid-")


def freeze_selection():
    path = ROOT / SELECTION
    if path.exists():
        raise SystemExit(f"{path} is frozen already")
    document = {"protocol": "docs/extended_evaluation_protocol.md",
                "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for package, method in PACKAGES.items():
        jobs = plan(package)
        finished = [j for j in jobs if run_status(j.run_dir) == "completed"]
        if not finished:
            continue
        if len(finished) != len(jobs):
            raise SystemExit(f"{package}: {len(jobs) - len(finished)} runs are not complete; nothing is selected")
        cells = []
        for name in dict.fromkeys(cell_of(j) for j in jobs):
            cells.append((name, validation_means([ROOT / j.run_dir for j in jobs if cell_of(j) == name])))
        document[method] = {"rule_R": rule_r(cells), "rule_Ov": rule_ov(cells),
                            "cells": {name: value["mean"] for name, value in cells}}
    path.write_bytes((json.dumps(document, indent=2) + "\n").encode("utf-8"))
    path.with_suffix(".sha256").write_bytes(f"{sha256_path(path)}  {path.name}\n".encode())
    log("comparator selection written; commit it before opening the test split")
    return 0


def open_test(device):
    selection = committed_selection(SELECTION)
    jobs, runs = [], []
    for package, method in PACKAGES.items():
        if method not in selection:
            continue
        jobs += plan(package)
        runs += [str(ROOT / j.run_dir) for j in plan(package) if cell_of(j) == selection[method]["rule_R"]
                 and not (ROOT / j.run_dir / "final_test_metrics.json").exists()]
    log(f"OPEN TEST comparators: {len(runs)} runs")
    if not runs:
        return 0
    code = subprocess.run([sys.executable, "-m", "tools.experiments.evaluate_sealed_test",
                           "--device", device, "--runs", *runs], cwd=ROOT).returncode
    update_registry(jobs)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--package", nargs="+", choices=sorted(PACKAGES))
    action.add_argument("--select", action="store_true")
    action.add_argument("--open-test", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    if args.prepare:
        return prepare() or 0
    if args.package:
        return run_packages(args.package, args.device, args.dry_run)
    if args.select:
        return freeze_selection()
    return open_test(args.device)


if __name__ == "__main__":
    raise SystemExit(main())
