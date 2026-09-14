"""Warm-start, prototype transition and resumable TaxPro-CL trainer."""

from __future__ import annotations

import json
import random
from pathlib import Path
from time import time

import numpy as np
import torch
from tqdm import tqdm

import utility.utility_function.tools as tools
import utility.utility_train.batch_test as batch_test
from config_path.config_path import checkpoint_file


def _restore_rng_state(payload):
    if not payload:
        return
    if payload.get("torch") is not None:
        torch.set_rng_state(payload["torch"].cpu())
    if torch.cuda.is_available() and payload.get("cuda") is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda"]])
    if payload.get("numpy") is not None:
        np.random.set_state(payload["numpy"])
    if payload.get("python") is not None:
        random.setstate(payload["python"])


def load_resume_checkpoint(
    path, model, optimizer, config, device, training_seed=None
):
    checkpoint = _load_checkpoint(path, device)
    metadata = checkpoint.get("taxprocl_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint lacks TaxProCL metadata")
    model.validate_checkpoint_metadata(metadata)
    checkpoint_config = checkpoint.get("configuration", {})
    for key in (
        "dataset",
        "top_K",
        "selection_K",
        "training_epochs",
        "early_stopping",
        "interval",
        "taxonomy_policy",
        "taxonomy_granularity",
        "augmentation_direction",
        "warm_start_epochs",
        "embedding_size",
        "GCN_layer",
        "learn_rate",
        "reg_lambda",
        "batch_size",
        "test_batch_size",
        "num_worker",
        "epsilon_max",
        "delta",
        "temperature",
        "ssl_lambda",
        "mu",
        "unknown_taxonomy_policy",
        "symmetric_info_nce",
        "sparsity_test",
    ):
        if str(checkpoint_config.get(key)) != str(config.get(key)):
            raise ValueError("Checkpoint configuration mismatch: {}".format(key))
    if training_seed is not None:
        checkpoint_seed = checkpoint.get("training_seed")
        if checkpoint_seed is None:
            raise ValueError("Checkpoint lacks training seed")
        if int(checkpoint_seed) != int(training_seed):
            raise ValueError("Checkpoint training seed mismatch")
    model.load_state_dict(checkpoint["model_state_dict"])
    if checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    _restore_rng_state(checkpoint.get("rng_state"))
    return checkpoint


def train_taxprocl(model, args, config, dataset, device, logger):
    model.to(device)
    parameter_count = tools.format_parameter_count(model)
    logger.info("Model parameters: %s", parameter_count)
    seed_value = getattr(args, "seed", None)
    model.training_seed = int(seed_value) if seed_value is not None else None
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learn_rate"]))
    total_epochs = int(config["training_epochs"])
    warm_start_epochs = int(config["warm_start_epochs"])
    validation_interval = int(config.get("interval", 1))
    if warm_start_epochs < 0 or warm_start_epochs >= total_epochs:
        raise ValueError("warm_start_epochs must be in [0, training_epochs)")
    if validation_interval <= 0:
        raise ValueError("interval must be positive")

    best_results = {
        "count": 0,
        "epoch": 0,
        "recall": [0.0 for _ in eval(config["top_K"])],
        "ndcg": [0.0 for _ in eval(config["top_K"])],
        "stop": 0,
    }
    start_epoch = 0
    resume_value = str(config.get("resume_checkpoint", "none")).strip()
    if resume_value.lower() not in {"", "none", "null"}:
        checkpoint = load_resume_checkpoint(
            Path(resume_value),
            model,
            optimizer,
            config,
            device,
            training_seed=model.training_seed,
        )
        start_epoch = int(checkpoint["epoch"])
        training_state = checkpoint.get("training_state", {})
        best_results["count"] = int(training_state.get("count", 0))
        best_results["epoch"] = int(
            training_state.get("epoch", checkpoint.get("epoch", 0))
        )
        best_results["recall"] = list(
            training_state.get("recall", best_results["recall"])
        )
        best_results["ndcg"] = list(
            training_state.get("ndcg", best_results["ndcg"])
        )
        best_results["stop"] = int(training_state.get("stop", 0))
        best_results["primary_metric"] = training_state.get(
            "primary_metric", checkpoint.get("selection_metric")
        )
        best_results["primary_value"] = float(
            training_state.get(
                "primary_value", checkpoint.get("selection_value", float("-inf"))
            )
        )
        logger.info("Resumed TaxProCL from %s at epoch %d", resume_value, start_epoch)

    loop_started = time()
    prototype_evidence = None
    for epoch_index in range(start_epoch, total_epochs):
        epoch_number = epoch_index + 1
        dataset.current_epoch = epoch_number
        if epoch_number == warm_start_epochs + 1 and "taxonomy_phase_lr" in config:
            taxonomy_phase_lr = float(config["taxonomy_phase_lr"])
            if taxonomy_phase_lr <= 0.0:
                raise ValueError("taxonomy_phase_lr must be positive")
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = taxonomy_phase_lr
            logger.info("Taxonomy phase learning rate: %s", taxonomy_phase_lr)
        if (
            epoch_number == warm_start_epochs + 1
            and not bool(model.prototype_bank.initialized.item())
        ):
            prototype_evidence = model.initialize_prototypes(warm_start_epochs)
            evidence_path = Path(dataset.training_output_dir) / "prototype_init.json"
            evidence_path.write_text(
                json.dumps(
                    {
                        **prototype_evidence,
                        "taxonomy_hash": model.taxonomy_hash,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            logger.info("Prototype initialization: %s", prototype_evidence)

        model.train()
        sampled = dataset.sample_data_to_train_all()
        users = torch.as_tensor(sampled[:, 0], dtype=torch.long, device=device)
        positives = torch.as_tensor(sampled[:, 1], dtype=torch.long, device=device)
        negatives = torch.as_tensor(sampled[:, 2], dtype=torch.long, device=device)
        users, positives, negatives = tools.shuffle(users, positives, negatives)
        totals = [0.0, 0.0, 0.0]
        batch_count = 0
        batch_size = int(config["batch_size"])
        total_batches = (len(users) + batch_size - 1) // batch_size
        phase = "warm_start" if epoch_number <= warm_start_epochs else "joint"
        batches = tools.mini_batch(
            users,
            positives,
            negatives,
            batch_size=batch_size,
        )
        with tqdm(
            enumerate(batches),
            desc="TaxPro-CL | {} | seed={} | epoch {}/{} ({})".format(
                config["dataset"], model.training_seed, epoch_number, total_epochs, phase
            ),
            total=total_batches,
        ) as progress:
            progress.set_postfix(params=parameter_count)
            for _, (batch_users, batch_positive, batch_negative) in progress:
                losses = model(
                    batch_users,
                    batch_positive,
                    batch_negative,
                    epoch=epoch_number,
                )
                total_loss = sum(losses)
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                batch_count += 1
                for index, value in enumerate(losses):
                    totals[index] += float(value.detach().item())
                progress.set_postfix(
                    params=parameter_count,
                    bpr="{:.4f}".format(totals[0] / batch_count),
                    reg="{:.4f}".format(totals[1] / batch_count),
                    cl="{:.4f}".format(totals[2] / batch_count),
                )
        logger.info(
            "TaxProCL epoch=%d phase=%s batches=%d bpr=%.8f reg=%.8f cl=%.8f stats=%s",
            epoch_number,
            phase,
            batch_count,
            totals[0] / max(batch_count, 1),
            totals[1] / max(batch_count, 1),
            totals[2] / max(batch_count, 1),
            model.last_batch_statistics,
        )

        if tools.should_run_validation(epoch_index, total_epochs, validation_interval):
            _, best_results = batch_test.general_test(
                dataset,
                model,
                device,
                config,
                epoch_index,
                best_results,
                optimizer,
                logger,
            )
            if epoch_number <= warm_start_epochs:
                best_results["count"] = 0
                best_results["stop"] = 0
        elapsed = time() - loop_started
        logger.info(
            "Epoch Progress: %d/%d elapsed=%s ETA=%s",
            epoch_number,
            total_epochs,
            tools.format_duration(elapsed),
            tools.estimate_remaining_time(elapsed, epoch_number, total_epochs),
        )
        if best_results.get("stop", 0) > 0 and epoch_number > warm_start_epochs:
            break

    logger.info(
        "TaxProCL training completed: best_epoch=%s prototype=%s",
        best_results.get("epoch"),
        prototype_evidence or model.checkpoint_metadata(),
    )
    return best_results


def _load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)
