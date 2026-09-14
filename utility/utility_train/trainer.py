import json
import random
from pathlib import Path

import numpy as np
import torch
from time import time
from tqdm import tqdm

import utility.utility_train.batch_test as batch_test
import utility.utility_function.tools as tools


def _restore_rng_state(payload):
    if not payload:
        return
    if payload.get("torch") is not None:
        # Checkpoints loaded with map_location="cuda" move byte tensors to
        # CUDA, whereas PyTorch's RNG setters require CPU byte tensors.
        torch.set_rng_state(payload["torch"].cpu())
    if torch.cuda.is_available() and payload.get("cuda") is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda"]])
    if payload.get("numpy") is not None:
        np.random.set_state(payload["numpy"])
    if payload.get("python") is not None:
        random.setstate(payload["python"])


def _load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _validate_resume_configuration(checkpoint_config, config):
    # The destination path and requested final epoch are safe to change on resume.
    ignored = {"resume_checkpoint", "training_epochs"}
    for key, checkpoint_value in checkpoint_config.items():
        if key in ignored or key not in config:
            continue
        if str(checkpoint_value) != str(config[key]):
            raise ValueError("Checkpoint configuration mismatch: {}".format(key))


def load_resume_checkpoint(path, model, optimizer, config, device, training_seed=None):
    """Restore a baseline checkpoint and return its serialized training state."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Resume checkpoint not found: {}".format(path))
    checkpoint = _load_checkpoint(path, device)
    _validate_resume_configuration(checkpoint.get("configuration", {}), config)
    checkpoint_seed = checkpoint.get("training_seed")
    if (
        training_seed is not None
        and checkpoint_seed is not None
        and int(checkpoint_seed) != int(training_seed)
    ):
        raise ValueError("Checkpoint training seed mismatch")
    model.load_state_dict(checkpoint["model_state_dict"])
    if checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    _restore_rng_state(checkpoint.get("rng_state"))
    return checkpoint


def prepare_resumed_training(model, optimizer, args, config, dataset, device, logger):
    """Initialize common early-stopping state and optionally restore a run."""
    seed = getattr(args, "seed", None)
    model.training_seed = int(seed) if seed is not None else None
    best_results = {
        "count": 0,
        "epoch": 0,
        "recall": [0.0 for _ in eval(config["top_K"])],
        "ndcg": [0.0 for _ in eval(config["top_K"])],
        "stop": 0,
    }
    resume_value = str(config.get("resume_checkpoint", "none")).strip()
    if resume_value.lower() in {"", "none", "null"}:
        return 0, best_results

    checkpoint = load_resume_checkpoint(
        resume_value, model, optimizer, config, device, training_seed=model.training_seed
    )
    start_epoch = int(checkpoint["epoch"])
    if start_epoch >= int(config["training_epochs"]):
        raise ValueError(
            "Checkpoint epoch {} is not below training_epochs {}".format(
                start_epoch, config["training_epochs"]
            )
        )
    state = checkpoint.get("training_state", {})
    for key in ("count", "epoch", "stop"):
        if key in state:
            best_results[key] = int(state[key])
    for key in ("recall", "ndcg"):
        if key in state:
            best_results[key] = list(state[key])
    best_results["primary_metric"] = state.get(
        "primary_metric", checkpoint.get("selection_metric")
    )
    best_results["primary_value"] = float(
        state.get("primary_value", checkpoint.get("selection_value", float("-inf")))
    )
    history_path = Path(dataset.training_output_dir) / "validation_metrics.json"
    if history_path.is_file():
        dataset.validation_history = json.loads(history_path.read_text(encoding="utf-8"))
    logger.info("Resumed training from %s at epoch %d", resume_value, start_epoch)
    return start_epoch, best_results


def universal_trainer(model, args, config, dataset, device, logger):
    loop_started_at = time()
    total_epochs = int(config["training_epochs"])
    validation_interval = int(config.get("interval", 1))
    if validation_interval <= 0:
        raise ValueError("interval must be a positive integer")
    model.to(device)
    parameter_count = tools.format_parameter_count(model)
    logger.info("Model parameters: %s", parameter_count)
    Optim = torch.optim.Adam(model.parameters(), lr=float(config["learn_rate"]))
    logger.info("Optimizer Initialized: Adam lr=%s", config["learn_rate"])
    logger.info("Validation interval: every %d epoch(s)", validation_interval)

    start_epoch, best_results = prepare_resumed_training(
        model, Optim, args, config, dataset, device, logger
    )
    for epoch in range(start_epoch, total_epochs):
        dataset.current_epoch = epoch + 1
        logger.info("Start Epoch: %d", epoch + 1)
        start_time = time()
        model.train()

        sample_data = dataset.sample_data_to_train_all()
        users = torch.Tensor(sample_data[:, 0]).long().to(device)
        pos_items = torch.Tensor(sample_data[:, 1]).long().to(device)
        neg_items = torch.Tensor(sample_data[:, 2]).long().to(device)
        users, pos_items, neg_items = tools.shuffle(users, pos_items, neg_items)
        num_batch = len(users) // int(config["batch_size"]) + 1
        total_loss_list = []

        batches = tools.mini_batch(users, pos_items, neg_items, batch_size=int(config["batch_size"]))
        with tqdm(
            enumerate(batches),
            desc="{} | {} | seed={} | epoch {}/{}".format(
                args.model, config["dataset"], args.seed, epoch + 1, total_epochs
            ),
            total=int(num_batch),
        ) as progress:
            progress.set_postfix(params=parameter_count)
            for batch_i, (batch_users, batch_positive, batch_negative) in progress:
                loss_list = model(batch_users, batch_positive, batch_negative)
                if batch_i == 0:
                    assert len(loss_list) >= 1
                    total_loss_list = [0.0] * len(loss_list)
                total_loss = 0.0
                for index, loss in enumerate(loss_list):
                    total_loss += loss
                    total_loss_list[index] += loss.item()
                Optim.zero_grad()
                total_loss.backward()
                Optim.step()
                loss_names = ("bpr", "reg", "ssl")
                postfix = {"params": parameter_count}
                for index in range(len(total_loss_list)):
                    name = loss_names[index] if index < len(loss_names) else "loss{}".format(index)
                    postfix[name] = "{:.4f}".format(
                        total_loss_list[index] / (batch_i + 1)
                    )
                progress.set_postfix(**postfix)

        elapsed = time() - start_time
        loss_strs = str(round(sum(total_loss_list) / num_batch, 6)) + " = " + " + ".join(
            str(round(value / num_batch, 6)) for value in total_loss_list
        )
        logger.info("Train Loss: epoch=%d loss=%s", epoch + 1, loss_strs)
        logger.info("End Epoch: %d training_time=%.3f", epoch + 1, elapsed)
        if tools.should_run_validation(epoch, total_epochs, validation_interval):
            _, best_results = batch_test.general_test(
                dataset, model, device, config, epoch, best_results, Optim, logger
            )
        elapsed_total = time() - loop_started_at
        logger.info(
            "Epoch Progress: %d/%d elapsed=%s ETA=%s",
            epoch + 1,
            total_epochs,
            tools.format_duration(elapsed_total),
            tools.estimate_remaining_time(elapsed_total, epoch + 1, total_epochs),
        )
        if best_results["stop"] > 0:
            break

    logger.info(
        "Training loop completed: best_epoch=%d best_recall=%s best_ndcg=%s",
        best_results["epoch"], best_results["recall"], best_results["ndcg"],
    )
