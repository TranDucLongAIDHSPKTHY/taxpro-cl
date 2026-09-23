"""Discover and select which trained checkpoint represents a (model, dataset) pair.

Checkpoint coverage in log/p0 is uneven: some (model, dataset) pairs have one
clearly labelled run, others have an unlabelled sweep of many run
directories with no naming convention marking a "best"/"final" one. This
module resolves that ambiguity with one explicit, auditable rule: pick the
run whose training reached the highest validation selection metric
(the same metric best_validation_model.pt was checkpointed on), never the
test metric, so nothing here is chosen by looking at the number being
reported. CHECKPOINT_OVERRIDES in config.py can still pin an exact run.

Two safeguards keep the choice tied to the canonical training commands even
when the output tree also holds other experiments:

* run families named with config.EXPLORATORY_FAMILY_PREFIXES (the A7
  comparable-budget grid) are never candidates;
* if more than one run family remains for a (model, dataset), the resolver
  raises AmbiguousCheckpointFamilyError instead of silently picking the
  highest-validation family -- pin it with CHECKPOINT_FAMILY_OVERRIDES or
  CHECKPOINT_OVERRIDES in config.py.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from config_path.config_path import model_result_dir, relative_to_project

from . import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckpointChoice:
    model_name: str
    dataset_name: str
    run_dir: Path
    seed: int
    validation_score: float
    num_candidates: int

    def as_row(self):
        return {
            "Model": self.model_name,
            "Dataset": self.dataset_name,
            "Checkpoint_Dir": relative_to_project(self.run_dir),
            "Seed": self.seed,
            "Validation_Score": self.validation_score,
            "Num_Candidates_Considered": self.num_candidates,
        }


class AmbiguousCheckpointFamilyError(RuntimeError):
    """More than one non-exploratory run family exists and none is pinned."""


def _display(path):
    try:
        return relative_to_project(path)
    except ValueError:
        return str(path)


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _seed_from_run_dir(run_dir):
    name = Path(run_dir).name
    if name.startswith("seed"):
        try:
            return int(name[len("seed"):])
        except ValueError:
            pass
    return -1


def _is_exploratory(run_dir, family_root):
    parts = Path(run_dir).relative_to(family_root).parts
    return any(part.startswith(config.EXPLORATORY_FAMILY_PREFIXES) for part in parts)


def find_candidate_runs(model_name, dataset_name):
    """Return every completed, non-exploratory run directory for the pair.

    If config.CHECKPOINT_FAMILY_OVERRIDES pins a family for the pair, only that
    family's runs are returned.
    """
    family_dir = model_result_dir(model_name, dataset_name)
    candidates = []
    if not family_dir.exists():
        return candidates
    pinned_family = config.CHECKPOINT_FAMILY_OVERRIDES.get((model_name, dataset_name))
    for manifest_path in sorted(family_dir.rglob("run_manifest.json")):
        run_dir = manifest_path.parent
        if _is_exploratory(run_dir, family_dir):
            continue
        if pinned_family is not None and Path(pinned_family) not in run_dir.parents:
            continue
        try:
            manifest = _read_json(manifest_path)
        except (json.JSONDecodeError, OSError):
            continue
        if manifest.get("status") != "completed":
            continue
        candidates.append(run_dir)
    return candidates


def score_candidate(run_dir):
    """Best validation selection-metric value reached while training run_dir.

    final_test_metrics.json["selection_value"] is the value that
    best_validation_model.pt was actually checkpointed on (it is computed on
    the validation split, despite living in a file named "final_test_..." --
    verified against best_validation_metrics in the same file); prefer it.
    Fall back to the max "primary_value" across validation_metrics.json's
    per-epoch log when the summary file is unavailable.
    """
    final_test_path = Path(run_dir) / "final_test_metrics.json"
    if final_test_path.exists():
        try:
            final_test = _read_json(final_test_path)
        except (json.JSONDecodeError, OSError):
            final_test = {}
        if "selection_value" in final_test:
            return float(final_test["selection_value"])

    logger.warning(
        "%s: final_test_metrics.json missing selection_value, falling back "
        "to max primary_value in validation_metrics.json.",
        relative_to_project(run_dir),
    )
    validation_path = Path(run_dir) / "validation_metrics.json"
    entries = _read_json(validation_path)
    values = [float(entry["primary_value"]) for entry in entries if "primary_value" in entry]
    if not values:
        raise ValueError("No usable validation metric found in {}".format(run_dir))
    return max(values)


def select_checkpoint(model_name, dataset_name):
    override = config.CHECKPOINT_OVERRIDES.get((model_name, dataset_name))
    if override is not None:
        run_dir = Path(override)
        return CheckpointChoice(
            model_name=model_name,
            dataset_name=dataset_name,
            run_dir=run_dir,
            seed=_seed_from_run_dir(run_dir),
            validation_score=score_candidate(run_dir),
            num_candidates=1,
        )

    candidates = find_candidate_runs(model_name, dataset_name)
    if not candidates:
        raise FileNotFoundError(
            "No completed checkpoint found for model={!r} dataset={!r} "
            "under {}".format(
                model_name, dataset_name, relative_to_project(model_result_dir(model_name, dataset_name))
            )
        )

    families = sorted({_display(run_dir.parent) for run_dir in candidates})
    if len(families) > 1:
        raise AmbiguousCheckpointFamilyError(
            "Several run families for model={!r} dataset={!r}: {}. Pin one in "
            "tests/Recommendation_system/config.py (CHECKPOINT_FAMILY_OVERRIDES).".format(
                model_name, dataset_name, ", ".join(families)
            )
        )

    scored = [(run_dir, score_candidate(run_dir)) for run_dir in candidates]
    best_score = max(score for _, score in scored)
    tied = [run_dir for run_dir, score in scored if score == best_score]
    if len(tied) > 1:
        preferred = [rd for rd in tied if _seed_from_run_dir(rd) == config.DEFAULT_SEED_PREFERENCE]
        tied = preferred or tied
        tied = sorted(tied, key=_seed_from_run_dir)
    best_run_dir = tied[0]

    return CheckpointChoice(
        model_name=model_name,
        dataset_name=dataset_name,
        run_dir=best_run_dir,
        seed=_seed_from_run_dir(best_run_dir),
        validation_score=best_score,
        num_candidates=len(candidates),
    )
