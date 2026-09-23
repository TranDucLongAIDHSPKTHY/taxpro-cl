"""The checkpoint resolver must tie a reported number to the canonical run
family, even when the output tree also holds exploratory (A7) sweeps."""

import json

import pytest

from tests.Recommendation_system import checkpoint_selection as cs
from tests.Recommendation_system import config


def _make_run(root, relative, selection_value):
    run_dir = root.joinpath(*relative.split("/"))
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    (run_dir / "final_test_metrics.json").write_text(
        json.dumps({"selection_value": selection_value}), encoding="utf-8"
    )
    return run_dir


@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "model_result_dir", lambda model, dataset: tmp_path)
    monkeypatch.setattr(config, "CHECKPOINT_OVERRIDES", {})
    monkeypatch.setattr(config, "CHECKPOINT_FAMILY_OVERRIDES", {})
    return tmp_path


def test_exploratory_family_is_never_selected_even_if_better_on_validation(tree):
    canonical = _make_run(tree, "seed42", 0.040)
    _make_run(tree, "seed0", 0.039)
    _make_run(tree, "A7-temp0.08/seed42", 0.900)

    choice = cs.select_checkpoint("NCL", "amazon-book")

    assert choice.run_dir == canonical
    assert choice.num_candidates == 2


def test_two_canonical_families_fail_closed(tree):
    _make_run(tree, "family-a/seed42", 0.040)
    _make_run(tree, "family-b/seed42", 0.041)

    with pytest.raises(cs.AmbiguousCheckpointFamilyError):
        cs.select_checkpoint("SimGCL", "musical-instruments")


def test_family_override_resolves_ambiguity_and_still_picks_seed_by_validation(tree, monkeypatch):
    _make_run(tree, "family-a/seed42", 0.040)
    best = _make_run(tree, "family-a/seed0", 0.043)
    _make_run(tree, "family-b/seed42", 0.100)
    monkeypatch.setattr(
        config, "CHECKPOINT_FAMILY_OVERRIDES", {("SimGCL", "musical-instruments"): tree / "family-a"}
    )

    choice = cs.select_checkpoint("SimGCL", "musical-instruments")

    assert choice.run_dir == best
