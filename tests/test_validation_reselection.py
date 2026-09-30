"""Unit tests for the validation-only re-selection rules (Online Resource 1, Section S35).

The tests use synthetic candidates, so they need neither checkpoints nor run logs."""
from tools.analysis import validation_reselection_audit as audit


def _cand(name, overall, long_tail, near_cold):
    return {"family": name, "val": {"overall": overall, "long_tail": long_tail, "near_cold": near_cold, "warm": 0.0}}


CANDIDATES = [
    _cand("best_overall", 0.130, 0.010, 0.002),
    _cand("best_long_tail", 0.120, 0.020, 0.004),   # 7.7% below the best Overall
    _cand("balanced", 0.1285, 0.015, 0.005),         # 1.2% below the best Overall, best Near-Cold
    _cand("close", 0.1295, 0.012, 0.003),            # 0.4% below the best Overall
]


def test_single_criterion_rules():
    assert audit.select(CANDIDATES, "overall")["family"] == "best_overall"
    assert audit.select(CANDIDATES, "lt")["family"] == "best_long_tail"
    assert audit.select(CANDIDATES, "nc")["family"] == "balanced"


def test_tolerance_rules_restrict_to_near_best_overall():
    # within 1%: best_overall and close are eligible; close has the higher Long-Tail
    assert audit.select(CANDIDATES, "lt_within_1")["family"] == "close"
    # within 2%: balanced becomes eligible and has the higher Long-Tail
    assert audit.select(CANDIDATES, "lt_within_2")["family"] == "balanced"
    # within 8%: every candidate is eligible, so the rule equals plain "lt"
    assert audit.select(CANDIDATES, "lt_within_8")["family"] == "best_long_tail"


def test_ties_are_broken_by_the_next_criterion():
    tied = [_cand("a", 0.120, 0.015, 0.004), _cand("b", 0.125, 0.015, 0.004)]
    assert audit.select(tied, "lt")["family"] == "b"          # equal LT -> higher Overall
    tied_nc = [_cand("a", 0.120, 0.014, 0.004), _cand("b", 0.120, 0.016, 0.004)]
    assert audit.select(tied_nc, "nc")["family"] == "b"       # equal NC -> higher LT


def test_config_normalization_fills_defaults_and_canonical_numbers():
    old_run = audit.normalize_config({"temperature": "0.1", "warm_start_epochs": 20})
    new_run = audit.normalize_config({"temperature": 0.1, "warm_start_epochs": "20",
                                      "prototype_weighting": "interaction", "isotropic_blend": 0})
    assert old_run["temperature"] == new_run["temperature"]
    assert old_run["prototype_weighting"] == new_run["prototype_weighting"] == "interaction"
    assert audit.differing_keys(old_run, new_run) == []


def test_tuned_keys_do_not_exclude_but_component_keys_do():
    main = audit.normalize_config({"temperature": 0.1, "same_leaf_weight": 0.0})
    tuned_variant = audit.normalize_config({"temperature": 0.2, "same_leaf_weight": 0.0})
    component_variant = audit.normalize_config({"temperature": 0.1, "same_leaf_weight": 0.4})
    assert audit.differing_keys(tuned_variant, main) == []
    assert audit.differing_keys(component_variant, main) == ["same_leaf_weight"]
