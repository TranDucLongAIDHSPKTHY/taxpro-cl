"""Calibration checks for the multiplicity-corrected inference of Table 11."""

import numpy as np

from tools.analysis import a2_multiplicity_correction as mc


def test_sign_flip_is_calibrated_under_the_null():
    rng = np.random.default_rng(0)
    rejections = 0
    trials = 300
    for _ in range(trials):
        # heavy-tailed, symmetric, mean zero
        values = rng.standard_t(df=3, size=150)
        if mc.sign_flip_pvalue(values, 400, rng) <= 0.05:
            rejections += 1
    # nominal 5%; allow generous Monte Carlo slack (binomial SD ~ 1.3%)
    assert rejections / trials < 0.09


def test_sign_flip_detects_a_real_shift():
    rng = np.random.default_rng(1)
    values = rng.normal(loc=0.3, scale=1.0, size=200)
    assert mc.sign_flip_pvalue(values, 2000, rng) < 0.001


def test_holm_step_down_stops_at_first_failure():
    cells = [{"p": 0.001}, {"p": 0.0104}, {"p": 0.0121}, {"p": 0.5}]
    mc.holm_bonferroni(cells, "p", "sig")
    # thresholds 0.05/4, 0.05/3, 0.05/2, 0.05/1 -> 0.0125, 0.01667, 0.025, 0.05
    assert [c["sig"] for c in cells] == [True, True, True, False]
    cells = [{"p": 0.001}, {"p": 0.02}, {"p": 0.021}, {"p": 0.5}]
    mc.holm_bonferroni(cells, "p", "sig")
    assert [c["sig"] for c in cells] == [True, False, False, False]
