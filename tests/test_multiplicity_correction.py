"""Calibration checks for the multiplicity-corrected inference of Table 11 and
of the RQ5 factorial (Online Resource 1, Tables S13e and S26)."""

import itertools

import numpy as np

from tools.analysis import a2_multiplicity_correction as mc
from tools.analysis import factorial_multiplicity as fm


def _exact_two_sided_p(values):
    """Exact sign-flip p-value by enumerating all 2^n sign patterns."""
    v = np.asarray(values, dtype=np.float64)
    observed = abs(v.sum())
    hits = sum(abs(np.dot(signs, v)) >= observed - 1e-12
               for signs in itertools.product((-1.0, 1.0), repeat=len(v)))
    return hits / 2 ** len(v)


def test_factorial_sign_flip_matches_exact_enumeration():
    # small samples: every sign pattern can be enumerated, so the Monte Carlo
    # p-value must agree with the exact one up to Monte Carlo error
    data = np.random.default_rng(7)
    cases = [
        [0.0] * 10,                                   # all differences zero: p = 1
        list(data.normal(0.8, 1.0, 12)),              # shifted
        list(data.normal(0.0, 1.0, 12)),              # no shift
        [0.1, -0.1, 0.2, -0.2, 0.3, 0.05, 0.0, 0.0, 0.15, -0.05, 0.25, 0.1],  # ties and zeros
    ]
    n_perm = 20_000
    for values in cases:
        exact = _exact_two_sided_p(values)
        monte_carlo = fm.sign_flip_pvalue(values, n_perm, np.random.default_rng(42))
        se = np.sqrt(max(exact * (1 - exact), 1.0 / n_perm) / n_perm)
        assert abs(monte_carlo - exact) < 5 * se + 1.0 / n_perm
        assert monte_carlo >= 1.0 / (n_perm + 1)


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
