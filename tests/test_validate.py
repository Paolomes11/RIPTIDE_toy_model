import numpy as np
from scipy.stats import norm

from riptide_toy.validate import bias_curve, coverage_curve, pull_histogram


def test_bias_and_pull_on_honest_synthetic_data():
    # dati costruiti apposta per essere onesti (nessun bias, sigma_hat corretta):
    # se questo test non passa il bug e' nella funzione, non nei dati (guida, Sez. 3).
    rng = np.random.default_rng(20260907)
    truth = rng.uniform(1, 5, 2000)
    estimate = truth + rng.normal(0, 0.2, 2000)
    sigma_hat = np.full(2000, 0.2)

    centers, bias = bias_curve(truth, estimate)
    assert centers.shape == (20,)
    assert bias.shape == (20,)
    assert np.nanmax(np.abs(bias)) < 0.05

    pull = pull_histogram(truth, estimate, sigma_hat)
    assert pull.shape == (2000,)
    assert abs(pull.mean()) < 0.05
    assert abs(pull.std() - 1.0) < 0.05


def test_coverage_curve_on_honest_synthetic_data():
    rng = np.random.default_rng(20260907)
    truth = rng.uniform(1, 5, 2000)
    estimate = truth + rng.normal(0, 0.2, 2000)
    sigma_hat = np.full(2000, 0.2)

    levels = np.array([0.68, 0.90, 0.95])
    z = norm.ppf(0.5 + levels / 2)
    half_widths = z[None, :] * sigma_hat[:, None]
    intervals = np.stack(
        [estimate[:, None] - half_widths, estimate[:, None] + half_widths], axis=-1
    )

    out_levels, coverage = coverage_curve(truth, intervals, levels)
    np.testing.assert_array_equal(out_levels, levels)
    assert np.all(np.abs(coverage - levels) < 0.03)
