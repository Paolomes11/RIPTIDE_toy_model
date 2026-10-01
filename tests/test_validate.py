import numpy as np
from scipy.stats import norm

from riptide_toy.validate import (
    angular_pull,
    angular_residual,
    bias_curve,
    coverage_curve,
    credible_interval,
    credible_region_contains,
    posterior_angular_resolution,
    posterior_mean_std,
    pull_histogram,
)


def test_bias_and_pull_on_honest_synthetic_data():
    # dati costruiti apposta per essere onesti (nessun bias, sigma_hat corretta):
    # se questo test non passa il bug e' nella funzione, non nei dati.
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


def test_angular_residual_known_geometry():
    # riga 12 (estensione validate.py, etichetta (d)): geometria nota, nessun dato
    # simulato necessario. Versori costruiti a mano, non da kinematics (validate.py
    # resta generico, nessun import dal progetto).
    identical = np.array([[0.0, 0.0, 1.0]])
    orthogonal_true = np.array([[0.0, 0.0, 1.0]])
    orthogonal_estimate = np.array([[1.0, 0.0, 0.0]])
    antipodal_estimate = np.array([[0.0, 0.0, -1.0]])

    assert angular_residual(identical, identical)[0] == 0.0
    np.testing.assert_allclose(
        angular_residual(orthogonal_true, orthogonal_estimate)[0], np.pi / 2, atol=1e-12
    )
    np.testing.assert_allclose(
        angular_residual(orthogonal_true, antipodal_estimate)[0], np.pi, atol=1e-12
    )


def test_posterior_angular_resolution_recovers_known_sigma():
    # riga 12: log-posterior gaussiano costruito a mano in un angolo theta attorno a
    # reference_hat = z_hat, con sigma nota. Per |theta| piccolo la distanza angolare
    # da z_hat coincide con |theta|: la deviazione pesata recuperata deve riprodurre
    # sigma (verifica di consistenza interna, non un caso di riferimento).
    sigma_true = 0.05
    theta = np.linspace(-0.3, 0.3, 4000)
    omega_hat_grid = np.stack([np.sin(theta), np.zeros_like(theta), np.cos(theta)], axis=-1)
    log_posterior = -0.5 * (theta / sigma_true) ** 2

    reference_hat = np.array([[0.0, 0.0, 1.0]])
    sigma_hat = posterior_angular_resolution(
        log_posterior[None, :], omega_hat_grid, reference_hat
    )
    assert sigma_hat.shape == (1,)
    assert abs(sigma_hat[0] - sigma_true) / sigma_true < 0.02


def test_angular_pull_is_plain_ratio():
    angular_dist = np.array([0.1, 0.2, 0.0])
    sigma_hat = np.array([0.1, 0.1, 0.05])
    pull = angular_pull(angular_dist, sigma_hat)
    np.testing.assert_allclose(pull, [1.0, 2.0, 0.0])


def test_posterior_mean_std_recovers_gaussian_and_reference_two_events():
    # riga 14: gaussiana costruita a mano su un asse lineare, con i valori del
    # caso di riferimento R2 (2.40 +- 0.18 MeV) come media/sigma note: verifica di consistenza interna.
    grid = np.linspace(0.0, 6.0, 4001)
    log_post = -0.5 * ((grid - 2.4) / 0.18) ** 2
    mean, std = posterior_mean_std(log_post[None, :], grid)
    assert mean.shape == (1,) and std.shape == (1,)
    assert abs(mean[0] - 2.4) < 1e-6
    assert abs(std[0] - 0.18) / 0.18 < 0.01


def test_credible_interval_gaussian_and_2d_marginal():
    x = np.linspace(-6.0, 6.0, 2001)
    log_post = np.stack([norm.logpdf(x, 0.3, 1.0), norm.logpdf(x, -1.0, 0.5)])
    levels = np.array([0.0, 0.68, 0.95])
    intervals = credible_interval(log_post, x, levels)
    assert intervals.shape == (2, 3, 2)
    z = norm.ppf(0.5 + levels / 2)
    for k, (mean, sd) in enumerate([(0.3, 1.0), (-1.0, 0.5)]):
        np.testing.assert_allclose(intervals[k, :, 0], mean - z * sd, atol=2e-3)
        np.testing.assert_allclose(intervals[k, :, 1], mean + z * sd, atol=2e-3)
    # griglia 2D appiattita: l'intervallo su un asse e' quello della marginale
    y = np.linspace(-3.0, 3.0, 31)
    x_2d, y_2d = np.meshgrid(x, y, indexing="ij")
    log_post_2d = (norm.logpdf(x_2d, 0.3, 1.0) + norm.logpdf(y_2d, 0.0, 0.7)).ravel()[None, :]
    np.testing.assert_allclose(credible_interval(log_post_2d, x_2d.ravel(), levels)[0],
                               intervals[0], atol=1e-9)


def test_credible_interval_and_region_calibrated_on_simulated_draws():
    # posterior gaussiano esatto (sd nota), verita' estratta dal posterior:
    # la coverage deve essere il livello nominale entro l'errore binomiale.
    rng = np.random.default_rng(20260907)
    n_exp, sd = 4000, 0.5
    x = np.linspace(-5.0, 5.0, 801)
    centers = rng.normal(0.0, 1.0, n_exp)
    truth = rng.normal(centers, sd)
    log_post = norm.logpdf(x[None, :], centers[:, None], sd)
    levels = np.array([0.68, 0.90, 0.95])
    _, coverage = coverage_curve(truth, credible_interval(log_post, x, levels), levels)
    np.testing.assert_allclose(coverage, levels, atol=0.025)

    true_index = np.argmin(np.abs(x[None, :] - truth[:, None]), axis=1)
    hpd = credible_region_contains(log_post, true_index, levels)
    assert hpd.shape == (n_exp, 3)
    np.testing.assert_allclose(hpd.mean(axis=0), levels, atol=0.025)
