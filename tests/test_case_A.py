import numpy as np
from scipy.integrate import trapezoid

from riptide_toy import forward_model, grids, kinematics, priors
from riptide_toy.constants import SEED


def test_energy_grid_monotone_and_immutable():
    g = grids.energy_grid()
    assert g.shape == (500,)
    assert np.all(np.diff(g) > 0)
    assert not g.flags.writeable


def test_energy_grid_cache_stable_same_args():
    assert grids.energy_grid() is grids.energy_grid()


def test_sphere_grid_different_args_do_not_collide():
    # errata (a): maxsize=1 svuoterebbe la cache tra chiamate con n diversi
    theta_a, phi_a = grids.sphere_grid(500)
    theta_b, phi_b = grids.sphere_grid()
    assert theta_a.shape == (500,)
    assert theta_b.shape == (3000,)
    assert not theta_a.flags.writeable and not phi_a.flags.writeable


def test_theta_p_grid_domain():
    tp = grids.theta_p_grid()
    assert tp[0] == 0.0
    assert abs(tp[-1] - np.pi / 2) < 1e-12
    assert not tp.flags.writeable


def test_proton_energy_known_angles():
    # Ep = En*cos^2(theta_p): 0 -> Ep=En, pi/4 -> Ep=En/2, pi/2 -> Ep=0
    En = np.array([1.0, 2.0, 4.0])
    theta = np.array([0.0, np.pi / 4, np.pi / 2])
    Ep = kinematics.proton_energy(En, theta)
    np.testing.assert_allclose(Ep, [1.0, 1.0, 0.0], atol=1e-10)


def test_proton_energy_montecarlo_mean_var():
    # theta_p ~ Uniform(0, pi/2) campionato direttamente (non tramite
    # sample_cm_angle: la sua formula e' un TODO aperto, vedi kinematics.py).
    # E[cos^2(theta)] = 1/2, Var[cos^2(theta)] = 1/8 per theta~U(0, pi/2)
    # (calcolo elementare, non un dato di letteratura).
    rng = np.random.default_rng(SEED)
    n = 2_000_000
    En = 1.0
    theta = rng.uniform(0.0, np.pi / 2, n)
    Ep = kinematics.proton_energy(En, theta)
    assert abs(Ep.mean() - 0.5) < 5e-3
    assert abs(Ep.var() - 0.125) < 5e-3


def test_recoil_angle_from_direction_shape_and_values():
    track_hat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    omega_n_hat = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
    angles = kinematics.recoil_angle_from_direction(track_hat, omega_n_hat)
    assert angles.shape == (2, 3)
    np.testing.assert_allclose(
        angles[0], [0.0, np.pi / 2, np.pi], atol=1e-10
    )


def test_energy_prior_is_proper_and_integrates_to_one():
    # fix errata (b): scipy.integrate.trapezoid al posto di np.trapz deprecato
    en_grid = grids.energy_grid()
    p = priors.energy_prior(en_grid)
    assert p.shape == en_grid.shape
    assert np.all(p > 0.0)
    assert abs(trapezoid(p, en_grid) - 1.0) < 1e-10


def test_loglik_roundtrip_argmax_near_truth():
    rng = np.random.default_rng(SEED)
    En_true, theta_true = 2.5, 0.4
    sigma_Ep, sigma_theta = 0.10, 0.08

    Ep_true = np.array([En_true * np.cos(theta_true) ** 2])
    theta_p_true = np.array([theta_true])
    D = forward_model.measure(Ep_true, theta_p_true, sigma_Ep, sigma_theta, rng)

    en_grid = grids.energy_grid()
    tp_grid = grids.theta_p_grid()
    log_prior_theta = np.full(tp_grid.shape, -np.log(tp_grid.shape[0]))

    ll = forward_model.loglik(D, en_grid, tp_grid, sigma_Ep, sigma_theta, log_prior_theta)
    assert ll.shape == (1, en_grid.shape[0])
    argmax_En = en_grid[np.argmax(ll[0])]
    assert abs(argmax_En - En_true) < 3 * sigma_Ep
