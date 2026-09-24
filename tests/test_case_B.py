import numpy as np

from riptide_toy import grids, kinematics, priors


def test_direction_from_theta_phi_unit_norm():
    theta = np.array([0.0, np.pi / 4, np.pi / 2, np.pi])
    phi = np.array([0.0, 1.3, 4.0, 2.5])
    v = kinematics.direction_from_theta_phi(theta, phi)
    assert v.shape == (4, 3)
    np.testing.assert_allclose(np.linalg.norm(v, axis=-1), 1.0, atol=1e-12)


def test_omega_n_parallel_to_z_reduces_to_case_A():
    # riga 8 (CLAUDE.md Sez. 5, test di limite): Omega_n || z. L'angolo 3D fra
    # track_hat e Omega_n deve ridarsi esattamente theta_p del Caso A,
    # qualunque sia l'azimut della traccia (arccos(track . z_hat) = theta_p
    # per costruzione, indipendente da phi).
    omega_n_hat = kinematics.direction_from_theta_phi(np.array([0.0]), np.array([0.0]))

    theta_p = np.array([0.0, 0.3, np.pi / 4, 1.0, np.pi / 2])
    phi_arbitrary = np.array([0.0, 1.0, 3.7, np.pi, 5.5])
    track_hat = kinematics.direction_from_theta_phi(theta_p, phi_arbitrary)

    angles = kinematics.recoil_angle_from_direction(track_hat, omega_n_hat)
    assert angles.shape == (5, 1)
    np.testing.assert_allclose(angles[:, 0], theta_p, atol=1e-10)


def test_sphere_grid_is_pixelization_with_equal_area():
    # fix errata (c): un indice = una direzione, area solida ~costante per
    # pixel. Punti uniformi sulla sfera hanno cos(theta) ~ Uniform(-1, 1)
    # (isotropia, Cap. 20): la media deve annullarsi.
    theta, phi = grids.sphere_grid(2000)
    assert theta.shape == (2000,)
    assert phi.shape == (2000,)
    assert abs(np.cos(theta).mean()) < 0.01
    assert np.all((phi >= 0.0) & (phi < 2 * np.pi))


def test_direction_prior_proper_and_sums_to_one():
    theta, phi = grids.sphere_grid()
    p = priors.direction_prior(theta, phi)
    assert p.shape == theta.shape
    assert np.all(p > 0.0)
    assert abs(p.sum() - 1.0) < 1e-10
