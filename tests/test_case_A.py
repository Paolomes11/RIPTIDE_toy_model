import numpy as np

from riptide_toy import grids


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
