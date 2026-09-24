import numpy as np

from riptide_toy import grids, posterior_A, priors


def test_example_38_1():
    # oracolo Es. 38.1: Ep_hat=1.40 MeV, theta_p_hat=0.50 rad -> En = 1.82 +/- 0.21 MeV
    en_grid = grids.energy_grid()
    D = (np.array([1.40]), np.array([0.50]))
    prior = priors.energy_prior(en_grid)

    logpost = posterior_A.single_event_posterior(D, en_grid, prior)
    assert logpost.shape == (1, en_grid.shape[0])

    argmax_En = en_grid[np.argmax(logpost[0])]
    assert abs(argmax_En - 1.82) < 0.03  # ~3 passi di griglia

    # sigma stimata come deviazione standard pesata sulla griglia (CLAUDE.md
    # Sez. 5: tol. 0.01 su sigma; risoluzione_caso_A non e' definita in guida,
    # decisione presa con l'utente: vedi docs/roadmap.md, sezione Deviazioni).
    w = np.exp(logpost[0] - logpost[0].max())
    mean = np.sum(w * en_grid) / np.sum(w)
    sigma = np.sqrt(np.sum(w * (en_grid - mean) ** 2) / np.sum(w))
    assert abs(sigma - 0.21) < 0.01
