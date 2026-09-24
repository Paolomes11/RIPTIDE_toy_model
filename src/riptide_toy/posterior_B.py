import numpy as np

from riptide_toy import forward_model, grids, kinematics, priors
from riptide_toy.constants import SIGMA_EP


def single_event_posterior(D: tuple[np.ndarray, np.ndarray],
                            shared_param_grid: tuple[np.ndarray, np.ndarray],
                            prior: np.ndarray) -> np.ndarray:
    """Log-posterior non normalizzato su Omega_n, con En come nuisance
    per-evento a prior largo (Caso B, CLAUDE.md Sez. 3).

    Assunzioni dichiarate: sorgente unica (1) + direzione ~costante,
    campo lontano (2). Theta_p e' calcolato geometricamente da track_hat
    e ogni candidato Omega_n (kinematics.recoil_angle_from_direction),
    non e' una nuisance separata da marginalizzare: e' invariante
    all'azimut della traccia attorno al candidato per costruzione.

    Args:
        D: (Ep_hat, track_hat), osservabili: energia di rinculo, MeV,
            forma (n_events,), e direzione 3D della traccia di rinculo,
            versori, forma (n_events, 3).
        shared_param_grid: (theta_grid, phi_grid), ipotesi su Omega_n,
            rad, ciascuno forma (n_candidates,) (grids.sphere_grid).
        prior: prior proprio su shared_param_grid, forma (n_candidates,)
            (priors.direction_prior).

    Ritorna:
        array (n_events, n_candidates), log-posterior non normalizzato
        per evento, con En gia' marginalizzato.
    """
    Ep_hat, track_hat = D
    theta_grid, phi_grid = shared_param_grid
    omega_n_hat = kinematics.direction_from_theta_phi(theta_grid, phi_grid)
    theta_p = kinematics.recoil_angle_from_direction(track_hat, omega_n_hat)

    en_grid = grids.energy_grid()
    log_prior_En = np.log(priors.energy_prior(en_grid))

    loglik = forward_model.loglik_marginal_En(Ep_hat, theta_p, en_grid, SIGMA_EP, log_prior_En)
    return loglik + np.log(prior)[None, :]
