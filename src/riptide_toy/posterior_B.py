"""Caso B: direzione Omega_n condivisa (campo lontano), E_n diversa per evento.

E_n e' una nuisance per evento, marginalizzata con il prior largo; il parametro
d'interesse e' la direzione sulla sfera. Assunzioni: sorgente unica + direzione ~costante.
"""
import numpy as np

from riptide_toy import forward_model, grids, kinematics, priors
from riptide_toy.constants import SIGMA_EP, SIGMA_THETA


def single_event_posterior(D: tuple[np.ndarray, np.ndarray],
                            shared_param_grid: tuple[np.ndarray, np.ndarray],
                            prior: np.ndarray) -> np.ndarray:
    """Log-posterior non normalizzato su Omega_n, con En come nuisance
    per-evento a prior largo (Caso B).

    Assunzioni dichiarate: sorgente unica (1) + direzione ~costante,
    campo lontano (2) + scattering isotropo in CM (toy 0.5-6 MeV), che da'
    il termine di traccia cos(theta_p)/pi. La traccia ha risoluzione
    angolare SIGMA_THETA: theta_p vero e' marginalizzato attorno
    all'angolo osservato fra track_hat e ogni candidato
    (forward_model.loglik_marginal_En_theta, kernel vMF esatto sulla sfera), senza
    taglio netto a pi/2: il posterior e' finito su tutta la sfera.

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
    theta_obs = kinematics.recoil_angle_from_direction(track_hat, omega_n_hat)

    en_grid = grids.energy_grid()
    log_prior_En = np.log(priors.energy_prior(en_grid))

    loglik = forward_model.loglik_marginal_En_theta(
        Ep_hat, theta_obs, en_grid, SIGMA_EP, SIGMA_THETA, log_prior_En
    )
    loglik += np.log(prior)[None, :]  # in place: nessuna copia (n_events, n_candidates)
    return loglik
