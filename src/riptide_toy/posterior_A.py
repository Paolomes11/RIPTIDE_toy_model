import numpy as np

from riptide_toy import forward_model, grids
from riptide_toy.constants import SIGMA_EP, SIGMA_THETA


def single_event_posterior(D: tuple[np.ndarray, np.ndarray], shared_param_grid: np.ndarray,
                            prior: np.ndarray) -> np.ndarray:
    """Log-posterior non normalizzato su En, per uno o piu' eventi, con
    theta_p marginalizzato come nuisance (Caso A).

    Assunzioni dichiarate: sorgente unica (1) + monoenergetica (En condiviso
    fra gli eventi, caso limite della 3); z non modellata (vedi roadmap,
    Deviazioni).

    Nota: diverge dalla guida (Sez. 3), che assume theta_p_hat gia'
    fissato dentro loglik. Qui forward_model.loglik marginalizza
    theta_p internamente su un prior piatto proprio su [0, pi/2]
    (CLAUDE.md Sez. 1: conflitto guida/codice segnalato, non risolto
    in silenzio). Vedi docs/roadmap.md, sezione Deviazioni.

    Args:
        D: (Ep_hat, theta_p_hat), osservabili, ciascuno forma (n_events,).
        shared_param_grid: griglia di ipotesi su En, MeV, forma (n_En,).
        prior: prior proprio su shared_param_grid, forma (n_En,)
            (es. priors.energy_prior(shared_param_grid)).

    Ritorna:
        array (n_events, n_En), log-posterior non normalizzato per evento.
    """
    theta_p_grid = grids.theta_p_grid()
    log_prior_theta = np.full(theta_p_grid.shape, -np.log(theta_p_grid.shape[0]))

    loglik = forward_model.loglik(D, shared_param_grid, theta_p_grid, SIGMA_EP, SIGMA_THETA, log_prior_theta)
    return loglik + np.log(prior)[None, :]
