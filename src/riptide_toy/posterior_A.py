"""Caso A: energia E_n condivisa da tutti gli eventi, direzione nota (asse z).

Posterior di singolo evento su E_n con theta_p marginalizzato; gli N eventi si
combinano poi con combine.combine_loglik. Assunzioni: sorgente unica + monoenergetica.
"""
import numpy as np

from riptide_toy import forward_model, grids, priors
from riptide_toy.constants import SIGMA_EP, SIGMA_THETA


def single_event_posterior(D: tuple[np.ndarray, np.ndarray], shared_param_grid: np.ndarray,
                            prior: np.ndarray, log_prior_theta: np.ndarray | None = None) -> np.ndarray:
    """Log-posterior non normalizzato su En, per uno o piu' eventi, con
    theta_p marginalizzato come nuisance (Caso A).

    Assunzioni dichiarate: sorgente unica (1) + monoenergetica (En condiviso
    fra gli eventi, caso limite della 3); z non modellata (vedi roadmap,
    Scelte di implementazione).

    Scelta di implementazione: theta_p non e' preso come noto (la sua
    misura theta_p_hat ha errore SIGMA_THETA); forward_model.loglik lo
    marginalizza internamente. Il prior di default su theta_p e' quello
    fisico per scattering isotropo in CM, sin(2 theta_p)
    (priors.log_recoil_angle_prior): un prior piatto su [0, pi/2] biasa E_n
    combinato quando gli eventi seguono la cinematica fisica.
    Vedi docs/roadmap.md, sezione Scelte di implementazione.

    Args:
        D: (Ep_hat, theta_p_hat), osservabili, ciascuno forma (n_events,).
        shared_param_grid: griglia di ipotesi su En, MeV, forma (n_En,).
        prior: prior proprio su shared_param_grid, forma (n_En,)
            (es. priors.energy_prior(shared_param_grid)).
        log_prior_theta: log-pesi del prior su grids.theta_p_grid(), forma
            (n_theta,), exp somma a 1; None = prior fisico sin(2 theta_p).

    Ritorna:
        array (n_events, n_En), log-posterior non normalizzato per evento.
    """
    theta_p_grid = grids.theta_p_grid()
    if log_prior_theta is None:
        log_prior_theta = priors.log_recoil_angle_prior(theta_p_grid)

    loglik = forward_model.loglik(D, shared_param_grid, theta_p_grid, SIGMA_EP, SIGMA_THETA, log_prior_theta)
    return loglik + np.log(prior)[None, :]
