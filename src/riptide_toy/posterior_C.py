import numpy as np

from riptide_toy import combine, forward_model, grids, kinematics, posterior_B, priors
from riptide_toy.constants import SIGMA_EP


def estimate_shared_direction(D_B: tuple[np.ndarray, np.ndarray],
                               theta_grid: np.ndarray, phi_grid: np.ndarray,
                               direction_prior: np.ndarray) -> np.ndarray:
    """Stadio 1 del Caso C: stima puntuale (MAP) di Omega_n riusando il
    Caso B, per fissarla prima della griglia 2D su (mu_E, sigma_E) --
    evita la maledizione della dimensionalita' di una griglia 4D bruta
    (CLAUDE.md Sez. 4: "griglia a due stadi (Omega_n da B, poi 2D su
    mu_E,sigma_E)").

    Nota: posterior_B.single_event_posterior somma gia' il prior sulla
    direzione una volta per evento (CLAUDE.md Sez. 3). Per combinare gli N
    eventi con combine.combine_loglik (che il prior lo aggiunge una sola
    volta sull'intero campione, come nelle righe 11/12) va prima sottratto,
    altrimenti verrebbe contato N volte invece di 1.

    Args:
        D_B: (Ep_hat, track_hat), osservabili di tutti gli N eventi, stessa
            forma richiesta da posterior_B.single_event_posterior.
        theta_grid: colatitudine dei candidati Omega_n, rad, forma (n_candidates,).
        phi_grid: azimut dei candidati Omega_n, rad, stessa forma di theta_grid.
        direction_prior: prior proprio su Omega_n, forma (n_candidates,)
            (priors.direction_prior).

    Ritorna:
        array (1, 3), versore MAP di Omega_n sul combinato degli N eventi.

    Solleva:
        ValueError: se il combinato e' -inf su tutti i candidati (nessun
            pixel tiene tutti gli eventi con theta_p <= pi/2). Senza questo
            controllo np.argmax restituirebbe in silenzio il pixel 0 (polo
            nord della griglia): e' la causa del "pixel a 53 gradi" in
            docs/report_caso_C_stadio1.md.
    """
    log_posterior_per_event = posterior_B.single_event_posterior(
        D_B, (theta_grid, phi_grid), direction_prior
    )
    raw_loglik = log_posterior_per_event - np.log(direction_prior)[None, :]
    combined = combine.combine_loglik(raw_loglik, np.log(direction_prior))

    if not np.any(np.isfinite(combined)):
        raise ValueError(
            "posterior combinato -inf su tutti i candidati Omega_n: nessun "
            "pixel e' cinematicamente compatibile con tutti gli eventi "
            "(theta_p <= pi/2); griglia troppo grossolana o tracce senza "
            "risoluzione angolare"
        )

    omega_n_hat_grid = kinematics.direction_from_theta_phi(theta_grid, phi_grid)
    map_idx = np.argmax(combined)
    return omega_n_hat_grid[map_idx:map_idx + 1]


def single_event_posterior(D: tuple[np.ndarray, np.ndarray, np.ndarray],
                            shared_param_grid: tuple[np.ndarray, np.ndarray],
                            prior: np.ndarray) -> np.ndarray:
    """Log-posterior non normalizzato su (mu_E, sigma_E), con Omega_n gia'
    fissato (stadio 1, estimate_shared_direction) ed En^(k) come nuisance
    per-evento a prior gerarchico N(mu_E, sigma_E) (Caso C, CLAUDE.md
    Sez. 3). Stadio 2 della griglia a due stadi (Sez. 4): qui la griglia
    e' solo su (mu_E, sigma_E), mai 4D.

    Assunzioni dichiarate: sorgente unica (1) + direzione ~costante, campo
    lontano (2) + energie simili tra loro (3, E_n^(k) ~ N(mu_E, sigma_E)).

    Nota: a differenza del template generico (D, shared_param_grid, prior)
    di posterior_A/B, qui D include anche omega_n_hat -- il risultato
    dello stadio 1, gia' fissato, non una nuisance o uno shared_param di
    questa funzione. Vedi docs/roadmap.md, sezione Deviazioni.

    Args:
        D: (Ep_hat, track_hat, omega_n_hat), osservabili piu' la stima
            puntuale di Omega_n dello stadio 1: energia di rinculo, MeV,
            forma (n_events,); direzione 3D della traccia, versori, forma
            (n_events, 3); Omega_n fissato, versore, forma (1, 3).
        shared_param_grid: (mu_grid, sigma_grid), ipotesi su (mu_E, sigma_E),
            MeV, ciascuno forma (n_hyper,) (grids.hyperparameter_grid).
        prior: prior proprio su shared_param_grid, forma (n_hyper,)
            (priors.hyperparameter_prior).

    Ritorna:
        array (n_events, n_hyper), log-posterior non normalizzato per
        evento, con En gia' marginalizzato dato ciascun candidato (mu_E, sigma_E).
    """
    Ep_hat, track_hat, omega_n_hat = D
    mu_grid, sigma_grid = shared_param_grid
    theta_p = kinematics.recoil_angle_from_direction(track_hat, omega_n_hat)[:, 0]

    en_grid = grids.energy_grid()
    log_prior_En_grid = np.log(priors.energy_prior_given_hyperparams(en_grid, mu_grid, sigma_grid))

    loglik = forward_model.loglik_marginal_En_hierarchical(
        Ep_hat, theta_p, en_grid, SIGMA_EP, log_prior_En_grid
    )
    return loglik + np.log(prior)[None, :]
