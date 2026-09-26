from typing import Callable

import numpy as np

from riptide_toy import combine, forward_model, grids, kinematics, posterior_B, priors
from riptide_toy.constants import (N_DIRECTION_CAP, N_MU_FINE, N_SIGMA_FINE, SIGMA_EP, SIGMA_THETA,
                                   WINDOW_DELTA_LOG)


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
        ValueError: se il combinato e' -inf su tutti i candidati. Senza
            questo controllo np.argmax restituirebbe in silenzio il pixel 0
            (polo nord della griglia): e' la causa del "pixel a 53 gradi" in
            docs/report_caso_C_stadio1.md, col vecchio taglio netto a
            theta_p = pi/2. Con la risoluzione angolare di posterior_B il
            combinato e' finito: il controllo resta come difesa.
    """
    combined = shared_direction_log_posterior(D_B, theta_grid, phi_grid, direction_prior)
    check_finite_combined(combined)
    omega_n_hat_grid = kinematics.direction_from_theta_phi(theta_grid, phi_grid)
    map_idx = np.argmax(combined)
    return omega_n_hat_grid[map_idx:map_idx + 1]


def shared_direction_log_posterior(D_B: tuple[np.ndarray, np.ndarray],
                                   theta_grid: np.ndarray, phi_grid: np.ndarray,
                                   direction_prior: np.ndarray) -> np.ndarray:
    """Log-posterior combinato degli N eventi su Omega_n (Caso B), prior
    contato una volta: posterior_B.single_event_posterior somma il prior per
    evento, qui lo si sottrae prima di combine.combine_loglik (vedi
    estimate_shared_direction). Assunzioni: quelle di posterior_B (1 + 2).

    Args:
        D_B: (Ep_hat, track_hat), come in estimate_shared_direction.
        theta_grid: colatitudine dei candidati Omega_n, rad, forma (n_candidates,).
        phi_grid: azimut dei candidati, rad, stessa forma.
        direction_prior: prior proprio sui candidati, forma (n_candidates,).

    Ritorna:
        array (n_candidates,), log-posterior combinato non normalizzato.
    """
    log_posterior_per_event = posterior_B.single_event_posterior(
        D_B, (theta_grid, phi_grid), direction_prior
    )
    raw_loglik = log_posterior_per_event - np.log(direction_prior)[None, :]
    return combine.combine_loglik(raw_loglik, np.log(direction_prior))


def check_finite_combined(combined: np.ndarray) -> None:
    """Guardia R1: un combinato -inf ovunque farebbe restituire a np.argmax
    il pixel 0 in silenzio (docs/report_caso_C_stadio1.md).

    Args:
        combined: log-posterior combinato, forma (n_candidates,).

    Ritorna:
        None; solleva ValueError se nessun candidato e' finito.
    """
    if not np.any(np.isfinite(combined)):
        raise ValueError(
            "posterior combinato -inf su tutti i candidati Omega_n: nessun "
            "pixel e' cinematicamente compatibile con tutti gli eventi "
            "(theta_p <= pi/2); griglia troppo grossolana o tracce senza "
            "risoluzione angolare"
        )


def single_event_posterior(D: tuple[np.ndarray, np.ndarray, np.ndarray],
                            shared_param_grid: tuple[np.ndarray, np.ndarray],
                            prior: np.ndarray) -> np.ndarray:
    """Log-posterior non normalizzato su (mu_E, sigma_E), con Omega_n gia'
    fissato (stadio 1, estimate_shared_direction) ed En^(k) come nuisance
    per-evento a prior gerarchico N(mu_E, sigma_E) (Caso C, CLAUDE.md
    Sez. 3). Stadio 2 della griglia a due stadi (Sez. 4): qui la griglia
    e' solo su (mu_E, sigma_E), mai 4D.

    Assunzioni dichiarate: sorgente unica (1) + direzione ~costante, campo
    lontano (2) + energie simili tra loro (3, E_n^(k) ~ N(mu_E, sigma_E)),
    piu' scattering isotropo in CM (termine di traccia cos(theta_p)/pi) e
    risoluzione angolare SIGMA_THETA sulla traccia, come in posterior_B.

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
    theta_obs = kinematics.recoil_angle_from_direction(track_hat, omega_n_hat)[:, 0]

    en_grid = grids.energy_grid()
    log_prior_En_grid = priors.log_energy_prior_given_hyperparams(en_grid, mu_grid, sigma_grid)

    loglik = forward_model.loglik_marginal_En_theta_hierarchical(
        Ep_hat, theta_obs, en_grid, SIGMA_EP, SIGMA_THETA, log_prior_En_grid
    )
    return loglik + np.log(prior)[None, :]


def direction_cap_grid(center_hat: np.ndarray, radius: float,
                       n_pixel: int) -> tuple[np.ndarray, np.ndarray]:
    """Calotta sferica di raggio `radius` attorno a center_hat, pixelizzata ad
    area ~costante con un reticolo di Fibonacci (come grids.sphere_grid, ma
    ristretto alla calotta): un prior uniforme per pixel resta uniforme in
    angolo solido. Serve al raffinamento locale di Omega_n (riga 14, R6).

    Args:
        center_hat: asse della calotta, versore, forma (1, 3) o (3,).
        radius: semiapertura della calotta, rad, in (0, pi].
        n_pixel: numero di pixel.

    Ritorna:
        (theta, phi) dei pixel in coordinate globali, rad, ciascuno forma (n_pixel,).
    """
    i = np.arange(n_pixel)
    # cos(alpha) uniforme in [cos(radius), 1] = area uguale per pixel
    cos_alpha = 1.0 - (1.0 - np.cos(radius)) * (i + 0.5) / n_pixel
    beta = i * np.pi * (3.0 - np.sqrt(5.0))  # angolo aureo, come sphere_grid
    axis = np.broadcast_to(np.reshape(center_hat, (1, 3)), (n_pixel, 3))
    v = kinematics.direction_around_axis(axis, np.arccos(cos_alpha), beta)
    return kinematics.theta_phi_from_direction(v)


def direction_cap_radius(coarse: np.ndarray, omega_grid: np.ndarray,
                         delta_log: float) -> float:
    """Raggio della calotta di raffinamento: angolo massimo dal MAP dei pixel
    grossolani con log-posterior > max - delta_log, piu' due passi della griglia
    grossolana (distanza fra il MAP e il pixel piu' vicino), limitato a pi.
    Il MAP si esclude per indice e non con angle > 0: arccos(best @ best)
    arrotondato vale ~1.5e-8 rad, non 0, e darebbe una calotta degenere
    (docs/report_caso_C_stadio1.md §8).

    Args:
        coarse: log-posterior combinato sulla griglia grossolana, forma (n_candidates,).
        omega_grid: candidati Omega_n, versori, forma (n_candidates, 3).
        delta_log: soglia sul log-posterior che definisce la regione tenuta.

    Ritorna:
        raggio della calotta, rad, scalare in (0, pi].
    """
    map_idx = np.argmax(coarse)
    angle = np.arccos(np.clip(omega_grid @ omega_grid[map_idx], -1.0, 1.0))
    step = np.min(np.delete(angle, map_idx))
    return float(min(np.max(angle[coarse > coarse.max() - delta_log]) + 2.0 * step, np.pi))


def direction_log_posterior_from_table(table: np.ndarray, track_hat: np.ndarray,
                                       theta_grid: np.ndarray, phi_grid: np.ndarray,
                                       direction_prior: np.ndarray) -> np.ndarray:
    """Come shared_direction_log_posterior, ma dalla tabella del Caso B gia'
    calcolata (forward_model.track_energy_table, prior largo su En come in
    posterior_B): la tabella non dipende dai candidati e si riusa fra griglia
    grossolana e calotta. Assunzioni: 1 + 2.

    Args:
        table: forma (n_events, n_theta_obs), da forward_model.track_energy_table.
        track_hat: direzione 3D della traccia, versori, forma (n_events, 3).
        theta_grid, phi_grid: candidati Omega_n, rad, forma (n_candidates,).
        direction_prior: prior proprio sui candidati, forma (n_candidates,).

    Ritorna:
        array (n_candidates,), log-posterior combinato non normalizzato.
    """
    omega_n_hat = kinematics.direction_from_theta_phi(theta_grid, phi_grid)
    theta_obs = kinematics.recoil_angle_from_direction(track_hat, omega_n_hat)
    loglik = forward_model.interpolate_track_table(table, theta_obs)
    return combine.combine_loglik(loglik, np.log(direction_prior))


def case_B_track_table(Ep_hat: np.ndarray) -> np.ndarray:
    """Tabella del Caso B per lo stadio 1 (prior largo su En, risoluzioni
    SIGMA_EP e SIGMA_THETA, come posterior_B.single_event_posterior).

    Args:
        Ep_hat: energia di rinculo osservata, MeV, forma (n_events,).

    Ritorna:
        array (n_events, N_THETA_OBS), vedi forward_model.track_energy_table.
    """
    en_grid = grids.energy_grid()
    log_prior_En = np.log(priors.energy_prior(en_grid))
    return forward_model.track_energy_table(Ep_hat, en_grid, SIGMA_EP, SIGMA_THETA, log_prior_En)


def refine_shared_direction(D_B: tuple[np.ndarray, np.ndarray],
                            theta_grid: np.ndarray, phi_grid: np.ndarray,
                            direction_prior: np.ndarray,
                            delta_log: float = WINDOW_DELTA_LOG,
                            n_cap: int = N_DIRECTION_CAP
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stadio 1 raffinato: combinato del Caso B sulla griglia grossolana, poi
    ricalcolo su una calotta fine attorno al MAP (riga 14, R6). Il raggio
    della calotta copre i pixel grossolani con log-posterior > max - delta_log
    piu' due passi della griglia grossolana (distanza del MAP dal pixel piu'
    vicino), cosi' la massa fuori dalla calotta e' < ~exp(-delta_log).
    Prior uniforme per pixel sulla calotta (area uguale), cioe' la stessa
    prior uniforme sulla sfera ristretta alla calotta. Assunzioni: 1 + 2.

    Args:
        D_B: (Ep_hat, track_hat), come in estimate_shared_direction.
        theta_grid, phi_grid: griglia grossolana, rad, forma (n_candidates,).
        direction_prior: prior sulla griglia grossolana, forma (n_candidates,).
        delta_log: soglia sul log-posterior che definisce la regione tenuta.
        n_cap: pixel della calotta fine.

    Ritorna:
        (omega_n_hat (1, 3) versore MAP sulla calotta,
         cap_log_post (n_cap,) log-posterior combinato non normalizzato,
         cap_theta, cap_phi (n_cap,) rad).
    """
    Ep_hat, track_hat = D_B
    table = case_B_track_table(Ep_hat)  # una volta: griglia grossolana e calotta
    coarse = direction_log_posterior_from_table(table, track_hat, theta_grid, phi_grid,
                                                direction_prior)
    check_finite_combined(coarse)
    omega_grid = kinematics.direction_from_theta_phi(theta_grid, phi_grid)
    best = omega_grid[np.argmax(coarse)]
    radius = direction_cap_radius(coarse, omega_grid, delta_log)

    cap_theta, cap_phi = direction_cap_grid(best, radius, n_cap)
    cap_prior = priors.direction_prior(cap_theta, cap_phi)
    cap_log_post = direction_log_posterior_from_table(table, track_hat, cap_theta, cap_phi,
                                                      cap_prior)
    check_finite_combined(cap_log_post)
    cap_grid = kinematics.direction_from_theta_phi(cap_theta, cap_phi)
    map_idx = np.argmax(cap_log_post)
    return cap_grid[map_idx:map_idx + 1], cap_log_post, cap_theta, cap_phi


def shared_hyperparameter_log_posterior(D: tuple[np.ndarray, np.ndarray, np.ndarray],
                                        mu_grid: np.ndarray, sigma_grid: np.ndarray,
                                        prior: np.ndarray) -> np.ndarray:
    """Log-posterior combinato degli N eventi su (mu_E, sigma_E), Omega_n
    fissato, prior contato una volta (single_event_posterior lo somma per
    evento: qui lo si sottrae prima di combine.combine_loglik). Conta per un
    prior non uniforme (robustezza al prior, riga 14). Assunzioni: 1 + 2 + 3.

    Args:
        D: (Ep_hat, track_hat, omega_n_hat), come in single_event_posterior.
        mu_grid, sigma_grid: ipotesi su (mu_E, sigma_E), MeV, forma (n_hyper,).
        prior: prior proprio sulla griglia, forma (n_hyper,).

    Ritorna:
        array (n_hyper,), log-posterior combinato non normalizzato.
    """
    per_event = single_event_posterior(D, (mu_grid, sigma_grid), prior)
    return combine.combine_loglik(per_event - np.log(prior)[None, :], np.log(prior))


def hyperparameter_log_posterior_from_base(base: np.ndarray, mu_grid: np.ndarray,
                                           sigma_grid: np.ndarray,
                                           prior: np.ndarray) -> np.ndarray:
    """Come shared_hyperparameter_log_posterior, ma dalla tabella per evento
    su En gia' calcolata (forward_model.hierarchical_base): non dipende dagli
    iperparametri e si riusa fra griglia globale e fine. Assunzioni: 1 + 2 + 3.

    Args:
        base: forma (n_events, n_En) su grids.energy_grid(), da hierarchical_base.
        mu_grid, sigma_grid: ipotesi su (mu_E, sigma_E), MeV, forma (n_hyper,).
        prior: prior proprio sulla griglia, forma (n_hyper,).

    Ritorna:
        array (n_hyper,), log-posterior combinato non normalizzato.
    """
    log_prior_En_grid = priors.log_energy_prior_given_hyperparams(
        grids.energy_grid(), mu_grid, sigma_grid
    )
    loglik = forward_model.marginalize_En_hierarchical(base, log_prior_En_grid)
    return combine.combine_loglik(loglik, np.log(prior))


def hyperparameter_window(log_post: np.ndarray, mu_grid: np.ndarray, sigma_grid: np.ndarray,
                          delta_log: float = WINDOW_DELTA_LOG
                          ) -> tuple[float, float, float, float]:
    """Finestra su (mu_E, sigma_E) che contiene i punti con log-posterior >
    max - delta_log, allargata di un passo della griglia per lato (in log
    per sigma_E, spaziata logaritmicamente).

    Args:
        log_post: log-posterior combinato, forma (n_hyper,).
        mu_grid, sigma_grid: griglia (grids.hyperparameter_grid), MeV, forma (n_hyper,).
        delta_log: soglia sul log-posterior.

    Ritorna:
        (mu_lo, mu_hi, sigma_lo, sigma_hi), MeV, float.
    """
    mu_step = np.min(np.diff(np.unique(mu_grid)))
    log_sigma_step = np.min(np.diff(np.log(np.unique(sigma_grid))))
    keep = log_post > log_post.max() - delta_log
    return (float(mu_grid[keep].min() - mu_step), float(mu_grid[keep].max() + mu_step),
            float(sigma_grid[keep].min() * np.exp(-log_sigma_step)),
            float(sigma_grid[keep].max() * np.exp(log_sigma_step)))


def refine_hyperparameters(D: tuple[np.ndarray, np.ndarray, np.ndarray],
                           mu_grid: np.ndarray, sigma_grid: np.ndarray,
                           prior_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                           delta_log: float = WINDOW_DELTA_LOG,
                           n_mu: int = N_MU_FINE, n_sigma: int = N_SIGMA_FINE
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stadio 2 raffinato: combinato su (mu_E, sigma_E) sulla griglia globale,
    poi ricalcolo su una griglia fine ristretta alla finestra di
    hyperparameter_window (riga 14, R6). Omega_n resta la stima puntuale
    dello stadio 1 (plug-in, non propagata: la coverage dice se basta).
    Assunzioni: 1 + 2 + 3.

    Args:
        D: (Ep_hat, track_hat, omega_n_hat), come in single_event_posterior.
        mu_grid, sigma_grid: griglia globale (grids.hyperparameter_grid), MeV.
        prior_fn: (mu_grid, sigma_grid) -> prior proprio sulla griglia, es.
            priors.hyperparameter_prior; valutata su entrambe le griglie.
        delta_log: soglia della finestra.
        n_mu, n_sigma: punti per asse della griglia fine.

    Ritorna:
        (log_post_fine (n_mu*n_sigma,) log-posterior combinato non normalizzato,
         mu_fine, sigma_fine (n_mu*n_sigma,) MeV, layout di grids.hyperparameter_grid_window).
    """
    Ep_hat, track_hat, omega_n_hat = D
    theta_obs = kinematics.recoil_angle_from_direction(track_hat, omega_n_hat)[:, 0]
    base = forward_model.hierarchical_base(Ep_hat, theta_obs, grids.energy_grid(),
                                           SIGMA_EP, SIGMA_THETA)  # una volta: due griglie
    coarse = hyperparameter_log_posterior_from_base(base, mu_grid, sigma_grid,
                                                    prior_fn(mu_grid, sigma_grid))
    mu_fine, sigma_fine = grids.hyperparameter_grid_window(
        *hyperparameter_window(coarse, mu_grid, sigma_grid, delta_log), n_mu, n_sigma
    )
    log_post_fine = hyperparameter_log_posterior_from_base(
        base, mu_fine, sigma_fine, prior_fn(mu_fine, sigma_fine)
    )
    return log_post_fine, mu_fine, sigma_fine
