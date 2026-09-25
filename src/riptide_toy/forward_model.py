import numpy as np
from scipy.special import logsumexp

from riptide_toy import kinematics
from riptide_toy.constants import N_THETA_OBS, N_THETA_TRACK

def measure(Ep_true: np.ndarray, theta_p_true: np.ndarray,
            sigma_Ep: float, sigma_theta: float,
            rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return: (Ep_hat, theta_p_hat), same dimension of the inputs,
    with added gaussian noise - they're the observables D."""
    Ep_hat = rng.normal(Ep_true, sigma_Ep, Ep_true.shape)
    theta_p_hat = rng.normal(theta_p_true, sigma_theta, theta_p_true.shape)
    return (Ep_hat, theta_p_hat)


def loglik(D: tuple[np.ndarray, np.ndarray], En_grid: np.ndarray,
           theta_p_grid: np.ndarray, sigma_Ep: float, sigma_theta: float,
           log_prior_theta: np.ndarray, chunk_size: int = 500) -> np.ndarray:
    """Log-verosimiglianza per ogni evento su ogni En, con theta_p
    gia' marginalizzato (pesato da log_prior_theta), a chunk di eventi
    per stare in RAM (evita la griglia piena (n_eventi, n_En, n_theta)).

    Nota: firma diversa da quella della guida (Sez. 3), che non
    marginalizza theta_p qui dentro. Deviazione mantenuta per
    prestazioni; posterior_A si adatta a questa firma. Vedi
    docs/roadmap.md, sezione Deviazioni.

    Args:
        D: (Ep_hat, theta_p_hat), osservabili, ciascuno forma (n_events,).
        En_grid: griglia di ipotesi su En, MeV, forma (n_En,).
        theta_p_grid: griglia su cui marginalizzare theta_p, rad, forma (n_theta,).
        sigma_Ep: risoluzione del detector su Ep, MeV.
        sigma_theta: risoluzione del detector su theta_p, rad.
        log_prior_theta: log-prior su theta_p_grid, forma (n_theta,).
        chunk_size: numero di eventi processati per lotto.

    Ritorna:
        array (n_events, n_En), log-verosimiglianza (float32).
    """
    Ep_hat, theta_p_hat = D
    Ep_hat = Ep_hat.astype(np.float32)
    theta_p_hat = theta_p_hat.astype(np.float32)
    n_events, n_En = Ep_hat.shape[0], En_grid.shape[0]

    theta = theta_p_grid[None, :].astype(np.float32)
    En = En_grid[:, None].astype(np.float32)
    Ep_pred = (En * np.cos(theta) ** 2)[None]

    out = np.empty((n_events, n_En), dtype=np.float32)
    for start in range(0, n_events, chunk_size):
        end = min(start + chunk_size, n_events)
        first = -0.5 * ((Ep_hat[start:end, None, None] - Ep_pred) / sigma_Ep) ** 2
        second = -0.5 * ((theta_p_hat[start:end, None, None] - theta[None]) / sigma_theta) ** 2
        joint = first + second + log_prior_theta[None, None, :]   # (chunk, n_En, n_theta)
        out[start:end] = logsumexp(joint, axis=2)                  # marginalizza theta, un lotto alla volta
    return out


def loglik_marginal_En(Ep_hat: np.ndarray, theta_p: np.ndarray, En_grid: np.ndarray,
                        sigma_Ep: float, log_prior_En: np.ndarray) -> np.ndarray:
    """Log-verosimiglianza marginalizzata su En (nuisance del Caso B), dato
    theta_p gia' noto per ogni evento/candidato (calcolato geometricamente
    da kinematics.recoil_angle_from_direction, non marginalizzato qui).

    Fisica: Ep = En*cos^2(theta_p), theta_lab <= pi/2 (CLAUDE.md Sez. 3,
    scattering elastico a masse uguali: il rinculo e' sempre in avanti). I
    candidati con theta_p > pi/2 sono cinematicamente vietati e ricevono
    log-verosimiglianza -inf.

    Args:
        Ep_hat: energia di rinculo osservata, MeV, forma (n_events,).
        theta_p: angolo di rinculo per candidato, rad, forma (n_events, n_candidates).
        En_grid: griglia su cui marginalizzare En, MeV, forma (n_En,).
        sigma_Ep: risoluzione del detector su Ep, MeV.
        log_prior_En: log-prior su En_grid (ampio ma proprio), forma (n_En,).

    Ritorna:
        array (n_events, n_candidates), log-verosimiglianza marginalizzata su En.
    """
    Ep_pred = En_grid[None, None, :] * np.cos(theta_p)[:, :, None] ** 2
    joint = -0.5 * ((Ep_hat[:, None, None] - Ep_pred) / sigma_Ep) ** 2
    joint = joint + log_prior_En[None, None, :]
    joint = np.where(theta_p[:, :, None] <= np.pi / 2, joint, -np.inf)
    return logsumexp(joint, axis=2)


def loglik_marginal_En_hierarchical(Ep_hat: np.ndarray, theta_p: np.ndarray,
                                     En_grid: np.ndarray, sigma_Ep: float,
                                     log_prior_En_grid: np.ndarray,
                                     chunk_size: int = 200) -> np.ndarray:
    """Log-verosimiglianza per evento marginalizzata su En, con un prior su
    En diverso per ogni candidato iperparametro (Caso C, stadio 2 della
    griglia a due stadi: Omega_n gia' fissato dal Caso B, qui si
    marginalizza solo su En dato ciascun candidato (mu_E, sigma_E) --
    CLAUDE.md Sez. 4, "mai griglia 4D bruta": la griglia e' solo su
    (mu_E, sigma_E), non su Omega_n.

    A differenza di loglik_marginal_En (Caso B, un solo log_prior_En
    condiviso da tutti i candidati direzione), qui ogni candidato
    iperparametro ha la propria pi(En | mu_E, sigma_E)
    (priors.energy_prior_given_hyperparams): il prior varia lungo l'asse
    dei candidati, da cui la funzione separata invece di riusare quella
    del Caso B.

    Args:
        Ep_hat: energia di rinculo osservata, MeV, forma (n_events,).
        theta_p: angolo di rinculo, rad, forma (n_events,) -- gia' fissato
            dalla stima di Omega_n dello stadio 1 (non e' una griglia di
            candidati come nel Caso B).
        En_grid: griglia su cui marginalizzare En, MeV, forma (n_En,).
        sigma_Ep: risoluzione del detector su Ep, MeV.
        log_prior_En_grid: log pi(En | candidato), forma (n_hyper, n_En)
            (log di priors.energy_prior_given_hyperparams).
        chunk_size: numero di candidati iperparametro processati per lotto
            (evita la griglia piena (n_events, n_hyper, n_En) in RAM).

    Ritorna:
        array (n_events, n_hyper), log-verosimiglianza marginalizzata su En.
    """
    n_events = Ep_hat.shape[0]
    n_hyper = log_prior_En_grid.shape[0]

    Ep_pred = En_grid[None, :] * np.cos(theta_p)[:, None] ** 2  # (n_events, n_En)
    base = -0.5 * ((Ep_hat[:, None] - Ep_pred) / sigma_Ep) ** 2
    base = np.where((theta_p <= np.pi / 2)[:, None], base, -np.inf)

    return marginalize_En_hierarchical(base, log_prior_En_grid, chunk_size)


def marginalize_En_hierarchical(base: np.ndarray, log_prior_En_grid: np.ndarray,
                                chunk_size: int = 200) -> np.ndarray:
    """log sum_En exp(base + log pi(En | candidato)) per ogni evento e
    candidato iperparametro (Caso C), a lotti di candidati.

    Args:
        base: log-verosimiglianza per evento su En_grid, forma (n_events, n_En).
        log_prior_En_grid: log pi(En | candidato), forma (n_hyper, n_En).
        chunk_size: candidati iperparametro per lotto (evita la griglia
            piena (n_events, n_hyper, n_En) in RAM).

    Ritorna:
        array (n_events, n_hyper), log-verosimiglianza marginalizzata su En.
    """
    n_events, n_hyper = base.shape[0], log_prior_En_grid.shape[0]
    out = np.empty((n_events, n_hyper), dtype=np.float64)
    for start in range(0, n_hyper, chunk_size):
        end = min(start + chunk_size, n_hyper)
        joint = base[:, None, :] + log_prior_En_grid[None, start:end, :]  # (n_events, chunk, n_En)
        out[:, start:end] = logsumexp(joint, axis=2)
    return out


def log_track_density(theta_p: np.ndarray) -> np.ndarray:
    """log p(traccia | Omega_n) per scattering isotropo in CM: cos(theta_p)/pi
    per steradiante sull'emisfero anteriore, 0 su quello posteriore.

    Args:
        theta_p: angolo fra traccia e Omega_n, rad, forma qualsiasi.

    Ritorna:
        array della stessa forma, log-densita' per steradiante (-inf per
        theta_p >= pi/2).
    """
    # (a) cos^2(theta_p) ~ U(0, 1) <=> densita' cos(theta_p)/pi su dOmega
    # (kinematics.sample_recoil_events); si annulla con continuita' a pi/2.
    with np.errstate(divide="ignore"):
        return np.log(np.clip(np.cos(theta_p), 0.0, None) / np.pi)


def log_track_kernel(theta_obs: np.ndarray, theta_grid: np.ndarray,
                     sigma_theta: float) -> np.ndarray:
    """Peso di quadratura in log per marginalizzare theta_p vero data la
    traccia osservata: p_track(theta) * [N(theta_obs; theta, s) +
    N(theta_obs; -theta, s)] * peso trapezoidale.

    Approssimazione (b): l'errore della traccia e' gaussiano nel piano
    tangente (kinematics.smear_direction) e la densita' della traccia varia
    solo lungo il meridiano per Omega_n, quindi la convoluzione 2D si riduce
    a una 1D sulla coordinata con segno s del meridiano (s < 0 = oltre il
    polo: da qui il termine specchiato in -theta). Curvatura della sfera
    trascurata, errore O(sigma_theta^2).

    Args:
        theta_obs: angolo osservato fra traccia e Omega_n, rad, forma (n,), >= 0.
        theta_grid: griglia uniforme su [0, pi/2] per theta vero, rad, forma (m,).
        sigma_theta: risoluzione angolare per componente, rad.

    Ritorna:
        array (n, m), log-pesi; exp(...).sum(axis=1) approssima la densita'
        per steradiante della traccia osservata.
    """
    step = theta_grid[1] - theta_grid[0]
    log_weight = np.full(theta_grid.shape, np.log(step))
    log_weight[[0, -1]] -= np.log(2.0)
    z_plus = (theta_obs[:, None] - theta_grid[None, :]) / sigma_theta
    z_minus = (theta_obs[:, None] + theta_grid[None, :]) / sigma_theta
    log_gauss = (np.logaddexp(-0.5 * z_plus ** 2, -0.5 * z_minus ** 2)
                 - np.log(sigma_theta * np.sqrt(2.0 * np.pi)))
    return log_gauss + (log_track_density(theta_grid) + log_weight)[None, :]


def loglik_marginal_En_theta(Ep_hat: np.ndarray, theta_obs: np.ndarray,
                             En_grid: np.ndarray, sigma_Ep: float, sigma_theta: float,
                             log_prior_En: np.ndarray, n_theta: int = N_THETA_TRACK,
                             n_theta_obs: int = N_THETA_OBS,
                             chunk_size: int = 20) -> np.ndarray:
    """Log-verosimiglianza di (Ep_hat, traccia) per evento e candidato Omega_n,
    marginalizzata su En e sul theta_p vero (Caso B), con termine di traccia
    cos(theta_p)/pi e risoluzione angolare sigma_theta. Sostituisce il
    taglio netto -inf di loglik_marginal_En a theta_p > pi/2.

    Il termine in En non dipende da Omega_n: si calcola
    h_k(theta_obs) = log int ds p_track N(theta_obs; s) exp g_k(s) una volta
    su una griglia uniforme theta_obs in [0, pi] e si interpola linearmente
    sui candidati (costo indipendente dal numero di candidati).

    Args:
        Ep_hat: energia di rinculo osservata, MeV, forma (n_events,).
        theta_obs: angolo fra traccia osservata e candidato, rad, forma
            (n_events, n_candidates) (kinematics.recoil_angle_from_direction).
        En_grid: griglia su cui marginalizzare En, MeV, forma (n_En,).
        sigma_Ep: risoluzione su Ep, MeV.
        sigma_theta: risoluzione angolare della traccia, rad.
        log_prior_En: log-prior su En_grid, forma (n_En,).
        n_theta: punti della griglia di theta vero su [0, pi/2].
        n_theta_obs: punti della griglia di interpolazione su [0, pi].
        chunk_size: eventi per lotto.

    Ritorna:
        array (n_events, n_candidates), log-verosimiglianza (a meno di una
        costante comune), finita per ogni candidato.
    """
    theta_grid = np.linspace(0.0, np.pi / 2, n_theta)
    obs_grid = np.linspace(0.0, np.pi, n_theta_obs)
    kernel = log_track_kernel(obs_grid, theta_grid, sigma_theta)  # (n_obs, n_theta)

    n_events = Ep_hat.shape[0]
    out = np.empty(theta_obs.shape, dtype=np.float64)
    for start in range(0, n_events, chunk_size):
        end = min(start + chunk_size, n_events)
        theta_true = np.broadcast_to(theta_grid, (end - start, n_theta))
        energy = loglik_marginal_En(Ep_hat[start:end], theta_true, En_grid,
                                    sigma_Ep, log_prior_En)          # (chunk, n_theta)
        h = logsumexp(energy[:, None, :] + kernel[None], axis=2)     # (chunk, n_obs)
        # indici di interpolazione per lotto: temporanei (chunk, n_candidates)
        # invece di (n_events, n_candidates), picco di memoria ~ output
        position = np.clip(theta_obs[start:end] / obs_grid[1], 0.0, n_theta_obs - 1 - 1e-9)
        index = position.astype(np.intp)
        lo = np.take_along_axis(h, index, axis=1)
        hi = np.take_along_axis(h, index + 1, axis=1)
        out[start:end] = lo + (position - index) * (hi - lo)
    return out


def loglik_marginal_En_theta_hierarchical(Ep_hat: np.ndarray, theta_obs: np.ndarray,
                                          En_grid: np.ndarray, sigma_Ep: float,
                                          sigma_theta: float,
                                          log_prior_En_grid: np.ndarray,
                                          n_theta: int = N_THETA_TRACK,
                                          chunk_size: int = 200,
                                          event_chunk_size: int = 20) -> np.ndarray:
    """Come loglik_marginal_En_hierarchical (Caso C, stadio 2, Omega_n
    fissato), ma con termine di traccia e risoluzione angolare: si
    marginalizza prima il theta_p vero (non dipende dagli iperparametri),
    poi En con il prior di ciascun candidato (mu_E, sigma_E).

    Args:
        Ep_hat: energia di rinculo osservata, MeV, forma (n_events,).
        theta_obs: angolo fra traccia osservata e Omega_n fissato, rad,
            forma (n_events,).
        En_grid: griglia su cui marginalizzare En, MeV, forma (n_En,).
        sigma_Ep: risoluzione su Ep, MeV.
        sigma_theta: risoluzione angolare della traccia, rad.
        log_prior_En_grid: log pi(En | candidato), forma (n_hyper, n_En).
        n_theta: punti della griglia di theta vero su [0, pi/2].
        chunk_size: candidati iperparametro per lotto.
        event_chunk_size: eventi per lotto nella marginalizzazione di theta
            (array (lotto, n_theta, n_En) in RAM).

    Ritorna:
        array (n_events, n_hyper), log-verosimiglianza marginalizzata su
        theta_p vero ed En (stessa costante comune di loglik_marginal_En_theta).
    """
    theta_grid = np.linspace(0.0, np.pi / 2, n_theta)
    Ep_pred = En_grid[None, :] * np.cos(theta_grid)[:, None] ** 2    # (n_theta, n_En)
    n_events = Ep_hat.shape[0]
    base = np.empty((n_events, En_grid.shape[0]), dtype=np.float64)
    for start in range(0, n_events, event_chunk_size):
        end = min(start + event_chunk_size, n_events)
        kernel = log_track_kernel(theta_obs[start:end], theta_grid, sigma_theta)
        energy = -0.5 * ((Ep_hat[start:end, None, None] - Ep_pred[None]) / sigma_Ep) ** 2
        base[start:end] = logsumexp(kernel[:, :, None] + energy, axis=1)
    return marginalize_En_hierarchical(base, log_prior_En_grid, chunk_size)