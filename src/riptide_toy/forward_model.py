import numpy as np
from scipy.special import logsumexp

from riptide_toy import kinematics

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

    out = np.empty((n_events, n_hyper), dtype=np.float64)
    for start in range(0, n_hyper, chunk_size):
        end = min(start + chunk_size, n_hyper)
        joint = base[:, None, :] + log_prior_En_grid[None, start:end, :]  # (n_events, chunk, n_En)
        out[:, start:end] = logsumexp(joint, axis=2)
    return out