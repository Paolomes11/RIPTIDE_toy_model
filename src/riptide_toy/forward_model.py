import numpy as np
from scipy.special import ive, logsumexp

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
           log_prior_theta: np.ndarray, chunk_size: int = 100) -> np.ndarray:
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
    inv_sigma_Ep = np.float32(1.0 / sigma_Ep)
    # termine in theta per evento, (n_events, 1, n_theta): non dipende da En
    second = (-0.5 * ((theta_p_hat[:, None] - theta) / sigma_theta) ** 2
              + log_prior_theta[None, :]).astype(np.float32)[:, None, :]

    out = np.empty((n_events, n_En), dtype=np.float32)
    for start in range(0, n_events, chunk_size):
        end = min(start + chunk_size, n_events)
        # log-sum-exp su theta a mano e in place: scipy.special.logsumexp era il
        # 75% del tempo (conversioni e passate extra, profilo in docs/roadmap.md)
        joint = (Ep_hat[start:end, None, None] - Ep_pred) * inv_sigma_Ep  # (chunk, n_En, n_theta)
        joint *= joint
        joint *= np.float32(-0.5)
        joint += second[start:end]
        joint_max = joint.max(axis=2, keepdims=True)  # finito: il prior su theta e' finito
        joint -= joint_max
        np.exp(joint, out=joint)
        out[start:end] = np.log(joint.sum(axis=2)) + joint_max[..., 0]
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
    return logsumexp_axis(joint, axis=2)


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


def logsumexp_axis(x: np.ndarray, axis: int) -> np.ndarray:
    """log sum exp(x) lungo un asse, stabile (spostamento del massimo).
    Sostituisce scipy.special.logsumexp nel path caldo: stesso risultato a
    precisione macchina, senza il suo overhead (conversioni array API,
    copie per il caso complesso), ~meta' del tempo nel profilo del Caso C.

    Args:
        x: array reale, forma qualsiasi, -inf ammessi.
        axis: asse da sommare.

    Ritorna:
        array con `axis` rimosso, stesso dtype di x; -inf dove tutti i
        termini sono -inf.
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    x_max[~np.isfinite(x_max)] = 0.0  # fette tutte -inf: exp(-inf - 0) = 0 -> -inf
    with np.errstate(divide="ignore"):
        out = np.log(np.sum(np.exp(x - x_max), axis=axis))
    return out + np.squeeze(x_max, axis=axis)


def log_matmul_exp(a: np.ndarray, b: np.ndarray, cell_chunk_size: int = 20_000) -> np.ndarray:
    """log sum_j exp(a[i, j] + b[k, j]) per ogni coppia (i, k), come prodotto
    di matrici (BLAS) invece del tensore (n_a, n_b, n_j) passato a logsumexp.

    Stabilita': a e b sono spostati del proprio massimo per riga, quindi
    ogni termine exp(...) e' in [0, 1] con errore relativo ~1e-16 e la somma
    di termini positivi e' ben condizionata. Le celle il cui prodotto scende
    sotto TINY (vicino ai subnormali, precisione persa) sono ricalcolate con
    logsumexp: il risultato resta esatto anche lontano dal massimo.

    Args:
        a: log-termini, forma (n_a, n_j).
        b: log-termini, forma (n_b, n_j).
        cell_chunk_size: celle ricalcolate per lotto nel ripiego.

    Ritorna:
        array (n_a, n_b), float64.
    """
    tiny = 1e-280  # log ~ -645: sopra i subnormali (~1e-308) con ampio margine
    a_max = np.max(a, axis=1, keepdims=True)
    b_max = np.max(b, axis=1, keepdims=True)
    # righe interamente -inf: lo spostamento 0 lascia exp(-inf) = 0 -> -inf
    a_max[~np.isfinite(a_max)] = 0.0
    b_max[~np.isfinite(b_max)] = 0.0
    product = np.exp(a - a_max) @ np.exp(b - b_max).T
    with np.errstate(divide="ignore"):
        out = np.log(product) + a_max + b_max.T

    row, col = np.nonzero(product < tiny)
    for start in range(0, row.shape[0], cell_chunk_size):
        r, c = row[start:start + cell_chunk_size], col[start:start + cell_chunk_size]
        out[r, c] = logsumexp(a[r] + b[c], axis=1)
    return out


def marginalize_En_hierarchical(base: np.ndarray, log_prior_En_grid: np.ndarray,
                                chunk_size: int = 200,
                                event_chunk_size: int = 50) -> np.ndarray:
    """log sum_En exp(base + log pi(En | candidato)) per ogni evento e
    candidato iperparametro (Caso C), come prodotto di matrici
    (log_matmul_exp): RAM O(n_events * n_hyper) invece del blocco 3D.

    Args:
        base: log-verosimiglianza per evento su En_grid, forma (n_events, n_En).
        log_prior_En_grid: log pi(En | candidato), forma (n_hyper, n_En).
        chunk_size: candidati iperparametro per lotto.
        event_chunk_size: eventi per lotto. Con il prodotto di matrici i
            lotti servono solo a limitare la RAM a N molto grande; il
            risultato non dipende dalla loro dimensione.

    Ritorna:
        array (n_events, n_hyper), log-verosimiglianza marginalizzata su En.
    """
    n_events, n_hyper = base.shape[0], log_prior_En_grid.shape[0]
    out = np.empty((n_events, n_hyper), dtype=np.float64)
    for ev_start in range(0, n_events, event_chunk_size):
        ev_end = min(ev_start + event_chunk_size, n_events)
        for start in range(0, n_hyper, chunk_size):
            end = min(start + chunk_size, n_hyper)
            out[ev_start:ev_end, start:end] = log_matmul_exp(
                base[ev_start:ev_end], log_prior_En_grid[start:end]
            )
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
    trascurata, errore O(sigma_theta^2): produce un bias di mu_E
    ~ -sigma_theta^2 mu_E nel Caso C. Sostituito nel motore da
    log_track_kernel_sphere; resta come confronto.

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


def log_track_kernel_sphere(theta_obs: np.ndarray, theta_grid: np.ndarray,
                            sigma_theta: float) -> np.ndarray:
    """Come log_track_kernel, ma con la convoluzione esatta sulla sfera:
    errore della traccia von Mises-Fisher di concentrazione
    kappa = 1/sigma_theta^2 (la gaussiana 2D nel piano tangente di
    kinematics.smear_direction a meno di O(sigma_theta^4)), integrato
    sull'azimut del theta vero attorno a Omega_n (a):

        K = kappa/(1 - e^{-2 kappa}) sin(theta) exp(kappa (cos(theta_obs - theta) - 1))
            * I0e(kappa sin(theta_obs) sin(theta)),

    per steradiante della traccia osservata e per unita' di theta. Per
    kappa sin(theta_obs) sin(theta) >> 1 si riduce a log_track_kernel per
    sqrt(sin(theta)/sin(theta_obs)): la curvatura trascurata nel kernel 1D,
    che spostava theta_p vero verso l'alto di ~sigma^2 cot(theta)/2 e
    produceva il bias di mu_E ~ -sigma_theta^2 mu_E (docs/report_caso_C_stadio1.md
    §7, verificato con MC). Il passaggio oltre il polo e' gia' dentro I0:
    nessun termine specchiato.

    Args:
        theta_obs: angolo osservato fra traccia e Omega_n, rad, forma (n,), in [0, pi].
        theta_grid: griglia uniforme su [0, pi/2] per theta vero, rad, forma (m,).
        sigma_theta: risoluzione angolare per componente, rad.

    Ritorna:
        array (n, m), log-pesi; exp(...).sum(axis=1) approssima la densita'
        per steradiante della traccia osservata.
    """
    kappa = 1.0 / sigma_theta ** 2
    step = theta_grid[1] - theta_grid[0]
    log_weight = np.full(theta_grid.shape, np.log(step))
    log_weight[[0, -1]] -= np.log(2.0)
    t_obs, t_true = theta_obs[:, None], theta_grid[None, :]
    # cos(d) - 1 = -2 sin^2(d/2): nessuna cancellazione per d << 1 (kappa grande)
    log_vmf = -2.0 * kappa * np.sin(0.5 * (t_obs - t_true)) ** 2
    with np.errstate(divide="ignore"):
        log_norm = np.log(kappa) - np.log1p(-np.exp(-2.0 * kappa)) + np.log(np.sin(theta_grid))
        log_bessel = np.log(ive(0, kappa * np.sin(t_obs) * np.sin(t_true)))
    return log_vmf + log_bessel + (log_norm + log_track_density(theta_grid) + log_weight)[None, :]


def track_energy_table(Ep_hat: np.ndarray, En_grid: np.ndarray, sigma_Ep: float,
                       sigma_theta: float, log_prior_En: np.ndarray,
                       n_theta: int = N_THETA_TRACK, n_theta_obs: int = N_THETA_OBS,
                       chunk_size: int = 20) -> np.ndarray:
    """Tabella h_k(theta_obs) = log int ds p_track N(theta_obs; s) exp g_k(s)
    del Caso B su una griglia uniforme theta_obs in [0, pi]: non dipende dai
    candidati Omega_n, quindi si calcola una volta e si riusa su piu' griglie
    di candidati (griglia grossolana e calotta, posterior_C).

    Args:
        Ep_hat: energia di rinculo osservata, MeV, forma (n_events,).
        En_grid: griglia su cui marginalizzare En, MeV, forma (n_En,).
        sigma_Ep: risoluzione su Ep, MeV.
        sigma_theta: risoluzione angolare della traccia, rad.
        log_prior_En: log-prior su En_grid, forma (n_En,).
        n_theta: punti della griglia di theta vero su [0, pi/2].
        n_theta_obs: punti della griglia theta_obs su [0, pi].
        chunk_size: eventi per lotto (temporaneo (chunk, n_theta, n_En)).

    Ritorna:
        array (n_events, n_theta_obs), log-verosimiglianza sulla griglia
        theta_obs = linspace(0, pi, n_theta_obs).
    """
    theta_grid = np.linspace(0.0, np.pi / 2, n_theta)
    obs_grid = np.linspace(0.0, np.pi, n_theta_obs)
    kernel = log_track_kernel_sphere(obs_grid, theta_grid, sigma_theta)  # (n_obs, n_theta)

    n_events = Ep_hat.shape[0]
    table = np.empty((n_events, n_theta_obs), dtype=np.float64)
    for start in range(0, n_events, chunk_size):
        end = min(start + chunk_size, n_events)
        theta_true = np.broadcast_to(theta_grid, (end - start, n_theta))
        energy = loglik_marginal_En(Ep_hat[start:end], theta_true, En_grid,
                                    sigma_Ep, log_prior_En)          # (chunk, n_theta)
        table[start:end] = log_matmul_exp(energy, kernel)
    return table


def interpolate_track_table(table: np.ndarray, theta_obs: np.ndarray,
                            chunk_size: int = 20) -> np.ndarray:
    """Interpolazione lineare della tabella di track_energy_table sugli angoli
    osservati dei candidati.

    Args:
        table: forma (n_events, n_theta_obs), griglia uniforme su [0, pi].
        theta_obs: angolo fra traccia osservata e candidato, rad, forma
            (n_events, n_candidates).
        chunk_size: eventi per lotto: temporanei (chunk, n_candidates)
            invece di (n_events, n_candidates), picco di memoria ~ output.

    Ritorna:
        array (n_events, n_candidates), log-verosimiglianza.
    """
    n_events, n_theta_obs = table.shape
    step = np.pi / (n_theta_obs - 1)
    out = np.empty(theta_obs.shape, dtype=np.float64)
    for start in range(0, n_events, chunk_size):
        end = min(start + chunk_size, n_events)
        position = np.clip(theta_obs[start:end] / step, 0.0, n_theta_obs - 1 - 1e-9)
        index = position.astype(np.intp)
        lo = np.take_along_axis(table[start:end], index, axis=1)
        hi = np.take_along_axis(table[start:end], index + 1, axis=1)
        out[start:end] = lo + (position - index) * (hi - lo)
    return out


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
    h_k(theta_obs) una volta su una griglia uniforme theta_obs in [0, pi]
    (track_energy_table) e si interpola linearmente sui candidati
    (interpolate_track_table): costo indipendente dal numero di candidati.

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
    table = track_energy_table(Ep_hat, En_grid, sigma_Ep, sigma_theta, log_prior_En,
                               n_theta, n_theta_obs, chunk_size)
    return interpolate_track_table(table, theta_obs, chunk_size)


def hierarchical_base(Ep_hat: np.ndarray, theta_obs: np.ndarray, En_grid: np.ndarray,
                      sigma_Ep: float, sigma_theta: float,
                      n_theta: int = N_THETA_TRACK,
                      event_chunk_size: int = 20) -> np.ndarray:
    """Log-verosimiglianza per evento su En_grid, con theta_p vero gia'
    marginalizzato (termine di traccia e risoluzione angolare), Omega_n
    fissato (Caso C, stadio 2). Non dipende dagli iperparametri: si calcola
    una volta e si riusa su piu' griglie (mu_E, sigma_E)
    (marginalize_En_hierarchical).

    Args:
        Ep_hat: energia di rinculo osservata, MeV, forma (n_events,).
        theta_obs: angolo fra traccia osservata e Omega_n fissato, rad,
            forma (n_events,).
        En_grid: griglia su cui marginalizzare En, MeV, forma (n_En,).
        sigma_Ep: risoluzione su Ep, MeV.
        sigma_theta: risoluzione angolare della traccia, rad.
        n_theta: punti della griglia di theta vero su [0, pi/2].
        event_chunk_size: eventi per lotto (array (lotto, n_theta, n_En) in RAM).

    Ritorna:
        array (n_events, n_En), log-verosimiglianza (a meno di una costante comune).
    """
    theta_grid = np.linspace(0.0, np.pi / 2, n_theta)
    Ep_pred = En_grid[None, :] * np.cos(theta_grid)[:, None] ** 2    # (n_theta, n_En)
    n_events = Ep_hat.shape[0]
    base = np.empty((n_events, En_grid.shape[0]), dtype=np.float64)
    for start in range(0, n_events, event_chunk_size):
        end = min(start + event_chunk_size, n_events)
        kernel = log_track_kernel_sphere(theta_obs[start:end], theta_grid, sigma_theta)
        energy = -0.5 * ((Ep_hat[start:end, None, None] - Ep_pred[None]) / sigma_Ep) ** 2
        base[start:end] = logsumexp_axis(kernel[:, :, None] + energy, axis=1)
    return base


def loglik_marginal_En_theta_hierarchical(Ep_hat: np.ndarray, theta_obs: np.ndarray,
                                          En_grid: np.ndarray, sigma_Ep: float,
                                          sigma_theta: float,
                                          log_prior_En_grid: np.ndarray,
                                          n_theta: int = N_THETA_TRACK,
                                          chunk_size: int = 200,
                                          event_chunk_size: int = 20) -> np.ndarray:
    """Come loglik_marginal_En_hierarchical (Caso C, stadio 2, Omega_n
    fissato), ma con termine di traccia e risoluzione angolare: si
    marginalizza prima il theta_p vero (hierarchical_base, non dipende dagli
    iperparametri), poi En con il prior di ciascun candidato (mu_E, sigma_E).

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
    base = hierarchical_base(Ep_hat, theta_obs, En_grid, sigma_Ep, sigma_theta,
                             n_theta, event_chunk_size)
    return marginalize_En_hierarchical(base, log_prior_En_grid, chunk_size)
