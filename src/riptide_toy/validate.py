from collections.abc import Callable

import numpy as np


def bias_curve(truth: np.ndarray, estimate: np.ndarray, n_bins: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Bias medio di una ricostruzione in funzione del valore vero, a bin (checklist di validazione, punto 1).

    Args:
        truth: valore vero per evento, forma (n_eventi,).
        estimate: valore ricostruito per evento, stessa forma di truth.
        n_bins: numero di bin equispaziati sul range di truth.

    Ritorna:
        (centri_bin, bias_medio), due array di forma (n_bins,); bias_medio[i]
        e' vuoto (nan) se nessun evento cade nel bin i.
    """
    edges = np.linspace(truth.min(), truth.max(), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(np.digitize(truth, edges) - 1, 0, n_bins - 1)
    residual = estimate - truth

    bias = np.full(n_bins, np.nan)
    counts = np.bincount(bin_idx, minlength=n_bins)
    sums = np.bincount(bin_idx, weights=residual, minlength=n_bins)
    nonzero = counts > 0
    bias[nonzero] = sums[nonzero] / counts[nonzero]
    return centers, bias


def pull_histogram(truth: np.ndarray, estimate: np.ndarray, sigma_hat: np.ndarray) -> np.ndarray:
    """Pull di una ricostruzione, atteso N(0, 1) se calibrata (checklist di validazione, punto 3).

    Args:
        truth: valore vero per evento, forma (n_eventi,).
        estimate: valore ricostruito per evento, stessa forma di truth.
        sigma_hat: incertezza dichiarata per evento, stessa forma di truth.

    Ritorna:
        array di forma (n_eventi,), (estimate - truth) / sigma_hat.
    """
    return (estimate - truth) / sigma_hat


def coverage_curve(truth: np.ndarray, intervals: np.ndarray, levels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Copertura empirica di intervalli credibili a piu' livelli nominali (checklist di validazione, punto 4).

    Nota: la guida (Sez. 3) non da' un type hint per `intervals`. Si sceglie qui la
    forma piu' generica possibile (estremi espliciti, non necessariamente simmetrici
    o gaussiani), cosi' che questa funzione resti utilizzabile sia con intervalli
    gaussiani (livello -> z*sigma_hat) sia con intervalli letti da un posterior a
    griglia (Caso B/C). Vedi docs/roadmap.md, sezione Deviazioni.

    Args:
        truth: valore vero per evento, forma (n_eventi,).
        intervals: estremi degli intervalli credibili, forma (n_eventi, n_livelli, 2),
            con [..., 0] = estremo inferiore e [..., 1] = estremo superiore.
        levels: livelli nominali (es. 0.68, 0.90), forma (n_livelli,).

    Ritorna:
        (levels, coperture_empiriche), due array di forma (n_livelli,).
    """
    covered = (truth[:, None] >= intervals[:, :, 0]) & (truth[:, None] <= intervals[:, :, 1])
    return levels, covered.mean(axis=0)


def run_checklist(reconstruction_fn: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]],
                  simulated_truths: np.ndarray, levels: np.ndarray,
                  n_bins: int = 20, **kwargs) -> dict:
    """Esegue la checklist di validazione (punti 1, 3, 4 in ordine) su una ricostruzione.

    Args:
        reconstruction_fn: callable(simulated_truths, **kwargs) -> (estimate, sigma_hat,
            intervals), con estimate/sigma_hat forma (n_eventi,) e intervals forma
            (n_eventi, n_livelli, 2) allineata a `levels`.
        simulated_truths: valori veri noti, forma (n_eventi,).
        levels: livelli nominali di copertura da testare, forma (n_livelli,).
        n_bins: numero di bin per bias_curve.
        **kwargs: passati a reconstruction_fn.

    Ritorna:
        dict con chiavi: bias_centers, bias, pull_mean, pull_width, levels, coverage.
    """
    estimate, sigma_hat, intervals = reconstruction_fn(simulated_truths, **kwargs)

    bias_centers, bias = bias_curve(simulated_truths, estimate, n_bins)
    pull = pull_histogram(simulated_truths, estimate, sigma_hat)
    _, coverage = coverage_curve(simulated_truths, intervals, levels)

    return {
        "bias_centers": bias_centers,
        "bias": bias,
        "pull_mean": pull.mean(),
        "pull_width": pull.std(),
        "levels": levels,
        "coverage": coverage,
    }


def angular_residual(omega_true_hat: np.ndarray, omega_estimate_hat: np.ndarray) -> np.ndarray:
    """Distanza angolare fra direzione vera e ricostruita (riga 12: estensione di
    validate.py per il Caso B, etichetta (d) — nessun valore di riferimento, verificata su dati
    simulati con Omega_n nota, vedi docs/roadmap.md).

    Args:
        omega_true_hat: direzione vera, versori, forma (n_eventi, 3).
        omega_estimate_hat: direzione ricostruita, versori, stessa forma.

    Ritorna:
        array (n_eventi,), angolo fra le due direzioni, rad, in [0, pi].
    """
    cos_angle = np.clip(np.sum(omega_true_hat * omega_estimate_hat, axis=-1), -1.0, 1.0)
    return np.arccos(cos_angle)


def posterior_angular_resolution(log_posterior: np.ndarray, omega_hat_grid: np.ndarray,
                                  reference_hat: np.ndarray) -> np.ndarray:
    """Deviazione angolare pesata sul posterior rispetto a una direzione di riferimento
    per evento: analogo sferico della deviazione standard pesata sulla griglia gia'
    usata per il Caso A (docs/roadmap.md, voce "risoluzione_caso_A"). Estensione (d).

    Args:
        log_posterior: log-posterior non normalizzato su Omega_n, forma
            (n_eventi, n_candidati) (es. combine.combine_loglik).
        omega_hat_grid: direzioni candidate, versori, forma (n_candidati, 3).
        reference_hat: direzione di riferimento per evento (verita' nota, o la stima
            puntuale stessa), versori, forma (n_eventi, 3).

    Ritorna:
        array (n_eventi,), sqrt(E_p[distanza_angolare^2]) in rad, con pesi
        p = exp(log_posterior - max(log_posterior)) normalizzati per evento.
    """
    weights = np.exp(log_posterior - log_posterior.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)

    cos_angle = np.clip(reference_hat @ omega_hat_grid.T, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    return np.sqrt(np.sum(weights * angle ** 2, axis=-1))


def posterior_mean_std(log_posterior: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Media e deviazione standard pesate su un posterior a griglia per un parametro
    scalare (momenti del posterior; generalizza il calcolo gia' usato inline per
    i casi di riferimento R1/R2 e per la voce "risoluzione_caso_A" di docs/roadmap.md; qui
    fattorizzata perche' riusata per i parametri scalari mu_E, sigma_E del Caso C,
    riga 14 -- analoga a posterior_angular_resolution ma per un asse lineare invece
    che sferico). Estensione additiva (CLAUDE.md Sez. 6, checkpoint Caso A).

    Args:
        log_posterior: log-posterior non normalizzato, forma (n_eventi, n_candidati).
        grid: valori del parametro sulla griglia, stessa unita' del parametro,
            forma (n_candidati,).

    Ritorna:
        (mean, std), due array di forma (n_eventi,): media e deviazione standard
        pesate, con pesi p = exp(log_posterior - max(log_posterior)) normalizzati
        per evento.
    """
    weights = np.exp(log_posterior - log_posterior.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    mean = np.sum(weights * grid[None, :], axis=-1)
    var = np.sum(weights * (grid[None, :] - mean[:, None]) ** 2, axis=-1)
    return mean, np.sqrt(var)


def angular_pull(angular_dist: np.ndarray, sigma_hat: np.ndarray) -> np.ndarray:
    """Pull della distanza angolare (checklist di validazione punto 3, estensione (d)). A differenza del
    pull 1D (pull_histogram, gia' verificato N(0,1) se calibrato), la distanza angolare
    e' una quantita' non negativa: un ricostruttore calibrato la cui incertezza
    dichiarata sigma_hat riflette correttamente lo scatter reale produce
    angular_dist/sigma_hat con scala ~1, non media 0 — da verificare su dati simulati
    con Omega_n nota (etichetta (d), ipotesi da testare, non un caso di riferimento).

    Args:
        angular_dist: distanza angolare, rad, forma (n_eventi,) (angular_residual).
        sigma_hat: incertezza angolare dichiarata per evento, rad, stessa forma
            (es. posterior_angular_resolution).

    Ritorna:
        array (n_eventi,), angular_dist / sigma_hat.
    """
    return angular_dist / sigma_hat


def credible_interval(log_posterior: np.ndarray, grid: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Intervalli credibili a code uguali letti da un posterior a griglia, per
    un parametro scalare (checklist di validazione punto 4, coverage del Caso C, riga 14).
    Se grid ha valori ripetuti (es. un asse di una griglia 2D appiattita) il
    posterior viene prima marginalizzato sommando sui punti con lo stesso
    valore. Quantili per interpolazione lineare della CDF ai centri delle
    celle; livello 0 = mediana. Estensione additiva (Sez. 6, checkpoint).

    Args:
        log_posterior: log-posterior non normalizzato, forma (n_eventi, n_candidati).
        grid: valori del parametro, forma (n_candidati,), stessa unita' del parametro.
        levels: livelli nominali in [0, 1), forma (n_livelli,).

    Ritorna:
        array (n_eventi, n_livelli, 2), [..., 0] estremo inferiore e [..., 1]
        superiore, stessa unita' di grid; stessa forma attesa da coverage_curve.
    """
    values, inverse = np.unique(grid, return_inverse=True)
    one_hot = np.zeros((grid.shape[0], values.shape[0]))
    one_hot[np.arange(grid.shape[0]), inverse] = 1.0
    weights = np.exp(log_posterior - log_posterior.max(axis=-1, keepdims=True))
    marginal = weights @ one_hot
    marginal /= marginal.sum(axis=-1, keepdims=True)
    cdf = np.cumsum(marginal, axis=-1) - 0.5 * marginal

    tail = (1.0 - levels) / 2.0
    quantiles = np.concatenate([tail, 1.0 - tail])
    upper_idx = np.clip(np.sum(cdf[:, :, None] < quantiles[None, None, :], axis=1), 1, values.shape[0] - 1)
    cdf_lo = np.take_along_axis(cdf, upper_idx - 1, axis=1)
    cdf_hi = np.take_along_axis(cdf, upper_idx, axis=1)
    frac = np.clip((quantiles[None, :] - cdf_lo) / (cdf_hi - cdf_lo), 0.0, 1.0)
    q_values = values[upper_idx - 1] + frac * (values[upper_idx] - values[upper_idx - 1])
    n_levels = levels.shape[0]
    return np.stack([q_values[:, :n_levels], q_values[:, n_levels:]], axis=-1)


def credible_region_contains(log_posterior: np.ndarray, true_index: np.ndarray,
                             levels: np.ndarray) -> np.ndarray:
    """Copertura con regioni di massima densita' (HPD) su una griglia di
    candidati ad area/volume uguale, es. la calotta di Omega_n (checklist di
    validazione punto 4, riga 14): il candidato vero e' nella regione a livello L se la massa
    dei candidati piu' probabili di lui e' < L. Estensione additiva.

    Args:
        log_posterior: log-posterior non normalizzato, forma (n_eventi, n_candidati).
        true_index: indice del candidato piu' vicino alla verita', int, forma (n_eventi,).
        levels: livelli nominali, forma (n_livelli,).

    Ritorna:
        array bool (n_eventi, n_livelli), True se la verita' e' coperta.
    """
    p = np.exp(log_posterior - log_posterior.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    p_true = np.take_along_axis(p, true_index[:, None], axis=1)
    mass_above = np.sum(np.where(p > p_true, p, 0.0), axis=-1)
    return mass_above[:, None] < levels[None, :]
