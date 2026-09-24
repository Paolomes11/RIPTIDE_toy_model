import numpy as np


def bias_curve(truth: np.ndarray, estimate: np.ndarray, n_bins: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Bias medio di una ricostruzione in funzione del valore vero, a bin (Cap. 40 punto 1).

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
    """Pull di una ricostruzione, atteso N(0, 1) se calibrata (Cap. 40 punto 3).

    Args:
        truth: valore vero per evento, forma (n_eventi,).
        estimate: valore ricostruito per evento, stessa forma di truth.
        sigma_hat: incertezza dichiarata per evento, stessa forma di truth.

    Ritorna:
        array di forma (n_eventi,), (estimate - truth) / sigma_hat.
    """
    return (estimate - truth) / sigma_hat


def coverage_curve(truth: np.ndarray, intervals: np.ndarray, levels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Copertura empirica di intervalli credibili a piu' livelli nominali (Cap. 40 punto 4).

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


def run_checklist(reconstruction_fn, simulated_truths: np.ndarray, levels: np.ndarray,
                   n_bins: int = 20, **kwargs) -> dict:
    """Esegue la checklist Cap. 40 (punti 1, 3, 4 in ordine) su una ricostruzione.

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
