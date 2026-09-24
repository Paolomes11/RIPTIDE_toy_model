import numpy as np


def combine_loglik(loglik_matrix: np.ndarray, log_prior: np.ndarray) -> np.ndarray:
    """Combina il log-posterior di piu' eventi sullo stesso parametro condiviso
    (Cap. 22/39): somma sull'asse eventi, prior contato una sola volta.

    Args:
        loglik_matrix: log-verosimiglianza per evento, forma (n_eventi, n_punti_griglia).
        log_prior: log-prior proprio sulla griglia condivisa, forma (n_punti_griglia,).

    Ritorna:
        array (n_punti_griglia,), log-posterior combinato non normalizzato.
    """
    return log_prior + np.sum(loglik_matrix, axis=0)


def expected_sigma_n(sigma_1: float, n: int) -> float:
    """Contrazione attesa della risoluzione con N eventi indipendenti (Cap. 39).

    Args:
        sigma_1: risoluzione (sigma) di un singolo evento.
        n: numero di eventi combinati.

    Ritorna:
        sigma_1 / sqrt(n), un singolo numero float.
    """
    return sigma_1 / np.sqrt(n)
