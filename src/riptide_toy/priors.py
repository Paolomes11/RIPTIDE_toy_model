import numpy as np
from scipy.integrate import trapezoid


def energy_prior(en_grid: np.ndarray) -> np.ndarray:
    """Prior su En: piatta ma propria, limitata al dominio di en_grid
    (Cap. 38, Es. 38.1: "a flat prior"). Mai flat impropria su un
    dominio illimitato (Cap. 38): qui e' propria perche' en_grid ha
    estremi finiti.

    Args:
        en_grid: griglia di ipotesi su En, MeV, forma (n,).

    Ritorna:
        array (n,), densita' di probabilita' costante; l'integrale
        trapezoidale su en_grid vale 1 (fix errata b: scipy.integrate.trapezoid).
    """
    width = en_grid[-1] - en_grid[0]
    return np.full_like(en_grid, 1.0 / width)


def direction_prior(theta_grid: np.ndarray, phi_grid: np.ndarray) -> np.ndarray:
    """Prior su Omega_n: piatta ma propria sulla sfera (Caso B/C).

    Args:
        theta_grid: colatitudine dei pixel della griglia sferica, rad,
            forma (n_pixel,) (grids.sphere_grid).
        phi_grid: azimut dei pixel, rad, stessa forma di theta_grid.

    Ritorna:
        array (n_pixel,), pesi costanti che sommano a 1. Uniforme perche'
        grids.sphere_grid pixelizza la sfera ad area solida ~costante
        per pixel (reticolo di Fibonacci).
    """
    n_pixel = theta_grid.shape[0]
    return np.full(n_pixel, 1.0 / n_pixel)


def energy_prior_given_hyperparams(en_grid: np.ndarray, mu_E: np.ndarray,
                                    sigma_E: np.ndarray) -> np.ndarray:
    """pi(En | mu_E, sigma_E): gaussiana troncata al dominio di en_grid e
    rinormalizzata (Caso C, prior gerarchico sull'energia condivisa,
    CLAUDE.md Sez. 3: "E_n^(k) ~ N(mu_E, sigma_E)"). Mai flat impropria
    (Cap. 38): la normalizzazione e' sul dominio finito di en_grid, non su
    tutta la retta reale, quindi resta propria anche per sigma_E grande
    rispetto al dominio (limite Caso B).

    Args:
        en_grid: griglia di ipotesi su En, MeV, forma (n_En,).
        mu_E: media dell'iperprior, MeV, forma (n_hyper,) o scalare.
        sigma_E: deviazione standard dell'iperprior, MeV, stessa forma di mu_E.

    Ritorna:
        array (n_hyper, n_En) se mu_E/sigma_E hanno forma (n_hyper,),
        altrimenti (n_En,); l'integrale trapezoidale su en_grid vale 1 per
        ogni riga (fix errata b: scipy.integrate.trapezoid).
    """
    scalar_input = np.ndim(mu_E) == 0
    mu = np.atleast_1d(mu_E)[:, None]
    sigma = np.atleast_1d(sigma_E)[:, None]
    en = en_grid[None, :]
    unnorm = np.exp(-0.5 * ((en - mu) / sigma) ** 2)
    norm = trapezoid(unnorm, en_grid, axis=1)[:, None]
    density = unnorm / norm
    return density[0] if scalar_input else density


def hyperparameter_prior(mu_grid: np.ndarray, sigma_grid: np.ndarray) -> np.ndarray:
    """Prior su (mu_E, sigma_E): piatta ma propria sulla griglia (Caso C).

    Pesi uniformi sui punti della griglia: dato che grids.hyperparameter_grid
    spazia sigma_E logaritmicamente (e' un parametro di scala), un peso
    uniforme per punto equivale a un prior uniforme in (mu_E, log(sigma_E))
    -- la scelta standard per un parametro di scala, che evita di favorire
    sigma_E grandi solo perche' occupano piu' "spazio" lineare.

    Args:
        mu_grid: ipotesi su mu_E, MeV, forma (n_hyper,) (grids.hyperparameter_grid).
        sigma_grid: ipotesi su sigma_E, MeV, stessa forma di mu_grid.

    Ritorna:
        array (n_hyper,), pesi costanti che sommano a 1.
    """
    n_hyper = mu_grid.shape[0]
    return np.full(n_hyper, 1.0 / n_hyper)
