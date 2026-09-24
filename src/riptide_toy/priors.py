import numpy as np


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
