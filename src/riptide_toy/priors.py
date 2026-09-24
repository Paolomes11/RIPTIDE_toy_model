import numpy as np

# direction_prior (Caso B/C, riga 9) rimandata alla Fase 2: vedi docs/roadmap.md


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
