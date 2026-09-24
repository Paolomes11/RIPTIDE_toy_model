import numpy as np
from functools import lru_cache

from riptide_toy.constants import EN_MAX, EN_MIN

# (a) errata guida: maxsize=1 svuota la cache se si chiama la funzione con
# argomenti diversi (es. sphere_grid(500) poi sphere_grid()); gli array
# cachati vanno resi non scrivibili perche' sono condivisi tra i chiamanti.
@lru_cache(maxsize=None)
def energy_grid(n: int = 500) -> np.ndarray:
    """Return: array (n,), values of En in MeV,
    increasing from EN_MIN to EN_MAX."""
    grid = np.linspace(EN_MIN, EN_MAX, n)
    grid.flags.writeable = False
    return grid

# (c) errata: theta e phi erano due linspace indipendenti (stessa lunghezza
# n_pixel ma nessun accoppiamento), quindi l'indice i non corrispondeva a una
# singola direzione sulla sfera. Fix: reticolo di Fibonacci (angolo aureo),
# un indice = una direzione, area solida quasi costante per pixel.
@lru_cache(maxsize=None)
def sphere_grid(n_pixel: int = 3000) -> tuple[np.ndarray, np.ndarray]:
    """Return: (theta, phi), twoo arrays of dimension (n_pixel,)
    in radians - the coordinates of the directions for the sphere."""
    i = np.arange(n_pixel)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    theta = np.arccos(1.0 - 2.0 * (i + 0.5) / n_pixel)
    phi = np.mod(i * golden_angle, 2 * np.pi)
    theta.flags.writeable = False
    phi.flags.writeable = False
    return (theta, phi)

@lru_cache(maxsize=None)
def theta_p_grid(n: int = 500) -> np.ndarray:
    """Return: array (n,), lab recoil angle theta_p in rad,
    increasing from 0 to pi/2 (theta_lab <= 90 deg, CLAUDE.md Sez. 3).
    Usata per marginalizzare la nuisance theta_p in posterior_A."""
    grid = np.linspace(0.0, np.pi / 2, n)
    grid.flags.writeable = False
    return grid