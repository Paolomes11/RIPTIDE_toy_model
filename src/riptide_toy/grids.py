import numpy as np
from functools import lru_cache

from riptide_toy.constants import EN_MAX, EN_MIN

@lru_cache(maxsize=1)
def energy_grid(n: int = 500) -> np.ndarray:
    """Return: array (n,), values of En in MeV,
    increasing from EN_MIN to EN_MAX."""
    return np.linspace(EN_MIN, EN_MAX, n)

@lru_cache(maxsize=1)
def sphere_grid(n_pixel: int = 3000) -> tuple[np.ndarray, np.ndarray]:
    """Return: (theta, phi), twoo arrays of dimension (n_pixel,)
    in radians - the coordinates of the directions for the sphere."""
    return (np.linspace(0.0, np.pi, n_pixel), np.linspace(0.0, 2*np.pi, n_pixel))