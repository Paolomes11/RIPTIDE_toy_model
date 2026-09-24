import numpy as np

from riptide_toy.constants import M_NEUTRON, M_PROTON

def proton_energy(En: np.ndarray, theta_p: np.ndarray) -> np.ndarray:
    """Ep = En * cos(theta_p)**2.
    Return: array of the same dimensionas En/theta_p (broadvasting numpy)
    scatter energy in MeV."""
    return En * (np.cos(theta_p) ** 2)

def sample_cm_angle(rng: np.random.Generator, n: int) -> np.ndarray:
    """Return: array of dimension (n,), angles in radians taken
    isotropicaly in CM."""
    return rng.uniform(0, 2*np.pi, n)

def recoil_angle_from_direction(track_hat: np.ndarray,
                                omega_n_hat: np.ndarray) -> np.ndarray:
    """Return: array of dimension (n_events,), angle in radians
    between the twoo versors (arccos of the scalar product; Case B/C)."""
    cos_theta = track_hat @ omega_n_hat.T
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))