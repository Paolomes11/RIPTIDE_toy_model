import numpy as np

from riptide_toy.constants import M_NEUTRON, M_PROTON

def proton_energy(En: np.ndarray, theta_p: np.ndarray) -> np.ndarray:
    """Ep = En * cos(theta_p)**2.
    Return: array of the same dimensionas En/theta_p (broadvasting numpy)
    scatter energy in MeV."""
    return En * (np.cos(theta_p) ** 2)

def sample_cm_angle(rng: np.random.Generator, n: int) -> np.ndarray:
    """Return: array of dimension (n,), angles in radians taken
    isotropicaly in CM.

    TODO (aperto, non ancora usato da nessun modulo a valle): questa
    funzione campiona uniform(0, 2*pi). Se e' intesa come l'angolo
    polare theta_CM di uno scattering isotropo in angolo solido, la
    formula corretta sarebbe arccos(uniform(-1, 1)), non uniforme.
    Nessuna fonte autorevole del progetto (guida, libro) conferma o
    smentisce quale sia l'interpretazione voluta qui: non si inventa,
    si verifica prima di usarla in un test formale (CLAUDE.md Sez. 1).
    Vedi docs/roadmap.md, sezione Deviazioni.
    """
    return rng.uniform(0, 2*np.pi, n)

def direction_from_theta_phi(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Converte coordinate sferiche in versori cartesiani (convenzione fisica:
    theta = colatitudine da z, phi = azimut).

    Args:
        theta: angolo polare, rad, forma (n,) o scalare.
        phi: angolo azimutale, rad, stessa forma di theta.

    Ritorna:
        array (..., 3), versori unitari; ultima dimensione = (x, y, z).
    """
    sin_theta = np.sin(theta)
    return np.stack(
        [sin_theta * np.cos(phi), sin_theta * np.sin(phi), np.cos(theta)],
        axis=-1,
    )

def recoil_angle_from_direction(track_hat: np.ndarray,
                                omega_n_hat: np.ndarray) -> np.ndarray:
    """Angolo tra le direzioni della traccia osservata e una griglia
    di ipotesi sulla direzione del neutrone (Caso B/C).

    Args:
        track_hat: versori osservati, forma (n_events, 3).
        omega_n_hat: versori di ipotesi sulla griglia, forma (n_candidates, 3).

    Ritorna:
        array (n_events, n_candidates), angolo in rad tra ogni coppia
        (arccos del prodotto scalare, clip per stabilita' numerica).
    """
    cos_theta = track_hat @ omega_n_hat.T
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))