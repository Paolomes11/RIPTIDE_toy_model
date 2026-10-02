"""Cinematica dello scattering elastico n-p a masse uguali.

E_p = E_n cos^2(theta_p), con theta_p <= 90 gradi. Con scattering isotropo nel
centro di massa cos(theta_CM) ~ U(-1, 1) e theta_p = (pi - theta_CM)/2, quindi
cos^2(theta_p) = (1 - cos theta_CM)/2 ~ U(0, 1) e E_p ~ U(0, E_n). Qui: generazione
degli eventi, rotazioni attorno alla direzione del neutrone, smearing della traccia.
"""
import numpy as np

from riptide_toy.constants import BIMODAL_HALF_SEPARATION, M_NEUTRON, M_PROTON

def proton_energy(En: np.ndarray, theta_p: np.ndarray) -> np.ndarray:
    """Energia del protone di rinculo, Ep = En * cos(theta_p)**2.

    Args:
        En: energia del neutrone incidente, MeV, forma (n,) o scalare.
        theta_p: angolo di rinculo in lab, rad, stessa forma di En.

    Ritorna:
        Ep in MeV, stessa forma degli input (broadcasting numpy).
    """
    return En * (np.cos(theta_p) ** 2)

def sample_cm_angle(rng: np.random.Generator, n: int) -> np.ndarray:
    """Angolo polare di scattering del neutrone nel CM, isotropo in angolo
    solido: cos(theta_CM) ~ Uniform(-1, 1).

    Args:
        rng: generatore numpy.
        n: numero di eventi.

    Ritorna:
        array (n,), theta_CM in rad, in [0, pi].
    """
    # (a) isotropia in angolo solido: uniforme in cos, non in theta.
    # Chiude la voce aperta in docs/roadmap.md (prima: uniform(0, 2*pi)).
    return np.arccos(rng.uniform(-1.0, 1.0, n))


def recoil_angle_from_cm(theta_cm: np.ndarray) -> np.ndarray:
    """Angolo del protone di rinculo in lab dall'angolo del neutrone in CM,
    theta_p = (pi - theta_CM) / 2.

    Args:
        theta_cm: angolo di scattering del neutrone nel CM, rad, forma (n,).

    Ritorna:
        array (n,), theta_p in rad, in [0, pi/2].
    """
    # (a) masse uguali: il protone rincula a pi - theta_CM nel CM e l'angolo
    # in lab e' la meta'. Con cos(theta_CM) uniforme segue Ep ~ U(0, En).
    return 0.5 * (np.pi - theta_cm)


def perpendicular_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Due versori ortonormali perpendicolari a ciascun asse.

    Args:
        axis: versori, forma (n, 3).

    Ritorna:
        (e1, e2), ciascuno forma (n, 3); (e1, e2, axis) e' una terna destrorsa.
    """
    # asse ausiliario x, o y se axis e' quasi parallelo a x (prodotto
    # vettoriale mal condizionato)
    helper = np.where(np.abs(axis[:, :1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    e1 = np.cross(axis, helper)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(axis, e1)
    return e1, e2


def direction_around_axis(axis: np.ndarray, theta: np.ndarray,
                          phi: np.ndarray) -> np.ndarray:
    """Versore a angolo polare theta e azimut phi attorno a un asse dato.

    Args:
        axis: versori, forma (n, 3).
        theta: angolo polare rispetto ad axis, rad, forma (n,).
        phi: azimut attorno ad axis, rad, forma (n,).

    Ritorna:
        array (n, 3), versori unitari.
    """
    e1, e2 = perpendicular_basis(axis)
    sin_theta = np.sin(theta)[:, None]
    return (np.cos(theta)[:, None] * axis
            + sin_theta * np.cos(phi)[:, None] * e1
            + sin_theta * np.sin(phi)[:, None] * e2)


def sample_recoil_events(rng: np.random.Generator, En: np.ndarray,
                         omega_n_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eventi di rinculo veri per scattering n-p isotropo in CM, con
    neutroni incidenti lungo omega_n_hat (Caso B/C).

    Args:
        rng: generatore numpy.
        En: energia vera del neutrone per evento, MeV, forma (n,).
        omega_n_hat: direzione del neutrone incidente, versore, forma (3,).

    Ritorna:
        (Ep_true, track_true): energia del protone in MeV, forma (n,), e
        direzione vera della traccia, versori, forma (n, 3), sempre con
        theta_p <= pi/2 rispetto a omega_n_hat.
    """
    n = En.shape[0]
    theta_p = recoil_angle_from_cm(sample_cm_angle(rng, n))
    phi = rng.uniform(0.0, 2 * np.pi, n)
    axis = np.broadcast_to(omega_n_hat, (n, 3))
    return proton_energy(En, theta_p), direction_around_axis(axis, theta_p, phi)


def sample_energy_spectrum(rng: np.random.Generator, shape: str, mean: float,
                           sd: float, n: int) -> np.ndarray:
    """Energie vere dei neutroni da uno spettro di forma data, con media e
    deviazione standard fissate (stress test dell'ipotesi di spettro gaussiano).

    Args:
        rng: generatore numpy.
        shape: "gauss", "uniform", "bimodal" o "lognormal".
        mean: media dello spettro, MeV, > 0.
        sd: deviazione standard dello spettro, MeV, > 0.
        n: numero di eventi.

    Ritorna:
        array (n,), E_n in MeV; media e sd attese uguali a mean e sd per ogni forma.
    """
    if shape == "gauss":
        # stessa chiamata della generazione standard: a parita' di seme si
        # riottengono esattamente gli stessi dataset
        return rng.normal(mean, sd, n)
    if shape == "uniform":
        half_width = np.sqrt(3.0) * sd  # varianza di U(-h, h) = h^2 / 3
        return rng.uniform(mean - half_width, mean + half_width, n)
    if shape == "bimodal":
        # varianza = a^2 sd^2 (fra righe) + (1 - a^2) sd^2 (entro riga) = sd^2
        a = BIMODAL_HALF_SEPARATION
        side = rng.choice(np.array([-1.0, 1.0]), n)
        return rng.normal(mean + side * a * sd, sd * np.sqrt(1.0 - a ** 2))
    if shape == "lognormal":
        # momenti della lognormale: mean = exp(m + s^2/2), (sd/mean)^2 = exp(s^2) - 1
        s2 = np.log1p((sd / mean) ** 2)
        return rng.lognormal(np.log(mean) - s2 / 2, np.sqrt(s2), n)
    raise ValueError(f"forma di spettro sconosciuta: {shape}")


def smear_direction(rng: np.random.Generator, track: np.ndarray,
                    sigma_theta: float) -> np.ndarray:
    """Risoluzione angolare del detector sulla traccia: spostamento
    gaussiano isotropo nel piano tangente, sigma_theta per componente.

    Args:
        rng: generatore numpy.
        track: direzioni vere, versori, forma (n, 3).
        sigma_theta: risoluzione angolare per componente, rad.

    Ritorna:
        array (n, 3), direzioni misurate, versori.
    """
    # (b) proiettato su qualunque piano meridiano l'errore e' N(0, sigma_theta):
    # stesso modello 1D di SIGMA_THETA usato nel Caso A (forward_model.measure).
    offset = rng.normal(0.0, sigma_theta, (track.shape[0], 2))
    delta = np.hypot(offset[:, 0], offset[:, 1])
    alpha = np.arctan2(offset[:, 1], offset[:, 0])
    return direction_around_axis(track, delta, alpha)

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
    # in place: un solo array (n_events, n_candidates) invece di tre
    np.clip(cos_theta, -1.0, 1.0, out=cos_theta)
    return np.arccos(cos_theta, out=cos_theta)

def theta_phi_from_direction(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inversa di direction_from_theta_phi.

    Args:
        v: versori, forma (..., 3).

    Ritorna:
        (theta, phi) in rad, ciascuno forma (...,): theta in [0, pi], phi in [0, 2 pi).
    """
    theta = np.arccos(np.clip(v[..., 2], -1.0, 1.0))
    phi = np.mod(np.arctan2(v[..., 1], v[..., 0]), 2 * np.pi)
    return theta, phi
