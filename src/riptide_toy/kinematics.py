"""Cinematica relativistica dello scattering elastico n-p a masse uguali m.

E_p = 2 m E_n cos^2(theta_p) / (2 m + E_n sin^2(theta_p)), con theta_p <= 90 gradi
(E_n, E_p energie cinetiche); per E_n << m si riduce a E_n cos^2(theta_p). L'energia
del protone resta lineare in cos(theta_CM): E_p = E_n (1 - cos theta_CM) / 2, quindi
con scattering isotropo nel centro di massa E_p ~ U(0, E_n) anche relativisticamente.
Qui: generazione degli eventi, rotazioni attorno alla direzione del neutrone,
smearing della traccia.
"""
import numpy as np
from numpy.polynomial import legendre

from riptide_toy.constants import BIMODAL_HALF_SEPARATION, N_CM_NEWTON, PROTON_MASS_MEV, STUDENT_T_DOF

def proton_energy(En: np.ndarray, theta_p: np.ndarray) -> np.ndarray:
    """Energia cinetica del protone di rinculo, relativistica a masse uguali:
    Ep = 2 m En cos^2(theta_p) / (2 m + En sin^2(theta_p)), m = PROTON_MASS_MEV.

    Args:
        En: energia cinetica del neutrone incidente, MeV, forma (n,) o scalare.
        theta_p: angolo di rinculo in lab, rad, stessa forma di En.

    Ritorna:
        Ep in MeV, stessa forma degli input (broadcasting numpy); per En << m
        tende a En * cos(theta_p)**2.
    """
    # (a) conservazione del quadrimpulso, bersaglio fermo; a 20 MeV e 45 gradi lo
    # scarto dalla forma non relativistica e' ~0.05 MeV, comune a tutti gli eventi.
    # Forma cos^2 / (1/En + sin^2/2m): con griglie (n_En, 1) x (1, n_theta) le
    # operazioni sulla tabella piena sono due invece di cinque (path caldo del Caso A).
    # En = 0 da' 1/En = inf ed Ep = 0, il limite corretto.
    with np.errstate(divide="ignore"):
        inv_En = np.divide(1.0, En)
    denom = inv_En + np.sin(theta_p) ** 2 / (2.0 * PROTON_MASS_MEV)
    if not isinstance(denom, np.ndarray):
        return np.cos(theta_p) ** 2 / denom
    # divisione in place: una tabella piena allocata in meno, e l'allocazione di
    # una tabella nuova a ogni chiamata costava piu' del calcolo stesso (profilo)
    return np.divide(np.cos(theta_p) ** 2, denom, out=denom)


def recoil_cos2_from_energy(Ep: np.ndarray, En: np.ndarray) -> np.ndarray:
    """Inversa di proton_energy in theta_p: cos^2(theta_p) dato (Ep, En),
    cos^2(theta_p) = Ep (2 m + En) / (En (2 m + Ep)).

    Args:
        Ep: energia cinetica del protone, MeV, forma qualsiasi, in [0, En].
        En: energia cinetica del neutrone, MeV, > 0, forma compatibile (broadcasting).

    Ritorna:
        cos^2(theta_p), adimensionale, in [0, 1], forma del broadcasting.
    """
    two_m = 2.0 * PROTON_MASS_MEV
    return Ep * (two_m + En) / (En * (two_m + Ep))


def log_recoil_density_ratio(theta_p: np.ndarray, En: np.ndarray) -> np.ndarray:
    """log del rapporto fra la densita' relativistica della direzione del protone
    (log_recoil_direction_density) e il suo limite non relativistico cos(theta_p)/pi:
    log[2 m (2 m + En) / (2 m + En sin^2(theta_p))^2].

    Serve a separare la parte che dipende da En (questo fattore, finito ovunque)
    da quella che non ne dipende (cos(theta_p)/pi), cosi' le tabelle in theta_p
    restano indipendenti da En.

    Args:
        theta_p: angolo di rinculo in lab, rad, forma qualsiasi.
        En: energia cinetica del neutrone, MeV, forma compatibile (broadcasting).

    Ritorna:
        log-rapporto adimensionale, forma del broadcasting; 0 per En -> 0,
        |.| <~ 0.011 a 20 MeV.
    """
    two_m = 2.0 * PROTON_MASS_MEV
    return np.log(two_m * (two_m + En)) - 2.0 * np.log(two_m + En * np.sin(theta_p) ** 2)


def log_recoil_direction_density(theta_p: np.ndarray, En: np.ndarray) -> np.ndarray:
    """log della densita' per steradiante della direzione del protone attorno a
    Omega_n per scattering isotropo in CM, relativistica a masse uguali:
    p = 4 m c (2 m + En) / (2 pi (2 m + En - En c^2)^2), c = cos(theta_p).

    Args:
        theta_p: angolo di rinculo in lab, rad, forma qualsiasi.
        En: energia cinetica del neutrone, MeV, forma compatibile (broadcasting).

    Ritorna:
        log-densita' per steradiante, forma del broadcasting; -inf per
        theta_p >= pi/2. Per En << m tende a log(cos(theta_p) / pi).
    """
    # (a) Ep ~ U(0, En) e cambio di variabile Ep -> c (proton_energy), azimut
    # uniforme: p = (dEp/dc) / (2 pi En)
    with np.errstate(divide="ignore"):
        return (np.log(np.clip(np.cos(theta_p), 0.0, None) / np.pi)
                + log_recoil_density_ratio(theta_p, En))


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


def sample_cm_angle_anisotropic(rng: np.random.Generator, En: np.ndarray, energies: np.ndarray,
                                coeffs: np.ndarray) -> np.ndarray:
    """Angolo polare di scattering del neutrone nel CM da una distribuzione
    angolare in serie di Legendre dipendente dall'energia:
    f(mu | En) = sum_l (2l+1)/2 a_l(En) P_l(mu), a_0 = 1, mu = cos(theta_CM).

    Args:
        rng: generatore numpy (un numero uniforme per evento, come sample_cm_angle).
        En: energia cinetica del neutrone incidente per evento, MeV, forma (n,).
        energies: energie della tabella dei coefficienti, MeV, crescenti, forma (n_E,).
        coeffs: coefficienti (a_1, ..., a_L) a ciascuna energia, forma (n_E, L);
            interpolati linearmente in En, costanti fuori da [energies[0], energies[-1]].

    Ritorna:
        array (n,), theta_CM in rad, in [0, pi]; con coeffs nulli coincide
        (a meno dell'arrotondamento) con l'isotropo di sample_cm_angle.
    """
    a = np.stack([np.interp(En, energies, coeffs[:, l]) for l in range(coeffs.shape[1])])
    degree = np.arange(coeffs.shape[1] + 1)[:, None]
    density = (2 * degree + 1) / 2 * np.concatenate([np.ones((1, En.shape[0])), a])
    cdf = legendre.legint(density, lbnd=-1.0, axis=0)
    u = rng.uniform(0.0, 1.0, En.shape[0])
    # inversione della CDF con Newton per evento, vettorizzata: f > 0 e lontana da
    # zero (scostamento dall'isotropia di pochi %), quindi il passo e' ben definito
    mu = 2.0 * u - 1.0
    for _ in range(N_CM_NEWTON):
        mu -= (legendre.legval(mu, cdf, tensor=False) - u) / legendre.legval(mu, density, tensor=False)
        np.clip(mu, -1.0, 1.0, out=mu)
    return np.arccos(mu)


def recoil_angle_from_cm(theta_cm: np.ndarray, En: np.ndarray) -> np.ndarray:
    """Angolo del protone di rinculo in lab dall'angolo del neutrone in CM,
    relativistico a masse uguali (per En << m: theta_p = (pi - theta_CM) / 2).

    Args:
        theta_cm: angolo di scattering del neutrone nel CM, rad, forma (n,).
        En: energia cinetica del neutrone incidente, MeV, forma (n,) o scalare.

    Ritorna:
        array (n,), theta_p in rad, in [0, pi/2].
    """
    # (a) Ep = En (1 - cos theta_CM) / 2 vale anche relativisticamente (energia
    # trasferita lineare nel quadrato del momento trasferito); l'angolo in lab
    # segue dall'inversa di proton_energy. Con cos(theta_CM) uniforme Ep ~ U(0, En).
    Ep = 0.5 * En * (1.0 - np.cos(theta_cm))
    cos2 = np.clip(recoil_cos2_from_energy(Ep, En), 0.0, 1.0)
    return np.arccos(np.sqrt(cos2))


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


def sample_recoil_angle(rng: np.random.Generator, En: np.ndarray,
                        legendre_table: tuple[np.ndarray, np.ndarray] | None = None
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Energia e angolo di rinculo veri per scattering n-p in CM, isotropo o
    anisotropo, rispetto alla direzione incidente (Caso A: direzione nota).

    Args:
        rng: generatore numpy.
        En: energia vera del neutrone per evento, MeV, forma (n,).
        legendre_table: None = isotropo in CM (sample_cm_angle); altrimenti
            (energies, coeffs) di sample_cm_angle_anisotropic.

    Ritorna:
        (Ep_true, theta_p): energia del protone in MeV e angolo di rinculo in
        lab in rad, in [0, pi/2], entrambi forma (n,).
    """
    theta_cm = (sample_cm_angle(rng, En.shape[0]) if legendre_table is None
                else sample_cm_angle_anisotropic(rng, En, *legendre_table))
    theta_p = recoil_angle_from_cm(theta_cm, En)
    return proton_energy(En, theta_p), theta_p


def sample_recoil_events(rng: np.random.Generator, En: np.ndarray, omega_n_hat: np.ndarray,
                         legendre_table: tuple[np.ndarray, np.ndarray] | None = None
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Eventi di rinculo veri per scattering n-p in CM, isotropo o anisotropo,
    con neutroni incidenti lungo omega_n_hat (Caso B/C).

    Args:
        rng: generatore numpy.
        En: energia vera del neutrone per evento, MeV, forma (n,).
        omega_n_hat: direzione del neutrone incidente, versore, forma (3,)
            (comune a tutti gli eventi) o (n, 3) (una per evento).
        legendre_table: None = isotropo in CM (sample_cm_angle); altrimenti
            (energies, coeffs) di sample_cm_angle_anisotropic.

    Ritorna:
        (Ep_true, track_true): energia del protone in MeV, forma (n,), e
        direzione vera della traccia, versori, forma (n, 3), sempre con
        theta_p <= pi/2 rispetto a omega_n_hat.
    """
    n = En.shape[0]
    Ep_true, theta_p = sample_recoil_angle(rng, En, legendre_table)
    phi = rng.uniform(0.0, 2 * np.pi, n)
    axis = np.broadcast_to(omega_n_hat, (n, 3))
    return Ep_true, direction_around_axis(axis, theta_p, phi)


def mixed_source_axes(rng: np.random.Generator, omega_n_hat: np.ndarray, n_events: int,
                      fraction: float, separation: float | None) -> np.ndarray:
    """Direzioni incidenti per evento con una frazione di eventi che non viene
    dalla sorgente principale (stress test dell'ipotesi di sorgente unica).

    Args:
        rng: generatore numpy (usato solo per la geometria della contaminazione).
        omega_n_hat: direzione della sorgente principale, versore, forma (3,).
        n_events: numero totale di eventi.
        fraction: frazione di eventi contaminanti, in [0, 1]; il loro numero e'
            round(fraction * n_events).
        separation: angolo fra la sorgente principale e una seconda sorgente
            puntiforme (azimut casuale), rad; None = fondo isotropo, una
            direzione indipendente e uniforme sulla sfera per ogni contaminante.

    Ritorna:
        array (n_events, 3), versori: i primi n_events - n_c uguali a
        omega_n_hat, gli ultimi n_c dai contaminanti.
    """
    n_contaminants = int(round(fraction * n_events))
    if separation is None:
        theta = np.arccos(rng.uniform(-1.0, 1.0, n_contaminants))
        phi = rng.uniform(0.0, 2 * np.pi, n_contaminants)
        others = direction_from_theta_phi(theta, phi)
    else:
        second = direction_around_axis(omega_n_hat[None, :], np.array([separation]),
                                       rng.uniform(0.0, 2 * np.pi, 1))
        others = np.broadcast_to(second, (n_contaminants, 3))
    main = np.broadcast_to(omega_n_hat, (n_events - n_contaminants, 3))
    return np.concatenate([main, others])


def sample_energy_spectrum(rng: np.random.Generator, shape: str, mean: float,
                           sd: float, n: int) -> np.ndarray:
    """Energie vere dei neutroni da uno spettro di forma data, con media e
    deviazione standard fissate (stress test dell'ipotesi di spettro gaussiano).

    Args:
        rng: generatore numpy.
        shape: "gauss", "uniform", "bimodal", "lognormal", "laplace" o "student_t"
            (le ultime due a code pesanti, curtosi 6 e 3 + 6 / (STUDENT_T_DOF - 4);
            come la lognormale non sono troncate al dominio di energia).
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
    if shape == "laplace":
        return rng.laplace(mean, sd / np.sqrt(2.0), n)  # varianza = 2 * scala^2
    if shape == "student_t":
        # varianza della t standard = dof / (dof - 2)
        return mean + sd * np.sqrt((STUDENT_T_DOF - 2) / STUDENT_T_DOF) * rng.standard_t(STUDENT_T_DOF, n)
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


def direction_from_tangent(axis: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Versori a coordinate (a, b) nel piano tangente in axis, con la mappa
    esponenziale: angolo sqrt(a**2 + b**2) da axis lungo a*e1 + b*e2, con (e1, e2)
    di perpendicular_basis. Coordinate locali senza singolarita' attorno a una
    stima puntuale, per derivate finite sulla sfera.

    Args:
        axis: versore di riferimento, forma (3,).
        offsets: coordinate tangenti (a, b), rad, forma (n, 2).

    Ritorna:
        array (n, 3), versori unitari; offsets nulli danno axis.
    """
    radius = np.hypot(offsets[:, 0], offsets[:, 1])
    azimuth = np.arctan2(offsets[:, 1], offsets[:, 0])
    return direction_around_axis(np.broadcast_to(axis, (offsets.shape[0], 3)), radius, azimuth)
