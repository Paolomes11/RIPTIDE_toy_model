import numpy as np
from functools import lru_cache

from riptide_toy.constants import EN_MAX, EN_MIN, SIGMA_E_MAX, SIGMA_E_MIN

# (a) errata guida: maxsize=1 svuota la cache se si chiama la funzione con
# argomenti diversi (es. sphere_grid(500) poi sphere_grid()); gli array
# cachati vanno resi non scrivibili perche' sono condivisi tra i chiamanti.
@lru_cache(maxsize=None)
def energy_grid(n: int = 500) -> np.ndarray:
    """Griglia di ipotesi su En, equispaziata su [EN_MIN, EN_MAX].

    Args:
        n: numero di punti.

    Ritorna:
        array (n,), En in MeV, crescente, non scrivibile.
    """
    grid = np.linspace(EN_MIN, EN_MAX, n)
    grid.flags.writeable = False
    return grid

# (c) errata: theta e phi erano due linspace indipendenti (stessa lunghezza
# n_pixel ma nessun accoppiamento), quindi l'indice i non corrispondeva a una
# singola direzione sulla sfera. Fix: reticolo di Fibonacci (angolo aureo),
# un indice = una direzione, area solida quasi costante per pixel.
@lru_cache(maxsize=None)
def sphere_grid(n_pixel: int = 3000) -> tuple[np.ndarray, np.ndarray]:
    """Direzioni sulla sfera unitaria, reticolo di Fibonacci.

    Args:
        n_pixel: numero di direzioni.

    Ritorna:
        (theta, phi), due array (n_pixel,) in rad, non scrivibili; l'indice i
        e' una direzione. Il pixel 0 sta a theta ~ sqrt(2/n_pixel) dal polo.
    """
    i = np.arange(n_pixel)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    theta = np.arccos(1.0 - 2.0 * (i + 0.5) / n_pixel)
    phi = np.mod(i * golden_angle, 2 * np.pi)
    theta.flags.writeable = False
    phi.flags.writeable = False
    return (theta, phi)

@lru_cache(maxsize=None)
def theta_p_grid(n: int = 500) -> np.ndarray:
    """Griglia sull'angolo di rinculo in lab, [0, pi/2] (theta_lab <= 90 deg,
    CLAUDE.md Sez. 3); usata per marginalizzare la nuisance theta_p in posterior_A.

    Args:
        n: numero di punti.

    Ritorna:
        array (n,), theta_p in rad, crescente, non scrivibile.
    """
    grid = np.linspace(0.0, np.pi / 2, n)
    grid.flags.writeable = False
    return grid

@lru_cache(maxsize=None)
def hyperparameter_grid(n_mu: int = 60, n_sigma: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Griglia 2D appiattita su (mu_E, sigma_E) per il Caso C (CLAUDE.md Sez. 4: "mai
    griglia 4D bruta", Omega_n va fissato a parte dal Caso B prima di usare questa
    griglia). mu_E copre lo stesso dominio di energy_grid; sigma_E e' spaziata
    logaritmicamente (e' un parametro di scala) per coprire con la stessa griglia
    sia il limite sigma_E->0 (Caso A) sia sigma_E->infinito (Caso B, Sez. 5).

    Nota: meshgrid con indexing="ij" poi ravel (ordine 'C'): l'indice flat
    i*n_sigma + j corrisponde a (mu_grid_1d[i], sigma_grid_1d[j]), quindi un
    array di lunghezza n_mu*n_sigma allineato a questa griglia si puo'
    ri-plasmare con .reshape(n_mu, n_sigma).

    Args:
        n_mu: punti su mu_E.
        n_sigma: punti su sigma_E.

    Ritorna:
        (mu_grid, sigma_grid), MeV, ciascuno forma (n_mu*n_sigma,), non scrivibili.
    """
    mu_1d = np.linspace(EN_MIN, EN_MAX, n_mu)
    sigma_1d = np.geomspace(SIGMA_E_MIN, SIGMA_E_MAX, n_sigma)
    mu_mesh, sigma_mesh = np.meshgrid(mu_1d, sigma_1d, indexing="ij")
    mu_grid = mu_mesh.ravel()
    sigma_grid = sigma_mesh.ravel()
    mu_grid.flags.writeable = False
    sigma_grid.flags.writeable = False
    return (mu_grid, sigma_grid)

def hyperparameter_grid_window(mu_lo: float, mu_hi: float, sigma_lo: float, sigma_hi: float,
                               n_mu: int, n_sigma: int) -> tuple[np.ndarray, np.ndarray]:
    """Griglia (mu_E, sigma_E) ristretta a una finestra, stessa costruzione di
    hyperparameter_grid (mu lineare, sigma logaritmica, ravel "ij"): serve al
    raffinamento locale del Caso C (riga 14), dove la griglia globale ha passo
    piu' largo del posterior. La finestra e' tagliata al dominio globale
    [EN_MIN, EN_MAX] x [SIGMA_E_MIN, SIGMA_E_MAX].

    Args:
        mu_lo, mu_hi: estremi della finestra su mu_E, MeV.
        sigma_lo, sigma_hi: estremi della finestra su sigma_E, MeV, > 0.
        n_mu, n_sigma: punti per asse.

    Ritorna:
        (mu_grid, sigma_grid), MeV, ciascuno forma (n_mu*n_sigma,), non
        scrivibili; reshape(n_mu, n_sigma) allinea l'asse 0 a mu.
    """
    mu_1d = np.linspace(max(mu_lo, EN_MIN), min(mu_hi, EN_MAX), n_mu)
    sigma_1d = np.geomspace(max(sigma_lo, SIGMA_E_MIN), min(sigma_hi, SIGMA_E_MAX), n_sigma)
    mu_mesh, sigma_mesh = np.meshgrid(mu_1d, sigma_1d, indexing="ij")
    mu_grid = mu_mesh.ravel()
    sigma_grid = sigma_mesh.ravel()
    mu_grid.flags.writeable = False
    sigma_grid.flags.writeable = False
    return (mu_grid, sigma_grid)
