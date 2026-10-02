"""Prior, tutti propri (normalizzati su un dominio finito): un prior improprio non
garantisce un posterior normalizzabile. E_n piatto su [EN_MIN, EN_MAX], direzione
uniforme sulla sfera, E_n | (mu_E, sigma_E) gaussiana troncata, iperparametri
uniformi in mu_E e in log sigma_E (sigma_E e' un parametro di scala).
Le prior alternative (uniforme in sigma_E, log-uniforme su E_n, von Mises-Fisher
su Omega_n) servono solo ai test di robustezza al prior."""
import numpy as np
from scipy.integrate import trapezoid
from scipy.special import logsumexp


def energy_prior(en_grid: np.ndarray) -> np.ndarray:
    """Prior su En: piatta ma propria, limitata al dominio di en_grid
    (caso di riferimento R1: prior piatto). Mai flat impropria su un
    dominio illimitato: qui e' propria perche' en_grid ha
    estremi finiti.

    Args:
        en_grid: griglia di ipotesi su En, MeV, forma (n,).

    Ritorna:
        array (n,), densita' di probabilita' costante; l'integrale
        trapezoidale su en_grid vale 1 (scipy.integrate.trapezoid: np.trapz e' deprecato in NumPy 2).
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
    E_n^(k) ~ N(mu_E, sigma_E)). Mai flat impropria (un prior improprio non
    da' un posterior normalizzabile garantito):
    la normalizzazione e' sul dominio finito di en_grid, non su
    tutta la retta reale, quindi resta propria anche per sigma_E grande
    rispetto al dominio (limite Caso B).

    Args:
        en_grid: griglia di ipotesi su En, MeV, forma (n_En,).
        mu_E: media dell'iperprior, MeV, forma (n_hyper,) o scalare.
        sigma_E: deviazione standard dell'iperprior, MeV, stessa forma di mu_E.

    Ritorna:
        array (n_hyper, n_En) se mu_E/sigma_E hanno forma (n_hyper,),
        altrimenti (n_En,); l'integrale trapezoidale su en_grid vale 1 per
        ogni riga (scipy.integrate.trapezoid: np.trapz e' deprecato in NumPy 2).
    """
    scalar_input = np.ndim(mu_E) == 0
    mu = np.atleast_1d(mu_E)[:, None]
    sigma = np.atleast_1d(sigma_E)[:, None]
    en = en_grid[None, :]
    unnorm = np.exp(-0.5 * ((en - mu) / sigma) ** 2)
    norm = trapezoid(unnorm, en_grid, axis=1)[:, None]
    density = unnorm / norm
    return density[0] if scalar_input else density


def log_energy_prior_given_hyperparams(en_grid: np.ndarray, mu_E: np.ndarray,
                                        sigma_E: np.ndarray) -> np.ndarray:
    """log pi(En | mu_E, sigma_E), stessa densita' di
    energy_prior_given_hyperparams ma calcolata in log: a sigma_E piccolo
    la gaussiana va in underflow a 0 lontano da mu_E e np.log(0) darebbe
    -inf con RuntimeWarning; qui resta finita.

    Args:
        en_grid: griglia di ipotesi su En, MeV, forma (n_En,), uniforme.
        mu_E: media dell'iperprior, MeV, forma (n_hyper,).
        sigma_E: deviazione standard dell'iperprior, MeV, forma (n_hyper,).

    Ritorna:
        array (n_hyper, n_En), log-densita' (1/MeV); exp integra a 1 con
        la regola trapezoidale su en_grid.
    """
    log_unnorm = -0.5 * ((en_grid[None, :] - mu_E[:, None]) / sigma_E[:, None]) ** 2
    # log dei pesi trapezoidali (griglia uniforme): log int exp(log_unnorm)
    log_weight = np.full(en_grid.shape, np.log(en_grid[1] - en_grid[0]))
    log_weight[[0, -1]] -= np.log(2.0)
    log_norm = logsumexp(log_unnorm + log_weight[None, :], axis=1, keepdims=True)
    return log_unnorm - log_norm


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


def hyperparameter_prior_uniform_sigma(mu_grid: np.ndarray, sigma_grid: np.ndarray) -> np.ndarray:
    """Prior alternativa su (mu_E, sigma_E), uniforme in (mu_E, sigma_E)
    invece che in (mu_E, log sigma_E): serve solo al test di robustezza al
    prior (ultimo punto della checklist di validazione, riga 14). Su una griglia
    con sigma_E spaziata logaritmicamente la cella ha larghezza ~sigma_E,
    quindi il peso per punto e' proporzionale a sigma_E.

    Args:
        mu_grid: ipotesi su mu_E, MeV, forma (n_hyper,).
        sigma_grid: ipotesi su sigma_E, MeV, stessa forma, spaziata
            logaritmicamente (grids.hyperparameter_grid[_window]).

    Ritorna:
        array (n_hyper,), pesi proporzionali a sigma_E che sommano a 1.
    """
    return sigma_grid / sigma_grid.sum()


def energy_prior_log_uniform(en_grid: np.ndarray) -> np.ndarray:
    """Prior alternativa su En, log-uniforme (densita' ~ 1/En) e propria sul
    dominio di en_grid: serve solo al test di robustezza al prior su E_n.
    Pesa di piu' le energie basse, dove a parita' di Ep l'angolo di rinculo
    e' piu' piccolo.

    Args:
        en_grid: griglia di ipotesi su En, MeV, forma (n,), En > 0.

    Ritorna:
        array (n,), densita' di probabilita' (1/MeV) proporzionale a 1/En;
        l'integrale trapezoidale su en_grid vale 1.
    """
    unnorm = 1.0 / en_grid
    return unnorm / trapezoid(unnorm, en_grid)


def direction_prior_von_mises_fisher(theta_grid: np.ndarray, phi_grid: np.ndarray,
                                     theta_axis: float, phi_axis: float,
                                     kappa: float) -> np.ndarray:
    """Prior alternativa su Omega_n, von Mises-Fisher di asse (theta_axis,
    phi_axis) e concentrazione kappa, propria sulla sfera: serve solo al test
    di robustezza al prior sulla direzione (kappa = 0 ridà la prior uniforme).

    Args:
        theta_grid: colatitudine dei pixel, rad, forma (n_pixel,), pixel ad
            area solida ~costante (grids.sphere_grid o calotta di posterior_C).
        phi_grid: azimut dei pixel, rad, stessa forma di theta_grid.
        theta_axis: colatitudine dell'asse della prior, rad.
        phi_axis: azimut dell'asse della prior, rad.
        kappa: concentrazione, >= 0 (larghezza angolare ~1/sqrt(kappa) rad).

    Ritorna:
        array (n_pixel,), pesi proporzionali a exp(kappa * cos(angolo dall'asse))
        che sommano a 1 (pixel ad area uguale: peso per pixel = densita').
    """
    cos_angle = (np.cos(theta_grid) * np.cos(theta_axis)
                 + np.sin(theta_grid) * np.sin(theta_axis) * np.cos(phi_grid - phi_axis))
    # exp(kappa * (cos - 1)) invece di exp(kappa * cos): niente overflow a kappa grande
    unnorm = np.exp(kappa * (cos_angle - 1.0))
    return unnorm / unnorm.sum()
