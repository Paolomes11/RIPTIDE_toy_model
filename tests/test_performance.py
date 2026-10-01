import time
from typing import Callable

import numpy as np
import pytest
from scipy.stats import norm

from riptide_toy import combine, forward_model, grids, kinematics, posterior_A, posterior_B, priors, validate
from riptide_toy.constants import SEED, SIGMA_EP, SIGMA_THETA

# Soglie = obiettivi di tempo fissati per ogni funzione del path caldo (elenco
# in docs/roadmap.md). Test di regressione: avvisano se si reintroduce un ciclo
# Python lento. Esclusi dalla CI e dal pre-commit (dipendono dalla macchina),
# da lanciare prima di ogni PR.


def best_time(fn: Callable[[], object], repeat: int = 3) -> float:
    """Miglior tempo di esecuzione su `repeat` chiamate, dopo un giro di
    riscaldamento (import, cache di grids, allocazioni).

    Args:
        fn: funzione senza argomenti da cronometrare.
        repeat: numero di chiamate cronometrate.

    Ritorna:
        float, secondi (minimo sulle ripetizioni, meno sensibile al rumore
        della macchina della media).
    """
    fn()
    elapsed = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        elapsed.append(time.perf_counter() - start)
    return min(elapsed)


def case_B_events(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Eventi sintetici Caso B (generatore fisico + risoluzioni), Omega_n = z.

    Ritorna:
        (Ep_hat, track_hat): MeV forma (n,), versori forma (n, 3).
    """
    rng = np.random.default_rng(SEED)
    Ep_true, track = kinematics.sample_recoil_events(
        rng, rng.uniform(1.0, 5.0, n), np.array([0.0, 0.0, 1.0])
    )
    return rng.normal(Ep_true, SIGMA_EP), kinematics.smear_direction(rng, track, SIGMA_THETA)


def case_A_events(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Osservabili sintetici Caso A (Ep_hat, theta_p_hat), En ~ U(1, 5).

    Ritorna:
        (Ep_hat, theta_p_hat): MeV e rad, ciascuno forma (n,).
    """
    rng = np.random.default_rng(SEED)
    theta_p = rng.uniform(0.0, np.pi / 2, n)
    Ep_true = kinematics.proton_energy(rng.uniform(1.0, 5.0, n), theta_p)
    return forward_model.measure(Ep_true, theta_p, SIGMA_EP, SIGMA_THETA, rng)


def test_sample_recoil_events_1e5_under_100ms():
    rng = np.random.default_rng(SEED)
    En = rng.uniform(1.0, 5.0, 100_000)
    omega = np.array([0.0, 0.0, 1.0])
    assert best_time(lambda: kinematics.sample_recoil_events(rng, En, omega)) < 0.1


def test_combine_loglik_20000_events_under_200ms():
    loglik = np.random.default_rng(SEED).normal(size=(20_000, 500))
    assert best_time(lambda: combine.combine_loglik(loglik, np.zeros(500))) < 0.2


def test_run_checklist_20000_events_under_1_minute():
    # scala del caso di riferimento R3; ricostruzione analitica (quella di R3), per
    # misurare il costo di validate e non quello del motore del Caso A.
    rng = np.random.default_rng(SEED)
    truth = rng.uniform(1.0, 5.0, 20_000)

    def reconstruction(t: np.ndarray, rng: np.random.Generator):
        sigma = 0.08 * t
        estimate = t + rng.normal(0.0, sigma)
        half = norm.ppf(0.84) * sigma
        return estimate, sigma, np.stack([estimate - half, estimate + half], axis=-1)[:, None, :]

    elapsed = best_time(lambda: validate.run_checklist(
        reconstruction, truth, levels=np.array([0.68]), rng=rng
    ), repeat=1)
    assert elapsed < 60.0


@pytest.mark.parametrize("n_pixel", [3000, 10_000])
def test_posterior_B_one_event_under_100ms(n_pixel: int):
    Ep_hat, track_hat = case_B_events(1)
    theta_grid, phi_grid = grids.sphere_grid(n_pixel)
    prior = priors.direction_prior(theta_grid, phi_grid)
    elapsed = best_time(lambda: posterior_B.single_event_posterior(
        (Ep_hat, track_hat), (theta_grid, phi_grid), prior
    ))
    assert elapsed < 0.1


def test_combine_case_B_100_events_under_5s():
    Ep_hat, track_hat = case_B_events(100)
    theta_grid, phi_grid = grids.sphere_grid(3000)
    prior = priors.direction_prior(theta_grid, phi_grid)

    def run() -> np.ndarray:
        logpost = posterior_B.single_event_posterior((Ep_hat, track_hat), (theta_grid, phi_grid), prior)
        return combine.combine_loglik(logpost - np.log(prior)[None, :], np.log(prior))

    assert best_time(run, repeat=1) < 5.0


def test_posterior_A_one_event_under_1ms():
    en_grid = grids.energy_grid()
    prior = priors.energy_prior(en_grid)
    D = case_A_events(1)
    assert best_time(lambda: posterior_A.single_event_posterior(D, en_grid, prior)) < 1e-3


# Target ridefinito (R4, docs/roadmap.md): l'obiettivo iniziale era 50 ms, ma la
# marginalizzazione esatta su theta_p tocca 1000 x 500 x 500 = 2.5e8 celle,
# ~1.0-1.2 s con numpy denso float32 (c). Una finestra su theta_p non e'
# esatta (differenze di loglik fino a 18.6 nelle code), numba non adottato.
def test_posterior_A_1000_events_under_2s():
    en_grid = grids.energy_grid()
    prior = priors.energy_prior(en_grid)
    D = case_A_events(1000)
    assert best_time(lambda: posterior_A.single_event_posterior(D, en_grid, prior), repeat=1) < 2.0
