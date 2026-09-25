import numpy as np
import pytest

from riptide_toy import (combine, forward_model, grids, kinematics, posterior_B, posterior_C,
                         priors)
from riptide_toy.constants import EN_MAX, EN_MIN, SEED, SIGMA_EP, SIGMA_THETA


def test_hyperparameter_grid_shape_and_domain():
    mu_grid, sigma_grid = grids.hyperparameter_grid(10, 8)
    assert mu_grid.shape == (80,)
    assert sigma_grid.shape == (80,)
    assert mu_grid.min() >= EN_MIN and mu_grid.max() <= EN_MAX
    assert sigma_grid.min() > 0.0
    # ravel di meshgrid(indexing="ij"): reshape(n_mu, n_sigma) allinea
    # l'asse 0 a mu, l'asse 1 a sigma (crescente, spaziatura log).
    sigma_2d = sigma_grid.reshape(10, 8)
    np.testing.assert_allclose(sigma_2d[0], sigma_2d[-1])
    assert np.all(np.diff(sigma_2d[0]) > 0.0)


def test_hyperparameter_prior_proper_and_sums_to_one():
    mu_grid, sigma_grid = grids.hyperparameter_grid(10, 10)
    p = priors.hyperparameter_prior(mu_grid, sigma_grid)
    assert p.shape == mu_grid.shape
    assert np.all(p > 0.0)
    assert abs(p.sum() - 1.0) < 1e-10


def test_energy_prior_given_hyperparams_integrates_to_one():
    en_grid = grids.energy_grid(400)
    mu_grid, sigma_grid = grids.hyperparameter_grid(6, 6)
    density = priors.energy_prior_given_hyperparams(en_grid, mu_grid, sigma_grid)
    assert density.shape == (36, 400)

    from scipy.integrate import trapezoid
    integrals = trapezoid(density, en_grid, axis=1)
    np.testing.assert_allclose(integrals, 1.0, atol=1e-6)


def test_log_energy_prior_given_hyperparams_matches_and_stays_finite_at_small_sigma():
    en_grid = grids.energy_grid(400)
    mu_grid, sigma_grid = grids.hyperparameter_grid(6, 6)
    log_density = priors.log_energy_prior_given_hyperparams(en_grid, mu_grid, sigma_grid)
    density = priors.energy_prior_given_hyperparams(en_grid, mu_grid, sigma_grid)
    np.testing.assert_allclose(np.exp(log_density), density, rtol=1e-10, atol=1e-300)

    # sigma_E piccolo: la densita' lineare va a 0 (log -> -inf con warning),
    # la versione log resta finita su tutta la griglia.
    tiny = priors.log_energy_prior_given_hyperparams(en_grid, np.array([3.0]), np.array([0.01]))
    assert np.all(np.isfinite(tiny))


def test_hierarchical_track_loglik_sigma_E_to_infinity_matches_case_B():
    # Con risoluzione angolare: la versione gerarchica (Caso C) a sigma_E
    # enorme deve coincidere con loglik_marginal_En_theta a prior piatto
    # (Caso B) sullo stesso evento e la stessa direzione.
    rng = np.random.default_rng(SEED)
    n_events = 15
    en_grid = grids.energy_grid(300)
    omega = np.array([0.0, 0.0, 1.0])
    Ep_true, track = kinematics.sample_recoil_events(rng, rng.uniform(EN_MIN, EN_MAX, n_events), omega)
    track_hat = kinematics.smear_direction(rng, track, SIGMA_THETA)
    Ep_hat = rng.normal(Ep_true, SIGMA_EP)
    theta_obs = kinematics.recoil_angle_from_direction(track_hat, omega[None, :])[:, 0]

    log_prior_hier = priors.log_energy_prior_given_hyperparams(
        en_grid, np.array([(EN_MIN + EN_MAX) / 2.0]), np.array([1000.0])
    )
    hier = forward_model.loglik_marginal_En_theta_hierarchical(
        Ep_hat, theta_obs, en_grid, SIGMA_EP, SIGMA_THETA, log_prior_hier
    )
    flat = forward_model.loglik_marginal_En_theta(
        Ep_hat, theta_obs[:, None], en_grid, SIGMA_EP, SIGMA_THETA,
        np.log(priors.energy_prior(en_grid)),
    )
    assert hier.shape == flat.shape == (n_events, 1)
    np.testing.assert_allclose(hier, flat, atol=0.01)


def test_sigma_E_to_infinity_reduces_to_case_B():
    # riga 13 (CLAUDE.md Sez. 5, test di limite): sigma_E -> infinito => Caso B.
    # Con sigma_E grande la gaussiana troncata su en_grid tende alla prior
    # piatta di priors.energy_prior (limite gia' verificato in REPL, Fase 3):
    # loglik_marginal_En_hierarchical con quel prior deve coincidere con
    # forward_model.loglik_marginal_En a prior piatto, sullo stesso evento.
    rng = np.random.default_rng(SEED)
    n_events = 15
    en_grid = grids.energy_grid(300)

    theta_p = rng.uniform(0.0, np.pi / 2 - 0.05, n_events)
    en_true = rng.uniform(EN_MIN, EN_MAX, n_events)
    Ep_hat = kinematics.proton_energy(en_true, theta_p) + rng.normal(0.0, SIGMA_EP, n_events)

    sigma_E_huge = np.array([1000.0])
    mu_E_mid = np.array([(EN_MIN + EN_MAX) / 2.0])
    log_prior_hier = priors.log_energy_prior_given_hyperparams(en_grid, mu_E_mid, sigma_E_huge)
    loglik_hier = forward_model.loglik_marginal_En_hierarchical(
        Ep_hat, theta_p, en_grid, SIGMA_EP, log_prior_hier
    )
    assert loglik_hier.shape == (n_events, 1)

    log_prior_flat = np.log(priors.energy_prior(en_grid))
    loglik_flat = forward_model.loglik_marginal_En(
        Ep_hat, theta_p[:, None], en_grid, SIGMA_EP, log_prior_flat
    )[:, 0]

    np.testing.assert_allclose(loglik_hier[:, 0], loglik_flat, atol=0.01)


def test_sigma_E_to_zero_recovers_shared_true_energy():
    # riga 13, secondo test di limite: sigma_E -> 0 => Caso A (En condiviso e
    # quasi fisso). N eventi mono-energetici (stesso En_true per tutti):
    # il MAP di mu_E sulla colonna a sigma_E piu' piccolo deve avvicinarsi a
    # En_true (gia' verificato in REPL: errore assoluto 0.042 MeV su N=40).
    rng = np.random.default_rng(SEED)
    n_events = 40
    en_true_shared = 2.5

    theta_p = rng.uniform(0.0, np.pi / 2 - 0.05, n_events)
    Ep_hat = kinematics.proton_energy(
        np.full(n_events, en_true_shared), theta_p
    ) + rng.normal(0.0, SIGMA_EP, n_events)

    en_grid = grids.energy_grid(300)
    mu_grid, sigma_grid = grids.hyperparameter_grid(60, 60)
    hprior = priors.hyperparameter_prior(mu_grid, sigma_grid)

    log_prior_hier = priors.log_energy_prior_given_hyperparams(en_grid, mu_grid, sigma_grid)
    loglik = forward_model.loglik_marginal_En_hierarchical(
        Ep_hat, theta_p, en_grid, SIGMA_EP, log_prior_hier
    )
    combined = combine.combine_loglik(loglik, np.log(hprior)).reshape(60, 60)

    map_mu_idx = np.argmax(combined[:, 0])
    mu_1d = mu_grid.reshape(60, 60)[:, 0]
    map_mu_E = mu_1d[map_mu_idx]

    assert abs(map_mu_E - en_true_shared) < 0.1


def test_posterior_C_end_to_end_direction_and_hyperparams():
    # Integrazione riga 13: estimate_shared_direction (stadio 1, riuso Caso B)
    # seguito da single_event_posterior (stadio 2, griglia 2D su mu_E,sigma_E),
    # su un dataset sintetico con Omega_n e (mu_E, sigma_E) noti. Tolleranze
    # larghe: singola realizzazione MC con N modesto, non una media d'insieme.
    rng = np.random.default_rng(SEED)
    n_events = 60
    true_theta, true_phi = 0.9, 2.1
    omega_n_true = kinematics.direction_from_theta_phi(
        np.array([true_theta]), np.array([true_phi])
    )[0]
    mu_E_true, sigma_E_true = 3.0, 0.3

    # generatore fisico isotropo in CM + risoluzioni SIGMA_EP, SIGMA_THETA
    # (R3; prima: tracce isotrope in lab tagliate a theta_p <= pi/2, senza
    # risoluzione angolare -- regime del "pixel a 53 gradi" del report)
    En_k = rng.normal(mu_E_true, sigma_E_true, n_events)
    Ep_true, track_true = kinematics.sample_recoil_events(rng, En_k, omega_n_true)
    track_hat = kinematics.smear_direction(rng, track_true, SIGMA_THETA)
    Ep_hat = rng.normal(Ep_true, SIGMA_EP)

    theta_grid, phi_grid = grids.sphere_grid()
    dprior = priors.direction_prior(theta_grid, phi_grid)
    omega_n_hat = posterior_C.estimate_shared_direction(
        (Ep_hat, track_hat), theta_grid, phi_grid, dprior
    )
    assert omega_n_hat.shape == (1, 3)
    angular_error = np.arccos(np.clip(omega_n_hat[0] @ omega_n_true, -1.0, 1.0))
    assert angular_error < np.deg2rad(15.0)

    mu_grid, sigma_grid = grids.hyperparameter_grid(50, 50)
    hprior = priors.hyperparameter_prior(mu_grid, sigma_grid)
    loglik = posterior_C.single_event_posterior(
        (Ep_hat, track_hat, omega_n_hat), (mu_grid, sigma_grid), hprior
    )
    assert loglik.shape == (len(Ep_hat), mu_grid.shape[0])
    assert np.isfinite(loglik).all()

    combined = combine.combine_loglik(loglik, np.log(hprior))
    best = np.argmax(combined)
    assert abs(mu_grid[best] - mu_E_true) < 0.5


def test_estimate_shared_direction_raises_when_combined_is_all_neginf(monkeypatch):
    # R1 (docs/report_caso_C_stadio1.md): se il combinato e' -inf su tutti i
    # candidati, np.argmax restituiva in silenzio il pixel 0 (polo nord della
    # griglia, 52-53 gradi dalla verita' nel report). Con la risoluzione
    # angolare (R3) posterior_B non produce piu' -inf: la guardia si testa
    # forzando il Caso B a -inf ovunque.
    theta_grid, phi_grid = grids.sphere_grid(800)
    dprior = priors.direction_prior(theta_grid, phi_grid)
    monkeypatch.setattr(
        posterior_B, "single_event_posterior",
        lambda D, grid, prior: np.full((D[0].shape[0], grid[0].shape[0]), -np.inf),
    )
    with pytest.raises(ValueError, match="-inf su tutti i candidati"):
        posterior_C.estimate_shared_direction(
            (np.full(4, 1.0), np.eye(3)[[0, 1, 2, 2]]), theta_grid, phi_grid, dprior
        )


def test_estimate_shared_direction_finite_when_no_direction_has_all_tracks_forward():
    # Il caso che col taglio netto dava -inf ovunque (R1): 4 tracce ai vertici
    # di un tetraedro regolare, che sommano a zero, quindi nessun Omega_n ha
    # t_k . Omega_n >= 0 per tutte. Con la risoluzione angolare il combinato
    # e' finito e lo stadio 1 restituisce un versore, senza errore.
    track_hat = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0],
                          [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]) / np.sqrt(3.0)
    Ep_hat = np.full(4, 1.0)

    theta_grid, phi_grid = grids.sphere_grid(800)
    dprior = priors.direction_prior(theta_grid, phi_grid)
    omega_n_hat = posterior_C.estimate_shared_direction(
        (Ep_hat, track_hat), theta_grid, phi_grid, dprior
    )
    assert omega_n_hat.shape == (1, 3)
    np.testing.assert_allclose(np.linalg.norm(omega_n_hat), 1.0)
