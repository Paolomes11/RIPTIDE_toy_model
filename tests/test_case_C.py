import numpy as np
import pytest
from scipy.special import logsumexp

from riptide_toy import (combine, forward_model, grids, kinematics, posterior_B, posterior_C,
                         priors, validate)
from riptide_toy.constants import EN_MAX, EN_MIN, SEED, SIGMA_E_MAX, SIGMA_E_MIN, SIGMA_EP, SIGMA_THETA


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


def test_stage1_error_contracts_at_large_N():
    # Regressione (report §13, punto D): a N=300 lo stadio 1 raffinato deve
    # stare entro pochi gradi. Il vecchio massimo sistematico (pixel 0, ~53
    # gradi) lo violerebbe. Su 100 seed: mediana 1.2, max 3.0 gradi (c).
    rng = np.random.default_rng(SEED)
    n_events = 300
    omega_n_true = kinematics.direction_from_theta_phi(np.array([0.9]), np.array([2.1]))[0]
    En_k = rng.normal(3.0, 0.3, n_events)
    Ep_true, track_true = kinematics.sample_recoil_events(rng, En_k, omega_n_true)
    track_hat = kinematics.smear_direction(rng, track_true, SIGMA_THETA)
    Ep_hat = rng.normal(Ep_true, SIGMA_EP)

    theta_grid, phi_grid = grids.sphere_grid()
    omega_n_hat = posterior_C.refine_shared_direction(
        (Ep_hat, track_hat), theta_grid, phi_grid, priors.direction_prior(theta_grid, phi_grid)
    )[0]
    angular_error = np.arccos(np.clip(omega_n_hat[0] @ omega_n_true, -1.0, 1.0))
    assert angular_error < np.deg2rad(5.0)


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


def test_hyperparameter_grid_window_layout_and_clipping():
    mu_grid, sigma_grid = grids.hyperparameter_grid_window(2.0, 3.0, 0.2, 0.8, 11, 7)
    assert mu_grid.shape == sigma_grid.shape == (77,)
    assert not mu_grid.flags.writeable and not sigma_grid.flags.writeable
    np.testing.assert_allclose(mu_grid.reshape(11, 7)[:, 0], np.linspace(2.0, 3.0, 11))
    np.testing.assert_allclose(sigma_grid.reshape(11, 7)[0], np.geomspace(0.2, 0.8, 7))
    # finestra oltre il dominio globale: tagliata a [EN_MIN, EN_MAX] x [SIGMA_E_MIN, SIGMA_E_MAX]
    mu_wide, sigma_wide = grids.hyperparameter_grid_window(-1.0, 99.0, 1e-5, 1e5, 5, 5)
    assert mu_wide.min() == EN_MIN and mu_wide.max() == EN_MAX
    np.testing.assert_allclose([sigma_wide.min(), sigma_wide.max()], [SIGMA_E_MIN, SIGMA_E_MAX])


def test_hyperparameter_prior_uniform_sigma_weights_proportional_to_sigma():
    mu_grid, sigma_grid = grids.hyperparameter_grid(10, 12)
    p = priors.hyperparameter_prior_uniform_sigma(mu_grid, sigma_grid)
    assert abs(p.sum() - 1.0) < 1e-12
    np.testing.assert_allclose(p / sigma_grid, p[0] / sigma_grid[0])


def test_direction_cap_grid_within_radius_and_equal_area():
    center = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    radius = 0.2
    theta, phi = posterior_C.direction_cap_grid(center, radius, 4000)
    angle = np.arccos(np.clip(kinematics.direction_from_theta_phi(theta, phi) @ center, -1.0, 1.0))
    assert angle.max() <= radius + 1e-12
    # area uguale: 1 - cos(angolo) uniforme su [0, 1 - cos(radius)]
    one_minus_cos = (1.0 - np.cos(angle)) / (1.0 - np.cos(radius))
    np.testing.assert_allclose(np.sort(one_minus_cos), (np.arange(4000) + 0.5) / 4000, atol=1e-9)


def test_direction_cap_radius_not_degenerate_when_map_self_angle_rounds_above_zero():
    # Regressione P2 (report §8): a N=1000 resta un solo pixel sopra soglia; se
    # arccos(best @ best) arrotonda a ~1e-8 invece di 0, il passo non deve
    # diventare quell'angolo (calotta di raggio ~1e-8, sigma dichiarata 0).
    omega_grid = kinematics.direction_from_theta_phi(*grids.sphere_grid())
    # stessa operazione (matrice-vettore) del codice: l'arrotondamento dipende da essa
    map_idx = next(i for i in range(omega_grid.shape[0]) if (omega_grid @ omega_grid[i])[i] < 1.0)
    coarse = np.full(omega_grid.shape[0], -1e3)
    coarse[map_idx] = 0.0
    radius = posterior_C.direction_cap_radius(coarse, omega_grid, 10.0)
    angle = np.arccos(np.clip(omega_grid @ omega_grid[map_idx], -1.0, 1.0))
    assert angle[map_idx] > 0.0  # il caso che rompeva angle > 0
    nearest = np.min(np.delete(angle, map_idx))
    np.testing.assert_allclose(radius, angle[map_idx] + 2.0 * nearest, rtol=1e-12)
    assert radius > np.deg2rad(1.0)


def test_hyperparameter_window_contains_region_above_threshold():
    mu_grid, sigma_grid = grids.hyperparameter_grid(30, 30)
    log_post = -0.5 * ((mu_grid - 3.0) / 0.2) ** 2 - 0.5 * (np.log(sigma_grid / 0.4) / 0.3) ** 2
    mu_lo, mu_hi, sigma_lo, sigma_hi = posterior_C.hyperparameter_window(log_post, mu_grid, sigma_grid, 10.0)
    keep = log_post > log_post.max() - 10.0
    assert mu_lo < mu_grid[keep].min() and mu_hi > mu_grid[keep].max()
    assert sigma_lo < sigma_grid[keep].min() and sigma_hi > sigma_grid[keep].max()
    # fuori dalla finestra la soglia non e' superata
    outside = (mu_grid < mu_lo) | (mu_grid > mu_hi) | (sigma_grid < sigma_lo) | (sigma_grid > sigma_hi)
    assert np.all(log_post[outside] <= log_post.max() - 10.0)


def case_C_dataset(n_events: int, rng: np.random.Generator
                   ) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Dataset sintetico Caso C (generatore fisico + risoluzioni), verita'
    Omega_n = (0.9, 2.1), mu_E = 3.0, sigma_E = 0.4 (setup del report).

    Ritorna:
        ((Ep_hat MeV (n,), track_hat versori (n, 3)), omega_true versore (3,)).
    """
    omega_true = kinematics.direction_from_theta_phi(np.array([0.9]), np.array([2.1]))[0]
    Ep_true, track = kinematics.sample_recoil_events(rng, rng.normal(3.0, 0.4, n_events), omega_true)
    track_hat = kinematics.smear_direction(rng, track, SIGMA_THETA)
    return (rng.normal(Ep_true, SIGMA_EP), track_hat), omega_true


def test_refined_stages_negligible_mass_at_window_edges():
    # Integrazione R6: la calotta e la finestra fine contengono il posterior
    # (massa ai bordi trascurabile) e il MAP raffinato e' vicino alla verita'.
    rng = np.random.default_rng(SEED)
    D_B, omega_true = case_C_dataset(80, rng)
    theta_grid, phi_grid = grids.sphere_grid(1500)
    dprior = priors.direction_prior(theta_grid, phi_grid)
    omega_hat, cap_log_post, cap_theta, cap_phi = posterior_C.refine_shared_direction(
        D_B, theta_grid, phi_grid, dprior, n_cap=2000
    )
    assert omega_hat.shape == (1, 3) and cap_log_post.shape == (2000,)
    # anello esterno della calotta (ultimo 10% dei pixel, cos(alpha) decrescente)
    assert np.max(cap_log_post[-200:]) < cap_log_post.max() - 5.0
    assert np.arccos(np.clip(omega_hat[0] @ omega_true, -1.0, 1.0)) < np.deg2rad(8.0)

    mu_grid, sigma_grid = grids.hyperparameter_grid(30, 30)
    log_post, mu_fine, sigma_fine = posterior_C.refine_hyperparameters(
        (*D_B, omega_hat), mu_grid, sigma_grid, priors.hyperparameter_prior, n_mu=20, n_sigma=20
    )
    assert log_post.shape == mu_fine.shape == sigma_fine.shape == (400,)
    grid_2d = log_post.reshape(20, 20)
    edge = np.concatenate([grid_2d[0], grid_2d[-1], grid_2d[:, 0], grid_2d[:, -1]])
    assert np.max(edge) < log_post.max() - 5.0
    best = np.argmax(log_post)
    assert abs(mu_fine[best] - 3.0) < 0.3
    assert 0.2 < sigma_fine[best] < 0.8


def test_marginalize_En_hierarchical_independent_of_chunk_sizes():
    rng = np.random.default_rng(SEED)
    base = rng.normal(0.0, 3.0, (37, 50))
    log_prior_En_grid = rng.normal(0.0, 3.0, (23, 50))
    reference = forward_model.marginalize_En_hierarchical(base, log_prior_En_grid, 1000, 1000)
    chunked = forward_model.marginalize_En_hierarchical(base, log_prior_En_grid, 7, 5)
    # prodotto di matrici: BLAS somma in ordine diverso a seconda della forma,
    # quindi uguaglianza solo a precisione macchina (non bit a bit)
    np.testing.assert_allclose(chunked, reference, rtol=1e-12, atol=0.0)


def test_log_matmul_exp_matches_logsumexp_including_underflow():
    rng = np.random.default_rng(SEED)
    a = rng.normal(0.0, 3.0, (30, 40))
    b = rng.normal(0.0, 3.0, (20, 40))
    a[0] = -2000.0 + np.arange(40) * 50.0  # righe con dinamica enorme: ripiego logsumexp
    b[1] = -np.inf
    b[1, 5] = -700.0
    expected = logsumexp(a[:, None, :] + b[None, :, :], axis=2)
    np.testing.assert_allclose(forward_model.log_matmul_exp(a, b), expected, rtol=1e-12)


def test_logsumexp_axis_matches_scipy_including_all_neginf_slices():
    rng = np.random.default_rng(SEED)
    x = rng.normal(0.0, 30.0, (6, 7, 9))
    x[0, 2, :] = -np.inf
    x[1, :, 3] = -np.inf
    for axis in range(3):
        np.testing.assert_allclose(forward_model.logsumexp_axis(x, axis), logsumexp(x, axis=axis),
                                   rtol=1e-12)


def test_table_reuse_matches_direct_posteriors():
    # le varianti "from_table"/"from_base" usate da refine_* devono dare lo
    # stesso posterior delle funzioni dirette (ricalcolo completo)
    rng = np.random.default_rng(SEED)
    (Ep_hat, track_hat), omega_true = case_C_dataset(15, rng)
    theta_grid, phi_grid = grids.sphere_grid(500)
    prior = priors.direction_prior(theta_grid, phi_grid)
    direct = posterior_C.shared_direction_log_posterior((Ep_hat, track_hat), theta_grid, phi_grid, prior)
    table = posterior_C.case_B_track_table(Ep_hat)
    reused = posterior_C.direction_log_posterior_from_table(table, track_hat, theta_grid, phi_grid, prior)
    np.testing.assert_allclose(reused, direct, rtol=1e-12)

    mu_grid, sigma_grid = grids.hyperparameter_grid(8, 8)
    hyper_prior = priors.hyperparameter_prior(mu_grid, sigma_grid)
    D = (Ep_hat, track_hat, omega_true[None, :])
    direct = posterior_C.shared_hyperparameter_log_posterior(D, mu_grid, sigma_grid, hyper_prior)
    theta_obs = kinematics.recoil_angle_from_direction(track_hat, omega_true[None, :])[:, 0]
    base = forward_model.hierarchical_base(Ep_hat, theta_obs, grids.energy_grid(), SIGMA_EP, SIGMA_THETA)
    reused = posterior_C.hyperparameter_log_posterior_from_base(base, mu_grid, sigma_grid, hyper_prior)
    np.testing.assert_allclose(reused, direct, rtol=1e-12)


def test_hierarchical_track_table_reduces_to_case_B_when_sigma_E_large():
    # Limite sigma_E -> inf => Caso B: la gaussiana troncata diventa il prior piatto proprio.
    rng = np.random.default_rng(SEED)
    D_B, _ = case_C_dataset(30, rng)
    np.testing.assert_allclose(posterior_C.hierarchical_track_table(D_B[0], 3.0, 1e5),
                               posterior_C.case_B_track_table(D_B[0]), rtol=1e-8, atol=1e-8)


def test_refine_shared_direction_hierarchical_near_truth_and_narrower():
    # Stadio 1 iterato (report §8.2): con il prior gerarchico plug-in il posterior
    # su Omega_n e' piu' stretto di quello col prior largo e resta vicino alla verita'.
    rng = np.random.default_rng(SEED)
    D_B, omega_true = case_C_dataset(80, rng)
    theta_grid, phi_grid = grids.sphere_grid(1500)
    dprior = priors.direction_prior(theta_grid, phi_grid)
    spreads = []
    for result in (posterior_C.refine_shared_direction(D_B, theta_grid, phi_grid, dprior, n_cap=2000),
                   posterior_C.refine_shared_direction_hierarchical(D_B, theta_grid, phi_grid, dprior,
                                                                    3.0, 0.4, n_cap=2000)):
        omega_hat, cap_log_post, cap_theta, cap_phi = result
        cap_grid = kinematics.direction_from_theta_phi(cap_theta, cap_phi)
        spreads.append(validate.posterior_angular_resolution(cap_log_post[None, :], cap_grid, omega_hat)[0])
        assert np.max(cap_log_post[-200:]) < cap_log_post.max() - 5.0
        assert np.arccos(np.clip(omega_hat[0] @ omega_true, -1.0, 1.0)) < np.deg2rad(8.0)
    assert spreads[1] < spreads[0]


def test_marginal_direction_reduces_to_conditional_for_point_like_cap():
    # Correzione 2: se la calotta dello stadio 1 e' puntiforme (raggio 1e-6 rad),
    # marginalizzare su Omega_n equivale a fissarlo in omega_hat (a meno di una costante).
    rng = np.random.default_rng(SEED)
    D_B, omega_true = case_C_dataset(60, rng)
    omega_hat = omega_true[None, :]
    cap_theta, cap_phi = posterior_C.direction_cap_grid(omega_hat, 1e-6, 50)
    cap_log_post = np.zeros(50)
    mu_grid, sigma_grid = grids.hyperparameter_grid(30, 30)
    conditional, mu_c, sigma_c = posterior_C.refine_hyperparameters(
        (*D_B, omega_hat), mu_grid, sigma_grid, priors.hyperparameter_prior, n_mu=15, n_sigma=15
    )
    marginal, mu_m, sigma_m = posterior_C.refine_hyperparameters_marginal_direction(
        D_B, omega_hat, cap_log_post, cap_theta, cap_phi, mu_grid, sigma_grid,
        priors.hyperparameter_prior, n_pixel=5, n_mu=15, n_sigma=15
    )
    assert np.array_equal(mu_m, mu_c) and np.array_equal(sigma_m, sigma_c)
    np.testing.assert_allclose(marginal - logsumexp(marginal),
                               conditional - logsumexp(conditional), atol=1e-4)


def test_marginal_direction_not_narrower_than_conditional():
    # Propagare l'incertezza su Omega_n (qualche grado a N=60) non restringe il
    # posterior su log(sigma_E) rispetto allo stadio 2 condizionato su omega_hat.
    rng = np.random.default_rng(SEED)
    D_B, _ = case_C_dataset(60, rng)
    theta_grid, phi_grid = grids.sphere_grid(1500)
    dprior = priors.direction_prior(theta_grid, phi_grid)
    stage_1 = posterior_C.refine_shared_direction(D_B, theta_grid, phi_grid, dprior, n_cap=2000)
    mu_grid, sigma_grid = grids.hyperparameter_grid(30, 30)
    conditional, _, sigma_c = posterior_C.refine_hyperparameters(
        (*D_B, stage_1[0]), mu_grid, sigma_grid, priors.hyperparameter_prior, n_mu=20, n_sigma=20
    )
    marginal, _, sigma_m = posterior_C.refine_hyperparameters_marginal_direction(
        D_B, *stage_1, mu_grid, sigma_grid, priors.hyperparameter_prior,
        n_pixel=40, n_mu=20, n_sigma=20
    )
    assert np.all(np.isfinite(marginal))
    _, std_c = validate.posterior_mean_std(conditional[None, :], np.log(sigma_c))
    _, std_m = validate.posterior_mean_std(marginal[None, :], np.log(sigma_m))
    assert std_m[0] >= 0.99 * std_c[0]
