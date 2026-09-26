import numpy as np
from scipy import stats
from scipy.integrate import trapezoid

from riptide_toy import combine, forward_model, grids, kinematics, posterior_B, priors, validate
from riptide_toy.constants import EN_MAX, EN_MIN, SEED, SIGMA_EP, SIGMA_THETA


def test_direction_from_theta_phi_unit_norm():
    theta = np.array([0.0, np.pi / 4, np.pi / 2, np.pi])
    phi = np.array([0.0, 1.3, 4.0, 2.5])
    v = kinematics.direction_from_theta_phi(theta, phi)
    assert v.shape == (4, 3)
    np.testing.assert_allclose(np.linalg.norm(v, axis=-1), 1.0, atol=1e-12)


def test_theta_phi_from_direction_inverts_direction_from_theta_phi():
    rng = np.random.default_rng(SEED)
    theta = np.arccos(rng.uniform(-1.0, 1.0, 200))
    phi = rng.uniform(0.0, 2 * np.pi, 200)
    theta_back, phi_back = kinematics.theta_phi_from_direction(
        kinematics.direction_from_theta_phi(theta, phi)
    )
    np.testing.assert_allclose(theta_back, theta, atol=1e-12)
    np.testing.assert_allclose(phi_back, phi, atol=1e-12)


def test_sample_cm_angle_isotropic_in_solid_angle():
    rng = np.random.default_rng(SEED)
    theta_cm = kinematics.sample_cm_angle(rng, 200_000)
    assert np.all((theta_cm >= 0.0) & (theta_cm <= np.pi))
    # (a) isotropia: cos(theta_CM) ~ U(-1, 1)
    assert stats.kstest((np.cos(theta_cm) + 1.0) / 2.0, "uniform").pvalue > 1e-3


def test_recoil_angle_from_cm_limits():
    theta_p = kinematics.recoil_angle_from_cm(np.array([0.0, np.pi / 2, np.pi]))
    np.testing.assert_allclose(theta_p, [np.pi / 2, np.pi / 4, 0.0], atol=1e-12)


def test_perpendicular_basis_orthonormal_including_axis_along_x():
    axis = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                     [-1.0, 0.0, 0.0], [0.6, 0.0, 0.8]])
    e1, e2 = kinematics.perpendicular_basis(axis)
    for u, v in ((e1, axis), (e2, axis), (e1, e2)):
        np.testing.assert_allclose(np.sum(u * v, axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(e1, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(e2, axis=1), 1.0, atol=1e-12)


def test_sample_recoil_events_mean_var_and_track_distribution():
    # Oracolo guida riga 2: isotropia in CM => Ep ~ U(0, En), media En/2,
    # varianza En^2/12 (a); equivalente a cos^2(theta_p) ~ U(0, 1).
    rng = np.random.default_rng(SEED)
    n = 200_000
    En = np.full(n, 3.0)
    omega = kinematics.direction_from_theta_phi(np.array(0.9), np.array(2.1))
    Ep, track = kinematics.sample_recoil_events(rng, En, omega)

    assert Ep.shape == (n,) and track.shape == (n, 3)
    assert abs(Ep.mean() - 1.5) < 0.01
    assert abs(Ep.var() - 9.0 / 12.0) < 0.01
    np.testing.assert_allclose(np.linalg.norm(track, axis=1), 1.0, atol=1e-12)

    theta_p = kinematics.recoil_angle_from_direction(track, omega[None, :])[:, 0]
    assert theta_p.max() <= np.pi / 2 + 1e-12
    np.testing.assert_allclose(Ep, kinematics.proton_energy(En, theta_p), atol=1e-9)
    assert stats.kstest(np.cos(theta_p) ** 2, "uniform").pvalue > 1e-3


def test_smear_direction_polar_deviation_matches_sigma_theta():
    # (b) errore gaussiano isotropo nel piano tangente: angolo totale di
    # Rayleigh (media sigma*sqrt(pi/2)); componente polare ~ N(0, sigma)
    # lontano dai bordi (theta_p in [0.3, 1.2] rad).
    rng = np.random.default_rng(SEED)
    n = 200_000
    sigma = 0.08
    omega = kinematics.direction_from_theta_phi(np.array(0.9), np.array(2.1))
    _, track = kinematics.sample_recoil_events(rng, np.full(n, 3.0), omega)
    smeared = kinematics.smear_direction(rng, track, sigma)

    np.testing.assert_allclose(np.linalg.norm(smeared, axis=1), 1.0, atol=1e-12)
    total = np.arccos(np.clip(np.sum(smeared * track, axis=1), -1.0, 1.0))
    assert abs(total.mean() - sigma * np.sqrt(np.pi / 2)) < 1e-3

    theta_true = kinematics.recoil_angle_from_direction(track, omega[None, :])[:, 0]
    theta_obs = kinematics.recoil_angle_from_direction(smeared, omega[None, :])[:, 0]
    mid = (theta_true > 0.3) & (theta_true < 1.2)
    assert abs((theta_obs - theta_true)[mid].std() - sigma) < 2e-3


def test_omega_n_parallel_to_z_reduces_to_case_A():
    # riga 8 (CLAUDE.md Sez. 5, test di limite): Omega_n || z. L'angolo 3D fra
    # track_hat e Omega_n deve ridarsi esattamente theta_p del Caso A,
    # qualunque sia l'azimut della traccia (arccos(track . z_hat) = theta_p
    # per costruzione, indipendente da phi).
    omega_n_hat = kinematics.direction_from_theta_phi(np.array([0.0]), np.array([0.0]))

    theta_p = np.array([0.0, 0.3, np.pi / 4, 1.0, np.pi / 2])
    phi_arbitrary = np.array([0.0, 1.0, 3.7, np.pi, 5.5])
    track_hat = kinematics.direction_from_theta_phi(theta_p, phi_arbitrary)

    angles = kinematics.recoil_angle_from_direction(track_hat, omega_n_hat)
    assert angles.shape == (5, 1)
    np.testing.assert_allclose(angles[:, 0], theta_p, atol=1e-10)


def test_sphere_grid_is_pixelization_with_equal_area():
    # fix errata (c): un indice = una direzione, area solida ~costante per
    # pixel. Punti uniformi sulla sfera hanno cos(theta) ~ Uniform(-1, 1)
    # (isotropia): la media deve annullarsi.
    theta, phi = grids.sphere_grid(2000)
    assert theta.shape == (2000,)
    assert phi.shape == (2000,)
    assert abs(np.cos(theta).mean()) < 0.01
    assert np.all((phi >= 0.0) & (phi < 2 * np.pi))


def test_direction_prior_proper_and_sums_to_one():
    theta, phi = grids.sphere_grid()
    p = priors.direction_prior(theta, phi)
    assert p.shape == theta.shape
    assert np.all(p > 0.0)
    assert abs(p.sum() - 1.0) < 1e-10


def test_log_track_density_isotropic_cm():
    theta = np.array([0.0, np.pi / 3, np.pi / 2 + 0.1, np.pi])
    out = forward_model.log_track_density(theta)
    np.testing.assert_allclose(out[:2], np.log(np.cos(theta[:2]) / np.pi), atol=1e-12)
    assert np.all(np.isneginf(out[2:]))


def test_log_track_kernel_normalized_per_steradian():
    # (b) la densita' della traccia osservata integra a 1 sulla sfera, entro
    # l'errore di curvatura O(sigma_theta^2) dell'approssimazione 1D.
    obs = np.linspace(0.0, np.pi, 4001)
    theta = np.linspace(0.0, np.pi / 2, 400)
    density = np.exp(forward_model.log_track_kernel(obs, theta, SIGMA_THETA)).sum(axis=1)
    total = trapezoid(density * 2 * np.pi * np.sin(obs), obs)
    assert abs(total - 1.0) < 0.01


def test_log_track_kernel_sphere_normalized_and_exact_first_moment():
    # (a) vMF: E[cos theta_obs] = E[cos theta_p] * (coth k - 1/k), k = 1/sigma^2
    # (teorema di addizione), E[cos theta_p] = 2/3 per cos(theta_p)/pi.
    # Il kernel piatto sbaglia questo momento di ~sigma^2/3 (la curvatura
    # all'origine del bias di mu_E, report Caso C §7).
    obs = np.linspace(0.0, np.pi, 4001)
    theta = np.linspace(0.0, np.pi / 2, 400)
    density = np.exp(forward_model.log_track_kernel_sphere(obs, theta, SIGMA_THETA)).sum(axis=1)
    total = trapezoid(density * 2 * np.pi * np.sin(obs), obs)
    mean_cos = trapezoid(density * np.cos(obs) * 2 * np.pi * np.sin(obs), obs)
    kappa = 1.0 / SIGMA_THETA ** 2
    assert abs(total - 1.0) < 1e-4
    assert abs(mean_cos - 2.0 / 3.0 * (1.0 / np.tanh(kappa) - 1.0 / kappa)) < 1e-4


def test_log_track_kernel_sphere_matches_smear_direction_monte_carlo():
    rng = np.random.default_rng(SEED)
    _, track = kinematics.sample_recoil_events(rng, np.full(400_000, 3.0), np.array([0.0, 0.0, 1.0]))
    cos_obs = kinematics.smear_direction(rng, track, SIGMA_THETA)[:, 2]
    obs = np.linspace(0.0, np.pi, 4001)
    theta = np.linspace(0.0, np.pi / 2, 400)
    density = np.exp(forward_model.log_track_kernel_sphere(obs, theta, SIGMA_THETA)).sum(axis=1)
    mean_cos = trapezoid(density * np.cos(obs) * 2 * np.pi * np.sin(obs), obs)
    assert abs(cos_obs.mean() - mean_cos) < 4.0 * cos_obs.std() / np.sqrt(cos_obs.size)


def test_log_track_kernel_sphere_reduces_to_flat_times_curvature_factor():
    # kappa sin(theta_obs) sin(theta) >> 1: I0e(z) ~ 1/sqrt(2 pi z), quindi
    # sfera / piatto -> sqrt(sin(theta)/sin(theta_obs)) vicino al picco
    theta = np.linspace(0.0, np.pi / 2, 400)
    obs = np.array([0.6, 1.0])
    near = np.abs(obs[:, None] - theta[None, :]) < 2.0 * SIGMA_THETA
    ratio = np.exp(forward_model.log_track_kernel_sphere(obs, theta, SIGMA_THETA)
                   - forward_model.log_track_kernel(obs, theta, SIGMA_THETA))
    expected = np.sqrt(np.sin(theta)[None, :] / np.sin(obs)[:, None])
    np.testing.assert_allclose(ratio[near], expected[near], rtol=2e-2)


def test_loglik_marginal_En_theta_sigma_to_zero_is_sharp_cut_plus_track_term():
    # Limite sigma_theta -> 0: si ritrova il vecchio loglik_marginal_En (taglio
    # netto) piu' log(cos(theta_p)/pi), dove Ep/cos^2 resta dentro il dominio
    # in energia (vicino a EN_MAX il vincolo in En e' ripido e il limite e'
    # raggiunto solo come O(sigma_theta)).
    en_grid = grids.energy_grid()
    log_prior_En = np.log(priors.energy_prior(en_grid))
    Ep_hat = np.array([1.0, 2.0, 0.5])
    theta_obs = np.tile(np.linspace(0.05, 1.1, 40), (3, 1))

    new = forward_model.loglik_marginal_En_theta(
        Ep_hat, theta_obs, en_grid, SIGMA_EP, 0.003, log_prior_En,
        n_theta=3000, n_theta_obs=4001,
    )
    sharp = (forward_model.loglik_marginal_En(Ep_hat, theta_obs, en_grid, SIGMA_EP, log_prior_En)
             + forward_model.log_track_density(theta_obs))
    interior = Ep_hat[:, None] / np.cos(theta_obs) ** 2 < 0.8 * EN_MAX
    assert interior.sum() > 60
    np.testing.assert_allclose(new[interior], sharp[interior], atol=2e-3)


def test_posterior_B_one_event_flat_prior_matches_analytic_form():
    # riga 10, test di limite "1 evento + prior piatto" -- ERRATA (R3): con il
    # termine di traccia (isotropia in CM) L_k(Omega_n) non e' ~costante
    # sull'emisfero anteriore ma, per sigma_theta -> 0,
    #   L ∝ cos(theta_p)/pi * [Phi((EN_MAX c - Ep)/s) - Phi((EN_MIN c - Ep)/s)] / c,
    # c = cos^2(theta_p) (integrale analitico del prior piatto su En). Con la
    # risoluzione SIGMA_THETA la forma e' smussata di O(sigma_theta^2): si
    # confronta a meno di una costante entro 45 gradi. Nessun -inf: oltre
    # pi/2 + 5 sigma_theta il posterior e' solo fortemente soppresso.
    track_hat = np.array([[0.0, 0.0, 1.0]])
    Ep_hat = np.array([1.0])

    theta_grid, phi_grid = grids.sphere_grid(4000)
    prior = priors.direction_prior(theta_grid, phi_grid)
    logpost = posterior_B.single_event_posterior(
        (Ep_hat, track_hat), (theta_grid, phi_grid), prior
    )
    assert logpost.shape == (1, theta_grid.shape[0])
    assert np.all(np.isfinite(logpost))

    omega_n_hat = kinematics.direction_from_theta_phi(theta_grid, phi_grid)
    theta_p = kinematics.recoil_angle_from_direction(track_hat, omega_n_hat)[0]

    front = theta_p < np.deg2rad(45.0)
    c = np.cos(theta_p[front]) ** 2
    analytic = (np.log(np.cos(theta_p[front]) / np.pi) - np.log(c)
                + np.log(stats.norm.cdf((EN_MAX * c - Ep_hat[0]) / SIGMA_EP)
                         - stats.norm.cdf((EN_MIN * c - Ep_hat[0]) / SIGMA_EP)))
    assert front.sum() > 100
    assert np.ptp(logpost[0][front] - analytic) < 0.02

    rear = theta_p > np.pi / 2 + 5 * SIGMA_THETA
    assert rear.sum() > 100
    assert logpost[0][rear].max() < logpost[0][front].max() - 20.0


def simulate_case_B_events(rng: np.random.Generator, omega_n_true: np.ndarray,
                           en_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eventi sintetici Caso B col generatore fisico (isotropia in CM) e le
    risoluzioni del detector SIGMA_EP, SIGMA_THETA.

    Ritorna:
        (Ep_hat, track_hat): MeV forma (n,), versori forma (n, 3).
    """
    Ep_true, track_true = kinematics.sample_recoil_events(rng, en_true, omega_n_true)
    track_hat = kinematics.smear_direction(rng, track_true, SIGMA_THETA)
    return rng.normal(Ep_true, SIGMA_EP), track_hat


def test_combine_reuse_angular_contraction_vs_N():
    # riga 11 (guida Sez. 4, riga 11: "combine.py (riuso)"): combine_loglik
    # e' generico (nessun import dal progetto, congelato dal checkpoint Caso
    # A) e si riusa invariato passandogli la matrice grezza di
    # forward_model.loglik_marginal_En_theta (non posterior_B.single_event_posterior,
    # che incorpora gia' il prior — stesso schema del Caso A: il prior va
    # passato una sola volta a combine_loglik).
    #
    # Dati sintetici: Omega_n_true = asse z (nessuna perdita di generalita',
    # come nel test riga 8), generatore fisico isotropo in CM e risoluzioni
    # SIGMA_EP, SIGMA_THETA (simulate_case_B_events). Rigenerati in R3 insieme
    # al termine di traccia: prima theta_p ~ U(0, pi/2 - 0.05) e tracce esatte.
    rng = np.random.default_rng(SEED)
    n_max = 100

    omega_n_true = kinematics.direction_from_theta_phi(np.array([0.0]), np.array([0.0]))
    en_true = rng.uniform(1.0, 5.0, n_max)
    Ep_hat, track_hat = simulate_case_B_events(rng, omega_n_true[0], en_true)

    theta_grid, phi_grid = grids.sphere_grid(1500)
    en_grid = grids.energy_grid(150)
    log_prior_En = np.log(priors.energy_prior(en_grid))
    log_prior_dir = np.log(priors.direction_prior(theta_grid, phi_grid))
    omega_n_hat = kinematics.direction_from_theta_phi(theta_grid, phi_grid)
    angle_from_truth = kinematics.recoil_angle_from_direction(omega_n_true, omega_n_hat)[0]

    theta_p_candidates = kinematics.recoil_angle_from_direction(track_hat, omega_n_hat)
    loglik_all = forward_model.loglik_marginal_En_theta(
        Ep_hat, theta_p_candidates, en_grid, SIGMA_EP, SIGMA_THETA, log_prior_En
    )

    def angular_sigma(n: int) -> float:
        combined = combine.combine_loglik(loglik_all[:n], log_prior_dir)
        w = np.exp(combined - combined.max())
        w /= w.sum()
        return float(np.sqrt(np.sum(w * angle_from_truth ** 2)))

    sigma_10 = angular_sigma(10)
    sigma_100 = angular_sigma(100)

    # Contrazione ~1/sqrt(N) (stesso andamento di combine.expected_sigma_n,
    # gia' verificato in Caso A, caso di riferimento R2): tolleranza larga perche' qui
    # sigma_10 e' una singola realizzazione Monte Carlo, non una media
    # d'insieme.
    predicted_100 = sigma_10 / np.sqrt(10.0)
    assert sigma_100 < sigma_10
    assert 0.4 < sigma_100 / predicted_100 < 2.5


def test_angular_bias_and_pull_on_simulated_omega_n():
    # riga 12 (guida Sez. 4, riga 12): "Bias/pull su distanza angolare", etichetta (d)
    # (nessun valore di riferimento; verificato su dati simulati con Omega_n nota, come richiesto dalla
    # guida). Si ripetono M esperimenti indipendenti (N=20 eventi sintetici ciascuno,
    # stessa costruzione "onesta" della riga 11), si stima Omega_n col MAP del
    # posterior combinato, e si confronta la distanza angolare vera (angular_residual)
    # con l'incertezza dichiarata dal posterior attorno alla propria stima
    # (posterior_angular_resolution, self-referenziale sul proprio MAP). Non essendo
    # un caso di riferimento, si verifica solo che il pull risultante (angular_pull) sia
    # d'ordine 1 (ne' fortemente sovrastimato ne' sottostimato), non una calibrazione
    # esatta N(0,1)/Rayleigh(1): la distanza angolare non e' gaussiana come nel caso 1D.
    rng = np.random.default_rng(SEED)
    n_events = 20
    n_experiments = 30

    theta_grid, phi_grid = grids.sphere_grid(800)
    en_grid = grids.energy_grid(100)
    log_prior_En = np.log(priors.energy_prior(en_grid))
    log_prior_dir = np.log(priors.direction_prior(theta_grid, phi_grid))
    omega_n_hat_grid = kinematics.direction_from_theta_phi(theta_grid, phi_grid)
    omega_n_true = kinematics.direction_from_theta_phi(np.array([0.0]), np.array([0.0]))

    pulls = np.empty(n_experiments)
    residuals = np.empty(n_experiments)
    for i in range(n_experiments):
        en_true = rng.uniform(1.0, 5.0, n_events)
        Ep_hat, track_hat = simulate_case_B_events(rng, omega_n_true[0], en_true)

        theta_p_candidates = kinematics.recoil_angle_from_direction(track_hat, omega_n_hat_grid)
        loglik_all = forward_model.loglik_marginal_En_theta(
            Ep_hat, theta_p_candidates, en_grid, SIGMA_EP, SIGMA_THETA, log_prior_En
        )
        combined = combine.combine_loglik(loglik_all, log_prior_dir)

        omega_estimate = omega_n_hat_grid[np.argmax(combined):np.argmax(combined) + 1]
        sigma_hat = validate.posterior_angular_resolution(
            combined[None, :], omega_n_hat_grid, omega_estimate
        )
        residuals[i] = validate.angular_residual(omega_n_true, omega_estimate)[0]
        pulls[i] = validate.angular_pull(residuals[i:i + 1], sigma_hat)[0]

    # bias: la stima deve mediamente cadere vicino alla verita', non a caso sulla sfera
    assert np.degrees(residuals.mean()) < 20.0
    # pull d'ordine 1: ne' sigma_hat inutile (pull >> 1), ne' falsamente stretta (pull << 1)
    assert 0.2 < np.median(pulls) < 3.0
