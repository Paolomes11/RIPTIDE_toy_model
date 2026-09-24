import numpy as np

from riptide_toy import combine, forward_model, grids, kinematics, posterior_B, priors
from riptide_toy.constants import SEED, SIGMA_EP


def test_direction_from_theta_phi_unit_norm():
    theta = np.array([0.0, np.pi / 4, np.pi / 2, np.pi])
    phi = np.array([0.0, 1.3, 4.0, 2.5])
    v = kinematics.direction_from_theta_phi(theta, phi)
    assert v.shape == (4, 3)
    np.testing.assert_allclose(np.linalg.norm(v, axis=-1), 1.0, atol=1e-12)


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
    # (isotropia, Cap. 20): la media deve annullarsi.
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


def test_posterior_B_flat_prior_one_event_constant_on_front_hemisphere():
    # riga 10 (CLAUDE.md Sez. 5, test di limite): 1 evento + prior piatto ->
    # L_k(Omega_n) ~costante sull'emisfero anteriore. Con En a prior piatto e
    # proprio (Ep = En*cos^2(theta_p)), l'integrale marginale su En varia con
    # theta_p come 1/cos^2(theta_p) (Jacobiano del cambio di variabile
    # En' = En*cos^2(theta_p)): in un cono centrale attorno alla traccia
    # osservata questa variazione resta modesta (entro il 20% fino a ~30 gradi),
    # mentre sull'emisfero posteriore (theta_p > pi/2) il rinculo e'
    # cinematicamente vietato (scattering elastico a masse uguali, CLAUDE.md
    # Sez. 3: theta_lab <= 90 gradi) -> log-verosimiglianza -inf. Il salto a
    # -inf, non la variazione residua entro l'emisfero anteriore, e' la
    # discontinuita' rilevante testata qui.
    track_hat = np.array([[0.0, 0.0, 1.0]])
    Ep_hat = np.array([1.0])

    theta_grid, phi_grid = grids.sphere_grid(4000)
    prior = priors.direction_prior(theta_grid, phi_grid)

    logpost = posterior_B.single_event_posterior(
        (Ep_hat, track_hat), (theta_grid, phi_grid), prior
    )
    assert logpost.shape == (1, theta_grid.shape[0])

    omega_n_hat = kinematics.direction_from_theta_phi(theta_grid, phi_grid)
    theta_p = kinematics.recoil_angle_from_direction(track_hat, omega_n_hat)[0]

    front_central = logpost[0][theta_p < np.deg2rad(30.0)]
    rear = logpost[0][theta_p > np.pi / 2]

    assert front_central.size > 100
    assert rear.size > 100
    weights = np.exp(front_central - front_central.max())
    assert weights.std() / weights.mean() < 0.2
    assert np.all(np.isneginf(rear))


def test_combine_reuse_angular_contraction_vs_N():
    # riga 11 (guida Sez. 4, riga 11: "combine.py (riuso)"): combine_loglik
    # e' generico (nessun import dal progetto, congelato dal checkpoint Caso
    # A) e si riusa invariato passandogli la matrice grezza di
    # forward_model.loglik_marginal_En (non posterior_B.single_event_posterior,
    # che incorpora gia' il prior — stesso schema del Caso A: il prior va
    # passato una sola volta a combine_loglik).
    #
    # Dati sintetici "onesti": Omega_n_true = asse z (nessuna perdita di
    # generalita', come nel test riga 8). theta_p_true e phi_true sono pescati
    # direttamente (non simulati a partire da un angolo di scattering in CM):
    # la formula di isotropia per kinematics.sample_cm_angle resta una
    # questione aperta (docs/roadmap.md, sezione Deviazioni) e il criterio di
    # accettazione di questa riga (andamento qualitativo ~1/sqrt(N), come Fig.
    # 39.1) non richiede quella formula. Ep_hat ha lo stesso rumore gaussiano
    # di forward_model.measure.
    rng = np.random.default_rng(SEED)
    n_max = 100

    omega_n_true = kinematics.direction_from_theta_phi(np.array([0.0]), np.array([0.0]))
    theta_p_true = rng.uniform(0.0, np.pi / 2 - 0.05, n_max)
    phi_true = rng.uniform(0.0, 2 * np.pi, n_max)
    track_hat = kinematics.direction_from_theta_phi(theta_p_true, phi_true)

    en_true = rng.uniform(1.0, 5.0, n_max)
    Ep_true = kinematics.proton_energy(en_true, theta_p_true)
    Ep_hat = Ep_true + rng.normal(0.0, SIGMA_EP, n_max)

    theta_grid, phi_grid = grids.sphere_grid(1500)
    en_grid = grids.energy_grid(150)
    log_prior_En = np.log(priors.energy_prior(en_grid))
    log_prior_dir = np.log(priors.direction_prior(theta_grid, phi_grid))
    omega_n_hat = kinematics.direction_from_theta_phi(theta_grid, phi_grid)
    angle_from_truth = kinematics.recoil_angle_from_direction(omega_n_true, omega_n_hat)[0]

    theta_p_candidates = kinematics.recoil_angle_from_direction(track_hat, omega_n_hat)
    loglik_all = forward_model.loglik_marginal_En(
        Ep_hat, theta_p_candidates, en_grid, SIGMA_EP, log_prior_En
    )

    def angular_sigma(n: int) -> float:
        combined = combine.combine_loglik(loglik_all[:n], log_prior_dir)
        w = np.exp(combined - combined.max())
        w /= w.sum()
        return float(np.sqrt(np.sum(w * angle_from_truth ** 2)))

    sigma_10 = angular_sigma(10)
    sigma_100 = angular_sigma(100)

    # Contrazione ~1/sqrt(N) (stesso andamento di combine.expected_sigma_n,
    # gia' verificato in Caso A, Es. 39.1): tolleranza larga perche' qui
    # sigma_10 e' una singola realizzazione Monte Carlo, non una media
    # d'insieme.
    predicted_100 = sigma_10 / np.sqrt(10.0)
    assert sigma_100 < sigma_10
    assert 0.4 < sigma_100 / predicted_100 < 2.5
