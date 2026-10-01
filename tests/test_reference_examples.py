import numpy as np
from scipy.stats import norm

from riptide_toy import grids, posterior_A, priors
from riptide_toy.combine import combine_loglik, expected_sigma_n
from riptide_toy.validate import run_checklist


def test_reference_single_event():
    # caso di riferimento R1 (singolo evento): Ep_hat=1.40 MeV, theta_p_hat=0.50 rad -> En = 1.82 +/- 0.21 MeV
    en_grid = grids.energy_grid()
    D = (np.array([1.40]), np.array([0.50]))
    prior = priors.energy_prior(en_grid)

    logpost = posterior_A.single_event_posterior(D, en_grid, prior)
    assert logpost.shape == (1, en_grid.shape[0])

    argmax_En = en_grid[np.argmax(logpost[0])]
    assert abs(argmax_En - 1.82) < 0.03  # ~3 passi di griglia

    # sigma stimata come deviazione standard pesata sulla griglia del
    # posterior, tolleranza 0.01 MeV sul valore atteso 0.21.
    w = np.exp(logpost[0] - logpost[0].max())
    mean = np.sum(w * en_grid) / np.sum(w)
    sigma = np.sqrt(np.sum(w * (en_grid - mean) ** 2) / np.sum(w))
    assert abs(sigma - 0.21) < 0.01


def test_reference_two_events():
    # caso di riferimento R2: due eventi combinati -> En = 2.40 +/- 0.18 MeV
    en_grid = np.linspace(0.5, 6.0, 2000)
    ll1 = -0.5 * ((en_grid - 2.70) / 0.45) ** 2
    ll2 = -0.5 * ((en_grid - 2.35) / 0.20) ** 2
    combined = combine_loglik(np.stack([ll1, ll2]), log_prior=np.zeros_like(en_grid))

    argmax_En = en_grid[combined.argmax()]
    assert abs(argmax_En - 2.40) < 0.03

    w = np.exp(combined - combined.max())
    mean = np.sum(w * en_grid) / np.sum(w)
    sigma = np.sqrt(np.sum(w * (en_grid - mean) ** 2) / np.sum(w))
    assert abs(sigma - 0.18) < 0.01


def test_expected_sigma_n_reference():
    assert expected_sigma_n(1.0, 4) == 0.5


def _faulty_reconstruction(truth: np.ndarray, rng: np.random.Generator):
    # caso di riferimento R3: errore di scala +3% (+0.05 MeV) sul valore vero, incertezze
    # dichiarate 30% troppo strette rispetto alla risoluzione vera sigma(En)=0.08*En.
    # Il rumore intrinseco del detector si applica DOPO l'errore di scala, non scalato
    # con esso: e' l'unica costruzione delle due plausibili che riproduce pull width
    # ~1.4 (rapporto 1/0.7 = 1.4286, contro 1.03/0.7 = 1.4714 se il rumore fosse
    # scalato anch'esso). Vedi docs/roadmap.md, sezione Scelte di implementazione.
    true_sigma = 0.08 * truth
    noise = rng.normal(0.0, true_sigma, truth.shape[0])
    estimate = 1.03 * truth + 0.05 + noise
    sigma_hat = 0.7 * true_sigma

    levels = np.array([0.68])
    z = norm.ppf(0.5 + levels / 2)
    half_widths = z[None, :] * sigma_hat[:, None]
    intervals = np.stack(
        [estimate[:, None] - half_widths, estimate[:, None] + half_widths], axis=-1
    )
    return estimate, sigma_hat, intervals


def test_reference_faulty_reconstruction():
    # caso di riferimento R3 (ricostruzione difettosa): bias 0.08->0.20 MeV, pull
    # mean~0.9, pull width~1.4, copertura nominale 68% -> ~45%. Questi valori attesi
    # sono approssimati ("~"/"≈"), a differenza di R1/R2 che hanno una tolleranza
    # esplicita su sigma (0.01 MeV): le tolleranze qui sono scelte in base a
    # quanto strette sono le cifre attese, non desunte da un numero esatto.
    rng = np.random.default_rng(20260907)
    truth = rng.uniform(1.0, 5.0, 20_000)

    result = run_checklist(
        _faulty_reconstruction, truth, levels=np.array([0.68]), rng=rng
    )

    assert abs(result["bias"][0] - 0.08) < 0.02
    assert abs(result["bias"][-1] - 0.20) < 0.02
    assert abs(result["pull_mean"] - 0.9) < 0.1
    assert abs(result["pull_width"] - 1.4) < 0.1
    assert abs(result["coverage"][0] - 0.45) < 0.05
