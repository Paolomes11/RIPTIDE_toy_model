"""Checklist di validazione del posterior di singolo evento del Caso A.

Ordine della checklist: bias -> pull -> coverage.
Due generatori della verita' E_n, con lo stesso posterior (prior piatto su [EN_MIN, EN_MAX]):
  (i)  E_n ~ U(EN_MIN, EN_MAX), theta_p ~ U(0, pi/2): la verita' e' estratta dal prior stesso.
       Gli intervalli credibili devono coprire esattamente al livello nominale (media sul prior).
  (ii) E_n ~ U(1, 5): la sotto-popolazione usata nel notebook 02. La coverage non e' piu'
       garantita, perche' si media su una distribuzione diversa dal prior.

Uso: python scripts/caso_A_checklist.py   (~40 s, un processo, < 1 GB)
Produce: la tabella su stdout.
"""
import numpy as np

from riptide_toy import forward_model, grids, kinematics, posterior_A, priors, validate
from riptide_toy.constants import EN_MAX, EN_MIN, SEED, SIGMA_EP, SIGMA_THETA

N_EVENTS = 20_000
CHUNK = 50  # eventi per blocco: la marginalizzazione su theta_p alloca (CHUNK, 500, 500)
LEVELS = np.array([0.68, 0.90, 0.95])
GENERATORS = {"(i) prior = generatore, U(0.5, 6)": (EN_MIN, EN_MAX), "(ii) U(1, 5)": (1.0, 5.0)}


def reconstruct(truth: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simula un evento per ogni E_n vero e lo ricostruisce col posterior del Caso A.

    Args:
        truth: E_n vere, MeV, forma (n,).
        rng: generatore numpy.

    Ritorna:
        (media posteriore in MeV forma (n,), deviazione standard posteriore in MeV forma (n,),
         intervalli credibili centrali in MeV forma (n, len(LEVELS), 2)).
    """
    en_grid = grids.energy_grid()
    prior = priors.energy_prior(en_grid)
    # theta_p ~ U(0, pi/2): lo stesso prior su theta usato dalla verosimiglianza del Caso A
    theta = rng.uniform(0.0, np.pi / 2, truth.size)
    Ep_hat, theta_hat = forward_model.measure(kinematics.proton_energy(truth, theta), theta,
                                              SIGMA_EP, SIGMA_THETA, rng)
    log_post = np.concatenate([
        posterior_A.single_event_posterior((Ep_hat[i:i + CHUNK], theta_hat[i:i + CHUNK]), en_grid, prior)
        for i in range(0, truth.size, CHUNK)
    ])
    estimate, sigma_hat = validate.posterior_mean_std(log_post, en_grid)
    return estimate, sigma_hat, validate.credible_interval(log_post, en_grid, LEVELS)


def main() -> None:
    """Esegue la checklist per i due generatori e stampa la tabella.

    Ritorna:
        None (solo stdout).
    """
    binomial = np.sqrt(LEVELS * (1 - LEVELS) / N_EVENTS)
    for name, (lo, hi) in GENERATORS.items():
        rng = np.random.default_rng(SEED)
        truth = rng.uniform(lo, hi, N_EVENTS)
        r = validate.run_checklist(reconstruct, truth, levels=LEVELS, n_bins=5, rng=rng)
        print(f"{name}, {N_EVENTS} eventi")
        print("  bias per bin [MeV]: " + "  ".join(f"{c:.2f}:{b:+.3f}" for c, b in zip(r["bias_centers"], r["bias"])))
        print(f"  pull: media {r['pull_mean']:+.3f} (+- {1 / np.sqrt(N_EVENTS):.3f}), larghezza {r['pull_width']:.3f}")
        print("  coverage 0.68 0.90 0.95: " + "  ".join(
            f"{c:.4f} (+-{e:.4f})" for c, e in zip(r["coverage"], binomial)))


if __name__ == "__main__":
    main()
