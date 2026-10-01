"""Riga 14: checklist di validazione sul Caso C (stadio 2 su omega_0, Omega_n da stadio 1 iterato).

Ordine della checklist: bias -> risoluzione -> pull ->
coverage -> contrazione ~1/sqrt(N) -> robustezza al prior.
M esperimenti indipendenti per ciascun N; per ogni esperimento verita' nuova:
mu_E ~ U(2.5, 4.0) MeV, sigma_E ~ log-U(0.2, 0.6) MeV, Omega_n isotropa.

Uso: OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=5G -p MemorySwapMax=0 \
         python scripts/caso_C_checklist.py [n_processi]
     (un thread BLAS per processo; picco ~0.6 GB per processo a N=1000; il
     tetto MemoryMax fa uccidere dal kernel solo lo script, non l'editor)
Produce: outputs/caso_C_checklist_results.pkl, outputs/caso_C_contrazione.png, outputs/caso_C_pull.png,
         outputs/caso_C_coverage.png, e la tabella su stdout.
"""
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from riptide_toy import grids, kinematics, posterior_C, priors, validate
from riptide_toy.constants import SEED, SIGMA_EP, SIGMA_THETA

# (N eventi, M esperimenti): M scala con 1/N; ~10 min con 3 processi (misurato 2026-09-26)
N_M = [(1000, 40), (300, 100), (150, 200), (50, 200)]
N_ROBUSTNESS = 150  # N a cui si ripete lo stadio 2 con prior uniforme in sigma_E
MU_RANGE = (2.5, 4.0)       # MeV
SIGMA_RANGE = (0.2, 0.6)    # MeV, log-uniforme
LEVELS = np.array([0.68, 0.90, 0.95])


def run_experiment(task: tuple[int, int]) -> dict:
    """Un esperimento: dataset sintetico, stadio 1 e 2 raffinati, riassunto.

    Ritorna:
        dict con verita', stime, sigma posteriori e flag di copertura.
    """
    n_events, index = task
    rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(n_events, index)))
    mu_true = rng.uniform(*MU_RANGE)
    sigma_true = np.exp(rng.uniform(*np.log(SIGMA_RANGE)))
    omega_true = kinematics.direction_from_theta_phi(
        np.arccos(rng.uniform(-1.0, 1.0, 1)), rng.uniform(0.0, 2 * np.pi, 1)
    )

    Ep_true, track = kinematics.sample_recoil_events(
        rng, rng.normal(mu_true, sigma_true, n_events), omega_true[0]
    )
    D_B = (rng.normal(Ep_true, SIGMA_EP), kinematics.smear_direction(rng, track, SIGMA_THETA))

    theta_grid, phi_grid = grids.sphere_grid()
    direction_prior = priors.direction_prior(theta_grid, phi_grid)
    mu_grid, sigma_grid = grids.hyperparameter_grid()
    # Stadio 1 a prior largo su En -> omega_0 -> stadio 2 (report §10): (mu_E, sigma_E)
    # restano condizionati su omega_0; lo stadio 1 iterato con prior su En
    # N(media, sd) della predittiva dello stadio 2 (report §8.3, §13 punto B)
    # serve solo alla stima di Omega_n
    omega_0 = posterior_C.refine_shared_direction(D_B, theta_grid, phi_grid, direction_prior)[0]
    D = (*D_B, omega_0)
    summary = {"N": n_events, "mu_true": mu_true, "sigma_true": sigma_true}
    prior_fns = {"logU": priors.hyperparameter_prior}
    if n_events == N_ROBUSTNESS:
        prior_fns["U"] = priors.hyperparameter_prior_uniform_sigma
    for name, prior_fn in prior_fns.items():
        log_post, mu_fine, sigma_fine = posterior_C.refine_hyperparameters(D, mu_grid, sigma_grid, prior_fn)
        if name == "logU":
            prior_En = posterior_C.predictive_energy_moments(log_post, mu_fine, sigma_fine)
        mu_mean, mu_std = validate.posterior_mean_std(log_post[None, :], mu_fine)
        ls_mean, ls_std = validate.posterior_mean_std(log_post[None, :], np.log(sigma_fine))
        mu_int = validate.credible_interval(log_post[None, :], mu_fine, np.concatenate([[0.0], LEVELS]))[0]
        ls_int = validate.credible_interval(log_post[None, :], np.log(sigma_fine), np.concatenate([[0.0], LEVELS]))[0]
        summary[name] = {
            "mu_mean": mu_mean[0], "mu_std": mu_std[0], "mu_median": mu_int[0, 0], "mu_int": mu_int[1:],
            "ls_mean": ls_mean[0], "ls_std": ls_std[0], "ls_median": ls_int[0, 0], "ls_int": ls_int[1:],
        }

    omega_hat, cap_log_post, cap_theta, cap_phi = posterior_C.refine_shared_direction_hierarchical(
        D_B, theta_grid, phi_grid, direction_prior, *prior_En
    )
    cap_grid = kinematics.direction_from_theta_phi(cap_theta, cap_phi)
    cap_pixel = np.sqrt(2 * np.pi * (1.0 - np.min(cap_grid @ omega_hat[0])) / cap_grid.shape[0])
    nearest = np.argmax(cap_grid @ omega_true[0])
    truth_in_cap = np.arccos(np.clip(cap_grid[nearest] @ omega_true[0], -1.0, 1.0)) < 3 * cap_pixel
    summary["omega_error"] = validate.angular_residual(omega_true, omega_hat)[0]
    summary["omega_sigma"] = validate.posterior_angular_resolution(cap_log_post[None, :], cap_grid, omega_hat)[0]
    summary["omega_covered"] = validate.credible_region_contains(
        cap_log_post[None, :], np.array([nearest]), LEVELS
    )[0] & truth_in_cap
    return summary


def collect(results: list[dict], n_events: int, prior: str = "logU") -> dict:
    """Raccoglie in array gli esperimenti a N fissato.

    Ritorna:
        dict di array, forma (M,) (o (M, n_livelli, 2) per gli intervalli).
    """
    rows = [r for r in results if r["N"] == n_events]
    out = {k: np.array([r[k] for r in rows]) for k in
           ("mu_true", "sigma_true", "omega_error", "omega_sigma", "omega_covered")}
    for k in rows[0][prior]:
        out[k] = np.array([r[prior][k] for r in rows])
    return out


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


def main() -> None:
    n_proc = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    tasks = [(n, i) for n, m in N_M for i in range(m)]
    with Pool(n_proc) as pool:
        results = pool.map(run_experiment, tasks, chunksize=1)
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    # risultati grezzi per il notebook 05_validazione_caso_C, senza rilanciare il run
    with open(out_dir / "caso_C_checklist_results.pkl", "wb") as f:
        pickle.dump({"N_M": N_M, "N_ROBUSTNESS": N_ROBUSTNESS, "LEVELS": LEVELS, "results": results}, f)

    n_values = sorted(n for n, _ in N_M)
    table = {n: collect(results, n) for n in n_values}
    print("Caso C, checklist di validazione (riga 14); mu, sigma in MeV, Omega in gradi")
    print("stime: mu = media posteriore, sigma = mediana posteriore di log sigma_E, Omega = MAP sulla calotta")
    for n in n_values:
        t = table[n]
        mu_res = t["mu_mean"] - t["mu_true"]
        ls_res = t["ls_median"] - np.log(t["sigma_true"])
        mu_pull = (t["mu_mean"] - t["mu_true"]) / t["mu_std"]
        ls_pull = (t["ls_mean"] - np.log(t["sigma_true"])) / t["ls_std"]
        om_pull = validate.angular_pull(t["omega_error"], t["omega_sigma"])
        _, mu_cov = validate.coverage_curve(t["mu_true"], t["mu_int"], LEVELS)
        _, ls_cov = validate.coverage_curve(np.log(t["sigma_true"]), t["ls_int"], LEVELS)
        om_cov = t["omega_covered"].mean(axis=0)
        m = len(mu_res)
        print(f"\nN={n}  (M={m})")
        print(f"  bias:        mu {mu_res.mean():+.4f} +- {mu_res.std() / np.sqrt(m):.4f}"
              f" | log sigma {ls_res.mean():+.3f} +- {ls_res.std() / np.sqrt(m):.3f}"
              f" | Omega err medio {np.degrees(t['omega_error'].mean()):.2f}")
        print(f"  risoluzione: rms mu {rms(mu_res):.4f} (post. std media {t['mu_std'].mean():.4f})"
              f" | rms log sigma {rms(ls_res):.3f} ({t['ls_std'].mean():.3f})"
              f" | rms Omega {np.degrees(rms(t['omega_error'])):.2f} ({np.degrees(t['omega_sigma'].mean()):.2f})")
        print(f"  pull:        mu {mu_pull.mean():+.2f} / {mu_pull.std():.2f}"
              f" | log sigma {ls_pull.mean():+.2f} / {ls_pull.std():.2f}"
              f" | Omega rms {rms(om_pull):.2f}   (media / larghezza; Omega: rms atteso ~1)")
        print("  coverage " + " ".join(f"{lv:.2f}" for lv in LEVELS) + ":"
              f"  mu {np.round(mu_cov, 2)} | log sigma {np.round(ls_cov, 2)} | Omega (HPD) {np.round(om_cov, 2)}"
              f"   (+-{np.sqrt(0.68 * 0.32 / m):.2f} binomiale a 0.68)")

    rms_curves = {
        "mu": [rms(table[n]["mu_mean"] - table[n]["mu_true"]) for n in n_values],
        "log sigma": [rms(table[n]["ls_median"] - np.log(table[n]["sigma_true"])) for n in n_values],
        "Omega (rad)": [rms(table[n]["omega_error"]) for n in n_values],
    }
    print("\ncontrazione (pendenza log-log di rms vs N, attesa -0.5):")
    for name, curve in rms_curves.items():
        slope = np.polyfit(np.log(n_values), np.log(curve), 1)[0]
        print(f"  {name}: {slope:+.3f}  rms*sqrt(N) = {np.round(np.array(curve) * np.sqrt(n_values), 3)}")

    # robustezza al prior: stesso dataset, stadio 2 con prior uniforme in sigma_E
    t_log = collect(results, N_ROBUSTNESS, "logU")
    t_lin = collect(results, N_ROBUSTNESS, "U")
    d_mu = (t_lin["mu_mean"] - t_log["mu_mean"]) / t_log["mu_std"]
    d_ls = (t_lin["ls_median"] - t_log["ls_median"]) / t_log["ls_std"]
    _, ls_cov_lin = validate.coverage_curve(np.log(t_lin["sigma_true"]), t_lin["ls_int"], LEVELS)
    print(f"\nrobustezza al prior (N={N_ROBUSTNESS}, uniforme in sigma_E vs in log sigma_E),"
          " shift in unita' di std posteriore:")
    print(f"  mu: medio {d_mu.mean():+.3f}, max |.| {np.abs(d_mu).max():.3f}"
          f" | log sigma: medio {d_ls.mean():+.3f}, max |.| {np.abs(d_ls).max():.3f}"
          f" | coverage log sigma con prior U: {np.round(ls_cov_lin, 2)}")
    print("  nota: SIGMA_E_MAX non entra: la griglia fine e' ristretta alla finestra locale")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name, curve in rms_curves.items():
        ax.loglog(n_values, curve, "o-", label=name)
        ax.loglog(n_values, curve[0] * np.sqrt(n_values[0] / np.array(n_values)), "k:", lw=0.8)
    ax.set_xlabel("N eventi")
    ax.set_ylabel("rms errore")
    ax.set_title("Caso C: contrazione (punteggiato: 1/sqrt(N))")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "caso_C_contrazione.png", dpi=150)

    t = table[N_ROBUSTNESS]
    pulls = {
        "mu": (t["mu_mean"] - t["mu_true"]) / t["mu_std"],
        "log sigma": (t["ls_mean"] - np.log(t["sigma_true"])) / t["ls_std"],
    }
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    x = np.linspace(-4, 4, 200)
    for ax, (name, pull) in zip(axes[:2], pulls.items()):
        ax.hist(pull, bins=25, range=(-4, 4), density=True, alpha=0.7)
        ax.plot(x, norm.pdf(x), "k-")
        ax.set_title(f"pull {name}, N={N_ROBUSTNESS}")
    om_pull = validate.angular_pull(t["omega_error"], t["omega_sigma"]) * np.sqrt(2.0)
    axes[2].hist(om_pull, bins=25, range=(0, 5), density=True, alpha=0.7)
    r = np.linspace(0, 5, 200)
    axes[2].plot(r, r * np.exp(-r ** 2 / 2), "k-")
    axes[2].set_title(f"pull Omega * sqrt(2) vs Rayleigh, N={N_ROBUSTNESS}")
    fig.tight_layout()
    fig.savefig(out_dir / "caso_C_pull.png", dpi=150)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for n in n_values:
        tn = table[n]
        _, cov = validate.coverage_curve(tn["mu_true"], tn["mu_int"], LEVELS)
        ax.plot(LEVELS, cov, "o-", label=f"mu, N={n}")
        ax.plot(LEVELS, tn["omega_covered"].mean(axis=0), "s--", label=f"Omega, N={n}")
    ax.plot([0.6, 1.0], [0.6, 1.0], "k:")
    ax.set_xlabel("livello nominale")
    ax.set_ylabel("coverage empirica")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "caso_C_coverage.png", dpi=150)


if __name__ == "__main__":
    main()
