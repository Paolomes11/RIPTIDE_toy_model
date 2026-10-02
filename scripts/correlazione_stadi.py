"""Caso C: i due stadi contro la stima congiunta locale di (Omega_n, mu_E, log sigma_E).

I due stadi stimano Omega_n e (mu_E, sigma_E) separatamente: lo stadio 2 tiene Omega_n
fissato, quindi ignora la correlazione posteriore fra direzione e iperparametri. Qui,
sugli stessi dataset della checklist del Caso C (spawn_key=(N, i)), si calcola
l'approssimazione di Laplace del posterior congiunto 4D partendo dalle stime dei due
stadi (Hessiana per differenze centrali su uno stencil 3^4, passi pari alle
larghezze dei due stadi; nessuna griglia 4D) e si misura:
  - correlazioni canoniche fra il blocco Omega_n e il blocco (mu_E, log sigma_E);
  - quanto si allargano le incertezze marginalizzando invece di fissare l'altro
    blocco (marginale / condizionata);
  - accordo fra Laplace e griglie dei due stadi (larghezze) e distanza fra il modo
    congiunto e le stime dei due stadi, in unita' di sigma;
  - pull di mu_E, log sigma_E e Omega_n per i due stadi e per il modo congiunto.

Uso: OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=5G -p MemorySwapMax=0 \
         python scripts/correlazione_stadi.py [n_processi]
     (picco ~0.25 GB per processo a N=300)
Produce: outputs/correlazione_stadi_results.pkl e la tabella su stdout.
"""
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from riptide_toy import grids, kinematics, posterior_C, priors, validate
from riptide_toy.constants import SEED, SIGMA_EP, SIGMA_THETA

N_M = [(300, 100), (150, 100), (50, 100)]  # i piu' lenti per primi
MU_RANGE = (2.5, 4.0)       # MeV, come la checklist del Caso C
SIGMA_RANGE = (0.2, 0.6)    # MeV, log-uniforme, come la checklist del Caso C
OMEGA_BLOCK = np.array([0, 1])
HYPER_BLOCK = np.array([2, 3])


def run_experiment(task: tuple[int, int]) -> dict:
    """Un esperimento: dataset della checklist, due stadi, Laplace congiunta.

    Ritorna:
        dict con verita' (mu_E in MeV, log sigma_E, Omega_n versore (3,)), stime e
        larghezze dei due stadi (MeV, adimensionale, rad), modo congiunto, covarianza
        di Laplace (4, 4) in (a, b in rad, mu_E in MeV, log sigma_E) e flag di
        Hessiana definita negativa.
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
    omega_0 = posterior_C.refine_shared_direction(D_B, theta_grid, phi_grid, direction_prior)[0]
    log_post, mu_fine, sigma_fine = posterior_C.refine_hyperparameters(
        (*D_B, omega_0), *grids.hyperparameter_grid(), priors.hyperparameter_prior)
    mu_mean, mu_std = validate.posterior_mean_std(log_post[None, :], mu_fine)
    ls_mean, ls_std = validate.posterior_mean_std(log_post[None, :], np.log(sigma_fine))
    omega_hat, cap_log_post, cap_theta, cap_phi = posterior_C.refine_shared_direction_hierarchical(
        D_B, theta_grid, phi_grid, direction_prior,
        *posterior_C.predictive_energy_moments(log_post, mu_fine, sigma_fine))
    cap_grid = kinematics.direction_from_theta_phi(cap_theta, cap_phi)
    omega_sigma = validate.posterior_angular_resolution(cap_log_post[None, :], cap_grid, omega_hat)[0]

    # passi dello stencil = larghezze dei due stadi (omega_sigma e' rms 2D: /sqrt(2) per asse)
    steps = np.array([omega_sigma / np.sqrt(2), omega_sigma / np.sqrt(2), mu_std[0], ls_std[0]])
    omega_mode, hyper_mode, hess, negative_definite = posterior_C.joint_laplace(
        D_B, omega_hat[0], mu_mean[0], ls_mean[0], steps, priors.hyperparameter_prior)
    return {"N": n_events, "mu_true": mu_true, "ls_true": np.log(sigma_true), "omega_true": omega_true,
            "two_stage": {"mu": mu_mean[0], "mu_std": mu_std[0], "ls": ls_mean[0], "ls_std": ls_std[0],
                          "omega": omega_hat, "omega_sigma": omega_sigma},
            "joint": {"mu": hyper_mode[0], "ls": hyper_mode[1], "omega": omega_mode,
                      "cov": np.linalg.inv(-hess) if negative_definite else None},
            "negative_definite": negative_definite}


def rms(x: np.ndarray) -> float:
    """Radice della media dei quadrati.

    Ritorna:
        float, stesse unita' di x.
    """
    return float(np.sqrt(np.mean(x ** 2)))


def print_table(rows: list[dict]) -> None:
    """Per N: correlazioni canoniche, allargamento marginale/condizionata, accordo
    Laplace-griglia, distanza modo congiunto - due stadi, pull dei due metodi.

    Ritorna:
        None (stdout).
    """
    print("Caso C: due stadi contro Laplace congiunta su (Omega_n, mu_E, log sigma_E)")
    for n_events in sorted({r["N"] for r in rows}):
        sel_all = [r for r in rows if r["N"] == n_events]
        sel = [r for r in sel_all if r["negative_definite"]]
        cov = np.array([r["joint"]["cov"] for r in sel])
        ts = {k: np.array([r["two_stage"][k] for r in sel]) for k in ("mu", "mu_std", "ls", "ls_std", "omega_sigma")}
        mu_true = np.array([r["mu_true"] for r in sel])
        ls_true = np.array([r["ls_true"] for r in sel])
        canon = np.array([validate.canonical_correlations(c, OMEGA_BLOCK)[0] for c in cov])
        hyper_cond = np.array([np.diag(validate.conditional_covariance(c, HYPER_BLOCK)) for c in cov])
        omega_cond = np.array([np.trace(validate.conditional_covariance(c, OMEGA_BLOCK)) for c in cov])
        hyper_marg = cov[:, HYPER_BLOCK, HYPER_BLOCK]
        omega_marg = cov[:, 0, 0] + cov[:, 1, 1]
        inflation = np.sqrt(np.column_stack([hyper_marg / hyper_cond, omega_marg / omega_cond]))
        joint_mu, joint_ls = (np.array([r["joint"][k] for r in sel]) for k in ("mu", "ls"))
        omega_true = np.concatenate([r["omega_true"] for r in sel])
        omega_ts = np.concatenate([r["two_stage"]["omega"] for r in sel])
        omega_joint = np.concatenate([r["joint"]["omega"] for r in sel])
        omega_shift = validate.angular_residual(omega_ts, omega_joint)

        print(f"\nN={n_events} (M={len(sel_all)}, Hessiana definita negativa in {len(sel)})")
        print(f"  correlazione canonica max Omega-(mu, log sigma): mediana {np.median(canon):.3f}"
              f" | 90% {np.quantile(canon, 0.9):.3f} | max {canon.max():.3f}")
        for j, label in enumerate(("mu", "log sigma", "Omega")):
            print(f"  allargamento marginale/condizionata {label:9s}: media {inflation[:, j].mean():.4f}"
                  f" | max {inflation[:, j].max():.4f}")
        print(f"  Laplace (Omega fissato) / griglia stadio 2: mu {np.mean(np.sqrt(hyper_cond[:, 0]) / ts['mu_std']):.3f}"
              f" | log sigma {np.mean(np.sqrt(hyper_cond[:, 1]) / ts['ls_std']):.3f}"
              f" | Omega rms Laplace / calotta {np.mean(np.sqrt(omega_marg) / ts['omega_sigma']):.3f}")
        print(f"  modo congiunto - due stadi, in sigma: mu {rms((joint_mu - ts['mu']) / ts['mu_std']):.3f}"
              f" | log sigma {rms((joint_ls - ts['ls']) / ts['ls_std']):.3f}"
              f" | Omega {rms(omega_shift / ts['omega_sigma']):.3f} (rms)")
        pulls = {
            "due stadi": ((ts["mu"] - mu_true) / ts["mu_std"], (ts["ls"] - ls_true) / ts["ls_std"],
                          validate.angular_pull(validate.angular_residual(omega_true, omega_ts), ts["omega_sigma"])),
            "congiunto": ((joint_mu - mu_true) / np.sqrt(hyper_marg[:, 0]),
                          (joint_ls - ls_true) / np.sqrt(hyper_marg[:, 1]),
                          validate.angular_pull(validate.angular_residual(omega_true, omega_joint), np.sqrt(omega_marg))),
        }
        for name, (p_mu, p_ls, p_omega) in pulls.items():
            print(f"  pull {name:9s}: mu {p_mu.mean():+.2f}/{p_mu.std():.2f}"
                  f" | log sigma {p_ls.mean():+.2f}/{p_ls.std():.2f} | Omega rms {rms(p_omega):.2f}")


def main() -> None:
    """Esegue gli esperimenti in parallelo, salva i risultati e stampa la tabella.

    Ritorna:
        None (pkl in outputs/, tabella su stdout).
    """
    n_proc = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    tasks = [(n, i) for n, m in N_M for i in range(m)]
    with Pool(n_proc) as pool:
        rows = pool.map(run_experiment, tasks, chunksize=1)
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "correlazione_stadi_results.pkl", "wb") as f:
        pickle.dump({"N_M": N_M, "results": rows}, f)
    print_table(rows)


if __name__ == "__main__":
    main()
