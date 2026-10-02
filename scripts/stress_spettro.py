"""Stress test dell'assunzione 3 (spettro gaussiano) nel Caso C.

Le energie vere sono generate da spettri con la stessa media e deviazione standard
di N(mu_E, sigma_E) ma forma diversa (gaussiana, uniforme, bimodale, lognormale);
la ricostruzione resta quella del Caso C (modello gaussiano, stadio 1 -> stadio 2 ->
stadio 1 iterato), invariata. Bersagli: mu_E = media vera dello spettro,
log sigma_E = log della sd vera, Omega_n.
Stessi seed della checklist del Caso C, spawn_key=(N, i), per tutte le forme:
a parita' di (N, i) la verita' (mu_E, sigma_E, Omega_n) e' la stessa e cambia solo
la forma dello spettro; con "gauss" si riottengono esattamente i dataset (e i numeri)
della checklist.

Uso: OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=5G -p MemorySwapMax=0 \
         python scripts/stress_spettro.py [n_processi]
     (picco ~0.6 GB per processo a N=1000, come la checklist del Caso C)
Produce: outputs/stress_spettro_results.pkl e la tabella su stdout.
"""
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from riptide_toy import grids, kinematics, posterior_C, priors, validate
from riptide_toy.constants import SEED, SIGMA_EP, SIGMA_THETA

SHAPES = ["gauss", "uniform", "bimodal", "lognormal"]
N_M = [(1000, 40), (300, 100), (150, 200), (50, 200)]  # come la checklist del Caso C
MU_RANGE = (2.5, 4.0)       # MeV, come la checklist del Caso C
SIGMA_RANGE = (0.2, 0.6)    # MeV, log-uniforme, come la checklist del Caso C
LEVELS = np.array([0.68, 0.90, 0.95])


def run_experiment(task: tuple[str, int, int]) -> dict:
    """Un esperimento: spettro della forma data, ricostruzione del Caso C, riassunto.

    Ritorna:
        dict con forma, N, verita' (mu_E e sigma_E in MeV, Omega_n versore (3,)),
        media/std/mediana/intervalli di mu_E (MeV) e log(sigma_E / MeV), errore,
        risoluzione (rad) e copertura HPD (bool, (len(LEVELS),)) di Omega_n.
    """
    shape, n_events, index = task
    rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(n_events, index)))
    mu_true = rng.uniform(*MU_RANGE)
    sigma_true = np.exp(rng.uniform(*np.log(SIGMA_RANGE)))
    omega_true = kinematics.direction_from_theta_phi(
        np.arccos(rng.uniform(-1.0, 1.0, 1)), rng.uniform(0.0, 2 * np.pi, 1)
    )
    En = kinematics.sample_energy_spectrum(rng, shape, mu_true, sigma_true, n_events)
    Ep_true, track = kinematics.sample_recoil_events(rng, En, omega_true[0])
    D_B = (rng.normal(Ep_true, SIGMA_EP), kinematics.smear_direction(rng, track, SIGMA_THETA))

    theta_grid, phi_grid = grids.sphere_grid()
    direction_prior = priors.direction_prior(theta_grid, phi_grid)
    mu_grid, sigma_grid = grids.hyperparameter_grid()
    omega_0 = posterior_C.refine_shared_direction(D_B, theta_grid, phi_grid, direction_prior)[0]
    log_post, mu_fine, sigma_fine = posterior_C.refine_hyperparameters(
        (*D_B, omega_0), mu_grid, sigma_grid, priors.hyperparameter_prior)
    out = {"shape": shape, "N": n_events, "mu_true": mu_true, "ls_true": np.log(sigma_true),
           "En_mean": En.mean(), "En_ls": np.log(En.std())}
    levels_with_median = np.concatenate([[0.0], LEVELS])
    for key, grid in (("mu", mu_fine), ("ls", np.log(sigma_fine))):
        mean, std = validate.posterior_mean_std(log_post[None, :], grid)
        intervals = validate.credible_interval(log_post[None, :], grid, levels_with_median)[0]
        out[key] = {"mean": mean[0], "std": std[0], "median": intervals[0, 0], "int": intervals[1:]}

    omega_hat, cap_log_post, cap_theta, cap_phi = posterior_C.refine_shared_direction_hierarchical(
        D_B, theta_grid, phi_grid, direction_prior,
        *posterior_C.predictive_energy_moments(log_post, mu_fine, sigma_fine)
    )
    cap_grid = kinematics.direction_from_theta_phi(cap_theta, cap_phi)
    # come nella checklist: la verita' e' "nella calotta" solo entro 3 pixel dal pixel piu' vicino
    cap_pixel = np.sqrt(2 * np.pi * (1.0 - np.min(cap_grid @ omega_hat[0])) / cap_grid.shape[0])
    nearest = np.argmax(cap_grid @ omega_true[0])
    truth_in_cap = np.arccos(np.clip(cap_grid[nearest] @ omega_true[0], -1.0, 1.0)) < 3 * cap_pixel
    out["omega_error"] = validate.angular_residual(omega_true, omega_hat)[0]
    out["omega_sigma"] = validate.posterior_angular_resolution(cap_log_post[None, :], cap_grid, omega_hat)[0]
    out["omega_covered"] = validate.credible_region_contains(
        cap_log_post[None, :], np.array([nearest]), LEVELS)[0] & truth_in_cap
    return out


def rms(x: np.ndarray) -> float:
    """Radice della media dei quadrati.

    Ritorna:
        float, stesse unita' di x.
    """
    return float(np.sqrt(np.mean(x ** 2)))


def print_table(rows: list[dict]) -> None:
    """Per forma e N: bias, risoluzione, pull e coverage di mu_E, log sigma_E e Omega_n.

    Ritorna:
        None (stdout).
    """
    print("Caso C con spettro non gaussiano; mu in MeV, Omega in gradi;"
          " bias = media dei residui +- errore sulla media")
    for n_events in sorted({r["N"] for r in rows}):
        print(f"\nN={n_events}")
        for shape in SHAPES:
            sel = [r for r in rows if r["N"] == n_events and r["shape"] == shape]
            m = len(sel)
            line = f"  {shape:9s} (M={m:3d})"
            for key, label in (("mu", "mu"), ("ls", "log sigma")):
                truth = np.array([r[f"{key}_true"] for r in sel])
                mean = np.array([r[key]["mean"] for r in sel])
                std = np.array([r[key]["std"] for r in sel])
                # stima puntuale come nella checklist: media per mu, mediana per log sigma
                point = mean if key == "mu" else np.array([r[key]["median"] for r in sel])
                res = point - truth
                pull = (mean - truth) / std
                cov = validate.coverage_curve(truth, np.array([r[key]["int"] for r in sel]), LEVELS)[1]
                line += (f"\n    {label:9s} bias {res.mean():+.4f} +- {res.std() / np.sqrt(m):.4f}"
                         f" | rms {rms(res):.4f} (std post. {std.mean():.4f})"
                         f" | pull {pull.mean():+.2f}/{pull.std():.2f} | coverage {np.round(cov, 2)}")
            err = np.array([r["omega_error"] for r in sel])
            sig = np.array([r["omega_sigma"] for r in sel])
            cov = np.array([r["omega_covered"] for r in sel]).mean(axis=0)
            line += (f"\n    Omega     rms {np.degrees(rms(err)):.2f} (sigma post. {np.degrees(sig.mean()):.2f})"
                     f" | pull rms {rms(validate.angular_pull(err, sig)):.2f} | coverage HPD {np.round(cov, 2)}")
            print(line)
        print(f"  (errore binomiale a 0.68: +-{np.sqrt(0.68 * 0.32 / m):.3f})")


def main() -> None:
    """Esegue gli esperimenti in parallelo, salva i risultati e stampa la tabella.

    Ritorna:
        None (pkl in outputs/, tabella su stdout).
    """
    n_proc = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    tasks = [(shape, n, i) for n, m in N_M for shape in SHAPES for i in range(m)]
    with Pool(n_proc) as pool:
        rows = pool.map(run_experiment, tasks, chunksize=1)
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "stress_spettro_results.pkl", "wb") as f:
        pickle.dump({"SHAPES": SHAPES, "N_M": N_M, "LEVELS": LEVELS, "results": rows}, f)
    print_table(rows)


if __name__ == "__main__":
    main()
