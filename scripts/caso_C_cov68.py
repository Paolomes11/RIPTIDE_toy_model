"""Riga 14, report §11.4: coverage al 68% di log sigma_E a N = 150-300 su seed nuovi.

Stesso generatore e stessa pipeline della checklist (scripts/caso_C_checklist.py,
v0.11), ma con seed indipendenti da quelli della checklist, spawn_key=(N, i, 68),
e M piu' grande. Lo stadio 2 (prior di default) e' calcolato due volte: su
omega_0 dello stadio 1 e sulla Omega_n vera, per separare il contributo dello
stadio 1.

Uso: OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=5G -p MemorySwapMax=0 \
         python scripts/caso_C_cov68.py [n_processi]
     (~11 min con 3 processi, misurato 2026-09-26)
Produce: outputs/caso_C_cov68_results.pkl e la tabella su stdout.
"""
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from riptide_toy import grids, kinematics, posterior_C, priors, validate
from riptide_toy.constants import SEED, SIGMA_EP, SIGMA_THETA

sys.path.insert(0, str(Path(__file__).parent))
from caso_C_checklist import LEVELS, MU_RANGE, SIGMA_RANGE  # noqa: E402  stesso generatore

N_M = [(150, 600), (300, 300)]
SEED_TAG = 68  # terzo elemento di spawn_key: seed disgiunti da quelli della checklist


def summarize(log_post: np.ndarray, mu_fine: np.ndarray, sigma_fine: np.ndarray) -> dict:
    """Riassunto dello stadio 2 come nella checklist.

    Ritorna:
        dict: mu_mean, mu_std [MeV], mu_int (n_livelli, 2) [MeV]; ls_mean, ls_std,
        ls_median, ls_int (n_livelli, 2) su log(sigma_E / MeV).
    """
    levels = np.concatenate([[0.0], LEVELS])
    mu_mean, mu_std = validate.posterior_mean_std(log_post[None, :], mu_fine)
    ls_mean, ls_std = validate.posterior_mean_std(log_post[None, :], np.log(sigma_fine))
    mu_int = validate.credible_interval(log_post[None, :], mu_fine, levels)[0]
    ls_int = validate.credible_interval(log_post[None, :], np.log(sigma_fine), levels)[0]
    return {"mu_mean": mu_mean[0], "mu_std": mu_std[0], "mu_int": mu_int[1:],
            "ls_mean": ls_mean[0], "ls_std": ls_std[0], "ls_median": ls_int[0, 0], "ls_int": ls_int[1:]}


def run_experiment(task: tuple[int, int, int]) -> dict:
    """Un esperimento: dataset sintetico, stadio 1 a prior largo, stadio 2 su omega_0 e su Omega vera.

    Args:
        task: (N eventi, indice dell'esperimento, tag del seed), spawn_key=(N, i, tag).

    Ritorna:
        dict con N, i, verita' (mu_true, sigma_true in MeV) e i riassunti "hat" e "true".
    """
    n_events, index, seed_tag = task
    rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(n_events, index, seed_tag)))
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
    mu_grid, sigma_grid = grids.hyperparameter_grid()
    omega_0 = posterior_C.refine_shared_direction(
        D_B, theta_grid, phi_grid, priors.direction_prior(theta_grid, phi_grid)
    )[0]
    out = {"N": n_events, "i": index, "mu_true": mu_true, "sigma_true": sigma_true}
    for name, omega in (("hat", omega_0), ("true", omega_true)):
        out[name] = summarize(*posterior_C.refine_hyperparameters(
            (*D_B, omega), mu_grid, sigma_grid, priors.hyperparameter_prior
        ))
    return out


def main() -> None:
    n_proc = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    tasks = [(n, i, SEED_TAG) for n, m in N_M for i in range(m)]
    with Pool(n_proc) as pool:
        results = pool.map(run_experiment, tasks, chunksize=1)
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "caso_C_cov68_results.pkl", "wb") as f:
        pickle.dump({"N_M": N_M, "LEVELS": LEVELS, "results": results}, f)
    for n, m in N_M:
        rows = [r for r in results if r["N"] == n]
        log_sigma_true = np.log([r["sigma_true"] for r in rows])
        for name in ("hat", "true"):
            intervals = np.array([r[name]["ls_int"] for r in rows])
            _, cov = validate.coverage_curve(log_sigma_true, intervals, LEVELS)
            print(f"N={n} M={m} [{name}] coverage log sigma_E {np.round(cov, 3)}")


if __name__ == "__main__":
    main()
