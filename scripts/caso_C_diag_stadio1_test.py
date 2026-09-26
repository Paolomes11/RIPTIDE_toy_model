"""Riga 14, report §13.4: margine dei test dello stadio 1 in tests/test_case_C.py.

Stesso generatore dei test (Omega_n a theta=0.9, phi=2.1 rad; En ~ N(3.0, 0.3) MeV),
su seed SEED + s:
    global  N=60,        estimate_shared_direction, 200 seed  (test end-to-end, soglia 15 gradi)
    refine  N=300, 1000, refine_shared_direction,   100 seed  (test_stage1_error_contracts_at_large_N, soglia 5 gradi)
Il seed s = 0 riproduce la realizzazione dei test.

Uso: OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=5G -p MemorySwapMax=0 \
         python scripts/caso_C_diag_stadio1_test.py [n_processi]
     (~3.5 min con 3 processi, misurato 2026-09-26)
Produce: outputs/caso_C_diag_stadio1_test.pkl e la tabella su stdout.
"""
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from riptide_toy import grids, kinematics, posterior_C, priors
from riptide_toy.constants import SEED, SIGMA_EP, SIGMA_THETA

OMEGA_THETA_PHI = (0.9, 2.1)  # rad, come in tests/test_case_C.py
EN_MEAN, EN_SD = 3.0, 0.3     # MeV, come in tests/test_case_C.py
TASKS = [(SEED + s, 60, "global") for s in range(200)] + \
        [(SEED + s, n, "refine") for n in (300, 1000) for s in range(100)]


def run_one(task: tuple[int, int, str]) -> tuple[int, int, str, float, float]:
    """Una realizzazione del test: dati sintetici e stima di Omega_n con il metodo richiesto.

    Args:
        task: (seed, N eventi, "global" per estimate_shared_direction o "refine" per refine_shared_direction).

    Ritorna:
        (seed, N, metodo, errore angolare in gradi, tempo della stima in s).
    """
    seed, n_events, method = task
    rng = np.random.default_rng(seed)
    omega = kinematics.direction_from_theta_phi(np.array([OMEGA_THETA_PHI[0]]), np.array([OMEGA_THETA_PHI[1]]))[0]
    Ep_true, track_true = kinematics.sample_recoil_events(rng, rng.normal(EN_MEAN, EN_SD, n_events), omega)
    track_hat = kinematics.smear_direction(rng, track_true, SIGMA_THETA)
    Ep_hat = rng.normal(Ep_true, SIGMA_EP)
    theta_grid, phi_grid = grids.sphere_grid()
    direction_prior = priors.direction_prior(theta_grid, phi_grid)
    start = time.perf_counter()
    if method == "global":
        omega_hat = posterior_C.estimate_shared_direction((Ep_hat, track_hat), theta_grid, phi_grid, direction_prior)
    else:
        omega_hat = posterior_C.refine_shared_direction((Ep_hat, track_hat), theta_grid, phi_grid, direction_prior)[0]
    elapsed = time.perf_counter() - start
    return seed, n_events, method, float(np.degrees(np.arccos(np.clip(omega_hat[0] @ omega, -1.0, 1.0)))), elapsed


def main() -> None:
    n_proc = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    with Pool(n_proc) as pool:
        results = pool.map(run_one, TASKS, chunksize=1)
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "caso_C_diag_stadio1_test.pkl", "wb") as f:
        pickle.dump(results, f)
    for n, method in ((60, "global"), (300, "refine"), (1000, "refine")):
        err = np.array([r[3] for r in results if r[1] == n and r[2] == method])
        dt = np.array([r[4] for r in results if r[1] == n and r[2] == method])
        print(f"N={n} [{method}] M={err.size}: errore mediano {np.median(err):.1f}, "
              f"95% {np.percentile(err, 95):.1f}, max {err.max():.1f} gradi; tempo mediano {np.median(dt):.1f} s")


if __name__ == "__main__":
    main()
