"""Caso C: da dove viene lo scarto modo congiunto - media su log sigma_E.

La relazione di Pearson (asimmetria del marginale) sbaglia il segno: il marginale di
log sigma_E ha la coda verso sigma_E piccoli, eppure il modo congiunto sta sotto la media.
Sugli stessi dataset di correlazione_stadi.py (spawn_key=(N, i)) lo scarto si scompone in:
  - asimmetria: modo del marginale di log sigma_E - media (griglia fine dello stadio 2);
  - volume di mu_E: modo 2D in (mu_E, log sigma_E) - modo del marginale (marginalizzare mu_E
    premia i sigma_E grandi, dove il posterior di mu_E e' piu' largo);
  - numerica: modo di joint_laplace con i passi d'uso (stencil = larghezze dei due stadi,
    2 passi di Newton; letto da outputs/correlazione_stadi_results.pkl) - modo ripartendo da
    li' con stencil STEP_SHRINK volte piu' stretto e N_NEWTON_FINE passi;
  - resto (direzione: Omega_n marginalizzato invece che fissato a omega_0 e partenza
    gerarchica): modo convergente di joint_laplace - modo 2D.
Tutto in unita' della std posteriore s dei due stadi.

Uso: OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=5G -p MemorySwapMax=0 \
         python scripts/modo_log_sigma.py [n_processi]
     (picco ~0.25 GB per processo a N=300)
Richiede: outputs/correlazione_stadi_results.pkl.
Produce: outputs/modo_log_sigma_results.pkl e la tabella su stdout.
"""
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from riptide_toy import grids, kinematics, posterior_C, priors, validate
from riptide_toy.constants import N_MU_FINE, N_SIGMA_FINE, SEED, SIGMA_EP, SIGMA_THETA

MU_RANGE = (2.5, 4.0)       # MeV, come la checklist del Caso C
SIGMA_RANGE = (0.2, 0.6)    # MeV, log-uniforme, come la checklist del Caso C
STEP_SHRINK = 4.0           # errore cubico del gradiente a differenze centrali ~ passo^2
N_NEWTON_FINE = 2  # si parte gia' dal modo d'uso: Newton converge quadraticamente


def parabolic_peak(x: np.ndarray, y: np.ndarray) -> float:
    """Ascissa del massimo di y(x) su griglia uniforme, con parabola sui tre punti attorno
    al massimo discreto (il massimo al bordo resta sul bordo).

    Ritorna:
        float, stessa unita' di x.
    """
    j = int(np.argmax(y))
    if j == 0 or j == x.size - 1:
        return float(x[j])
    denom = y[j - 1] - 2 * y[j] + y[j + 1]
    return float(x[j] + 0.5 * (x[1] - x[0]) * (y[j - 1] - y[j + 1]) / denom)


def run_experiment(task: tuple[int, int, np.ndarray, float, float]) -> dict:
    """Un esperimento: dataset di correlazione_stadi.py, griglia fine dello stadio 2 e
    joint_laplace convergente a partire dal modo d'uso (Omega_n versore (3,), mu_E in MeV,
    log sigma_E) passato nel task.

    Ritorna:
        dict con N, log sigma_E vero, media, std e mediana dei due stadi, modo del marginale,
        modo 2D, modo di joint_laplace d'uso e convergente (tutti log sigma_E, adimensionale)
        e i due flag di Hessiana definita negativa.
    """
    n_events, index, omega_usual, mu_usual, ls_usual = task
    # stesso flusso di numeri casuali di correlazione_stadi.py: stessi dataset
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
    ls_median = validate.credible_interval(log_post[None, :], np.log(sigma_fine), np.array([0.0]))[0, 0, 0]

    table = log_post.reshape(N_MU_FINE, N_SIGMA_FINE)  # asse 0 = mu_E, asse 1 = sigma_E
    ls_axis = np.log(sigma_fine.reshape(N_MU_FINE, N_SIGMA_FINE)[0])
    weights = np.exp(table - table.max())
    marginal_mode = parabolic_peak(ls_axis, np.log(weights.sum(axis=0)))
    profile_mode = parabolic_peak(ls_axis, table.max(axis=0))

    omega_hat, cap_log_post, cap_theta, cap_phi = posterior_C.refine_shared_direction_hierarchical(
        D_B, theta_grid, phi_grid, direction_prior,
        *posterior_C.predictive_energy_moments(log_post, mu_fine, sigma_fine))
    cap_grid = kinematics.direction_from_theta_phi(cap_theta, cap_phi)
    omega_sigma = validate.posterior_angular_resolution(cap_log_post[None, :], cap_grid, omega_hat)[0]
    steps = np.array([omega_sigma / np.sqrt(2), omega_sigma / np.sqrt(2), mu_std[0], ls_std[0]])
    _, hyper_fine, _, nd_fine = posterior_C.joint_laplace(
        D_B, omega_usual, mu_usual, ls_usual, steps / STEP_SHRINK, priors.hyperparameter_prior,
        n_newton=N_NEWTON_FINE)
    return {"N": n_events, "ls_true": float(np.log(sigma_true)),
            "mean": ls_mean[0], "std": ls_std[0], "median": float(ls_median),
            "marginal_mode": marginal_mode, "profile_mode": profile_mode,
            "laplace_usual": ls_usual, "laplace_fine": hyper_fine[1], "nd_fine": nd_fine}


def print_table(rows: list[dict]) -> None:
    """Per N: scomposizione media (+- errore della media) dello scarto modo congiunto - media
    in unita' di s, e pull medio di ciascuna stima puntuale.

    Ritorna:
        None (stdout).
    """
    print("Caso C, log sigma_E: scomposizione di (modo congiunto - media)/s")
    for n_events in sorted({r["N"] for r in rows}):
        sel = [r for r in rows if r["N"] == n_events and r["nd_fine"]]
        a = {k: np.array([r[k] for r in sel]) for k in sel[0] if k not in ("N", "nd_fine")}
        s = a["std"]
        parts = {
            "asimmetria (modo marg. - media)": (a["marginal_mode"] - a["mean"]) / s,
            "volume mu (modo 2D - modo marg.)": (a["profile_mode"] - a["marginal_mode"]) / s,
            "direzione (Laplace conv. - modo 2D)": (a["laplace_fine"] - a["profile_mode"]) / s,
            "numerica (Laplace d'uso - conv.)": (a["laplace_usual"] - a["laplace_fine"]) / s,
            "totale (Laplace d'uso - media)": (a["laplace_usual"] - a["mean"]) / s,
        }
        print(f"\nN={n_events} (M={len(sel)} con Hessiane definite negative)")
        for name, x in parts.items():
            print(f"  {name:38s} {x.mean():+.3f} +- {x.std(ddof=1) / np.sqrt(x.size):.3f}"
                  f"   | /s: {np.mean(x / s):+.2f}")
        for name in ("mean", "median", "marginal_mode", "profile_mode", "laplace_fine", "laplace_usual"):
            pull = (a[name] - a["ls_true"]) / s
            print(f"  pull {name:14s} {pull.mean():+.3f} / {pull.std():.3f}")


def main() -> None:
    """Esegue gli esperimenti in parallelo, salva i risultati e stampa la tabella.

    Ritorna:
        None (pkl in outputs/, tabella su stdout).
    """
    n_proc = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    with open(out_dir / "correlazione_stadi_results.pkl", "rb") as f:
        usual = pickle.load(f)
    # solo gli esperimenti dove la Laplace d'uso e' arrivata al modo (Hessiana definita negativa)
    tasks = [(n, i, r["joint"]["omega"][0], r["joint"]["mu"], r["joint"]["ls"])
             for (n, i), r in zip(((n, i) for n, m in usual["N_M"] for i in range(m)), usual["results"])
             if r["negative_definite"]]
    with Pool(n_proc) as pool:
        rows = pool.map(run_experiment, tasks, chunksize=1)
    with open(out_dir / "modo_log_sigma_results.pkl", "wb") as f:
        pickle.dump({"N_M": usual["N_M"], "results": rows}, f)
    print_table(rows)


if __name__ == "__main__":
    main()
