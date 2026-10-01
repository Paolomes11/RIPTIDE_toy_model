"""Checklist di validazione quantitativa del Caso B: direzione condivisa Omega_n, E_n per evento.

Ordine della checklist: bias -> risoluzione -> pull -> coverage -> contrazione ~1/sqrt(N).
M esperimenti indipendenti per ciascun N; per ogni esperimento verita' nuova:
Omega_n isotropa, E_n^(k) ~ U(EN_MIN, EN_MAX) per evento, cioe' estratta dallo stesso prior
largo che il Caso B usa per marginalizzare E_n. Con verita' estratta dal prior la regione
credibile deve coprire al livello nominale (in media sugli esperimenti).
Stima: combinato del Caso B su griglia sferica grossolana + calotta fine attorno al MAP
(posterior_C.refine_shared_direction, che e' il Caso B puro a prior largo).

Uso: OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 \
         python scripts/caso_B_checklist.py [n_processi]
     (un thread BLAS per processo; il tetto MemoryMax fa uccidere dal kernel solo lo
     script, non l'editor)
Produce: outputs/caso_B_checklist_results.pkl, outputs/caso_B_contrazione.png e la tabella su stdout.
"""
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from riptide_toy import grids, kinematics, posterior_C, priors, validate
from riptide_toy.constants import EN_MAX, EN_MIN, SEED, SIGMA_EP, SIGMA_THETA

# (N eventi, M esperimenti): M scala circa con 1/N per tenere costante il costo per N
N_M = [(1000, 40), (300, 100), (100, 200), (30, 300), (10, 400)]
LEVELS = np.array([0.68, 0.90, 0.95])


def run_experiment(task: tuple[int, int]) -> dict:
    """Un esperimento: dataset sintetico del Caso B, direzione stimata sulla calotta fine.

    Args:
        task: (N eventi, indice dell'esperimento), usati anche per il seme.

    Ritorna:
        dict con N, errore angolare (rad), risoluzione dichiarata (rad) e flag di
        copertura HPD, array bool forma (len(LEVELS),).
    """
    n_events, index = task
    rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(n_events, index)))
    omega_true = kinematics.direction_from_theta_phi(
        np.arccos(rng.uniform(-1.0, 1.0, 1)), rng.uniform(0.0, 2 * np.pi, 1)
    )
    Ep_true, track = kinematics.sample_recoil_events(
        rng, rng.uniform(EN_MIN, EN_MAX, n_events), omega_true[0]
    )
    D_B = (rng.normal(Ep_true, SIGMA_EP), kinematics.smear_direction(rng, track, SIGMA_THETA))

    theta_grid, phi_grid = grids.sphere_grid()
    direction_prior = priors.direction_prior(theta_grid, phi_grid)
    omega_hat, cap_log_post, cap_theta, cap_phi = posterior_C.refine_shared_direction(
        D_B, theta_grid, phi_grid, direction_prior
    )
    cap_grid = kinematics.direction_from_theta_phi(cap_theta, cap_phi)
    # la verita' conta come "nella calotta" solo se dista meno di 3 pixel dal pixel piu' vicino:
    # altrimenti e' fuori dalla regione calcolata e non puo' essere coperta
    cap_pixel = np.sqrt(2 * np.pi * (1.0 - np.min(cap_grid @ omega_hat[0])) / cap_grid.shape[0])
    nearest = np.argmax(cap_grid @ omega_true[0])
    truth_in_cap = np.arccos(np.clip(cap_grid[nearest] @ omega_true[0], -1.0, 1.0)) < 3 * cap_pixel
    return {
        "N": n_events,
        "omega_error": validate.angular_residual(omega_true, omega_hat)[0],
        "omega_sigma": validate.posterior_angular_resolution(cap_log_post[None, :], cap_grid, omega_hat)[0],
        "omega_covered": validate.credible_region_contains(
            cap_log_post[None, :], np.array([nearest]), LEVELS
        )[0] & truth_in_cap,
    }


def rms(x: np.ndarray) -> float:
    """Radice della media dei quadrati.

    Ritorna:
        float, stesse unita' di x.
    """
    return float(np.sqrt(np.mean(x ** 2)))


def main() -> None:
    """Esegue gli esperimenti in parallelo, salva i risultati grezzi e stampa la tabella.

    Ritorna:
        None (pkl e png in outputs/, tabella su stdout).
    """
    n_proc = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    tasks = [(n, i) for n, m in N_M for i in range(m)]
    with Pool(n_proc) as pool:
        results = pool.map(run_experiment, tasks, chunksize=1)
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    # risultati grezzi per il notebook 03, senza rilanciare il run
    with open(out_dir / "caso_B_checklist_results.pkl", "wb") as f:
        pickle.dump({"N_M": N_M, "LEVELS": LEVELS, "results": results}, f)

    n_values = sorted(n for n, _ in N_M)
    print("Caso B, checklist di validazione; angoli in gradi; stima = MAP sulla calotta")
    curve = []
    for n in n_values:
        rows = [r for r in results if r["N"] == n]
        err = np.array([r["omega_error"] for r in rows])
        sig = np.array([r["omega_sigma"] for r in rows])
        cov = np.array([r["omega_covered"] for r in rows]).mean(axis=0)
        m = len(rows)
        curve.append(rms(err))
        # per una gaussiana 2D isotropa con sigma per asse s, E[d^2] = 2 s^2: il pull
        # d / sqrt(E_post[d^2]) ha rms atteso 1
        print(f"N={n:5d} (M={m:3d})  err medio {np.degrees(err.mean()):6.2f}"
              f" | rms err {np.degrees(rms(err)):6.2f} (risoluzione dichiarata media {np.degrees(sig.mean()):6.2f})"
              f" | pull rms {rms(validate.angular_pull(err, sig)):.2f}"
              f" | coverage HPD {np.round(cov, 2)} (+-{np.sqrt(0.68 * 0.32 / m):.2f} binomiale a 0.68)")
    slope = np.polyfit(np.log(n_values), np.log(curve), 1)[0]
    print(f"\ncontrazione: pendenza log-log di rms vs N {slope:+.3f} (attesa -0.5)"
          f"  rms*sqrt(N) [rad] = {np.round(np.array(curve) * np.sqrt(n_values), 3)}")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.loglog(n_values, np.degrees(curve), "o-", label="rms errore Omega_n")
    ax.loglog(n_values, np.degrees(curve[0]) * np.sqrt(n_values[0] / np.array(n_values)), "k:",
              label="1/sqrt(N)")
    ax.set_xlabel("N eventi")
    ax.set_ylabel("rms errore angolare [gradi]")
    ax.set_title("Caso B: contrazione")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "caso_B_contrazione.png", dpi=150)


if __name__ == "__main__":
    main()
