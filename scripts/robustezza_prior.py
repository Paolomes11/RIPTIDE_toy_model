"""Robustezza al prior (ultimo punto della checklist di validazione) per i Casi A, B e C.

Stessi dataset ricostruiti col prior di default e con prior alternativi propri; metrica:
spostamento della stima in unita' della deviazione standard posteriore del default, e coverage
col prior alternativo (verita' estratta dai generatori di default). Attesa: spostamento -> 0 con N.
  A (E_n condivisa): prior piatto su [EN_MIN, EN_MAX] vs piatto su [EN_MIN, EN_MAX_WIDE],
     log-uniforme su [EN_MIN, EN_MAX], log-uniforme su [EN_MIN, EN_MAX_WIDE].
  B (Omega_n, E_n nuisance): prior su E_n come sopra (piatto largo, log-uniforme), e prior su
     Omega_n von Mises-Fisher (concentrazione KAPPA) con asse a 90 gradi dalla verita'.
  C (iperparametri): prior uniforme in log sigma_E vs uniforme in sigma_E, a tutti gli N.
I seed di B e C sono quelli delle checklist (spawn_key=(N, i)): col prior di default si
ritrovano i loro numeri (controllo di consistenza).

Uso: OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=5G -p MemorySwapMax=0 \
         python scripts/robustezza_prior.py [n_processi]
     (~0.5 GB per processo; ~15 min con 3 processi, stima da un esperimento per caso, 2026-10-02)
Produce: outputs/robustezza_prior_results.pkl e le tabelle su stdout.
"""
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from riptide_toy import forward_model, grids, kinematics, posterior_A, posterior_C, priors, validate
from riptide_toy.constants import EN_MAX, EN_MIN, SEED, SIGMA_EP, SIGMA_THETA

LEVELS = np.array([0.68, 0.90, 0.95])
# (N eventi, M esperimenti) per caso
# Caso A fino a N = 100: a N = 1000 la std posteriore (~0.01 MeV) e' pari al passo della
# griglia su E_n e media/intervalli sarebbero dominati dalla discretizzazione
N_M_A = [(1, 400), (3, 400), (10, 400), (30, 200), (100, 200)]
N_M_B = [(10, 200), (30, 200), (100, 100), (300, 50)]
N_M_C = [(50, 200), (150, 200), (300, 100), (1000, 40)]
KAPPA = 2.0                 # prior su Omega_n larga ~1/sqrt(KAPPA) rad ~ 40 gradi
PRIOR_AXIS_ANGLE = np.pi / 2  # asse della prior a 90 gradi dalla verita': gradiente massimo
MU_RANGE = (2.5, 4.0)       # MeV, come la checklist del Caso C
SIGMA_RANGE = (0.2, 0.6)    # MeV, log-uniforme, come la checklist del Caso C
CASE_A_TAG = 1              # spawn_key=(CASE_A_TAG, N, i): disgiunti dai seed (N, i) di B e C


def energy_priors() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Griglie e prior su E_n confrontati; il primo e' il default.

    Ritorna:
        dict nome -> (griglia En in MeV forma (n,), densita' 1/MeV forma (n,)).
    """
    default, wide = grids.energy_grid(), grids.energy_grid_wide()
    return {
        "piatto": (default, priors.energy_prior(default)),
        "piatto largo": (wide, priors.energy_prior(wide)),
        "logU": (default, priors.energy_prior_log_uniform(default)),
        "logU largo": (wide, priors.energy_prior_log_uniform(wide)),
    }


def run_case_A(task: tuple[int, int]) -> dict:
    """Caso A: N eventi con E_n condivisa, posterior combinato con ciascun prior su E_n.

    Ritorna:
        dict con N, E_n vera (MeV) e, per prior, media/std (MeV) e intervalli (len(LEVELS), 2).
    """
    n_events, index = task
    rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(CASE_A_TAG, n_events, index)))
    En_true = rng.uniform(EN_MIN, EN_MAX)
    theta = rng.uniform(0.0, np.pi / 2, n_events)
    D = forward_model.measure(kinematics.proton_energy(np.full(n_events, En_true), theta), theta,
                              SIGMA_EP, SIGMA_THETA, rng)
    out = {"N": n_events, "truth": En_true}
    for name, (en_grid, prior) in energy_priors().items():
        # prior contato una volta: si somma il log-posterior per evento e si tolgono N-1 prior
        log_post = (posterior_A.single_event_posterior(D, en_grid, prior).sum(axis=0)
                    - (n_events - 1) * np.log(prior))[None, :]
        mean, std = validate.posterior_mean_std(log_post, en_grid)
        out[name] = {"mean": mean[0], "std": std[0],
                     "int": validate.credible_interval(log_post, en_grid, LEVELS)[0]}
    return out


def direction_summary(result: tuple, omega_true: np.ndarray) -> dict:
    """Riassunto di un raffinamento di Omega_n: stima, risoluzione, copertura HPD.

    Ritorna:
        dict con omega_hat (3,), omega_error e omega_sigma in rad, omega_covered bool (len(LEVELS),).
    """
    omega_hat, cap_log_post, cap_theta, cap_phi = result
    cap_grid = kinematics.direction_from_theta_phi(cap_theta, cap_phi)
    # come nelle checklist: la verita' e' "nella calotta" solo entro 3 pixel dal pixel piu' vicino
    cap_pixel = np.sqrt(2 * np.pi * (1.0 - np.min(cap_grid @ omega_hat[0])) / cap_grid.shape[0])
    nearest = np.argmax(cap_grid @ omega_true[0])
    truth_in_cap = np.arccos(np.clip(cap_grid[nearest] @ omega_true[0], -1.0, 1.0)) < 3 * cap_pixel
    return {
        "omega_hat": omega_hat[0],
        "omega_error": validate.angular_residual(omega_true, omega_hat)[0],
        "omega_sigma": validate.posterior_angular_resolution(cap_log_post[None, :], cap_grid, omega_hat)[0],
        "omega_covered": validate.credible_region_contains(
            cap_log_post[None, :], np.array([nearest]), LEVELS)[0] & truth_in_cap,
    }


def run_case_B(task: tuple[int, int]) -> dict:
    """Caso B: dataset della checklist (stessi seed), Omega_n con ciascun prior.

    Ritorna:
        dict con N e, per prior, il riassunto di direction_summary.
    """
    n_events, index = task
    rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(n_events, index)))
    omega_true = kinematics.direction_from_theta_phi(
        np.arccos(rng.uniform(-1.0, 1.0, 1)), rng.uniform(0.0, 2 * np.pi, 1)
    )
    Ep_true, track = kinematics.sample_recoil_events(
        rng, rng.uniform(EN_MIN, EN_MAX, n_events), omega_true[0]
    )
    Ep_hat, track_hat = rng.normal(Ep_true, SIGMA_EP), kinematics.smear_direction(rng, track, SIGMA_THETA)
    # asse della prior su Omega_n: estratto dopo il dataset, che resta identico alla checklist
    axis = kinematics.direction_around_axis(omega_true, np.array([PRIOR_AXIS_ANGLE]),
                                            rng.uniform(0.0, 2 * np.pi, 1))
    axis_theta, axis_phi = kinematics.theta_phi_from_direction(axis)

    theta_grid, phi_grid = grids.sphere_grid()
    out = {"N": n_events}
    for name in ("piatto", "piatto largo", "logU"):
        en_grid, prior = energy_priors()[name]
        table = forward_model.track_energy_table(Ep_hat, en_grid, SIGMA_EP, SIGMA_THETA, np.log(prior))
        out[name] = direction_summary(posterior_C.refine_direction_from_table_with_prior(
            table, track_hat, theta_grid, phi_grid, priors.direction_prior), omega_true)
        if name == "piatto":
            out["vMF"] = direction_summary(posterior_C.refine_direction_from_table_with_prior(
                table, track_hat, theta_grid, phi_grid,
                lambda t, p: priors.direction_prior_von_mises_fisher(t, p, axis_theta[0], axis_phi[0], KAPPA)),
                omega_true)
    return out


def run_case_C(task: tuple[int, int]) -> dict:
    """Caso C: dataset della checklist (stessi seed), stadio 2 con i due prior su sigma_E.

    Ritorna:
        dict con N, verita' e, per prior, media/std/intervalli di mu_E (MeV) e log(sigma_E / MeV).
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
    mu_grid, sigma_grid = grids.hyperparameter_grid()
    omega_0 = posterior_C.refine_shared_direction(
        D_B, theta_grid, phi_grid, priors.direction_prior(theta_grid, phi_grid))[0]
    out = {"N": n_events, "mu_true": mu_true, "ls_true": np.log(sigma_true)}
    for name, prior_fn in (("logU", priors.hyperparameter_prior),
                           ("U", priors.hyperparameter_prior_uniform_sigma)):
        log_post, mu_fine, sigma_fine = posterior_C.refine_hyperparameters(
            (*D_B, omega_0), mu_grid, sigma_grid, prior_fn)
        summary = {}
        for key, grid in (("mu", mu_fine), ("ls", np.log(sigma_fine))):
            mean, std = validate.posterior_mean_std(log_post[None, :], grid)
            summary[key] = {"mean": mean[0], "std": std[0],
                            "int": validate.credible_interval(log_post[None, :], grid, LEVELS)[0]}
        out[name] = summary
    return out


def run_task(task: tuple[str, int, int]) -> tuple[str, dict]:
    """Smista un esperimento al caso giusto.

    Ritorna:
        (caso, dict dell'esperimento).
    """
    case, n_events, index = task
    fn = {"A": run_case_A, "B": run_case_B, "C": run_case_C}[case]
    return case, fn((n_events, index))


def covered(truth: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    """Frazione di intervalli centrali che contengono la verita', per livello.

    Ritorna:
        array (len(LEVELS),).
    """
    return validate.coverage_curve(truth, intervals, LEVELS)[1]


def print_scalar_table(title: str, rows: list[dict], names: list[str], key: str | None,
                       truth_key: str) -> None:
    """Tabella per un parametro scalare: spostamento vs default in unita' di std, pull, coverage.

    Ritorna:
        None (stdout).
    """
    print(f"\n{title}")
    get = (lambda r, n: r[n]) if key is None else (lambda r, n: r[n][key])
    for n_events in sorted({r["N"] for r in rows}):
        sel = [r for r in rows if r["N"] == n_events]
        truth = np.array([r[truth_key] for r in sel])
        ref_mean = np.array([get(r, names[0])["mean"] for r in sel])
        ref_std = np.array([get(r, names[0])["std"] for r in sel])
        print(f"  N={n_events:5d} (M={len(sel):3d})")
        for name in names:
            mean = np.array([get(r, name)["mean"] for r in sel])
            std = np.array([get(r, name)["std"] for r in sel])
            shift = (mean - ref_mean) / ref_std
            pull = (mean - truth) / std
            cov = covered(truth, np.array([get(r, name)["int"] for r in sel]))
            print(f"    {name:12s} shift/std medio {shift.mean():+.3f} max|.| {np.abs(shift).max():.3f}"
                  f" | std/std_def {np.mean(std / ref_std):.3f}"
                  f" | pull {pull.mean():+.2f}/{pull.std():.2f} | coverage {np.round(cov, 3)}")
        print(f"    (errore binomiale a 0.68: +-{np.sqrt(0.68 * 0.32 / len(sel)):.3f})")


def print_direction_table(rows: list[dict], names: list[str]) -> None:
    """Tabella del Caso B: distanza angolare dalla stima di default in unita' di risoluzione.

    Ritorna:
        None (stdout).
    """
    print("\nCaso B, Omega_n; spostamento = angolo fra le stime / risoluzione del default")
    for n_events in sorted({r["N"] for r in rows}):
        sel = [r for r in rows if r["N"] == n_events]
        ref_hat = np.array([r[names[0]]["omega_hat"] for r in sel])
        ref_sigma = np.array([r[names[0]]["omega_sigma"] for r in sel])
        print(f"  N={n_events:5d} (M={len(sel):3d})")
        for name in names:
            hat = np.array([r[name]["omega_hat"] for r in sel])
            shift = validate.angular_residual(ref_hat, hat) / ref_sigma
            err = np.array([r[name]["omega_error"] for r in sel])
            sig = np.array([r[name]["omega_sigma"] for r in sel])
            cov = np.array([r[name]["omega_covered"] for r in sel]).mean(axis=0)
            pull = validate.angular_pull(err, sig)
            print(f"    {name:12s} shift/sigma medio {shift.mean():.3f} max {shift.max():.3f}"
                  f" | rms err {np.degrees(np.sqrt(np.mean(err ** 2))):.2f} gradi"
                  f" | pull rms {np.sqrt(np.mean(pull ** 2)):.2f} | coverage HPD {np.round(cov, 3)}")
        print(f"    (errore binomiale a 0.68: +-{np.sqrt(0.68 * 0.32 / len(sel)):.3f})")


def main() -> None:
    """Esegue gli esperimenti dei tre casi in parallelo, salva i risultati e stampa le tabelle.

    Ritorna:
        None (pkl in outputs/, tabelle su stdout).
    """
    n_proc = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    tasks = [(case, n, i) for case, n_m in (("C", N_M_C), ("B", N_M_B), ("A", N_M_A))
             for n, m in n_m for i in range(m)]
    with Pool(n_proc) as pool:
        results = pool.map(run_task, tasks, chunksize=1)
    rows = {case: [r for c, r in results if c == case] for case in "ABC"}
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "robustezza_prior_results.pkl", "wb") as f:
        pickle.dump({"LEVELS": LEVELS, "KAPPA": KAPPA, "results": rows}, f)

    print_scalar_table("Caso A, E_n condivisa [MeV]", rows["A"], list(energy_priors()), None, "truth")
    print_direction_table(rows["B"], ["piatto", "piatto largo", "logU", "vMF"])
    print_scalar_table("Caso C, mu_E [MeV]", rows["C"], ["logU", "U"], "mu", "mu_true")
    print_scalar_table("Caso C, log sigma_E", rows["C"], ["logU", "U"], "ls", "ls_true")


if __name__ == "__main__":
    main()
