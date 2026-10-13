"""Stress test dell'isotropia in CM dello scattering n-p, Casi A, B e C.

Le direzioni di rinculo vere sono generate con la distribuzione angolare anisotropa
dei coefficienti di Legendre di constants (NP_LEGENDRE_*); la ricostruzione resta
quella dei tre casi, che assume l'isotropia, invariata. Ogni esperimento e' fatto due
volte con lo stesso seme, spawn_key=(N, i): isotropo e anisotropo consumano lo stesso
flusso RNG (un uniforme per evento per l'angolo in CM), quindi i due dataset differiscono
solo per l'anisotropia e la differenza appaiata isola il suo effetto. Le righe isotrope
di B e C coincidono con i dataset delle rispettive checklist.
  A: E_n ~ U(11, 19) per esperimento, condivisa; Omega_n = z nota; rinculi generati
     dal CM (non da theta_p ~ U(0, pi/2) come nella checklist del Caso A), posterior_A
     + combine_loglik. Bersaglio: E_n.
  B: come la checklist del Caso B. Bersaglio: Omega_n.
  C: come la checklist del Caso C (spettro gaussiano). Bersagli: mu_E, log sigma_E, Omega_n.
Criterio: anisotropia trascurabile se, al N massimo, |bias aniso - bias iso| < 0.2 volte
la deviazione standard media del posterior e la coverage resta nei limiti binomiali.

Uso: OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=5G -p MemorySwapMax=0 \
         python scripts/stress_anisotropia.py [n_processi] [casi, es. ABC]
     (picco ~0.6 GB per processo a N=1000 nel Caso C, come la sua checklist)
Produce: outputs/stress_anisotropia_results.pkl e la tabella su stdout.
"""
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from riptide_toy import combine, forward_model, grids, kinematics, posterior_A, posterior_C, priors, validate
from riptide_toy.constants import (EN_MAX, EN_MIN, NP_LEGENDRE_COEFFS, NP_LEGENDRE_ENERGIES, SEED, SIGMA_EP,
                                   SIGMA_THETA)

N_M = {
    "A": [(1000, 40), (100, 200), (10, 400)],
    "B": [(1000, 40), (300, 100), (100, 200), (30, 300), (10, 400)],  # come la checklist del Caso B
    "C": [(1000, 40), (300, 100), (150, 200), (50, 200)],             # come la checklist del Caso C
}
EN_RANGE_A = (11.0, 19.0)   # MeV, come il generatore (ii) della checklist del Caso A
MU_RANGE = (13.0, 17.0)     # MeV, come la checklist del Caso C
SIGMA_RANGE = (0.8, 2.4)    # MeV, log-uniforme, come la checklist del Caso C
LEVELS = np.array([0.68, 0.90, 0.95])
CHUNK = 50  # eventi per blocco nel Caso A: la marginalizzazione su theta_p alloca (CHUNK, 500, 500)
BIAS_TOLERANCE = 0.2  # frazione della deviazione standard del posterior


def legendre_table(anisotropic: bool) -> tuple[np.ndarray, np.ndarray] | None:
    """Tabella dei coefficienti per sample_recoil_events.

    Ritorna:
        None (isotropo) oppure (energie in MeV forma (n_E,), coefficienti forma (n_E, 6)).
    """
    if not anisotropic:
        return None
    return np.array(NP_LEGENDRE_ENERGIES), np.array(NP_LEGENDRE_COEFFS)


def scalar_summary(log_post: np.ndarray, grid: np.ndarray) -> dict:
    """Media, deviazione standard, mediana e intervalli credibili di un posterior 1D.

    Ritorna:
        dict: "mean", "std", "median" (unita' della griglia), "int" forma (len(LEVELS), 2).
    """
    mean, std = validate.posterior_mean_std(log_post[None, :], grid)
    intervals = validate.credible_interval(log_post[None, :], grid, np.concatenate([[0.0], LEVELS]))[0]
    return {"mean": mean[0], "std": std[0], "median": intervals[0, 0], "int": intervals[1:]}


def direction_summary(omega_true: np.ndarray, omega_hat: np.ndarray, cap_log_post: np.ndarray,
                      cap_theta: np.ndarray, cap_phi: np.ndarray) -> dict:
    """Errore, risoluzione e copertura HPD della direzione stimata sulla calotta fine.

    Ritorna:
        dict: "omega_error" e "omega_sigma" in rad, "omega_covered" bool forma (len(LEVELS),).
    """
    cap_grid = kinematics.direction_from_theta_phi(cap_theta, cap_phi)
    # come nelle checklist: la verita' e' "nella calotta" solo entro 3 pixel dal pixel piu' vicino
    cap_pixel = np.sqrt(2 * np.pi * (1.0 - np.min(cap_grid @ omega_hat[0])) / cap_grid.shape[0])
    nearest = np.argmax(cap_grid @ omega_true[0])
    truth_in_cap = np.arccos(np.clip(cap_grid[nearest] @ omega_true[0], -1.0, 1.0)) < 3 * cap_pixel
    return {
        "omega_error": validate.angular_residual(omega_true, omega_hat)[0],
        "omega_sigma": validate.posterior_angular_resolution(cap_log_post[None, :], cap_grid, omega_hat)[0],
        "omega_covered": validate.credible_region_contains(
            cap_log_post[None, :], np.array([nearest]), LEVELS)[0] & truth_in_cap,
    }


def experiment_A(rng: np.random.Generator, n_events: int, anisotropic: bool) -> dict:
    """Caso A con rinculi generati dal CM: E_n condivisa, direzione z nota.

    Ritorna:
        dict con "En_true" (MeV) e "En" (scalar_summary del posterior combinato, MeV).
    """
    En_true = rng.uniform(*EN_RANGE_A)
    En = np.full(n_events, En_true)
    Ep_true, track = kinematics.sample_recoil_events(rng, En, np.array([0.0, 0.0, 1.0]),
                                                     legendre_table(anisotropic))
    theta_p = np.arccos(np.clip(track[:, 2], -1.0, 1.0))
    Ep_hat, theta_hat = forward_model.measure(Ep_true, theta_p, SIGMA_EP, SIGMA_THETA, rng)
    en_grid = grids.energy_grid()
    log_prior = np.log(priors.energy_prior(en_grid))
    # single_event_posterior include il prior per evento: si toglie prima di combinare
    loglik = np.concatenate([
        posterior_A.single_event_posterior((Ep_hat[i:i + CHUNK], theta_hat[i:i + CHUNK]), en_grid,
                                           np.exp(log_prior)) - log_prior[None, :]
        for i in range(0, n_events, CHUNK)
    ])
    return {"En_true": En_true, "En": scalar_summary(combine.combine_loglik(loglik, log_prior), en_grid)}


def experiment_B(rng: np.random.Generator, n_events: int, anisotropic: bool) -> dict:
    """Caso B come nella sua checklist, con la distribuzione angolare scelta.

    Ritorna:
        dict di direction_summary.
    """
    omega_true = kinematics.direction_from_theta_phi(
        np.arccos(rng.uniform(-1.0, 1.0, 1)), rng.uniform(0.0, 2 * np.pi, 1))
    Ep_true, track = kinematics.sample_recoil_events(
        rng, rng.uniform(EN_MIN, EN_MAX, n_events), omega_true[0], legendre_table(anisotropic))
    D_B = (rng.normal(Ep_true, SIGMA_EP), kinematics.smear_direction(rng, track, SIGMA_THETA))
    theta_grid, phi_grid = grids.sphere_grid()
    direction_prior = priors.direction_prior(theta_grid, phi_grid)
    omega_hat, cap_log_post, cap_theta, cap_phi = posterior_C.refine_shared_direction(
        D_B, theta_grid, phi_grid, direction_prior)
    return direction_summary(omega_true, omega_hat, cap_log_post, cap_theta, cap_phi)


def experiment_C(rng: np.random.Generator, n_events: int, anisotropic: bool) -> dict:
    """Caso C come nella sua checklist (spettro gaussiano), con la distribuzione angolare scelta.

    Ritorna:
        dict con "mu_true" (MeV), "ls_true" (log(sigma_E / MeV)), scalar_summary di "mu"
        e "ls", e le chiavi di direction_summary.
    """
    mu_true = rng.uniform(*MU_RANGE)
    sigma_true = np.exp(rng.uniform(*np.log(SIGMA_RANGE)))
    omega_true = kinematics.direction_from_theta_phi(
        np.arccos(rng.uniform(-1.0, 1.0, 1)), rng.uniform(0.0, 2 * np.pi, 1))
    En = kinematics.sample_energy_spectrum(rng, "gauss", mu_true, sigma_true, n_events)
    Ep_true, track = kinematics.sample_recoil_events(rng, En, omega_true[0], legendre_table(anisotropic))
    D_B = (rng.normal(Ep_true, SIGMA_EP), kinematics.smear_direction(rng, track, SIGMA_THETA))

    theta_grid, phi_grid = grids.sphere_grid()
    direction_prior = priors.direction_prior(theta_grid, phi_grid)
    mu_grid, sigma_grid = grids.hyperparameter_grid()
    omega_0 = posterior_C.refine_shared_direction(D_B, theta_grid, phi_grid, direction_prior)[0]
    log_post, mu_fine, sigma_fine = posterior_C.refine_hyperparameters(
        (*D_B, omega_0), mu_grid, sigma_grid, priors.hyperparameter_prior)
    out = {"mu_true": mu_true, "ls_true": np.log(sigma_true),
           "mu": scalar_summary(log_post, mu_fine), "ls": scalar_summary(log_post, np.log(sigma_fine))}
    omega_hat, cap_log_post, cap_theta, cap_phi = posterior_C.refine_shared_direction_hierarchical(
        D_B, theta_grid, phi_grid, direction_prior,
        *posterior_C.predictive_energy_moments(log_post, mu_fine, sigma_fine))
    out.update(direction_summary(omega_true, omega_hat, cap_log_post, cap_theta, cap_phi))
    return out


EXPERIMENTS = {"A": experiment_A, "B": experiment_B, "C": experiment_C}


def run_experiment(task: tuple[str, int, int, bool]) -> dict:
    """Un esperimento di un caso, isotropo o anisotropo, col seme (N, i) delle checklist.

    Ritorna:
        dict con "case", "N", "index", "aniso" e le chiavi dell'esperimento del caso.
    """
    case, n_events, index, anisotropic = task
    rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(n_events, index)))
    out = EXPERIMENTS[case](rng, n_events, anisotropic)
    out.update({"case": case, "N": n_events, "index": index, "aniso": anisotropic})
    return out


def rms(x: np.ndarray) -> float:
    """Radice della media dei quadrati.

    Ritorna:
        float, stesse unita' di x.
    """
    return float(np.sqrt(np.mean(x ** 2)))


def paired(rows: list[dict], case: str, n_events: int) -> tuple[list[dict], list[dict]]:
    """Righe isotrope e anisotrope dello stesso caso e N, ordinate per indice (appaiate).

    Ritorna:
        (iso, aniso), liste di dict della stessa lunghezza.
    """
    sel = sorted((r for r in rows if r["case"] == case and r["N"] == n_events), key=lambda r: r["index"])
    return [r for r in sel if not r["aniso"]], [r for r in sel if r["aniso"]]


def scalar_report(iso: list[dict], aniso: list[dict], key: str, truth_key: str, label: str,
                  use_median: bool) -> tuple[str, float]:
    """Bias, pull e coverage di un parametro scalare, iso vs aniso, e differenza appaiata del bias.

    Ritorna:
        (testo della riga, |differenza di bias| / std media del posterior).
    """
    text, residuals, stds = f"    {label:9s}", [], []
    for name, sel in (("iso", iso), ("aniso", aniso)):
        truth = np.array([r[truth_key] for r in sel])
        mean = np.array([r[key]["mean"] for r in sel])
        std = np.array([r[key]["std"] for r in sel])
        point = np.array([r[key]["median"] for r in sel]) if use_median else mean
        res = point - truth
        cov = validate.coverage_curve(truth, np.array([r[key]["int"] for r in sel]), LEVELS)[1]
        pull = (mean - truth) / std
        residuals.append(res)
        stds.append(std)
        text += (f"\n      {name:5s} bias {res.mean():+.4f} +- {res.std() / np.sqrt(len(sel)):.4f}"
                 f" | rms {rms(res):.4f} (std post. {std.mean():.4f})"
                 f" | pull {pull.mean():+.2f}/{pull.std():.2f} | coverage {np.round(cov, 2)}")
    diff = residuals[1] - residuals[0]
    ratio = abs(diff.mean()) / np.mean(stds[1])
    text += (f"\n      aniso - iso: bias {diff.mean():+.4f} +- {diff.std() / np.sqrt(diff.size):.4f}"
             f" = {ratio:.2f} std post.")
    return text, ratio


def direction_report(iso: list[dict], aniso: list[dict]) -> str:
    """Errore rms, pull e coverage HPD di Omega_n, iso vs aniso.

    Ritorna:
        testo della riga (angoli in gradi).
    """
    text = "    Omega"
    for name, sel in (("iso", iso), ("aniso", aniso)):
        err = np.array([r["omega_error"] for r in sel])
        sig = np.array([r["omega_sigma"] for r in sel])
        cov = np.array([r["omega_covered"] for r in sel]).mean(axis=0)
        text += (f"\n      {name:5s} rms {np.degrees(rms(err)):.2f} (sigma post. {np.degrees(sig.mean()):.2f})"
                 f" | pull rms {rms(validate.angular_pull(err, sig)):.2f} | coverage HPD {np.round(cov, 2)}")
    return text


def print_table(rows: list[dict], cases: str) -> None:
    """Per caso e N: confronto iso/aniso e verdetto sul bias al N massimo.

    Ritorna:
        None (stdout).
    """
    print("Isotropia in CM: generazione anisotropa, ricostruzione isotropa; E_n e mu in MeV,"
          " Omega in gradi; bias = media dei residui +- errore sulla media")
    for case in cases:
        n_max = max(n for n, _ in N_M[case])
        for n_events in sorted(n for n, _ in N_M[case]):
            iso, aniso = paired(rows, case, n_events)
            print(f"\nCaso {case}, N={n_events} (M={len(iso)}, errore binomiale a 0.68:"
                  f" +-{np.sqrt(0.68 * 0.32 / len(iso)):.3f})")
            scalars = {"A": [("En", "En_true", "E_n", False)],
                       "C": [("mu", "mu_true", "mu", False), ("ls", "ls_true", "log sigma", True)]}
            for key, truth_key, label, use_median in scalars.get(case, []):
                text, ratio = scalar_report(iso, aniso, key, truth_key, label, use_median)
                print(text)
                if n_events == n_max:
                    verdict = "trascurabile" if ratio < BIAS_TOLERANCE else "NON trascurabile"
                    print(f"      -> bias da anisotropia {verdict} (soglia {BIAS_TOLERANCE} std post.)")
            if case in "BC":
                print(direction_report(iso, aniso))


def main() -> None:
    """Esegue gli esperimenti in parallelo, salva i risultati e stampa la tabella.

    Ritorna:
        None (pkl in outputs/, tabella su stdout).
    """
    n_proc = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    cases = sys.argv[2] if len(sys.argv) > 2 else "ABC"
    tasks = [(case, n, i, aniso) for case in cases for n, m in N_M[case]
             for i in range(m) for aniso in (False, True)]
    with Pool(n_proc) as pool:
        rows = pool.map(run_experiment, tasks, chunksize=1)
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "stress_anisotropia_results.pkl", "wb") as f:
        pickle.dump({"N_M": N_M, "LEVELS": LEVELS, "cases": cases, "results": rows}, f)
    print_table(rows, cases)


if __name__ == "__main__":
    main()
