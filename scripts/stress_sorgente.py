"""Stress test dell'assunzione 1 (sorgente unica) nei Casi B e C.

Una frazione f degli eventi non viene dalla sorgente principale: viene da una seconda
sorgente puntiforme a separazione Delta (10, 30, 90 gradi) oppure da un fondo isotropo
(una direzione indipendente per evento). La ricostruzione resta quella a sorgente
unica, invariata: Caso B (stadio 1 raffinato) e Caso C (stadio 1 -> stadio 2 ->
stadio 1 iterato). Si misurano bias, pull e coverage di Omega_n (e di mu_E,
log sigma_E nel Caso C) rispetto alla sorgente principale, e la potenza di una
diagnostica di violazione: KS fra gli angoli osservati delle tracce rispetto a
Omega_n stimata e la ripartizione attesa per sorgente unica. Omega_n e' stimata
dagli stessi dati, quindi la soglia KS e' il 95-esimo percentile della statistica
a f = 0 (stesso caso, stesso N), non quella tabulata.
Seed: flusso principale con spawn_key=(N, i) come nelle checklist (a f = 0 si
riottengono i loro dataset); geometria della contaminazione da un generatore a parte,
spawn_key=(CONTAMINATION_TAG, N, i), quindi a parita' di (N, i) verita' ed energie
sono le stesse per ogni configurazione e cambia solo la contaminazione.

Uso: OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=5G -p MemorySwapMax=0 \
         python scripts/stress_sorgente.py [n_processi]
Produce: outputs/stress_sorgente_results.pkl e le tabelle su stdout.
"""
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from riptide_toy import forward_model, grids, kinematics, posterior_C, priors, validate
from riptide_toy.constants import EN_MAX, EN_MIN, SEED, SIGMA_EP, SIGMA_THETA

N_VALUES = {"B": [30, 300], "C": [50, 300]}
M_NULL = 200                # esperimenti a f = 0: fissano la soglia KS al 95%
M_CONTAMINATED = 100        # esperimenti per configurazione contaminata
FRACTIONS = [0.05, 0.1, 0.3]
SEPARATIONS_DEG = [10.0, 30.0, 90.0, None]  # None = fondo isotropo
MU_RANGE = (2.5, 4.0)       # MeV, come la checklist del Caso C
SIGMA_RANGE = (0.2, 0.6)    # MeV, log-uniforme, come la checklist del Caso C
LEVELS = np.array([0.68, 0.90, 0.95])
CONTAMINATION_TAG = 2       # disgiunto dai seed (N, i) e da (1, N, i) della robustezza al prior
KS_QUANTILE = 0.95


def run_experiment(task: tuple[str, int, float, float | None, int]) -> dict:
    """Un esperimento: dataset contaminato, ricostruzione a sorgente unica, riassunto.

    Ritorna:
        dict con caso, N, f, separazione (gradi o None), errore, risoluzione (rad) e
        copertura HPD (bool, (len(LEVELS),)) di Omega_n, spostamento verso la seconda
        sorgente (rad, nan se assente), statistica KS; nel Caso C anche media/std/
        intervalli di mu_E (MeV) e log(sigma_E / MeV).
    """
    case, n_events, fraction, separation_deg, index = task
    rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(n_events, index)))
    rng_contamination = np.random.default_rng(
        np.random.SeedSequence(SEED, spawn_key=(CONTAMINATION_TAG, n_events, index)))
    out = {"case": case, "N": n_events, "f": fraction, "sep": separation_deg}
    if case == "C":
        mu_true = rng.uniform(*MU_RANGE)
        sigma_true = np.exp(rng.uniform(*np.log(SIGMA_RANGE)))
        out.update(mu_true=mu_true, ls_true=np.log(sigma_true))
    omega_true = kinematics.direction_from_theta_phi(
        np.arccos(rng.uniform(-1.0, 1.0, 1)), rng.uniform(0.0, 2 * np.pi, 1)
    )
    En = (rng.normal(mu_true, sigma_true, n_events) if case == "C"
          else rng.uniform(EN_MIN, EN_MAX, n_events))
    separation = None if separation_deg is None else np.radians(separation_deg)
    axes = kinematics.mixed_source_axes(rng_contamination, omega_true[0], n_events, fraction, separation)
    Ep_true, track = kinematics.sample_recoil_events(rng, En, axes)
    D_B = (rng.normal(Ep_true, SIGMA_EP), kinematics.smear_direction(rng, track, SIGMA_THETA))

    theta_grid, phi_grid = grids.sphere_grid()
    direction_prior = priors.direction_prior(theta_grid, phi_grid)
    result = posterior_C.refine_shared_direction(D_B, theta_grid, phi_grid, direction_prior)
    if case == "C":
        log_post, mu_fine, sigma_fine = posterior_C.refine_hyperparameters(
            (*D_B, result[0]), *grids.hyperparameter_grid(), priors.hyperparameter_prior)
        for key, grid in (("mu", mu_fine), ("ls", np.log(sigma_fine))):
            mean, std = validate.posterior_mean_std(log_post[None, :], grid)
            out[key] = {"mean": mean[0], "std": std[0],
                        "int": validate.credible_interval(log_post[None, :], grid, LEVELS)[0]}
        result = posterior_C.refine_shared_direction_hierarchical(
            D_B, theta_grid, phi_grid, direction_prior,
            *posterior_C.predictive_energy_moments(log_post, mu_fine, sigma_fine))

    omega_hat, cap_log_post, cap_theta, cap_phi = result
    cap_grid = kinematics.direction_from_theta_phi(cap_theta, cap_phi)
    # come nelle checklist: la verita' e' "nella calotta" solo entro 3 pixel dal pixel piu' vicino
    cap_pixel = np.sqrt(2 * np.pi * (1.0 - np.min(cap_grid @ omega_hat[0])) / cap_grid.shape[0])
    nearest = np.argmax(cap_grid @ omega_true[0])
    truth_in_cap = np.arccos(np.clip(cap_grid[nearest] @ omega_true[0], -1.0, 1.0)) < 3 * cap_pixel
    out["omega_error"] = validate.angular_residual(omega_true, omega_hat)[0]
    out["omega_sigma"] = validate.posterior_angular_resolution(cap_log_post[None, :], cap_grid, omega_hat)[0]
    out["omega_covered"] = validate.credible_region_contains(
        cap_log_post[None, :], np.array([nearest]), LEVELS)[0] & truth_in_cap
    # spostamento verso la seconda sorgente: separazione vera - distanza della stima da essa
    out["toward_second"] = (separation - validate.angular_residual(axes[-1:], omega_hat)[0]
                            if separation is not None and fraction > 0 else np.nan)
    theta_obs = validate.angular_residual(np.broadcast_to(omega_hat, D_B[1].shape), D_B[1])
    out["ks"] = validate.ks_uniform_statistic(
        validate.probability_integral_transform(theta_obs, *forward_model.track_angle_cdf(SIGMA_THETA)))
    return out


def rms(x: np.ndarray) -> float:
    """Radice della media dei quadrati.

    Ritorna:
        float, stesse unita' di x.
    """
    return float(np.sqrt(np.mean(x ** 2)))


def config_label(fraction: float, separation_deg: float | None) -> str:
    """Etichetta leggibile di una configurazione di contaminazione.

    Ritorna:
        str.
    """
    if fraction == 0:
        return "f=0"
    return f"f={fraction:.2f} " + ("isotropo" if separation_deg is None else f"{separation_deg:.0f} gradi")


def print_tables(rows: list[dict]) -> None:
    """Per caso, N e configurazione: Omega_n (rms, spostamento verso la seconda
    sorgente, pull, coverage), potenza KS e, nel Caso C, bias e pull di mu_E e log sigma_E.

    Ritorna:
        None (stdout).
    """
    configs = [(0.0, None)] + [(f, s) for s in SEPARATIONS_DEG for f in FRACTIONS]
    for case in N_VALUES:
        for n_events in N_VALUES[case]:
            sel_n = [r for r in rows if r["case"] == case and r["N"] == n_events]
            null_ks = np.array([r["ks"] for r in sel_n if r["f"] == 0])
            threshold = np.quantile(null_ks, KS_QUANTILE)
            print(f"\nCaso {case}, N={n_events}; Omega in gradi; soglia KS al {KS_QUANTILE:.0%} da f=0:"
                  f" {threshold:.4f} (1.36/sqrt(N) = {1.36 / np.sqrt(n_events):.4f})")
            for fraction, separation_deg in configs:
                sel = [r for r in sel_n if r["f"] == fraction and r["sep"] == separation_deg]
                err = np.array([r["omega_error"] for r in sel])
                sig = np.array([r["omega_sigma"] for r in sel])
                cov = np.array([r["omega_covered"] for r in sel]).mean(axis=0)
                toward = np.array([r["toward_second"] for r in sel])
                power = np.mean(np.array([r["ks"] for r in sel]) > threshold)
                line = (f"  {config_label(fraction, separation_deg):20s} (M={len(sel):3d})"
                        f" Omega rms {np.degrees(rms(err)):6.2f} (sigma {np.degrees(sig.mean()):.2f})"
                        f" verso 2a {np.degrees(np.nanmean(toward)) if np.any(np.isfinite(toward)) else np.nan:+6.2f}"
                        f" | pull rms {rms(validate.angular_pull(err, sig)):5.2f}"
                        f" | cov {np.round(cov, 2)} | KS > soglia {power:.2f}")
                if case == "C":
                    for key, label in (("mu", "mu"), ("ls", "log sigma")):
                        truth = np.array([r[f"{key}_true"] for r in sel])
                        mean = np.array([r[key]["mean"] for r in sel])
                        pull = (mean - truth) / np.array([r[key]["std"] for r in sel])
                        cov_k = validate.coverage_curve(truth, np.array([r[key]["int"] for r in sel]), LEVELS)[1]
                        line += (f"\n      {label:9s} bias {np.mean(mean - truth):+.4f}"
                                 f" | pull {pull.mean():+.2f}/{pull.std():.2f} | cov {np.round(cov_k, 2)}")
                print(line)


def main() -> None:
    """Esegue gli esperimenti in parallelo, salva i risultati e stampa le tabelle.

    Ritorna:
        None (pkl in outputs/, tabelle su stdout).
    """
    n_proc = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    tasks = []
    for case, n_values in N_VALUES.items():
        for n_events in sorted(n_values, reverse=True):  # i piu' lenti per primi
            tasks += [(case, n_events, 0.0, None, i) for i in range(M_NULL)]
            tasks += [(case, n_events, f, s, i) for s in SEPARATIONS_DEG for f in FRACTIONS
                      for i in range(M_CONTAMINATED)]
    with Pool(n_proc) as pool:
        rows = pool.map(run_experiment, tasks, chunksize=1)
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "stress_sorgente_results.pkl", "wb") as f:
        pickle.dump({"N_VALUES": N_VALUES, "FRACTIONS": FRACTIONS, "SEPARATIONS_DEG": SEPARATIONS_DEG,
                     "LEVELS": LEVELS, "results": rows}, f)
    print_tables(rows)


if __name__ == "__main__":
    main()
