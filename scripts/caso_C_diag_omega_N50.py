"""Riga 14, report §13.2: diagnostica della sovra-confidenza di Omega_n a N = 50.

Per ogni esperimento (stesso generatore della checklist) rifa' lo stadio 1'
(refine_shared_direction_hierarchical) con piu' prior gaussiani su E_n:
    plugin       N(E[mu], exp(E[log sigma]))           prior della checklist fino alla v0.13
    oracolo      N(mu_true, sigma_true)                 riferimento (report §8.2)
    predittiva   predictive_energy_moments              prior della checklist dalla v0.14
    plugin_+1sd  N(E[mu], exp(E[log sigma] + std[log sigma]))
e, sui seed della checklist, anche lo stadio 1 a prior largo ("largo").
Per ogni variante salva errore, sigma dichiarata, coverage HPD e le misure
delle ipotesi B2/B3 (verita' dentro la calotta, massa nell'anello esterno).

Seed: senza tag, spawn_key=(50, i) come la checklist (M=200); con tag,
spawn_key=(50, i, tag). Nel report: tag 50 e 51, M=600 ciascuno (il run col
tag 50 aveva solo plugin e oracolo; lo script ora calcola tutte le varianti).

Uso: OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=5G -p MemorySwapMax=0 \
         python scripts/caso_C_diag_omega_N50.py [n_processi] [M] [tag]
     (~1.5 min per M=200 senza tag, ~3.5 min per M=600 con tag, 3 processi, misurato 2026-09-26)
Produce: outputs/caso_C_diag_omega_N50[_tag<tag>].pkl e la tabella su stdout.
"""
import pickle
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from riptide_toy import grids, kinematics, posterior_C, priors, validate
from riptide_toy.constants import SEED, SIGMA_EP, SIGMA_THETA

sys.path.insert(0, str(Path(__file__).parent))
from caso_C_checklist import LEVELS, MU_RANGE, SIGMA_RANGE, rms  # noqa: E402  stesso generatore

N_EVENTS = 50
EDGE_FRACTION = 0.8  # anello esterno della calotta: angolo dal centro > 0.8 R (ipotesi B3)
VARIANTS = ("largo", "plugin", "oracolo", "predittiva", "plugin_+1sd")


def cap_stats(result: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], omega_true: np.ndarray) -> dict:
    """Misure sulla calotta di uno stadio 1 / 1', con lo stesso criterio di copertura della checklist.

    Args:
        result: (omega_hat (1, 3), log_post (n_cap,), theta (n_cap,) rad, phi (n_cap,) rad).
        omega_true: direzione vera, forma (1, 3).

    Ritorna:
        dict: err, sigma [rad]; in_cap (bool); cov (n_livelli,) bool, con e senza (cov_raw)
        il filtro in_cap; edge_mass (massa oltre EDGE_FRACTION * R); R [rad]; true_angle_over_R.
    """
    omega_hat, log_post, theta, phi = result
    cap_grid = kinematics.direction_from_theta_phi(theta, phi)
    angle = np.arccos(np.clip(cap_grid @ omega_hat[0], -1.0, 1.0))
    radius = angle.max()
    cap_pixel = np.sqrt(2 * np.pi * (1.0 - np.cos(radius)) / cap_grid.shape[0])
    nearest = np.argmax(cap_grid @ omega_true[0])
    in_cap = np.arccos(np.clip(cap_grid[nearest] @ omega_true[0], -1.0, 1.0)) < 3 * cap_pixel
    p = np.exp(log_post - log_post.max())
    p /= p.sum()
    covered = validate.credible_region_contains(log_post[None, :], np.array([nearest]), LEVELS)[0]
    err = validate.angular_residual(omega_true, omega_hat)[0]
    return {"err": err,
            "sigma": validate.posterior_angular_resolution(log_post[None, :], cap_grid, omega_hat)[0],
            "in_cap": bool(in_cap), "cov": covered & in_cap, "cov_raw": covered,
            "edge_mass": p[angle > EDGE_FRACTION * radius].sum(), "R": radius,
            "true_angle_over_R": err / radius}


def run_experiment(index: int, seed_tag: int | None = None) -> dict:
    """Un esperimento a N = 50: stadio 1, stadio 2 e stadio 1' con ciascun prior su E_n.

    Args:
        index: indice dell'esperimento.
        seed_tag: None per i seed della checklist, spawn_key=(50, i); altrimenti (50, i, tag).

    Ritorna:
        dict con i, sigma_true, sigma_plugin, pred_sd [MeV] e cap_stats per ogni variante.
    """
    key = (N_EVENTS, index) if seed_tag is None else (N_EVENTS, index, seed_tag)
    rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=key))
    mu_true = rng.uniform(*MU_RANGE)
    sigma_true = np.exp(rng.uniform(*np.log(SIGMA_RANGE)))
    omega_true = kinematics.direction_from_theta_phi(
        np.arccos(rng.uniform(-1.0, 1.0, 1)), rng.uniform(0.0, 2 * np.pi, 1)
    )
    Ep_true, track = kinematics.sample_recoil_events(
        rng, rng.normal(mu_true, sigma_true, N_EVENTS), omega_true[0]
    )
    D_B = (rng.normal(Ep_true, SIGMA_EP), kinematics.smear_direction(rng, track, SIGMA_THETA))

    theta_grid, phi_grid = grids.sphere_grid()
    direction_prior = priors.direction_prior(theta_grid, phi_grid)
    mu_grid, sigma_grid = grids.hyperparameter_grid()
    stage1 = posterior_C.refine_shared_direction(D_B, theta_grid, phi_grid, direction_prior)
    log_post, mu_fine, sigma_fine = posterior_C.refine_hyperparameters(
        (*D_B, stage1[0]), mu_grid, sigma_grid, priors.hyperparameter_prior
    )
    mu_mean, _ = validate.posterior_mean_std(log_post[None, :], mu_fine)
    ls_mean, ls_std = validate.posterior_mean_std(log_post[None, :], np.log(sigma_fine))
    predictive = posterior_C.predictive_energy_moments(log_post, mu_fine, sigma_fine)
    prior_En = {"plugin": (mu_mean[0], np.exp(ls_mean[0])),
                "oracolo": (mu_true, sigma_true),
                "predittiva": predictive,
                "plugin_+1sd": (mu_mean[0], np.exp(ls_mean[0] + ls_std[0]))}

    out = {"i": index, "sigma_true": sigma_true, "sigma_plugin": np.exp(ls_mean[0]), "pred_sd": predictive[1]}
    if seed_tag is None:
        out["largo"] = cap_stats(stage1, omega_true)
    for name, (m, s) in prior_En.items():
        out[name] = cap_stats(posterior_C.refine_shared_direction_hierarchical(
            D_B, theta_grid, phi_grid, direction_prior, m, s
        ), omega_true)
    return out


def main() -> None:
    n_proc = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    n_exp = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    seed_tag = int(sys.argv[3]) if len(sys.argv) > 3 else None
    with Pool(n_proc) as pool:
        results = pool.map(partial(run_experiment, seed_tag=seed_tag), range(n_exp), chunksize=1)
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    suffix = "" if seed_tag is None else f"_tag{seed_tag}"
    with open(out_dir / f"caso_C_diag_omega_N50{suffix}.pkl", "wb") as f:
        pickle.dump({"N": N_EVENTS, "seed_tag": seed_tag, "LEVELS": LEVELS, "results": results}, f)
    for name in VARIANTS:
        if name not in results[0]:
            continue
        rows = [r[name] for r in results]
        cov = np.mean([r["cov"] for r in rows], axis=0)
        pull = validate.angular_pull(np.array([r["err"] for r in rows]), np.array([r["sigma"] for r in rows]))
        print(f"N={N_EVENTS} M={n_exp} tag={seed_tag} [{name}] coverage HPD {np.round(cov, 3)}, "
              f"pull rms {rms(pull):.2f}, verita' fuori calotta {sum(not r['in_cap'] for r in rows)}, "
              f"massa al bordo max {max(r['edge_mass'] for r in rows):.1e}")


if __name__ == "__main__":
    main()
