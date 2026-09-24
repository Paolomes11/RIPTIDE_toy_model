"""Riproduce la Fig. 39.1 del libro (Cap. 39, Esercizio 3) e lo shift sistematico 2%.

Uso: python scripts/ch39_fig_repro.py
Produce: outputs/fig_39_1.png, outputs/fig_39_1_systematic_shift.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid

from riptide_toy import forward_model, grids, kinematics, priors
from riptide_toy.combine import combine_loglik, expected_sigma_n
from riptide_toy.constants import SEED, SIGMA_EP, SIGMA_THETA

EN_TRUE = 2.5  # MeV, energia condivisa della Fig. 39.1
N_VALUES = [1, 2, 10, 100]  # stesso range della Fig. 39.1 del libro

# il plateau di contrazione sotto sistematico non e' ancora visibile a N=100 (lo shift
# in Ep e' piccolo rispetto a sigma_Ep, quindi a N=100 e' ancora sotto il rumore
# statistico): serve un range di N piu' esteso per renderlo visibile (verificato in
# REPL: offset ~0.028 a N=100, ~0.050 a N=1000/5000, mentre sigma continua a scendere).
N_VALUES_SHIFT = [1, 2, 10, 100, 1000, 5000]


def weighted_mean_sigma(logpost: np.ndarray, en_grid: np.ndarray) -> tuple[float, float]:
    w = np.exp(logpost - logpost.max())
    mean = np.sum(w * en_grid) / np.sum(w)
    sigma = np.sqrt(np.sum(w * (en_grid - mean) ** 2) / np.sum(w))
    return mean, sigma


def contraction_curve(ll_matrix: np.ndarray, log_prior_en: np.ndarray,
                       en_grid: np.ndarray, n_values: list[int]) -> tuple[list[float], list[float]]:
    sigmas, offsets = [], []
    for n in n_values:
        combined = combine_loglik(ll_matrix[:n], log_prior_en)
        mean, sigma = weighted_mean_sigma(combined, en_grid)
        sigmas.append(sigma)
        offsets.append(abs(mean - EN_TRUE))
    return sigmas, offsets


def main() -> None:
    rng = np.random.default_rng(SEED)
    n_max = max(N_VALUES_SHIFT)

    # theta_p campionato direttamente Uniform(0, pi/2), non tramite
    # kinematics.sample_cm_angle (formula aperta, vedi docs/roadmap.md).
    theta_p_true = rng.uniform(0.0, np.pi / 2, n_max)
    ep_true = kinematics.proton_energy(EN_TRUE, theta_p_true)
    D = forward_model.measure(ep_true, theta_p_true, SIGMA_EP, SIGMA_THETA, rng)

    en_grid = grids.energy_grid()
    theta_p_grid = grids.theta_p_grid()
    log_prior_theta = np.full(theta_p_grid.shape, -np.log(theta_p_grid.shape[0]))
    log_prior_en = np.log(priors.energy_prior(en_grid))

    ll_matrix = forward_model.loglik(D, en_grid, theta_p_grid, SIGMA_EP, SIGMA_THETA, log_prior_theta)

    outputs_dir = Path(__file__).resolve().parent.parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for n in N_VALUES:
        combined = combine_loglik(ll_matrix[:n], log_prior_en)
        w = np.exp(combined - combined.max())
        axes[0].plot(en_grid, w / trapezoid(w, en_grid), label=f"N={n}")
    axes[0].axvline(EN_TRUE, color="k", linestyle=":", label="En (true)")
    axes[0].set_xlim(1.5, 4.0)
    axes[0].set_xlabel("neutron energy En [MeV]")
    axes[0].set_ylabel("posterior P(En | D1:N)")
    axes[0].set_title("(a) combined posterior vs event count")
    axes[0].legend()

    sigmas, offsets = contraction_curve(ll_matrix, log_prior_en, en_grid, N_VALUES)
    n_arr = np.array(N_VALUES)
    guide = expected_sigma_n(sigmas[0], n_arr)
    axes[1].loglog(n_arr, sigmas, "o-", label="posterior s.d.")
    axes[1].loglog(n_arr, offsets, "s-", label="|mean - En true|")
    axes[1].loglog(n_arr, guide, "k--", label="sigma_1 / sqrt(N)")
    axes[1].set_xlabel("number of combined events N")
    axes[1].set_ylabel("width / offset on En [MeV]")
    axes[1].set_title("(b) contraction rate")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(outputs_dir / "fig_39_1.png", dpi=150)
    plt.close(fig)

    # shift sistematico 2% su ogni evento (stesso D_theta, Ep osservato scalato):
    # comune a tutti gli eventi, non si mediazza via con N (Cap. 39, "common mistakes").
    ep_true_shifted = 1.02 * ep_true
    D_shifted = forward_model.measure(ep_true_shifted, theta_p_true, SIGMA_EP, SIGMA_THETA, rng)
    ll_matrix_shifted = forward_model.loglik(
        D_shifted, en_grid, theta_p_grid, SIGMA_EP, SIGMA_THETA, log_prior_theta
    )
    sigmas_shifted, offsets_shifted = contraction_curve(
        ll_matrix_shifted, log_prior_en, en_grid, N_VALUES_SHIFT
    )
    n_arr_shift = np.array(N_VALUES_SHIFT)
    guide_shift = expected_sigma_n(sigmas_shifted[0], n_arr_shift)

    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    ax2.loglog(n_arr_shift, sigmas_shifted, "o-", label="posterior s.d. (shift 2%)")
    ax2.loglog(n_arr_shift, offsets_shifted, "s-", label="|mean - En true| (shift 2%)")
    ax2.loglog(n_arr_shift, guide_shift, "k--", label="sigma_1 / sqrt(N) (senza shift)")
    ax2.set_xlabel("number of combined events N")
    ax2.set_ylabel("width / offset on En [MeV]")
    ax2.set_title("shift sistematico 2%: l'offset non converge a zero")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(outputs_dir / "fig_39_1_systematic_shift.png", dpi=150)
    plt.close(fig2)

    print(f"(c) senza sistematico: sigma(N={N_VALUES}) = {sigmas}")
    print(f"(c) senza sistematico: |offset|(N={N_VALUES}) = {offsets}")
    print(f"(c) con shift 2%: sigma(N={N_VALUES_SHIFT}) = {sigmas_shifted}")
    print(f"(c) con shift 2%: |offset|(N={N_VALUES_SHIFT}) = {offsets_shifted}")
    print(f"scritto: {outputs_dir / 'fig_39_1.png'}")
    print(f"scritto: {outputs_dir / 'fig_39_1_systematic_shift.png'}")


if __name__ == "__main__":
    main()
