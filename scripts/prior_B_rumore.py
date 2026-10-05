"""Caso B: lo spostamento di ~1 sigma della stima di Omega_n al cambiare del prior su E_n
e' rumore o bias, e perche' non cala con N?

Ipotesi. Col prior di default uguale al prior del generatore, la stima di default e' (quasi)
la media posteriore sotto il modello vero, quindi il suo errore e' ortogonale a qualunque
funzione dei dati, in particolare allo spostamento d = stima_alt - stima_def. Ne segue
    E|err_alt|^2 = E|err_def|^2 + E|d|^2      (relazione tipo Hausman)
cioe' lo spostamento e' rumore che si somma in quadratura, non un bias. Se il prior
alternativo spreca in ogni evento la stessa frazione di informazione su Omega_n, il
rapporto rms(err_alt)/rms(err_def) e' costante con N, e allora anche |d|/sigma lo e'.

Previsioni (scritte prima del calcolo), per i prior "piatto largo" e "logU":
  1. vettore medio di d nel piano tangente (base e_theta, e_phi alla verita') compatibile
     con zero: il prior agisce sull'energia, non su un verso nello spazio;
  2. correlazione normalizzata <err_def . d> / sqrt(<|err_def|^2><|d|^2>) ~ 0, e quindi
     rms|d| ~ sqrt(rms(err_alt)^2 - rms(err_def)^2) entro l'errore;
  3. rapporto rms(err_alt)/rms(err_def) indipendente da N.
La stima puntuale e' il MAP sulla calotta, non la media: l'ortogonalita' e' attesa solo
approssimata.

Solo lettura: outputs/robustezza_prior_results.pkl (Caso B). La verita' Omega_n si rigenera
dai seed (primi due numeri del flusso spawn_key=(N, i), come in robustezza_prior.py).

Uso: python scripts/prior_B_rumore.py
Produce: la tabella su stdout.
"""
import pickle
from pathlib import Path

import numpy as np

from riptide_toy import kinematics
from riptide_toy.constants import SEED

# ordine dei task del Caso B in robustezza_prior.py (pool.map conserva l'ordine)
N_M_B = [(10, 200), (30, 200), (100, 100), (300, 50)]
DEFAULT = "piatto"
ALTERNATIVES = ["piatto largo", "logU"]
N_BOOTSTRAP = 2000


def true_direction(n_events: int, index: int) -> np.ndarray:
    """Omega_n vera dell'esperimento (N, i), rigenerata dal suo seed.

    Ritorna:
        versore, forma (3,).
    """
    rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(n_events, index)))
    return kinematics.direction_from_theta_phi(
        np.arccos(rng.uniform(-1.0, 1.0, 1)), rng.uniform(0.0, 2 * np.pi, 1))[0]


def tangent_coordinates(omega_hat: np.ndarray, omega_true: np.ndarray) -> np.ndarray:
    """Coordinate di omega_hat nel piano tangente a omega_true (mappa logaritmica sulla
    sfera), nella base (e_theta, e_phi) locale.

    Args:
        omega_hat: stime, versori, forma (M, 3).
        omega_true: verita', versori, forma (M, 3).

    Ritorna:
        array (M, 2) in rad; la norma e' la distanza angolare.
    """
    cos_angle = np.clip(np.sum(omega_hat * omega_true, axis=1), -1.0, 1.0)
    perp = omega_hat - cos_angle[:, None] * omega_true
    norm = np.linalg.norm(perp, axis=1)
    log_map = perp * (np.arccos(cos_angle) / np.where(norm > 0, norm, 1.0))[:, None]
    theta, phi = kinematics.theta_phi_from_direction(omega_true)
    e_theta = np.column_stack([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)])
    e_phi = np.column_stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)])
    return np.column_stack([np.sum(log_map * e_theta, axis=1), np.sum(log_map * e_phi, axis=1)])


def hausman_statistics(err_def: np.ndarray, err_alt: np.ndarray) -> np.ndarray:
    """Statistiche di Hausman su vettori di errore nel piano tangente.

    Args:
        err_def, err_alt: errori delle due stime, rad, forma (M, 2).

    Ritorna:
        array (4,): correlazione normalizzata err_def . d, rms|d| osservato e previsto da
        Hausman sqrt(max(rms(err_alt)^2 - rms(err_def)^2, 0)) in rad, rapporto
        rms(err_alt)/rms(err_def).
    """
    d = err_alt - err_def
    m_def, m_alt, m_d = (np.mean(np.sum(x ** 2, axis=1)) for x in (err_def, err_alt, d))
    corr = np.mean(np.sum(err_def * d, axis=1)) / np.sqrt(m_def * m_d)
    return np.array([corr, np.sqrt(m_d), np.sqrt(max(m_alt - m_def, 0.0)), np.sqrt(m_alt / m_def)])


def main() -> None:
    """Stampa, per N e prior alternativo, le tre verifiche con errori bootstrap.

    Ritorna:
        None (stdout).
    """
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    with open(out_dir / "robustezza_prior_results.pkl", "rb") as f:
        rows = pickle.load(f)["results"]["B"]
    tasks = [(n, i) for n, m in N_M_B for i in range(m)]
    omega_true = np.array([true_direction(n, i) for n, i in tasks])
    # controllo: la verita' rigenerata ridà gli errori angolari salvati
    hat_def = np.array([r[DEFAULT]["omega_hat"] for r in rows])
    saved_err = np.array([r[DEFAULT]["omega_error"] for r in rows])
    regen_err = np.linalg.norm(tangent_coordinates(hat_def, omega_true), axis=1)
    print(f"controllo verita' rigenerata: max |errore salvato - ricalcolato| = "
          f"{np.max(np.abs(saved_err - regen_err)):.1e} rad")

    rng = np.random.default_rng(SEED)
    n_of_row = np.array([n for n, _ in tasks])
    print("\nCaso B: spostamento d = stima_alt - stima_def nel piano tangente alla verita'")
    for n_events in sorted(set(n_of_row)):
        sel = n_of_row == n_events
        m = int(sel.sum())
        sigma = np.array([r[DEFAULT]["omega_sigma"] for r in rows])[sel]
        err_def = tangent_coordinates(hat_def[sel], omega_true[sel])
        print(f"\nN={n_events} (M={m})")
        for name in ALTERNATIVES:
            hat_alt = np.array([r[name]["omega_hat"] for r in rows])[sel]
            err_alt = tangent_coordinates(hat_alt, omega_true[sel])
            d = err_alt - err_def
            scale = np.mean(sigma)
            d_mean = d.mean(axis=0) / scale
            d_sem = d.std(axis=0, ddof=1) / np.sqrt(m) / scale
            stats = hausman_statistics(err_def, err_alt)
            boot_idx = rng.integers(0, m, size=(N_BOOTSTRAP, m))
            spread = np.array([hausman_statistics(err_def[b], err_alt[b]) for b in boot_idx]).std(axis=0)
            print(f"  {name:12s} <d>/sigma (e_theta, e_phi) {d_mean[0]:+.3f} +- {d_sem[0]:.3f},"
                  f" {d_mean[1]:+.3f} +- {d_sem[1]:.3f} | corr(err_def, d) {stats[0]:+.3f} +- {spread[0]:.3f}")
            print(f"  {'':12s} rms|d|/sigma osservato {stats[1] / scale:.3f} +- {spread[1] / scale:.3f}"
                  f" | previsto (Hausman) {stats[2] / scale:.3f} +- {spread[2] / scale:.3f}"
                  f" | rms err alt/def {stats[3]:.3f} +- {spread[3]:.3f}")


if __name__ == "__main__":
    main()
