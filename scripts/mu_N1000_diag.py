"""Caso C a N = 1000: la sovra-copertura di mu_E (coverage 90% = 0.95 su M = 240) e' una
fluttuazione o un effetto?

Dati (solo lettura): outputs/caso_C_checklist_results.pkl (N = 1000, i = 0-39) e
outputs/n1000_extra_results.pkl (Caso C, i = 40-239), prior "logU".

Previsioni (scritte prima del calcolo):
  1. sotto l'ipotesi di calibrazione (pull iid N(0,1), M = 240) la larghezza del pull ha
     errore ~1/sqrt(2M) = 0.046: 0.92 e' a ~1.7 sigma; le coverage 90% e 95% sono a ~2.5 sigma
     ciascuna, ma sono fortemente correlate fra loro e con la larghezza. Una p-value
     Monte Carlo che tiene conto delle correlazioni (stesse statistiche su M normali
     standard) attesa ~0.02-0.05: tensione lieve, non un effetto netto;
  2. il passo della griglia fine non c'entra: la finestra copre ~+-4.5 std con N_MU_FINE
     punti, passo ~0.23 std, e la somma su griglia uniforme di una gaussiana con passo
     h << std ridà media e varianza con errore relativo ~exp(-2 pi^2 std^2 / h^2), nullo;
  3. quindi nessuna struttura per terzili di sigma_true e di mu_true: larghezza del pull e
     coverage 90% compatibili fra gli strati.

Uso: python scripts/mu_N1000_diag.py
Produce: la tabella su stdout.
"""
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import norm

from riptide_toy.constants import N_MU_FINE, SEED

N_EVENTS = 1000
N_CHECKLIST = 40    # primi task di caso_C_checklist.py: N = 1000, i = 0-39
N_BOOTSTRAP = 4000
N_NULL = 20000      # pseudo-esperimenti sotto l'ipotesi di calibrazione
N_STRATA = 3
WINDOW_HALF_WIDTH_STD = 4.5  # semi-ampiezza tipica della finestra fine, in std posteriori


def load_rows(out_dir: Path) -> tuple[list[dict], np.ndarray]:
    """Righe del Caso C a N = 1000 dai due pkl, nell'ordine degli indici i = 0-239.

    Ritorna:
        (righe, livelli): lista di M dict con mu_true, sigma_true (MeV) e il riassunto
        "logU"; livelli di credibilita', forma (L,).
    """
    with open(out_dir / "caso_C_checklist_results.pkl", "rb") as f:
        checklist = pickle.load(f)["results"][:N_CHECKLIST]
    with open(out_dir / "n1000_extra_results.pkl", "rb") as f:
        extra = pickle.load(f)
    rows = checklist + extra["results"]["C"]
    assert all(r["N"] == N_EVENTS for r in rows)
    return rows, extra["LEVELS"]


def summary(pull: np.ndarray, covered: np.ndarray) -> np.ndarray:
    """Larghezza del pull e coverage ai livelli.

    Args:
        pull: forma (M,) o (B, M).
        covered: booleani, forma (..., M, L).

    Ritorna:
        array (..., 1 + L): std del pull e frazioni coperte.
    """
    return np.concatenate([pull.std(axis=-1)[..., None], covered.mean(axis=-2)], axis=-1)


def null_distribution(rng: np.random.Generator, m: int, levels: np.ndarray) -> np.ndarray:
    """Le stesse statistiche su pseudo-esperimenti calibrati: pull N(0,1), intervallo
    centrale gaussiano al livello.

    Ritorna:
        array (N_NULL, 1 + L).
    """
    pull = rng.standard_normal((N_NULL, m))
    half = norm.ppf(0.5 + levels / 2)
    return summary(pull, np.abs(pull)[..., None] <= half)


def main() -> None:
    """Stampa significativita' globale, controllo del passo di griglia e stratificazione.

    Ritorna:
        None (stdout).
    """
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    rows, levels = load_rows(out_dir)
    m = len(rows)
    mu_true = np.array([r["mu_true"] for r in rows])
    sigma_true = np.array([r["sigma_true"] for r in rows])
    mean = np.array([r["logU"]["mu_mean"] for r in rows])
    std = np.array([r["logU"]["mu_std"] for r in rows])
    interval = np.array([r["logU"]["mu_int"] for r in rows])          # (M, L, 2)
    pull = (mean - mu_true) / std
    covered = (interval[..., 0] <= mu_true[:, None]) & (mu_true[:, None] <= interval[..., 1])
    rng = np.random.default_rng(SEED)

    obs = summary(pull, covered)
    null = null_distribution(rng, m, levels)
    expected = np.concatenate([[1.0], levels])
    null_sd = null.std(axis=0)
    print(f"Caso C, mu_E, N={N_EVENTS}, M={m}")
    print("  statistica      osservato  atteso  sd(H0)   z")
    for name, o, e, s in zip(["larghezza pull"] + [f"coverage {lv:.0%}" for lv in levels],
                             obs, expected, null_sd):
        print(f"  {name:15s} {o:9.3f} {e:7.3f} {s:7.3f} {(o - e) / s:+5.2f}")
    # statistica globale: distanza di Mahalanobis dal valore atteso con la covarianza di H0
    cov_inv = np.linalg.pinv(np.cov(null, rowvar=False))
    d2 = lambda x: np.einsum("...i,ij,...j->...", x - expected, cov_inv, x - expected)  # noqa: E731
    print(f"  globale (Mahalanobis, covarianza H0): d2 = {d2(obs):.2f},"
          f" p-value MC = {np.mean(d2(null) >= d2(obs)):.3f}")
    print(f"  solo larghezza: p-value MC (una coda, <= osservato) = {np.mean(null[:, 0] <= obs[0]):.3f}")

    step = 2 * WINDOW_HALF_WIDTH_STD / (N_MU_FINE - 1)
    print(f"\nPasso della griglia fine su mu_E ~ {step:.2f} std posteriori:"
          f" errore relativo sui momenti ~ exp(-2 pi^2 / h^2) = {np.exp(-2 * np.pi ** 2 / step ** 2):.1e}")
    print(f"std posteriore * sqrt(N) / sigma_true: media {np.mean(std * np.sqrt(N_EVENTS) / sigma_true):.3f}"
          f" (risoluzione per evento oltre sigma_E)")

    for label, key in (("sigma_true", sigma_true), ("mu_true", mu_true)):
        edges = np.quantile(key, np.linspace(0, 1, N_STRATA + 1))
        stratum = np.clip(np.searchsorted(edges, key, side="right") - 1, 0, N_STRATA - 1)
        print(f"\nTerzili di {label} (MeV)")
        for k in range(N_STRATA):
            sel = stratum == k
            boot = rng.integers(0, sel.sum(), size=(N_BOOTSTRAP, sel.sum()))
            p, c = pull[sel], covered[sel]
            spread = summary(p[boot], c[boot]).std(axis=0)
            s = summary(p, c)
            print(f"  [{edges[k]:.2f}, {edges[k + 1]:.2f}] M={sel.sum():3d}"
                  f" | pull {p.mean():+.3f} / {s[0]:.3f} +- {spread[0]:.3f}"
                  f" | coverage 90% {s[2]:.3f} +- {spread[2]:.3f}")


if __name__ == "__main__":
    main()
