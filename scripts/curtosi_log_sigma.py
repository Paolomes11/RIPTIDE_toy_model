"""Caso C con spettro non gaussiano: la larghezza del pull di log sigma_E e' governata dalla
curtosi dello spettro?

Ipotesi. L'errore di log sigma_E rispetto alla sd vera ha due parti: quella intrinseca
(la sd campionaria delle N energie vere fluttua attorno alla sd vera) e quella di
ricostruzione (la stima rispetto alla sd campionaria). Per una distribuzione di curtosi k
    var(log s) ~ (k - 1) / (4 N),
mentre il modello gaussiano del Caso C mette nel posterior il valore gaussiano 2 / (4 N).
Se la parte di ricostruzione non dipende dalla forma, per una forma X
    var(err_X) ~ var(err_gauss) + (k_X - 3) / (4 N)
e la larghezza del pull scala come sqrt(var(err_X) / var(err_gauss)) a pari std posteriore.

Previsioni (scritte prima del calcolo), per N = 50/150/300/1000:
  1. var(En_ls - ls_true) compatibile con (k - 1) / (4 N) per ogni forma (k: gauss 3,
     uniforme 1.8, bimodale a^4 + 6 a^2 (1 - a^2) + 3 (1 - a^2)^2, lognormale dalla sua
     formula esperimento per esperimento);
  2. var(ls_mean - En_ls) (ricostruzione) uguale fra le forme entro l'errore;
  3. larghezza del pull prevista dalla sola forma gaussiana e da k compatibile con
     l'osservata per uniforme, bimodale e lognormale.
Se reggono, la stessa formula da' la previsione numerica per spettri a code pesanti
(Laplace k = 6, Student-t con STUDENT_T_DOF_PREVIEW gradi di liberta').

Aggiunta dopo il primo calcolo (dichiarata come tale): la varianza degli errori e' dominata
da pochi esperimenti con std posteriore grande, quindi la previsione 3 si riscrive per
esperimento, pull^2 medio = w_gauss^2 + < (k - 3) / (4 N std^2) >, e si misura su uniforme
e bimodale il fattore di amplificazione A che la porterebbe sull'osservato:
w_X^2 - w_gauss^2 = A < (k - 3) / (4 N std^2) >. Per le code pesanti si stampano le due
previsioni, con A = 1 (solo termine intrinseco) e con A misurato.

Solo lettura: outputs/stress_spettro_results.pkl (En_ls = log della sd campionaria delle
energie vere, salvata dallo stress test).

Uso: python scripts/curtosi_log_sigma.py
Produce: la tabella su stdout.
"""
import pickle
from pathlib import Path

import numpy as np

from riptide_toy.constants import BIMODAL_HALF_SEPARATION, SEED

N_BOOTSTRAP = 2000
STUDENT_T_DOF_PREVIEW = 5   # curtosi 3 + 6 / (dof - 4) = 9, finita
KURTOSIS_PREVIEW = {"laplace": 6.0, "student_t": 3.0 + 6.0 / (STUDENT_T_DOF_PREVIEW - 4)}


def kurtosis(shape: str, mu_true: np.ndarray, sigma_true: np.ndarray) -> np.ndarray:
    """Curtosi (momento quarto standardizzato) dello spettro, esperimento per esperimento.

    Args:
        shape: forma dello spettro dello stress test.
        mu_true, sigma_true: media e sd vere, MeV, forma (M,).

    Ritorna:
        array (M,), adimensionale.
    """
    if shape == "gauss":
        return np.full(mu_true.shape, 3.0)
    if shape == "uniform":
        return np.full(mu_true.shape, 1.8)
    if shape == "bimodal":
        a2 = BIMODAL_HALF_SEPARATION ** 2
        return np.full(mu_true.shape, a2 ** 2 + 6 * a2 * (1 - a2) + 3 * (1 - a2) ** 2)
    if shape == "lognormal":
        w = 1.0 + (sigma_true / mu_true) ** 2   # exp(s^2) della lognormale
        return w ** 4 + 2 * w ** 3 + 3 * w ** 2 - 3
    raise ValueError(f"forma sconosciuta: {shape}")


def arrays(rows: list[dict], shape: str, n_events: int) -> dict[str, np.ndarray]:
    """Array (M,) per forma e N: verita', log sd campionaria, media e std posteriori di
    log sigma_E (adimensionali) e curtosi.

    Ritorna:
        dict di array (M,).
    """
    sel = [r for r in rows if r["shape"] == shape and r["N"] == n_events]
    out = {k: np.array([r[k] for r in sel]) for k in ("mu_true", "ls_true", "En_ls")}
    out["mean"] = np.array([r["ls"]["mean"] for r in sel])
    out["std"] = np.array([r["ls"]["std"] for r in sel])
    out["kappa"] = kurtosis(shape, out["mu_true"], np.exp(out["ls_true"]))
    return out


def var_with_error(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """Varianza campionaria e suo errore bootstrap.

    Ritorna:
        (varianza, errore), unita' di x al quadrato.
    """
    boot = rng.integers(0, x.size, size=(N_BOOTSTRAP, x.size))
    return float(x.var(ddof=1)), float(x[boot].var(axis=1, ddof=1).std())


def main() -> None:
    """Stampa, per N e forma, le tre verifiche e la previsione per le code pesanti.

    Ritorna:
        None (stdout).
    """
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    with open(out_dir / "stress_spettro_results.pkl", "rb") as f:
        data = pickle.load(f)
    rows, shapes = data["results"], data["SHAPES"]
    rng = np.random.default_rng(SEED)
    for n_events in sorted({r["N"] for r in rows}):
        g = arrays(rows, "gauss", n_events)
        var_g = np.var(g["mean"] - g["ls_true"], ddof=1)
        width_g = np.std((g["mean"] - g["ls_true"]) / g["std"])
        amplification = []
        print(f"\nN={n_events} (M={g['ls_true'].size}); varianze x 4N")
        print("  forma      k     intrinseca oss. (prev. k-1) | ricostruzione | pull oss. / prev.")
        for shape in shapes:
            t = arrays(rows, shape, n_events)
            k = float(np.mean(t["kappa"]))
            intr, intr_err = var_with_error(t["En_ls"] - t["ls_true"], rng)
            reco, reco_err = var_with_error(t["mean"] - t["En_ls"], rng)
            pull = (t["mean"] - t["ls_true"]) / t["std"]
            boot = rng.integers(0, pull.size, size=(N_BOOTSTRAP, pull.size))
            predicted = (width_g * np.sqrt((var_g + (k - 3) / (4 * n_events)) / var_g)
                         * np.mean(g["std"]) / np.mean(t["std"]))
            f = 4 * n_events
            print(f"  {shape:9s} {k:4.2f}  {intr * f:5.2f} +- {intr_err * f:4.2f} ({k - 1:4.2f})"
                  f"          | {reco * f:5.2f} +- {reco_err * f:4.2f} |"
                  f" {pull.std():.2f} +- {pull[boot].std(axis=1).std():.2f} / {predicted:.2f}")
            excess = np.mean((t["kappa"] - 3) / (4 * n_events * t["std"] ** 2))
            print(f"  {'':9s} per esperimento: pull previsto {np.sqrt(width_g ** 2 + excess):.2f}"
                  f" | mediana std^2 x 4N {np.median(t['std'] ** 2) * f:5.2f}")
            if shape in ("uniform", "bimodal"):
                amplification.append((pull.std() ** 2 - width_g ** 2) / excess)
        a = float(np.mean(amplification))
        excess_unit = np.mean(1.0 / (4 * n_events * g["std"] ** 2))
        print(f"  amplificazione A (uniforme, bimodale): {np.round(amplification, 1)} -> media {a:.1f}")
        for shape, k in KURTOSIS_PREVIEW.items():
            print(f"  previsione {shape:9s} k={k:.0f}: larghezza del pull"
                  f" ~ {np.sqrt(width_g ** 2 + (k - 3) * excess_unit):.2f} (A = 1)"
                  f" / {np.sqrt(width_g ** 2 + a * (k - 3) * excess_unit):.2f} (A = {a:.1f})")


if __name__ == "__main__":
    main()
