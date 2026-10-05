"""Caso C: lo scarto fra modo congiunto e media dei due stadi su log sigma_E e' asimmetria?

Il modo congiunto di (Omega_n, mu_E, log sigma_E) ha un pull medio negativo su log sigma_E,
mentre la media posteriore dei due stadi e' centrata. Ipotesi: e' la differenza fra moda e
media di un marginale asimmetrico (coda verso sigma_E grandi), non la correlazione fra
direzione e iperparametri.

Previsione (scritta prima del calcolo): per una densita' poco asimmetrica vale la relazione
empirica di Pearson moda ~ media - 3 (media - mediana). Quindi, con s = std posteriore:
  - esperimento per esperimento (modo - media)/s ~ -3 (media - mediana)/s:
    correlazione alta e pendenza ~1 della retta osservato vs previsto;
  - per N, pull medio del modo ~ pull medio della media - 3 <(media - mediana)/s>.

Esito: smentita. La previsione ha il segno opposto all'osservato (+0.25 contro -0.29 a N = 50):
il marginale ha la coda verso sigma_E piccoli. La scomposizione dello scarto e' in
modo_log_sigma.py.

Solo lettura: outputs/correlazione_stadi_results.pkl (modo congiunto, due stadi) e
outputs/caso_C_checklist_results.pkl (mediana del marginale di log sigma_E, stessi dataset:
spawn_key=(N, i), stessa ricostruzione a due stadi). Nessun ricalcolo.

Uso: python scripts/asimmetria_log_sigma.py
Produce: la tabella su stdout.
"""
import pickle
from pathlib import Path

import numpy as np

# ordine dei task nei due script che hanno prodotto i pkl (pool.map conserva l'ordine)
CHECKLIST_N_M = [(1000, 40), (300, 100), (150, 200), (50, 200)]
PEARSON_FACTOR = 3.0  # moda ~ media - 3 (media - mediana) per asimmetria debole


def index_by_task(rows: list[dict], n_m: list[tuple[int, int]]) -> dict[tuple[int, int], dict]:
    """Associa a ogni riga del pkl il suo task (N, i), nell'ordine in cui e' stato generato.

    Ritorna:
        dict {(N, i): riga}.
    """
    tasks = [(n, i) for n, m in n_m for i in range(m)]
    return dict(zip(tasks, rows))


def joined_arrays(stages: dict[tuple[int, int], dict], checklist: dict[tuple[int, int], dict],
                  n_events: int) -> dict[str, np.ndarray]:
    """Array allineati per N fissato (solo esperimenti con Hessiana definita negativa).

    Ritorna:
        dict di array (M,): verita', media, std, mediana dei due stadi e modo e std di Laplace
        del congiunto, tutti in log sigma_E (adimensionale); mismatch = scarto massimo fra le
        medie dei due pkl (controllo del join, atteso ~0).
    """
    keys = sorted(k for k in stages if k[0] == n_events and stages[k]["negative_definite"])
    st = [stages[k] for k in keys]
    ck = [checklist[k]["logU"] for k in keys]
    out = {
        "true": np.array([r["ls_true"] for r in st]),
        "mean": np.array([r["two_stage"]["ls"] for r in st]),
        "std": np.array([r["two_stage"]["ls_std"] for r in st]),
        "median": np.array([c["ls_median"] for c in ck]),
        "mode": np.array([r["joint"]["ls"] for r in st]),
        "mode_std": np.array([np.sqrt(r["joint"]["cov"][3, 3]) for r in st]),
    }
    out["mismatch"] = np.max(np.abs(out["mean"] - np.array([c["ls_mean"] for c in ck])))
    return out


def main() -> None:
    """Stampa, per N, previsione di Pearson vs osservato per lo scarto modo - media.

    Ritorna:
        None (stdout).
    """
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    with open(out_dir / "correlazione_stadi_results.pkl", "rb") as f:
        stage_data = pickle.load(f)
    with open(out_dir / "caso_C_checklist_results.pkl", "rb") as f:
        checklist_data = pickle.load(f)
    stages = index_by_task(stage_data["results"], stage_data["N_M"])
    checklist = index_by_task(checklist_data["results"], CHECKLIST_N_M)

    print("Caso C, log sigma_E: modo congiunto - media due stadi vs Pearson -3 (media - mediana)")
    for n_events in sorted({k[0] for k in stages}):
        t = joined_arrays(stages, checklist, n_events)
        m = t["true"].size
        observed = (t["mode"] - t["mean"]) / t["std"]
        predicted = -PEARSON_FACTOR * (t["mean"] - t["median"]) / t["std"]
        slope, intercept = np.polyfit(predicted, observed, 1)
        corr = np.corrcoef(predicted, observed)[0, 1]
        pull_mean = (t["mean"] - t["true"]) / t["std"]
        pull_mode = (t["mode"] - t["true"]) / t["mode_std"]
        # il pull del modo usa la std di Laplace: si riscala lo scarto previsto con s/s_Laplace
        pull_mode_pred = ((t["mean"] + predicted * t["std"]) - t["true"]) / t["mode_std"]
        sem = lambda x: x.std(ddof=1) / np.sqrt(x.size)  # noqa: E731
        print(f"\nN={n_events} (M={m}; controllo join: max |media ckl - media stadi| = {t['mismatch']:.1e})")
        print(f"  (modo - media)/s  osservato media {observed.mean():+.3f} +- {sem(observed):.3f}"
              f" | previsto {predicted.mean():+.3f} +- {sem(predicted):.3f}")
        print(f"  per esperimento: correlazione {corr:.3f} | pendenza {slope:.3f}"
              f" | intercetta {intercept:+.3f} | rms residuo {np.std(observed - predicted):.3f}")
        print(f"  pull medio: media due stadi {pull_mean.mean():+.3f} | modo osservato {pull_mode.mean():+.3f}"
              f" | modo previsto {pull_mode_pred.mean():+.3f}   (+-{sem(pull_mode):.3f})")


if __name__ == "__main__":
    main()
