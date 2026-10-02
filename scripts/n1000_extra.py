"""Casi B e C a N = 1000: esperimenti aggiuntivi per ridurre l'errore binomiale sulla coverage.

Le checklist (scripts/caso_B_checklist.py, scripts/caso_C_checklist.py) usano a N = 1000 solo
M = 40 esperimenti, indici 0-39 in spawn_key=(N, i). Qui si rilancia la stessa pipeline, con le
stesse funzioni run_experiment, sugli indici 40 ... 40 + M_EXTRA - 1: seed disgiunti da quelli
della checklist, che si possono quindi unire ai risultati gia' salvati.
Per il Caso B si rifa' anche la pendenza di contrazione con la statistica nuova a N = 1000, e si
guarda la forma del pull a N = 10 (code, mediana) per il regime non gaussiano.

Uso: OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=5G -p MemorySwapMax=0 \
         python scripts/n1000_extra.py [n_processi]
     (~0.3 GB per processo; ~15 min con 3 processi, misurato 2026-10-02)
Richiede: outputs/caso_B_checklist_results.pkl e outputs/caso_C_checklist_results.pkl.
Produce: outputs/n1000_extra_results.pkl e le tabelle su stdout.
"""
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from riptide_toy import validate

sys.path.insert(0, str(Path(__file__).parent))
import caso_B_checklist as checklist_B  # noqa: E402  stessa pipeline del Caso B
import caso_C_checklist as checklist_C  # noqa: E402  stessa pipeline del Caso C

N_EVENTS = 1000
M_EXTRA = 200
FIRST_INDEX = 40  # la checklist usa gli indici 0-39 a N = 1000
LEVELS = checklist_C.LEVELS


def run_task(task: tuple[str, int]) -> tuple[str, dict]:
    """Un esperimento del Caso B o C a N = N_EVENTS.

    Args:
        task: ("B" o "C", indice dell'esperimento).

    Ritorna:
        (caso, dict del run_experiment della checklist corrispondente).
    """
    case, index = task
    module = checklist_B if case == "B" else checklist_C
    return case, module.run_experiment((N_EVENTS, index))


def omega_summary(rows: list[dict]) -> str:
    """Riga di riassunto su Omega_n: rms errore, risoluzione dichiarata, pull rms, coverage HPD.

    Ritorna:
        str, angoli in gradi.
    """
    err = np.array([r["omega_error"] for r in rows])
    sig = np.array([r["omega_sigma"] for r in rows])
    cov = np.array([r["omega_covered"] for r in rows]).mean(axis=0)
    m = len(rows)
    return (f"M={m:3d}  rms err {np.degrees(checklist_B.rms(err)):.2f}"
            f" (dichiarata {np.degrees(sig.mean()):.2f}) | pull rms {checklist_B.rms(validate.angular_pull(err, sig)):.2f}"
            f" | coverage HPD {np.round(cov, 3)} (+-{np.sqrt(0.68 * 0.32 / m):.3f} a 0.68)")


def hyper_summary(rows: list[dict]) -> str:
    """Riga di riassunto su mu_E e log sigma_E del Caso C: bias, pull, coverage.

    Ritorna:
        str, mu in MeV, sigma come log(sigma_E / MeV).
    """
    t = checklist_C.collect(rows, N_EVENTS)
    mu_res = t["mu_mean"] - t["mu_true"]
    ls_res = t["ls_median"] - np.log(t["sigma_true"])
    mu_pull = mu_res / t["mu_std"]
    ls_pull = (t["ls_mean"] - np.log(t["sigma_true"])) / t["ls_std"]
    _, mu_cov = validate.coverage_curve(t["mu_true"], t["mu_int"], LEVELS)
    _, ls_cov = validate.coverage_curve(np.log(t["sigma_true"]), t["ls_int"], LEVELS)
    m = len(mu_res)
    return (f"bias mu {mu_res.mean():+.4f} +- {mu_res.std() / np.sqrt(m):.4f}"
            f" | bias log sigma {ls_res.mean():+.3f} +- {ls_res.std() / np.sqrt(m):.3f}"
            f" | pull mu {mu_pull.mean():+.2f}/{mu_pull.std():.2f}"
            f" | pull log sigma {ls_pull.mean():+.2f}/{ls_pull.std():.2f}"
            f"\n      coverage mu {np.round(mu_cov, 3)} | log sigma {np.round(ls_cov, 3)}")


def main() -> None:
    """Esegue gli esperimenti aggiuntivi, li unisce a quelli della checklist e stampa le tabelle.

    Ritorna:
        None (pkl in outputs/, tabelle su stdout).
    """
    n_proc = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    with open(out_dir / "caso_B_checklist_results.pkl", "rb") as f:
        old_B = pickle.load(f)["results"]
    with open(out_dir / "caso_C_checklist_results.pkl", "rb") as f:
        old_C = pickle.load(f)["results"]

    # il Caso C per primo: costa il doppio, cosi' i processi finiscono insieme
    tasks = [(case, i) for case in ("C", "B") for i in range(FIRST_INDEX, FIRST_INDEX + M_EXTRA)]
    with Pool(n_proc) as pool:
        results = pool.map(run_task, tasks, chunksize=1)
    new = {case: [r for c, r in results if c == case] for case in ("B", "C")}
    with open(out_dir / "n1000_extra_results.pkl", "wb") as f:
        pickle.dump({"N": N_EVENTS, "first_index": FIRST_INDEX, "LEVELS": LEVELS, "results": new}, f)

    old = {"B": [r for r in old_B if r["N"] == N_EVENTS], "C": [r for r in old_C if r["N"] == N_EVENTS]}
    for case in ("B", "C"):
        print(f"\nCaso {case}, N = {N_EVENTS}, Omega_n (gradi)")
        for label, rows in (("checklist", old[case]), ("nuovi", new[case]), ("uniti", old[case] + new[case])):
            print(f"  {label:9s} {omega_summary(rows)}")
    print(f"\nCaso C, N = {N_EVENTS}, iperparametri")
    for label, rows in (("checklist", old["C"]), ("nuovi", new["C"]), ("uniti", old["C"] + new["C"])):
        print(f"  {label:9s} {hyper_summary(rows)}")

    # Caso B: contrazione con N = 1000 a statistica piena
    merged_B = [r for r in old_B if r["N"] != N_EVENTS] + old["B"] + new["B"]
    n_values = sorted({r["N"] for r in merged_B})
    curve = np.array([checklist_B.rms(np.array([r["omega_error"] for r in merged_B if r["N"] == n]))
                      for n in n_values])
    print(f"\nCaso B, contrazione; rms*sqrt(N) [rad] = {np.round(curve * np.sqrt(n_values), 3)} per N = {n_values}")
    for lo, hi in ((10, 1000), (30, 300), (30, 1000)):
        sel = [k for k, n in enumerate(n_values) if lo <= n <= hi]
        slope = np.polyfit(np.log(np.array(n_values)[sel]), np.log(curve[sel]), 1)[0]
        print(f"  pendenza log-log N = {lo}-{hi}: {slope:+.3f} (attesa -0.5)")

    # Caso B, N = 10: forma del pull (atteso per gaussiana 2D: pull*sqrt(2) ~ Rayleigh)
    rows_10 = [r for r in old_B if r["N"] == 10]
    pull_10 = validate.angular_pull(np.array([r["omega_error"] for r in rows_10]),
                                    np.array([r["omega_sigma"] for r in rows_10])) * np.sqrt(2.0)
    rayleigh_median = np.sqrt(2.0 * np.log(2.0))
    print(f"\nCaso B, N = 10 (M={len(rows_10)}): pull*sqrt(2) mediana {np.median(pull_10):.3f}"
          f" (Rayleigh {rayleigh_median:.3f}); frazione > 3: {np.mean(pull_10 > 3):.4f}"
          f" (Rayleigh {np.exp(-4.5):.4f})")


if __name__ == "__main__":
    main()
