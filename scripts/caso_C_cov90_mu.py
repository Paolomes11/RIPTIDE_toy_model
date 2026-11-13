"""Coverage di mu_E a N = 300 su seed nuovi: la checklist regge con M = 600?

Stessa pipeline di scripts/caso_C_cov68.py (stadio 1 a prior largo, stadio 2
su omega_0 e sulla Omega_n vera), con seed disgiunti sia dalla checklist sia
da caso_C_cov68: spawn_key=(N, i, 90).

Dominio 10-20 MeV, E_n per evento dalla gaussiana troncata. Nella checklist (M = 100) a
N = 300 mu_E ha coverage 68% 0.68 e pull +0.18 / 0.92.
Previsioni (scritte prima del calcolo):
  1. [hat] coverage 68 / 90 / 95% nominale entro 2 errori binomiali (+-0.019 / 0.012 / 0.009);
  2. pull medio |media| < 0.1 (errore ~0.04 con M = 600: il +0.18 della checklist, a ~1.8
     errori con M = 100, e' una fluttuazione) e larghezza 1.00 +- 0.06;
  3. [true] - [hat] entro ~0.02 sulla coverage, come in caso_C_cov68.

Uso: OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=5G -p MemorySwapMax=0 \
         python scripts/caso_C_cov90_mu.py [n_processi]
     (~10.5 min con 3 processi, misurato 2026-09-26 a 0.5-6 MeV)
Produce: outputs/caso_C_cov90_mu_results.pkl e la tabella su stdout.
"""
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from riptide_toy import validate

sys.path.insert(0, str(Path(__file__).parent))
from caso_C_checklist import LEVELS  # noqa: E402
from caso_C_cov68 import run_experiment  # noqa: E402  stessa pipeline, tag diverso

N_M = [(300, 600)]
SEED_TAG = 90  # terzo elemento di spawn_key: seed disgiunti da checklist (nessun tag) e cov68 (68)


def main() -> None:
    n_proc = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    tasks = [(n, i, SEED_TAG) for n, m in N_M for i in range(m)]
    with Pool(n_proc) as pool:
        results = pool.map(run_experiment, tasks, chunksize=1)
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "caso_C_cov90_mu_results.pkl", "wb") as f:
        pickle.dump({"N_M": N_M, "LEVELS": LEVELS, "results": results}, f)
    for n, m in N_M:
        rows = [r for r in results if r["N"] == n]
        mu_true = np.array([r["mu_true"] for r in rows])
        for name in ("hat", "true"):
            intervals = np.array([r[name]["mu_int"] for r in rows])
            _, cov = validate.coverage_curve(mu_true, intervals, LEVELS)
            pull = (np.array([r[name]["mu_mean"] for r in rows]) - mu_true) / np.array([r[name]["mu_std"] for r in rows])
            print(f"N={n} M={m} [{name}] coverage mu_E {np.round(cov, 3)}, "
                  f"pull media {pull.mean():+.3f} sd {pull.std():.3f}")


if __name__ == "__main__":
    main()
