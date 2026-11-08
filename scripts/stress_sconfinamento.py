"""Stress test dello sconfinamento dal dominio di energia nel Caso C.

Stessi esperimenti di stress_spettro.py, ma con energie vere non troncate: una frazione
degli eventi cade fuori da [EN_MIN, EN_MAX], dove la ricostruzione non integra. Il fit e'
invariato; bersagli = media e sd nominali. Misura il danno quando il dominio del prior e'
sbagliato; il confronto con stress_spettro.py, a pari seed, ne isola l'effetto.

Uso: OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=5G -p MemorySwapMax=0 \
         python scripts/stress_sconfinamento.py [n_processi]
     (picco ~0.6 GB per processo a N=1000, come la checklist del Caso C)
Produce: outputs/stress_sconfinamento_results.pkl e la tabella su stdout.
"""
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stress_spettro import HEAVY_TAILED_SHAPES, LEVELS, N_M, SHAPES, print_table, run_experiment  # noqa: E402


def main() -> None:
    """Esegue gli esperimenti non troncati in parallelo, salva e stampa la tabella.

    Ritorna:
        None (pkl in outputs/, tabella su stdout).
    """
    n_proc = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    shapes = SHAPES + HEAVY_TAILED_SHAPES
    tasks = [(shape, n, i, False) for n, m in N_M for shape in shapes for i in range(m)]
    with Pool(n_proc) as pool:
        rows = pool.map(run_experiment, tasks, chunksize=1)
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "stress_sconfinamento_results.pkl", "wb") as f:
        pickle.dump({"SHAPES": shapes, "N_M": N_M, "LEVELS": LEVELS, "results": rows}, f)
    print_table(rows, shapes)


if __name__ == "__main__":
    main()
