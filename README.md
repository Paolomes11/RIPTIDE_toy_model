# riptide-toy

Toy model Python per la ricostruzione statistica da singolo scattering n–p (RIPTIDE),
Casi A/B/C. Progetto satellite di un articolo, separato dalla tesi.

- **Risultati in sintesi**: `docs/resoconto.md`
- **Stato riga per riga e scelte di implementazione**: `docs/roadmap.md`
- **Diario tecnico del Caso C**: `docs/report_caso_C_stadio1.md`

## I tre casi

| Caso | Si ricostruisce | Per ogni evento si marginalizza |
|---|---|---|
| A | energia E_n condivisa, direzione nota | angolo di rinculo θ_p |
| B | direzione Ω_n condivisa | energia E_n (prior largo) |
| C | Ω_n e la distribuzione delle energie N(μ_E, σ_E) | E_n di ogni evento |

## Mappa del repo

```
src/riptide_toy/   codice riusabile e testato
  constants.py       risoluzioni, range, seed
  grids.py           griglie in E_n, θ, sfera, (μ_E, σ_E)
  kinematics.py      E_p = E_n cos²θ_p, campionamento degli eventi
  forward_model.py   verosimiglianze per evento
  priors.py          prior propri
  posterior_A/B/C.py un file per caso
  combine.py         somma delle log-verosimiglianze sugli eventi (prior contato una volta)
  validate.py        bias, pull, coverage, contrazione
tests/             pytest (test_performance.py = soglie di tempo)
scripts/           run di validazione; scrivono in outputs/ (non versionata)
notebooks/         esempi guidati, dalla teoria ai numeri
docs/              resoconto, roadmap, report del Caso C
```

## Notebook, in ordine di lettura

1. `01_cinematica`: cinematica dello scattering, spettro del protone e distribuzione dell'angolo
   delle tracce (base della diagnostica di sorgente unica).
2. `02_caso_A`: casi di riferimento R1–R3, checklist, contrazione, plateau sistematico e robustezza
   al prior.
3. `03_caso_B`: verosimiglianza di un evento, checklist del Caso B, robustezza al prior su E_n e
   stress con una seconda sorgente.
4. `04_caso_C`: modello gerarchico, limiti σ_E → 0 e σ_E → ∞, spettri non gaussiani.
5. `05_validazione_caso_C`: checklist completa del Caso C e stress test (N = 1000 a statistica piena,
   prior, forma dello spettro, seconda sorgente, due stadi contro stima congiunta), letti dai
   risultati in `outputs/`.

Le versioni intermedie della validazione del Caso C sono state rimosse; restano nella storia git.

I notebook sono salvati senza output. Eseguili in VS Code/Jupyter, oppure installa nel venv
`pip install nbclient ipykernel`. I notebook 02, 03 e 05 leggono i pkl prodotti dagli script in `scripts/`.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -c "import riptide_toy"   # se fallisce: venv non attivo o manca pip install -e
```

## Test

```bash
pytest tests/ -v --ignore=tests/test_performance.py   # test funzionali
pytest tests/test_reference_examples.py -v            # solo i casi di riferimento R1-R3
pytest tests/ -v                                       # anche i test di prestazione
```

## Licenza

Apache License 2.0: vedi `LICENSE` e `NOTICE`. Copyright 2026 Giulio Mesini.
