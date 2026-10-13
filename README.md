# riptide-toy

Toy model Python per la ricostruzione statistica da singolo scattering n–p (RIPTIDE),
Casi A/B/C. Progetto satellite di un articolo, separato dalla tesi.

Dominio di energia 10–20 MeV, cinematica n–p relativistica esatta, scattering isotropo in CM
(verificato contro la distribuzione angolare anisotropa di ENDF/B-VIII.0: effetto trascurabile).
Stato: Casi A e B calibrati a 10–20 MeV; Caso C con un punto aperto sulla calibrazione
(vedi `docs/roadmap.md`).

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
  kinematics.py      cinematica relativistica E_p(E_n, θ_p), campionamento degli eventi
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
5. `05_validazione_caso_C`: checklist completa del Caso C a 10–20 MeV, diagnostica dello spettro
   fuori dal dominio, robustezza al prior, e riassunto degli stress test fatti a 0.5–6 MeV, letti dai
   risultati in `outputs/`.
6. `06_modifiche_10_20_MeV`: cosa è cambiato col passaggio a 10–20 MeV e perché (cinematica
   relativistica, anisotropia ENDF/B-VIII.0, prior sin 2θ_p nel Caso A, `loglik` sotto 1 ms) e il punto
   aperto del Caso C.

Le versioni intermedie della validazione del Caso C sono state rimosse; restano nella storia git.

I notebook sono salvati senza output. Eseguili in VS Code/Jupyter, oppure installa nel venv
`pip install nbclient ipykernel`. I notebook 02, 03, 05 e 06 leggono i pkl prodotti dagli script in
`scripts/`.

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
