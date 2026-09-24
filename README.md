# riptide-toy

Toy model Python per la ricostruzione statistica da singolo scattering n–p (RIPTIDE),
Casi A/B/C. Progetto satellite di un articolo, separato dalla tesi.

Vedi `docs/roadmap.md` per lo stato di avanzamento.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -c "import riptide_toy"   # se fallisce: venv non attivo o manca pip install -e
```

## Test

```bash
pytest tests/ -v --ignore=tests/test_performance.py   # test funzionali
pytest tests/test_against_book.py -v                  # solo gli oracoli del libro
pytest tests/ -v                                       # anche i test di prestazione
```

## Stato roadmap

Vedi `docs/roadmap.md` per lo stato dettagliato riga per riga.
