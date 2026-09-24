# Roadmap — riptide-toy

v0.1 — 2026-09-24

## Stato

| Riga | Modulo | Stato |
|---|---|---|
| 1 | `constants` + `grids` | fatto (fix errata a applicato) |
| 2 | `kinematics` | fatto |
| 3 | `forward_model` | fatto (firma diversa dalla guida, vedi Deviazioni) |
| 4 | `posterior_A` | in corso |
| 5 | `combine` | da fare |
| 6 | `validate` | da fare |
| 7 | `scripts/ch39_fig_repro.py` | da fare |
| — | **checkpoint Caso A** | non ancora raggiunto |
| 8 | `kinematics` 3D | da fare |
| 9 | `priors` (direction) | da fare |
| 10 | `posterior_B` | da fare |
| 11 | `combine` su Ω_n | da fare |
| 12 | `validate` su distanza angolare | da fare |
| 13 | `posterior_C` | da fare |
| 14 | `validate` finale su C | da fare |

## Deviazioni dalla guida (documentate, non silenziose)

- **`forward_model.loglik`**: la guida (Sez. 3) specifica una firma più semplice, senza
  marginalizzazione di θ_p. Il codice esistente marginalizza già θ_p internamente
  (`sigma_Ep, sigma_theta, log_prior_theta, chunk_size`, a chunk per stare in RAM).
  Decisione: si mantiene questa versione (ottimizzata) invece di riscriverla; `posterior_A`
  si adatta alla firma reale. Motivo: priorità a prestazioni e tempo di sviluppo (indicazione
  esplicita dell'utente).
- **`sample_cm_angle` / `sample_recoil`**: sono la stessa funzione; nessun tempo di volo
  nello scope del toy. Il riferimento a `sample_recoil` nella tabella prestazioni della guida
  (Sez. 5) va letto come `sample_cm_angle`.
- **`risoluzione_caso_A`**: non definita in guida. Si usa la deviazione standard pesata sulla
  griglia del posterior: `σ² = Σ p(En)(En−⟨En⟩)² / Σp(En)` con `p(En) = exp(log_post − max(log_post))`.
  Grounded in Cap. 19 (momenti del posterior) e nell'Esempio 38.1 del libro.
- **Caso A, nuisance `z`**: CLAUDE.md Sez. 3 elenca `θ_p, z` come nuisance del Caso A, ma né
  la guida né `forward_model.py` usano `z`. `posterior_A` marginalizza solo `θ_p`; `z` è
  fuori scope per ora (nessuna sorgente di dato/osservabile per `z` definita da nessuna fonte).
- **`kinematics.sample_cm_angle`**: campiona `uniform(0, 2π)`. Da verificare se questo è
  corretto per uno scattering isotropo in CM (in tal caso l'angolo polare andrebbe campionato
  come `arccos(uniform(-1,1))`, non uniforme). Nessuna fonte autorevole del progetto conferma
  o smentisce la formula attuale — **aperto**, etichetta (d), non usato ancora da nessun
  modulo a valle: non blocca la riga 4.

## Errata applicate

- (a) `grids.py`: `@lru_cache(maxsize=1)` → `maxsize=None`; array resi non scrivibili.
