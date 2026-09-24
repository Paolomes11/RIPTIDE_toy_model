# Roadmap — riptide-toy

v0.6 — 2026-09-24

## Stato

| Riga | Modulo | Stato |
|---|---|---|
| 1 | `constants` + `grids` | fatto (fix errata a applicato), testato |
| 2 | `kinematics` | fatto, testato |
| 3 | `forward_model` | fatto (firma diversa dalla guida, vedi Deviazioni), testato |
| 4 | `posterior_A` (+ `priors.energy_prior`, `constants.SIGMA_EP/SIGMA_THETA`) | fatto, oracolo Es. 38.1 verde (argmax 1.80 MeV, σ pesata 0.219 MeV vs atteso 1.82±0.21) |
| 5 | `combine` | fatto, oracolo Es. 39.1 verde (argmax 2.4067 MeV, σ pesata 0.1828 vs atteso 2.40±0.18); `expected_sigma_n(1.0,4)==0.5` |
| 6 | `validate` | fatto, oracolo Es. 40.1 verde (bias 0.08→0.20 MeV, pull mean 0.90, pull width 1.44, copertura 0.433 vs atteso ~45%) |
| 7 | `scripts/ch39_fig_repro.py` | fatto, Fig. 39.1 riprodotta (`outputs/fig_39_1.png`, ignorato da git); plateau di contrazione sotto shift sistematico 2% visibile in `outputs/fig_39_1_systematic_shift.png` (offset ≈0.05 MeV da N≥1000, sigma continua a scendere a ~0.001 MeV) |
| — | **checkpoint Caso A** | raggiunto: `combine.py`/`validate.py` congelati (solo funzioni nuove additive da qui in poi) |
| 8 | `kinematics` 3D | fatto (`direction_from_theta_phi`), testato; limite `Ω_n ∥ z ⇒ Caso A` verde |
| 9 | `priors` (direction) | fatto (`direction_prior`, fix errata c su `grids.sphere_grid`), testato |
| 10 | `posterior_B` | fatto (`forward_model.loglik_marginal_En` + `posterior_B.single_event_posterior`), testato; limite 1 evento + prior piatto verde |
| 11 | `combine` su Ω_n (riuso) | fatto, nessuna modifica a `combine.py`; contrazione angolare σ(N=10)→σ(N=100) coerente con 1/√N entro tolleranza larga (singola realizzazione MC) |
| 12 | `validate` su distanza angolare | da fare |
| 13 | `posterior_C` | da fare |
| 14 | `validate` finale su C | da fare |

## Deviazioni dalla guida (documentate, non silenziose)

- **`forward_model.loglik`**: la guida (Sez. 3) specifica una firma più semplice, senza
  marginalizzazione di θ_p. Il codice esistente marginalizza già θ_p internamente
  (`sigma_Ep, sigma_theta, log_prior_theta, chunk_size`, a chunk per stare in RAM).
  Decisione: si mantiene questa versione (ottimizzata) invece di riscriverla; `posterior_A`
  si adatta alla firma reale. Motivo: priorità a prestazioni e tempo di sviluppo.
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
  modulo a valle: non ha bloccato la riga 4, né la riga 11 (il test di contrazione angolare
  del Caso B genera `theta_p_true`/`phi_true` sintetici pescandoli direttamente, senza passare
  da questa funzione, perché il criterio di accettazione della riga 11 è l'andamento
  qualitativo ~1/√N, non un oracolo numerico legato alla formula di isotropia in CM). Resta
  aperto per un'eventuale futura generazione di eventi a partire dalla fisica dello scattering
  vero e proprio (angolo CM → lab), non ancora necessaria in nessun test del progetto.
- **`posterior_A.single_event_posterior`**: la guida (Sez. 3) assume `theta_p_hat` già fissato
  dentro `loglik`. Qui `forward_model.loglik` marginalizza `theta_p` internamente su un prior
  piatto proprio su `[0, π/2]` (coerente con la deviazione già presa per `forward_model`), e
  `single_event_posterior` somma il prior su `En` una sola volta sopra il risultato. Verificato
  contro l'oracolo Es. 38.1 (vedi tabella Stato, riga 4).
- **`validate.coverage_curve`**: la guida (Sez. 3) non dà un type hint per il parametro
  `intervals`. Si sceglie la forma più generica possibile, `(n_eventi, n_livelli, 2)` con
  estremi inferiore/superiore espliciti (non necessariamente simmetrici o gaussiani), così
  da restare utilizzabile sia con intervalli gaussiani (livello → z·σ_hat) sia con intervalli
  letti da un posterior a griglia (Caso B/C, dove la forma dell'intervallo non è nota a priori).
- **Ricostruzione "faulty" dell'Es. 40.1**: il libro descrive un errore di scala sistematico
  (+3%, +0.05 MeV) e incertezze dichiarate 30% troppo strette rispetto alla vera risoluzione
  σ(En)=0.08·En, ma non specifica se il rumore intrinseco del detector vada applicato prima o
  dopo il fattore di scala. Le due costruzioni sono entrambe plausibili dal solo testo; si è
  scelta quella con rumore applicato **dopo** l'errore di scala (non scalato insieme ad esso),
  perché riproduce pull width ≈1.4 molto più fedelmente (rapporto atteso 1/0.7≈1.429 contro
  1.03/0.7≈1.471 nell'altra costruzione). I valori del libro per questo esempio sono dati con
  "~"/"≈" (a differenza di Es. 38.1/39.1, che hanno tolleranza numerica esplicita — CLAUDE.md
  Sez. 5): le tolleranze del test (`test_example_40_1`) sono scelte in proporzione a quanto
  approssimative sono le cifre citate (0.02 MeV su bias, 0.1 su pull mean/width, 0.05 su
  copertura, quest'ultima la meno precisa delle quattro: "~45%" è la cifra più tonda del testo).

- **`scripts/ch39_fig_repro.py`**: il libro (Cap. 39, Esercizio 3) usa N=1,2,10,100 sia per
  il posterior combinato sia per la curva di contrazione con shift sistematico 2%. A quel
  range lo shift su Ep (2%, piccolo rispetto a σ_Ep) produce un offset ancora dentro il
  rumore statistico: a N=100 l'offset osservato (~0.02-0.08 MeV a seconda del run) non è
  chiaramente separato dal rumore, quindi il plateau richiesto da CLAUDE.md Sez. 6
  ("il plateau della contrazione è visibile, non converge a zero") non sarebbe dimostrabile
  con questi soli quattro punti. Si usa quindi un range esteso, `N=1,2,10,100,1000,5000`,
  solo per la figura del contraction plot con shift; la figura del posterior vs N e del
  contraction plot senza shift restano su `N=1,2,10,100` come nel libro. Con il range esteso
  l'offset si stabilizza a ≈0.05 MeV (coerente con la stima analitica 0.02·En_true=0.05 MeV)
  mentre σ continua a scendere fino a ~0.001 MeV: il plateau è così chiaramente visibile.

- **`posterior_B` / `forward_model.loglik_marginal_En`**: la guida (Sez. 3, Sez. 5) descrive
  il Caso B come θ_p derivato geometricamente dalla traccia 3D osservata e dal candidato Ω_n
  (`kinematics.recoil_angle_from_direction`), non come nuisance da marginalizzare su una
  griglia separata — a differenza del Caso A, dove θ_p è genuinamente ignoto. Questo angolo è
  invariante all'azimut della traccia attorno al candidato (per costruzione), quindi basta
  un'unica marginalizzazione su `En_grid`, con broadcasting a due assi (candidati × `En_grid`,
  come indicato in guida per le prestazioni del Caso B), invece di una griglia a tre assi con
  una seconda marginalizzazione su θ_p. I candidati con θ_p > π/2 (emisfero posteriore) sono
  esclusi (log-verosimiglianza −∞): il rinculo è sempre in avanti per scattering elastico a
  masse uguali (CLAUDE.md Sez. 3, `θ_lab ≤ 90°`). Nota sul test di limite: con prior piatto su
  `En`, l'integrale marginale su `En` varia con θ_p come `1/cos²θ_p` (Jacobiano del cambio di
  variabile `En' = En·cos²θ_p`) — non è esattamente costante, solo approssimativamente entro
  un cono centrale attorno alla direzione osservata; il test verifica questo comportamento
  qualitativo (nessun picco netto sull'emisfero anteriore, in contrasto con il taglio netto a
  −∞ sull'emisfero posteriore), non una costanza esatta.

- **Riga 11, dati sintetici per il test di contrazione angolare**: la guida chiede "Contrazione
  angolare vs N, stesso andamento della Fig. 39.1", senza specificare come generare gli eventi
  sintetici. Si è scelto di pescare `theta_p_true` e `phi_true` (angolo e azimut del rinculo
  rispetto a `Omega_n_true`, fissato sull'asse z senza perdita di generalità, come nel test
  della riga 8) direttamente da distribuzioni uniformi indipendenti, invece di simulare l'intera
  catena fisica a partire da un angolo di scattering in CM — perché quest'ultima richiederebbe
  la formula di isotropia di `kinematics.sample_cm_angle`, ancora aperta (vedi sopra). Etichetta
  **(b)**: dato sintetico "onesto" con assunzioni esplicite, analogo alla generazione diretta
  già usata nei test di `validate.py` per il Caso A (verità pescata direttamente, non simulata
  da una catena fisica completa). Il criterio di accettazione verificato è qualitativo (la
  deviazione standard angolare pesata sul posterior diminuisce da N=10 a N=100 in modo
  compatibile con 1/√N entro una tolleranza larga, dato che si tratta di una singola
  realizzazione Monte Carlo e non di una media d'insieme), coerente con quanto richiesto dalla
  guida per questa riga.

## Pubblicazione su GitHub

- **la guida di progetto locale non va pubblicata su GitHub** (2026-09-24): è
  stato rimosso dalla cronologia dei commit (tutti i branch) e aggiunto a `.gitignore`; resta
  presente solo in locale, fuori dal tracking git. Stessa esclusione già in vigore per
  il materiale di riferimento locale (il libro). L'unico file sotto `docs/` che resta
  tracciato in git è questo roadmap.

## Errata applicate

- (a) `grids.py`: `@lru_cache(maxsize=1)` → `maxsize=None`; array resi non scrivibili.
- (b) `priors.py` (verifica): `np.trapz` deprecato in NumPy ≥ 2.0 → `scipy.integrate.trapezoid`.
- (c) `grids.sphere_grid`: la costruzione precedente usava due array `linspace` indipendenti
  per `theta`/`phi` (stessa lunghezza `n_pixel` ma nessun accoppiamento tra i due) — non era
  una vera pixelizzazione della sfera, perché l'indice `i` non corrispondeva a un'unica
  direzione. Sostituita con un reticolo di Fibonacci (angolo aureo): un indice = una
  direzione, area solida quasi costante per pixel (costruzione numerica standard, non una
  formula fisica). Verificato con `cos(theta)` a media nulla su punti uniformi (isotropia,
  Cap. 20).
