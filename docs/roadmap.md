# Roadmap — riptide-toy

v0.8 — 2026-09-24

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
| 12 | `validate` su distanza angolare | fatto (`angular_residual`, `posterior_angular_resolution`, `angular_pull`, additive); verificato su geometria nota + su Caso B simulato (M=30 esperimenti, Ω_n nota): bias medio <20°, pull mediano d'ordine 1 |
| 13 | `posterior_C` | fatto (`grids.hyperparameter_grid`, `priors.energy_prior_given_hyperparams`/`hyperparameter_prior`, `forward_model.loglik_marginal_En_hierarchical`, `posterior_C.estimate_shared_direction`/`single_event_posterior`), testato; entrambi i limiti di Sez. 5 verdi (σ_E→∞ ≈ Caso B entro atol=0.01 sulla log-verosimiglianza; σ_E→0 recupera l'energia condivisa vera entro 0.1 MeV) |
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

- **Riga 12, `validate.py` su distanza angolare**: la guida (Sez. 4, riga 12) chiede
  "bias/pull su distanza angolare", senza dare firme né un oracolo numerico (etichetta
  **(d)**: ipotesi da testare, non un fatto del libro — la distanza angolare non è
  gaussiana, quindi non ci si aspetta un pull esattamente N(0,1) come nel caso 1D, solo
  scala ~1 se il ricostruttore è calibrato). Tre funzioni additive in `validate.py`
  (`combine.py`/`validate.py` restano congelati per le funzioni esistenti, CLAUDE.md
  Sez. 6): `angular_residual` (distanza angolare fra due versori), `posterior_angular_resolution`
  (analogo sferico della deviazione standard pesata già usata per `risoluzione_caso_A`,
  qui attorno a una direzione di riferimento per evento — verità nota, o la stima stessa),
  `angular_pull` (rapporto puro, senza assunzioni di normalità). Test di integrazione:
  M=30 esperimenti indipendenti, N=20 eventi sintetici ciascuno (stessa costruzione
  "onesta" della riga 11), stima di Ω_n via MAP del posterior combinato (`combine.combine_loglik`
  riusato, nessuna modifica), `sigma_hat` calcolata in modo self-referenziale (attorno
  alla propria stima MAP, non alla verità — l'unica scelta sensata quando si vuole
  un'incertezza dichiarata dal posterior senza conoscere la verità in un caso reale).
  Griglie (`sphere_grid(800)`, `energy_grid(100)`) e M scelti per tenere il test sotto i
  2s: confrontati con una configurazione più grande (`sphere_grid(1500)`, `energy_grid(150)`,
  M=200, 18.9s) le statistiche del pull (media/mediana/std) restano dello stesso ordine,
  confermando che la riduzione non altera la conclusione qualitativa. Soglie di
  accettazione volutamente larghe (bias medio <20°, pull mediano in [0.2, 3.0]), coerenti
  con l'etichetta (d).

- **Riga 13, `posterior_C` importa anche `combine`**: la guida (Sez. 3, tabella import) elenca
  per `posterior_C` solo `kinematics, forward_model, priors, grids, posterior_B`. Lo stadio 1
  (`estimate_shared_direction`) deve combinare gli N eventi su `Ω_n` prima di passare allo
  stadio 2 sulla griglia `(μ_E, σ_E)` — la griglia a due stadi mandata da CLAUDE.md Sez. 4
  ("mai griglia 4D bruta") richiede proprio questo. `combine.py` non ha import dal progetto ed
  è pensato per essere generico, chiamato da qualunque layer superiore (già usato direttamente
  nei test del Caso B, righe 11/12): l'import diretto in `posterior_C` è quindi una deviazione
  minima e coerente con l'architettura esistente, non una violazione dello strato.
- **Riga 13, doppio conteggio del prior nello stadio 1**: `posterior_B.single_event_posterior`
  somma già il prior sulla direzione una volta per evento (CLAUDE.md Sez. 3). Per riusarlo con
  `combine.combine_loglik` (che aggiunge il prior una sola volta sull'intero campione, come
  nelle righe 11/12) va prima sottratto il prior già sommato (`log_posterior_per_event -
  log(direction_prior)`), altrimenti verrebbe contato N volte invece di 1.
- **Riga 13, griglia iperparametri `(μ_E, σ_E)`**: `μ_E` copre lo stesso dominio di
  `energy_grid` (lineare); `σ_E` è spaziata logaritmicamente (`np.geomspace`,
  `SIGMA_E_MIN, SIGMA_E_MAX = 0.01, 50.0` MeV) perché è un parametro di scala. `SIGMA_E_MIN`
  è scelto ≪ `SIGMA_EP` (lo scatter fra le energie vere degli eventi diventa indistinguibile
  dal rumore di misura, limite Caso A); `SIGMA_E_MAX` è scelto ≫ `EN_MAX − EN_MIN` = 5.5 MeV
  (la gaussiana troncata sul dominio di `energy_grid` è già ~piatta, limite Caso B). Conseguenza:
  `priors.hyperparameter_prior` è uniforme **per punto della griglia**, non letteralmente piatta
  in `(μ_E, σ_E)` — con `σ_E` log-spaziata questo equivale a un prior uniforme in
  `(μ_E, log σ_E)`, la convenzione standard per un parametro di scala (evita di favorire `σ_E`
  grandi solo perché occupano più "spazio" lineare). Entrambi i limiti sono verificati
  numericamente in `tests/test_case_C.py` (vedi tabella Stato, riga 13).

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
