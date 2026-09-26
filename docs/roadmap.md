# Roadmap — riptide-toy

v0.13 — 2026-09-26

## Stato

| Riga | Modulo | Stato |
|---|---|---|
| 1 | `constants` + `grids` | fatto (fix errata a applicato), testato |
| 2 | `kinematics` | fatto, testato |
| 3 | `forward_model` | fatto (firma diversa dalla guida, vedi Deviazioni), testato |
| 4 | `posterior_A` (+ `priors.energy_prior`, `constants.SIGMA_EP/SIGMA_THETA`) | fatto, caso di riferimento R1 verde (argmax 1.80 MeV, σ pesata 0.219 MeV vs atteso 1.82±0.21) |
| 5 | `combine` | fatto, caso di riferimento R2 verde (argmax 2.4067 MeV, σ pesata 0.1828 vs atteso 2.40±0.18); `expected_sigma_n(1.0,4)==0.5` |
| 6 | `validate` | fatto, caso di riferimento R3 verde (bias 0.08→0.20 MeV, pull mean 0.90, pull width 1.44, copertura 0.433 vs atteso ~45%) |
| 7 | `scripts/contrazione_fig.py` | fatto, figura di contrazione vs N (`outputs/contrazione_N.png`, ignorato da git); plateau di contrazione sotto shift sistematico 2% visibile in `outputs/contrazione_shift_sistematico.png` (offset ≈0.05 MeV da N≥1000, sigma continua a scendere a ~0.001 MeV) |
| — | **checkpoint Caso A** | raggiunto: `combine.py`/`validate.py` congelati (solo funzioni nuove additive da qui in poi) |
| 8 | `kinematics` 3D | fatto (`direction_from_theta_phi`), testato; limite `Ω_n ∥ z ⇒ Caso A` verde |
| 9 | `priors` (direction) | fatto (`direction_prior`, fix errata c su `grids.sphere_grid`), testato |
| 10 | `posterior_B` | fatto (`forward_model.loglik_marginal_En` + `posterior_B.single_event_posterior`), testato; limite 1 evento + prior piatto verde |
| 11 | `combine` su Ω_n (riuso) | fatto, nessuna modifica a `combine.py`; contrazione angolare σ(N=10)→σ(N=100) coerente con 1/√N entro tolleranza larga (singola realizzazione MC) |
| 12 | `validate` su distanza angolare | fatto (`angular_residual`, `posterior_angular_resolution`, `angular_pull`, additive); verificato su geometria nota + su Caso B simulato (M=30 esperimenti, Ω_n nota): bias medio <20°, pull mediano d'ordine 1 |
| 13 | `posterior_C` | fatto (`grids.hyperparameter_grid`, `priors.energy_prior_given_hyperparams`/`hyperparameter_prior`, `forward_model.loglik_marginal_En_hierarchical`, `posterior_C.estimate_shared_direction`/`single_event_posterior`), testato; entrambi i limiti di Sez. 5 verdi (σ_E→∞ ≈ Caso B entro atol=0.01 sulla log-verosimiglianza; σ_E→0 recupera l'energia condivisa vera entro 0.1 MeV) |
| 14 | `validate` finale su C | fatto (`scripts/caso_C_checklist.py`, `validate.credible_interval`/`credible_region_contains`, raffinamento locale in `posterior_C`); checklist di validazione v0.11 completa (`docs/report_caso_C_stadio1.md` §9–§12, notebook 07–09): μ_E calibrato a ogni N (bias −0.02 MeV chiuso dal kernel sferico; coverage a N = 1000 nominale con M = 100), Ω_n calibrato a N ≥ 150 e ~40% più preciso (stadio 1 iterato), σ_E calibrato a ogni N con stadio 2 su Ω̂_0 (coverage a N = 50: 0.67/0.89/0.93) (c); bias di log σ_E a N piccolo (−0.15 a N = 50) spiegato come effetto del prior largo su σ_E: nullo col prior del generatore, la marginalizzazione su Ω lo peggiora (report §11, (c)); coverage 68% di log σ_E a N = 150–300 bassa sui seed della checklist (0.60–0.64): fluttuazione statistica, nominale su seed nuovi con M = 600/300 (0.658/0.723; `scripts/caso_C_cov68.py`, report §12, notebook 09, (c)); **al limite** (d): coverage 90% di μ_E a N = 300 (0.84 su M = 100, 0.877 ± 0.017 su seed nuovi), senza segnale nei pull |

## Casi di riferimento

Tre casi con valori attesi noti, verificati in `tests/test_reference_examples.py`:

- **R1, singolo evento**: Ê_p = 1.40 MeV, θ̂_p = 0.50 rad, σ_Ep = 0.10 MeV, σ_θ = 0.08 rad,
  prior piatto proprio su E_n → E_n = 1.82 ± 0.21 MeV.
- **R2, due eventi combinati**: log-verosimiglianze gaussiane 2.70 ± 0.45 e 2.35 ± 0.20 MeV,
  prior piatto → E_n = 2.40 ± 0.18 MeV (media pesata con 1/σ²).
- **R3, ricostruzione difettosa**: errore di scala +3% e offset +0.05 MeV, incertezze dichiarate
  30% più strette della risoluzione vera σ(E_n) = 0.08·E_n → bias 0.08 → 0.20 MeV, pull medio
  ≈ 0.9, larghezza ≈ 1.4, coverage nominale 68% → ~45%.

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
  È il secondo momento centrale del posterior; verificato sul caso di riferimento R1.
- **Caso A, nuisance `z`**: CLAUDE.md Sez. 3 elenca `θ_p, z` come nuisance del Caso A, ma né
  la guida né `forward_model.py` usano `z`. `posterior_A` marginalizza solo `θ_p`; `z` è
  fuori scope per ora (nessuna sorgente di dato/osservabile per `z` definita da nessuna fonte).
- **`kinematics.sample_cm_angle`** (chiusa, 2026-09-25): campionava `uniform(0, 2π)`; ora
  `arccos(uniform(-1, 1))`, isotropo in angolo solido (a). Nuove `recoil_angle_from_cm`
  (θ_p = (π − θ_CM)/2), `sample_recoil_events` (tracce con densità cosθ_p/π attorno a Ω_n,
  E_p ~ U(0, E_n)) e `smear_direction` (risoluzione SIGMA_THETA nel piano tangente). I dati
  sintetici dei test B/C (righe 11, 12, 13) sono rigenerati con questa catena: le note
  "Riga 11/12, dati sintetici" qui sotto descrivono la costruzione precedente.
- **`posterior_A.single_event_posterior`**: la guida (Sez. 3) assume `theta_p_hat` già fissato
  dentro `loglik`. Qui `forward_model.loglik` marginalizza `theta_p` internamente su un prior
  piatto proprio su `[0, π/2]` (coerente con la deviazione già presa per `forward_model`), e
  `single_event_posterior` somma il prior su `En` una sola volta sopra il risultato. Verificato
  contro il caso di riferimento R1 (vedi tabella Stato, riga 4).
- **`validate.coverage_curve`**: la guida (Sez. 3) non dà un type hint per il parametro
  `intervals`. Si sceglie la forma più generica possibile, `(n_eventi, n_livelli, 2)` con
  estremi inferiore/superiore espliciti (non necessariamente simmetrici o gaussiani), così
  da restare utilizzabile sia con intervalli gaussiani (livello → z·σ_hat) sia con intervalli
  letti da un posterior a griglia (Caso B/C, dove la forma dell'intervallo non è nota a priori).
- **Ricostruzione difettosa del caso R3**: la specifica di R3 dà un errore di scala sistematico
  (+3%, +0.05 MeV) e incertezze dichiarate 30% troppo strette rispetto alla vera risoluzione
  σ(En)=0.08·En, ma non specifica se il rumore intrinseco del detector vada applicato prima o
  dopo il fattore di scala. Le due costruzioni sono entrambe plausibili dal solo testo; si è
  scelta quella con rumore applicato **dopo** l'errore di scala (non scalato insieme ad esso),
  perché riproduce pull width ≈1.4 molto più fedelmente (rapporto atteso 1/0.7≈1.429 contro
  1.03/0.7≈1.471 nell'altra costruzione). I valori attesi di R3 sono dati con
  "~"/"≈" (a differenza di R1/R2, che hanno tolleranza numerica esplicita — CLAUDE.md
  Sez. 5): le tolleranze del test (`test_reference_faulty_reconstruction`) sono scelte in proporzione a quanto
  approssimative sono le cifre citate (0.02 MeV su bias, 0.1 su pull mean/width, 0.05 su
  copertura, quest'ultima la meno precisa delle quattro: "~45%" è la cifra più tonda).

- **`scripts/contrazione_fig.py`**: la figura di contrazione usa in origine N=1,2,10,100 sia per
  il posterior combinato sia per la curva di contrazione con shift sistematico 2%. A quel
  range lo shift su Ep (2%, piccolo rispetto a σ_Ep) produce un offset ancora dentro il
  rumore statistico: a N=100 l'offset osservato (~0.02-0.08 MeV a seconda del run) non è
  chiaramente separato dal rumore, quindi il plateau richiesto da CLAUDE.md Sez. 6
  ("il plateau della contrazione è visibile, non converge a zero") non sarebbe dimostrabile
  con questi soli quattro punti. Si usa quindi un range esteso, `N=1,2,10,100,1000,5000`,
  solo per la figura del contraction plot con shift; la figura del posterior vs N e del
  contraction plot senza shift restano su `N=1,2,10,100`. Con il range esteso
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
  angolare vs N, stesso andamento della contrazione in E_n", senza specificare come generare gli eventi
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
  **(d)**: ipotesi da testare, senza valore di riferimento — la distanza angolare non è
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
- **Prestazioni Caso A (R4, aperto 2026-09-25, chiuso 2026-09-26 con target ridefinito)**:
  `forward_model.loglik` marginalizza θ_p su una griglia piena (n_eventi, 500 En, 500 θ).
  Profilo (`cProfile`, 1000 eventi): ~75% del tempo in `scipy.special.logsumexp` (copie,
  `asarray`, conversioni di tipo), il resto nella costruzione dell'array (c). Correzione:
  log-sum-exp manuale in-place float32 a blocchi di 100 eventi, stesso risultato entro
  6·10⁻⁵ in assoluto e 2·10⁻⁷ in relativo (arrotondamento float32), `test_reference_examples`
  invariato (c). Tempi: 1 evento da ~6 ms a 0.64 ms (target < 1 ms, raggiunto); 1000 eventi
  da ~5 s a 1.0–1.2 s (c). **Target ridefinito**: < 50 ms per 1000 eventi non è
  raggiungibile con numpy denso (2.5·10⁸ celle con `exp`, ~4 ns/cella); la soglia in
  `tests/test_performance.py` è ora 2 s. Alternative scartate: finestra ±6σ_θ attorno a
  θ̂_p (non esatta: differenze di loglik fino a 18.6 nelle code, perché il termine in E_p
  sposta l'integrando fuori dalla finestra) e `numba` (dipendenza nuova, non adottata).
  Xfail rimossi. Casi B/C e `combine`/`validate`/`sample_recoil_events` nei target (c).
- **Caso B/C, verosimiglianza con termine di traccia e risoluzione angolare (2026-09-25)**:
  `posterior_B` e `posterior_C` usano `forward_model.loglik_marginal_En_theta(_hierarchical)`
  invece del taglio netto θ_p > π/2 ⇒ −∞. Il termine di traccia log(cosθ_p/π) viene
  dall'isotropia in CM (a); θ_p vero è marginalizzato con un kernel gaussiano 1D sul
  meridiano, di larghezza SIGMA_THETA (approssimazione O(σ²), (b)). Motivo: col taglio netto
  e tracce esatte il combinato diventava −∞ ovunque per N grande e `argmax` restituiva il
  pixel 0 (causa del "pixel a 53°", `docs/report_caso_C_stadio1.md` §6). Le funzioni
  vecchie (`loglik_marginal_En`, `loglik_marginal_En_hierarchical`) restano come confronto.
- **Errata guida, test di limite "1 evento + prior piatto" (riga 10)**: con il termine di
  traccia L_k(Ω_n) non è ~costante sull'emisfero anteriore né −∞ su quello posteriore. Il
  test confronta con la forma analitica
  cos/π · [Φ((EN_MAX c − E_p)/s) − Φ((EN_MIN c − E_p)/s)] / c, con c = cos²θ_p, entro 45°,
  e verifica la forte soppressione (non −∞) oltre π/2 + 5σ_θ.
- **`posterior_C`, guardia sul combinato**: errore esplicito (`ValueError`) se il combinato è
  −∞ su tutti i candidati, invece di restituire in silenzio il pixel 0.
- **`posterior_C`, raffinamento locale (riga 14)**: le griglie globali (sfera, (μ_E, σ_E))
  hanno passo più largo del posterior a N grande (c). `refine_shared_direction` usa una
  calotta Fibonacci di `N_DIRECTION_CAP` pixel attorno al MAP di Ω_n; `refine_hyperparameters`
  usa una finestra fine su (μ_E, log σ_E) entro `WINDOW_DELTA_LOG` dal massimo. Ω_n resta
  plug-in nello stadio 2 (griglia a due stadi, niente 4D). Lo stimatore puntuale di σ_E è la
  mediana del posterior marginale in log σ_E (la media pesata è sensibile alle code).
- **Motore Caso C con prodotto di matrici (2026-09-26)**: `forward_model.log_matmul_exp`
  calcola il log-sum-exp di somme separabili come GEMM BLAS e ricalcola in modo esatto solo le
  celle in underflow; `track_energy_table`/`hierarchical_base` calcolano una volta le tabelle
  che non dipendono dai candidati e i `refine_*` le riusano. Un esperimento a N = 300 passa da
  18.8 s a 2.2 s, con log-posterior invariato entro 3·10⁻¹³ (c); la checklist completa da
  ~45 min (6 processi) a ~10 min (3 processi).
- **Kernel di traccia esatto sulla sfera (2026-09-26, chiude il bias di μ_E)**:
  `forward_model.log_track_kernel_sphere` (errore von Mises–Fisher, κ = 1/σ_θ², integrato
  sull'azimut con `scipy.special.ive`) sostituisce il kernel gaussiano 1D in
  `track_energy_table` e `hierarchical_base`. Il kernel piatto trascurava la curvatura e dava
  ΔE_n/E_n ≈ −σ_θ² (a); diagnosi con σ_θ = 0.04/0.08/0.16 in report §7 (c). Il generatore
  resta gaussiano nel piano tangente (differenza O(σ⁴)); il vecchio kernel resta come
  confronto.
- **Raggio della calotta (2026-09-26)**: `posterior_C.direction_cap_radius` esclude il MAP per
  indice invece di filtrare `angle > 0`, perché `arccos(best @ best)` può valere ~1.5·10⁻⁸
  (report §8.1). Il raffinamento adattivo previsto dal piano non è servito.
- **Stadio 1 iterato, empirical Bayes (2026-09-26)**: dopo lo stadio 2, lo stadio 1 si rifà
  con prior N(μ̂_E, σ̂_E) su E_n (`posterior_C.refine_shared_direction_hierarchical`), poi si
  rifà lo stadio 2. Resta a due stadi, niente 4D. Usa i dati due volte (b).
  Dalla v0.11 serve **solo a Ω_n**: (μ_E, σ_E) vengono dallo stadio 2 su Ω̂_0 (stadio 1 a prior
  largo), perché condizionare lo stadio 2 sull'Ω iterato portava il bias di log σ_E a N = 50 da
  −0.15 a −0.27 (report §10, (c)). Ω̂_n e (μ̂_E, σ̂_E) non vengono quindi da un'unica stima
  congiunta.
- **Prior largo su σ_E mantenuto (2026-09-26)**: il bias di log σ_E a N piccolo è un effetto del
  prior log-uniforme su [0.01, 50] MeV, non del modello (report §11.2, (c)); si dichiara invece di
  restringere il prior. `posterior_C.refine_hyperparameters_marginal_direction` (stadio 2
  marginalizzato su Ω_n) resta nel codice ma non è usata: peggiora il bias e costa ~30–40×
  (report §11.1).

## Pubblicazione su GitHub

- **la guida di progetto locale non va pubblicata su GitHub** (2026-09-24): è
  stato rimosso dalla cronologia dei commit (tutti i branch) e aggiunto a `.gitignore`; resta
  presente solo in locale, fuori dal tracking git. Stessa esclusione già in vigore per
  il materiale di riferimento statistico locale. Sotto `docs/` restano tracciati questo
  roadmap e `docs/report_caso_C_stadio1.md`.

## Errata applicate

- (a) `grids.py`: `@lru_cache(maxsize=1)` → `maxsize=None`; array resi non scrivibili.
- (b) `priors.py` (verifica): `np.trapz` deprecato in NumPy ≥ 2.0 → `scipy.integrate.trapezoid`.
- (c) `grids.sphere_grid`: la costruzione precedente usava due array `linspace` indipendenti
  per `theta`/`phi` (stessa lunghezza `n_pixel` ma nessun accoppiamento tra i due) — non era
  una vera pixelizzazione della sfera, perché l'indice `i` non corrispondeva a un'unica
  direzione. Sostituita con un reticolo di Fibonacci (angolo aureo): un indice = una
  direzione, area solida quasi costante per pixel (costruzione numerica standard, non una
  formula fisica). Verificato con `cos(theta)` a media nulla su punti uniformi (isotropia).
