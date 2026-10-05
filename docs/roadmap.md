# Roadmap — riptide-toy

v0.18 — 2026-10-05

Sintesi dei risultati: `docs/resoconto.md`.

## Stato

| Riga | Modulo | Stato |
|---|---|---|
| 1 | `constants` + `grids` | fatto (fix errata a applicato), testato |
| 2 | `kinematics` | fatto, testato |
| 3 | `forward_model` | fatto (marginalizza θ_p internamente, vedi Scelte di implementazione), testato |
| 4 | `posterior_A` (+ `priors.energy_prior`, `constants.SIGMA_EP/SIGMA_THETA`) | fatto, caso di riferimento R1 verde (argmax 1.80 MeV, σ pesata 0.219 MeV vs atteso 1.82±0.21) |
| 5 | `combine` | fatto, caso di riferimento R2 verde (argmax 2.4067 MeV, σ pesata 0.1828 vs atteso 2.40±0.18); `expected_sigma_n(1.0,4)==0.5` |
| 6 | `validate` | fatto, caso di riferimento R3 verde (bias 0.08→0.20 MeV, pull mean 0.90, pull width 1.44, copertura 0.433 vs atteso ~45%) |
| 7 | `scripts/contrazione_fig.py` | fatto, figura di contrazione vs N (`outputs/contrazione_N.png`, ignorato da git); plateau di contrazione sotto shift sistematico 2% visibile in `outputs/contrazione_shift_sistematico.png` (offset ≈0.05 MeV da N≥1000, sigma continua a scendere a ~0.001 MeV) |
| — | **checkpoint Caso A** | raggiunto: `combine.py`/`validate.py` congelati (solo funzioni nuove additive da qui in poi) |
| 8 | `kinematics` 3D | fatto (`direction_from_theta_phi`), testato; limite `Ω_n ∥ z ⇒ Caso A` verde |
| 9 | `priors` (direction) | fatto (`direction_prior`, fix errata c su `grids.sphere_grid`), testato |
| 10 | `posterior_B` | fatto (`forward_model.loglik_marginal_En` + `posterior_B.single_event_posterior`), testato; limite 1 evento + prior piatto verde |
| 11 | `combine` su Ω_n (riuso) | fatto, nessuna modifica a `combine.py`; contrazione angolare σ(N=10)→σ(N=100) coerente con 1/√N entro tolleranza larga (singola realizzazione MC) |
| 12 | `validate` su distanza angolare | fatto (`angular_residual`, `posterior_angular_resolution`, `angular_pull`, additive); verificato su geometria nota + su Caso B simulato con il generatore fisico (M=30 esperimenti, Ω_n nota); checklist quantitativa del Caso B (`scripts/caso_B_checklist.py`, M = 400…40, N = 10…1000): pull rms 0.99–1.03 e coverage nominale fino a N = 300 (c) |
| 13 | `posterior_C` | fatto (`grids.hyperparameter_grid`, `priors.energy_prior_given_hyperparams`/`hyperparameter_prior`, `forward_model.loglik_marginal_En_hierarchical`, `posterior_C.estimate_shared_direction`/`single_event_posterior`), testato; entrambi i test di limite verdi (σ_E→∞ ≈ Caso B entro atol=0.01 sulla log-verosimiglianza; σ_E→0 recupera l'energia condivisa vera entro 0.1 MeV) |
| 14 | `validate` finale su C | fatto (`scripts/caso_C_checklist.py`, `validate.credible_interval`/`credible_region_contains`, raffinamento locale in `posterior_C`); checklist di validazione v0.11 completa (`docs/report_caso_C_stadio1.md` §9–§12): μ_E calibrato a ogni N (bias −0.02 MeV chiuso dal kernel sferico; coverage a N = 1000 nominale con M = 100), Ω_n calibrato a N ≥ 150 e ~40% più preciso (stadio 1 iterato), σ_E calibrato a ogni N con stadio 2 su Ω̂_0 (coverage a N = 50: 0.67/0.89/0.93) (c); bias di log σ_E a N piccolo (−0.15 a N = 50) spiegato come effetto del prior largo su σ_E: nullo col prior del generatore, la marginalizzazione su Ω lo peggiora (report §11, (c)); coverage 68% di log σ_E a N = 150–300 bassa sui seed della checklist (0.60–0.64): fluttuazione statistica, nominale su seed nuovi con M = 600/300 (0.658/0.723; `scripts/caso_C_cov68.py`, report §12, (c)); coverage 90% di μ_E a N = 300: fluttuazione, 0.890 ± 0.010 su 900 seed nuovi (`scripts/caso_C_cov90_mu.py`, report §13.1, (c)); Ω_n a N = 50 leggermente sovra-confidente col prior plug-in dello stadio 1′ (0.884/0.938 al 90/95%, M = 1400; l'oracolo è nominale): sostituito dal prior predittivo `posterior_C.predictive_energy_moments` (0.899/0.945, M = 800), checklist v0.14 rilanciata, stadio 2 invariato (report §13.2, (c)); run deterministici a `OMP_NUM_THREADS=1` (§13.3); test di regressione `test_stage1_error_contracts_at_large_N` (§13.4); diagnostiche in `scripts/caso_C_diag_omega_N50.py` e `scripts/caso_C_diag_stadio1_test.py`, riepilogo nel notebook `05_validazione_caso_C` (le versioni intermedie sono in `notebooks/storico/`). Nessun punto aperto sul Caso C |

## Casi di riferimento

Tre casi con valori attesi noti, verificati in `tests/test_reference_examples.py`:

- **R1, singolo evento**: Ê_p = 1.40 MeV, θ̂_p = 0.50 rad, σ_Ep = 0.10 MeV, σ_θ = 0.08 rad,
  prior piatto proprio su E_n → E_n = 1.82 ± 0.21 MeV.
- **R2, due eventi combinati**: log-verosimiglianze gaussiane 2.70 ± 0.45 e 2.35 ± 0.20 MeV,
  prior piatto → E_n = 2.40 ± 0.18 MeV (media pesata con 1/σ²).
- **R3, ricostruzione difettosa**: errore di scala +3% e offset +0.05 MeV, incertezze dichiarate
  30% più strette della risoluzione vera σ(E_n) = 0.08·E_n → bias 0.08 → 0.20 MeV, pull medio
  ≈ 0.9, larghezza ≈ 1.4, coverage nominale 68% → ~45%.

## Scelte di implementazione

Scelte non ovvie, con la motivazione. "Specifica" indica la specifica di codice locale da cui il
progetto è partito (non pubblicata).

- **`forward_model.loglik`**: la specifica prevedeva una firma più semplice, senza
  marginalizzazione di θ_p. Il codice esistente marginalizza già θ_p internamente
  (`sigma_Ep, sigma_theta, log_prior_theta, chunk_size`, a chunk per stare in RAM).
  Decisione: si mantiene questa versione (ottimizzata) invece di riscriverla; `posterior_A`
  si adatta alla firma reale. Motivo: priorità a prestazioni e tempo di sviluppo.
- **`sample_cm_angle` / `sample_recoil`**: in origine lette come la stessa funzione. Dal
  2026-09-25 l'evento intero è campionato da `sample_recoil_events`, ed è questa la funzione del
  target "10^5 eventi < 0.1 s" in `tests/test_performance.py`. Nessun tempo
  di volo nello scope del toy.
- **`risoluzione_caso_A`**: non definita nella specifica. Si usa la deviazione standard pesata sulla
  griglia del posterior: `σ² = Σ p(En)(En−⟨En⟩)² / Σp(En)` con `p(En) = exp(log_post − max(log_post))`.
  È il secondo momento centrale del posterior; verificato sul caso di riferimento R1.
- **Caso A, nuisance `z`**: la tabella dei casi elenca `θ_p, z` come nuisance del Caso A, ma né
  la specifica né `forward_model.py` usano `z`. `posterior_A` marginalizza solo `θ_p`; `z` è
  fuori scope per ora (nessuna sorgente di dato/osservabile per `z` definita da nessuna fonte).
- **`kinematics.sample_cm_angle`** (chiusa, 2026-09-25): campionava `uniform(0, 2π)`; ora
  `arccos(uniform(-1, 1))`, isotropo in angolo solido (a). Nuove `recoil_angle_from_cm`
  (θ_p = (π − θ_CM)/2), `sample_recoil_events` (tracce con densità cosθ_p/π attorno a Ω_n,
  E_p ~ U(0, E_n)) e `smear_direction` (risoluzione SIGMA_THETA nel piano tangente). I dati
  sintetici dei test B/C (righe 11, 12, 13) sono rigenerati con questa catena: le note
  "Riga 11/12, dati sintetici" qui sotto descrivono la costruzione precedente.
- **`posterior_A.single_event_posterior`**: la specifica assume `theta_p_hat` già fissato
  dentro `loglik`. Qui `forward_model.loglik` marginalizza `theta_p` internamente su un prior
  piatto proprio su `[0, π/2]` (coerente con la deviazione già presa per `forward_model`), e
  `single_event_posterior` somma il prior su `En` una sola volta sopra il risultato. Verificato
  contro il caso di riferimento R1 (vedi tabella Stato, riga 4).
- **`validate.coverage_curve`**: la specifica non dà un type hint per il parametro
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
  "~"/"≈" (a differenza di R1/R2, che hanno tolleranza numerica esplicita): le tolleranze del test (`test_reference_faulty_reconstruction`) sono scelte in proporzione a quanto
  approssimative sono le cifre citate (0.02 MeV su bias, 0.1 su pull mean/width, 0.05 su
  copertura, quest'ultima la meno precisa delle quattro: "~45%" è la cifra più tonda).

- **`scripts/contrazione_fig.py`**: la figura di contrazione usa in origine N=1,2,10,100 sia per
  il posterior combinato sia per la curva di contrazione con shift sistematico 2%. A quel
  range lo shift su Ep (2%, piccolo rispetto a σ_Ep) produce un offset ancora dentro il
  rumore statistico: a N=100 l'offset osservato (~0.02-0.08 MeV a seconda del run) non è
  chiaramente separato dal rumore, quindi il plateau richiesto
  ("il plateau della contrazione è visibile, non converge a zero") non sarebbe dimostrabile
  con questi soli quattro punti. Si usa quindi un range esteso, `N=1,2,10,100,1000,5000`,
  solo per la figura del contraction plot con shift; la figura del posterior vs N e del
  contraction plot senza shift restano su `N=1,2,10,100`. Con il range esteso
  l'offset si stabilizza a ≈0.05 MeV (coerente con la stima analitica 0.02·En_true=0.05 MeV)
  mentre σ continua a scendere fino a ~0.001 MeV: il plateau è così chiaramente visibile.

- **`posterior_B` / `forward_model.loglik_marginal_En`**: la specifica descrive
  il Caso B come θ_p derivato geometricamente dalla traccia 3D osservata e dal candidato Ω_n
  (`kinematics.recoil_angle_from_direction`), non come nuisance da marginalizzare su una
  griglia separata — a differenza del Caso A, dove θ_p è genuinamente ignoto. Questo angolo è
  invariante all'azimut della traccia attorno al candidato (per costruzione), quindi basta
  un'unica marginalizzazione su `En_grid`, con broadcasting a due assi (candidati × `En_grid`,
  per le prestazioni del Caso B), invece di una griglia a tre assi con
  una seconda marginalizzazione su θ_p. I candidati con θ_p > π/2 (emisfero posteriore) sono
  esclusi (log-verosimiglianza −∞): il rinculo è sempre in avanti per scattering elastico a
  masse uguali (`θ_lab ≤ 90°`). Nota sul test di limite: con prior piatto su
  `En`, l'integrale marginale su `En` varia con θ_p come `1/cos²θ_p` (Jacobiano del cambio di
  variabile `En' = En·cos²θ_p`) — non è esattamente costante, solo approssimativamente entro
  un cono centrale attorno alla direzione osservata; il test verifica questo comportamento
  qualitativo (nessun picco netto sull'emisfero anteriore, in contrasto con il taglio netto a
  −∞ sull'emisfero posteriore), non una costanza esatta.

- **Riga 11, dati sintetici per il test di contrazione angolare**: la specifica chiede "Contrazione
  angolare vs N, stesso andamento della contrazione in E_n", senza specificare come generare gli eventi
  sintetici. Si è scelto di pescare `theta_p_true` e `phi_true` (angolo e azimut del rinculo
  rispetto a `Omega_n_true`, fissato sull'asse z senza perdita di generalità, come nel test
  della riga 8) direttamente da distribuzioni uniformi indipendenti, invece di simulare l'intera
  catena fisica a partire da un angolo di scattering in CM — perché quest'ultima richiedeva
  la formula di isotropia di `kinematics.sample_cm_angle`, allora aperta (chiusa il 2026-09-25,
  vedi sopra; i dati sono stati poi rigenerati con la catena fisica). Etichetta
  **(b)**: dato sintetico "onesto" con assunzioni esplicite, analogo alla generazione diretta
  già usata nei test di `validate.py` per il Caso A (verità pescata direttamente, non simulata
  da una catena fisica completa). Il criterio di accettazione verificato è qualitativo (la
  deviazione standard angolare pesata sul posterior diminuisce da N=10 a N=100 in modo
  compatibile con 1/√N entro una tolleranza larga, dato che si tratta di una singola
  realizzazione Monte Carlo e non di una media d'insieme), coerente con quanto richiesto per
  questa riga.

- **Riga 12, `validate.py` su distanza angolare**: la riga 12 chiede
  "bias/pull su distanza angolare", senza dare firme né un oracolo numerico (etichetta
  **(d)**: ipotesi da testare, senza valore di riferimento — la distanza angolare non è
  gaussiana, quindi non ci si aspetta un pull esattamente N(0,1) come nel caso 1D, solo
  scala ~1 se il ricostruttore è calibrato). Tre funzioni additive in `validate.py`
  (`combine.py`/`validate.py` restano congelati per le funzioni esistenti dopo il
  checkpoint del Caso A): `angular_residual` (distanza angolare fra due versori), `posterior_angular_resolution`
  (analogo sferico della deviazione standard pesata già usata per `risoluzione_caso_A`,
  qui attorno a una direzione di riferimento per evento — verità nota, o la stima stessa),
  `angular_pull` (rapporto puro, senza assunzioni di normalità). Test di integrazione:
  M=30 esperimenti indipendenti, N=20 eventi sintetici ciascuno (stessa costruzione
  "onesta" della riga 11), stima di Ω_n via MAP del posterior combinato (`combine.combine_loglik`
  riusato, nessuna modifica), `sigma_hat` calcolata in modo self-referenziale (attorno
  alla propria stima MAP, non alla verità — l'unica scelta sensata quando si vuole
  un'incertezza dichiarata dal posterior senza conoscere la verità in un caso reale). Nota: la costruzione dei dati descritta qui è quella originale; dal
  2026-09-25 il test usa `simulate_case_B_events` (generatore fisico con risoluzioni).
  Griglie (`sphere_grid(800)`, `energy_grid(100)`) e M scelti per tenere il test sotto i
  2s: confrontati con una configurazione più grande (`sphere_grid(1500)`, `energy_grid(150)`,
  M=200, 18.9s) le statistiche del pull (media/mediana/std) restano dello stesso ordine,
  confermando che la riduzione non altera la conclusione qualitativa. Soglie di
  accettazione volutamente larghe (bias medio <20°, pull mediano in [0.2, 3.0]), coerenti
  con l'etichetta (d).

- **Riga 13, `posterior_C` importa anche `combine`**: la tabella degli import elenca
  per `posterior_C` solo `kinematics, forward_model, priors, grids, posterior_B`. Lo stadio 1
  (`estimate_shared_direction`) deve combinare gli N eventi su `Ω_n` prima di passare allo
  stadio 2 sulla griglia `(μ_E, σ_E)` — la griglia a due stadi (niente griglia 4D bruta:
  costerebbe n_pixel·n_μ·n_σ·n_E) richiede proprio questo. `combine.py` non ha import dal progetto ed
  è pensato per essere generico, chiamato da qualunque layer superiore (già usato direttamente
  nei test del Caso B, righe 11/12): l'import diretto in `posterior_C` è quindi una deviazione
  minima e coerente con l'architettura esistente, non una violazione dello strato.
- **Riga 13, doppio conteggio del prior nello stadio 1**: `posterior_B.single_event_posterior`
  somma già il prior sulla direzione una volta per evento. Per riusarlo con
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
- **Errata della specifica, test di limite "1 evento + prior piatto" (riga 10)**: con il termine di
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

- **Stress test della v0.17** (2026-10-03): solo funzioni nuove, nessuna modifica alle esistenti.
  Generatori: `kinematics.mixed_source_axes` (asse per evento con seconda sorgente o fondo
  isotropo), `kinematics.sample_energy_spectrum` (spettri con stessa media e sd della gaussiana),
  `kinematics.direction_from_tangent`. Prior alternativi, tutti propri: `priors.energy_prior_log_uniform`,
  `priors.direction_prior_von_mises_fisher`, `priors.hyperparameter_prior_uniform_sigma`, più
  `posterior_C.refine_direction_from_table_with_prior` per applicare un prior di direzione anche
  alla calotta fine. Diagnostica: `forward_model.track_angle_cdf`,
  `validate.probability_integral_transform`, `validate.ks_uniform_statistic`. Stima congiunta del
  Caso C: `posterior_C.joint_log_posterior_local`, `stencil_gradient_hessian` e `joint_laplace`
  (stencil 3^4 di differenze centrali in (a, b, μ_E, log σ_E), con a, b coordinate nel piano
  tangente a Ω_n e passi pari alle larghezze dei due stadi; `N_LAPLACE_NEWTON` = 2 passi di
  Newton; nessuna griglia 4D), più `validate.conditional_covariance` e
  `validate.canonical_correlations`. Costanti nuove: `EN_MAX_WIDE`, `BIMODAL_HALF_SEPARATION`,
  `N_LAPLACE_NEWTON`.

- **Verifiche della v0.18** (2026-10-05): quattro ipotesi chiuse con una previsione scritta prima del
  calcolo (punti 4, 5, 6, 8 sotto). Codice nuovo solo additivo: forme `"laplace"` e `"student_t"`
  in `kinematics.sample_energy_spectrum`, costante `STUDENT_T_DOF`. Script di analisi:
  `prior_B_rumore.py`, `mu_N1000_diag.py`, `curtosi_log_sigma.py`, `asimmetria_log_sigma.py`,
  `modo_log_sigma.py`.

## Punti aperti (v0.18, 2026-10-05)

Nessun punto aperto sul codice. I punti 1–3 erano chiusi nella v0.16; i punti 4–8 chiudono o
quantificano i limiti dichiarati della v0.16. Dettagli e tabelle in `docs/resoconto.md` §4–§6.

1. **Calibrazione del Caso A: chiuso (c).** `scripts/caso_A_checklist.py`, 20 000 eventi. Con la
   verità estratta dal prior (E_n ~ U(0.5, 6), θ_p ~ U(0, π/2)): pull +0.003 / 0.998, coverage
   0.683 / 0.899 / 0.949, cioè nominale. Con il generatore U(1, 5) del notebook 02: pull +0.115 / 0.927,
   coverage 0.735 / 0.925 / 0.964. La sovra-copertura viene quindi dal generatore più stretto del prior,
   non dal posterior. Il bias per bin di E_n (+0.51 … −0.69 MeV) mostra l'attrazione verso il centro
   del prior. Test di regressione in `tests/test_case_A.py`.
2. **Checklist quantitativa del Caso B: chiuso (c).** `scripts/caso_B_checklist.py`; E_n per evento
   distribuita come il suo prior, Ω_n isotropa, N = 10/30/100/300/1000 con M = 400/300/200/100/40.
   Pull rms 1.01 / 0.99 / 1.00 / 1.03 / 0.88; coverage al 68% 0.66 / 0.68 / 0.69 / 0.66 / 0.78
   (±0.07 a N = 1000); pendenza di contrazione −0.59. Tabella in `docs/resoconto.md` §4.
3. **Test del Caso B con Ω_n vera sul polo: chiuso il 2026-09-26 (c).**
4. **Statistica a N = 1000: chiuso (c).** `scripts/n1000_extra.py`: 200 esperimenti in più per caso
   su seed disgiunti (M = 240, ±0.03 sulla coverage 68%). Caso B: pull rms 0.96, coverage
   0.70 / 0.91 / 0.95. La pendenza di contrazione è −0.53 su 30–1000 e −0.57 su 10–1000 (la −0.59
   veniva da M = 40). A N = 10 la distribuzione del pull ha code pesanti (frazione > 3: 0.023 contro
   0.011 di Rayleigh), quindi il regime non è gaussiano. Caso C: calibrato. μ_E è appena
   sovra-coperto al 90% (0.95 ± 0.019); causa non indagata.
5. **Robustezza al prior: chiuso (c).** `scripts/robustezza_prior.py`. I prior sui parametri
   condivisi vengono dimenticati con N. Il prior su E_n per evento del Caso B no: Ω̂_n si sposta di
   ~1 σ a ogni N con un prior largo, ~0.3 σ con uno log-uniforme. La coverage resta nominale.
6. **Spettro non gaussiano, Caso C: chiuso (c).** `scripts/stress_spettro.py`. μ_E e Ω_n sono
   robusti. log σ_E è sovra-coperto per spettri a curtosi bassa (uniforme, bimodale): coverage 68%
   0.75–0.81 a N = 150–300. Con code pesanti (`"laplace"`, `"student_t"` in
   `kinematics.sample_energy_spectrum`, run `stress_spettro.py ... code`) è sotto-coperto: coverage
   68% 0.56–0.65, larghezza del pull 1.17–1.39 a N = 150–300. La previsione scritta prima,
   var(log s) ≈ (κ − 1)/(4N) aggiunta alla larghezza gaussiana (`scripts/curtosi_log_sigma.py`),
   regge fino a N = 300 (c); a N = 1000 l'osservato è più largo di ~2.3σ (M = 40, d). Con la
   Student-t μ_E ha pull medio +0.16 ± 0.04 (c, causa non indagata).
7. **Sorgente unica, Casi B e C: quantificato (c).** `scripts/stress_sorgente.py`. Ω̂_n si sposta
   verso la seconda sorgente di circa f·Δ. La coverage crolla con N: per f = 0.05 a 30° la coverage
   68% scende da 0.41 a N = 30 a 0.16 a N = 300. La diagnostica KS ha potenza al livello del falso
   allarme per Δ ≲ 30°. Resta un limite (sotto).
8. **Due stadi contro stima congiunta, Caso C: chiuso (c).** `scripts/correlazione_stadi.py`, con
   Laplace locale 4D (`posterior_C.joint_laplace`). La correlazione canonica fra Ω_n e (μ_E, log σ_E) ha mediana
   0.17 / 0.10 / 0.08 a N = 50 / 150 / 300. Marginalizzare invece di fissare allarga le incertezze in
   media dell'1.3% / 0.4% / 0.3% (massimo 6%), quindi la fattorizzazione è giustificata. Il modo
   congiunto di log σ_E è meno centrato della media dei due stadi (pull −0.27 a N = 50). Non è
   l'asimmetria del marginale: Pearson prevede il segno opposto (`scripts/asimmetria_log_sigma.py`).
   `scripts/modo_log_sigma.py` scompone lo scarto. Asimmetria (+) e volume di μ_E (−) quasi si
   cancellano; domina il termine dovuto a Ω_n lasciata libera nel massimo congiunto (−0.28 / −0.20
   / −0.14 s), con scala ~s², come un effetto di volume (c; interpretazione d).

## Limiti dichiarati (da riportare nell'articolo)

- **Assunzione 1 necessaria e non verificabile per sorgenti vicine (c).** Basta una contaminazione
  di pochi per cento a Δ ≲ 30° per perdere la calibrazione di Ω_n e σ_E a N grande, senza che la
  diagnostica KS se ne accorga. Un modello a mistura che stimi f è sviluppo futuro.
- **Assunzione 3:** la coverage di log σ_E dipende dalla curtosi dello spettro (c, κ da 1.7 a 9):
  sovra-coperto per spettri piatti, sotto-coperto per code pesanti (coverage 68% ~0.6). μ_E e Ω_n
  restano calibrati; con la Student-t μ_E ha un bias di +0.16 σ (c).
- **Prior su E_n nel Caso B** da scegliere sullo spettro fisico atteso: non si dimentica con N (c).
  Nel Caso C log σ_E dipende dal prior (0.3 σ a N = 50, < 0.1 σ da N ≈ 300, c).
- **Nuisance `z` del Caso A** fuori scope (nessun osservabile); **isotropia in CM** assunta nel
  range 0.5–6 MeV, da verificare oltre ~10 MeV (d); **target `loglik`** per N = 1000 ridefinito a 2 s.

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
