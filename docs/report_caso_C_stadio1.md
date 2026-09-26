# Caso C — resoconto diagnostico dello stadio 1 (direzione) e del profilo in σ_E

Etichette come da convenzione del progetto: **(a)** fatto consolidato · **(b)** stima con assunzioni esplicite · **(c)** verificato via MC/numerico · **(d)** ipotesi da testare.

## 1. Sintesi

> **Aggiornamento 2026-09-25:** la causa vera è un'altra (combinato −∞ ⇒ pixel 0) e l'ipotesi del §4 è smentita; correzioni e checklist della riga 14 sono in §6.

- Il motore dello stadio 2 (`forward_model.loglik_marginal_En_hierarchical`, `priors.energy_prior_given_hyperparams`) è corretto: con la direzione vera il profilo di verosimiglianza in σ_E ha un massimo interno vicino al valore vero **(c)**.
- Lo stadio 1 (`posterior_C.estimate_shared_direction`, che riusa il Caso B con prior piatto su E_n) produce un massimo a posteriori **sistematicamente sbagliato** quando le energie vere sono concentrate; l'errore non si riduce con N, anzi la frequenza dei fallimenti cresce **(c)**.
- L'errore di direzione si propaga allo stadio 2 e fa apparire σ_E sempre più grande al crescere di N (fino al limite superiore della griglia) **(c)**.
- La causa proposta (fattore 1/cos²θ_p del prior piatto in E_n, con modello mal specificato) è coerente con i dati ma **non ancora verificata (d)**.
- Il checklist Cap. 40 sulla riga 14 non può essere formalizzato in modo onesto per σ_E finché lo stadio 1 non è risolto. Per μ_E i risultati preliminari sono sani **(c)**, ma condizionati a N piccolo.

## 2. Setup dei diagnostici

Dataset sintetici di Caso C: Ω_n vera a (θ, φ) = (0.9, 2.1) rad; μ_E = 3.0 MeV, σ_E = 0.4 MeV (0.3 nel test di integrazione esistente); tracce isotrope; rumore su E_p con σ_Ep = SIGMA_EP. Sono tenuti solo gli eventi con θ_p ≤ π/2.

## 3. Risultati

### 3.1 Filtro cinematico basato sulla direzione stimata (c)

Gli eventi vanno ri-filtrati con `θ_p_hat = recoil_angle_from_direction(track_hat, Ω̂_n) ≤ π/2`, cioè con la direzione **stimata**, non con quella vera (ignota in un'analisi reale). Un evento cinematicamente vietato sotto la stima riceve log-verosimiglianza −∞ per ogni candidato (μ_E, σ_E); la somma su eventi porta l'intera riga combinata a −∞ e `posterior_mean_std` restituisce NaN. Il comportamento di `posterior_C` (−∞ per candidati vietati) è coerente con la convenzione già usata nel Caso B; il filtro va applicato dal chiamante.

### 3.2 Checklist preliminare, 25 esperimenti × N=50 (c)

| Grandezza | Risultato |
|---|---|
| μ_E: bias medio | 0.044 MeV |
| μ_E: σ̂ medio | 0.297 MeV |
| μ_E: pull media / dev. std. | −0.041 / 0.879 |
| σ_E: bias medio | 1.148 MeV |
| σ_E: σ̂ medio | 2.118 MeV |
| σ_E: pull media / dev. std. | 0.115 / 0.553 |
| Errore angolare, mediana | 1.04° |

Il bias di σ_E è dominato da poche realizzazioni con stima molto gonfiata (fino a 17.6 MeV in un esperimento con 14 eventi tenuti); la maggior parte è nell'intervallo 0.29–0.56 MeV. Un esperimento singolo ben condotto (N=50) dà MAP (μ, σ) = (2.756, 0.329) e media pesata di σ_E 0.345, compatibili col valore vero.

### 3.3 Il bias di σ_E peggiora con N (c)

Media della stima di σ_E su 20 esperimenti:

| N eventi generati | media σ̂_E (MeV) |
|---|---|
| 50 | 3.22 |
| 150 | 10.30 |
| 300 | 26.49 |

A N=300 anche la mediana raggiunge 27.8 MeV, cioè si avvicina a `SIGMA_E_MAX = 50`. Un estimatore consistente dovrebbe concentrarsi vicino a 0.4 all'aumentare di N; qui accade l'opposto.

### 3.4 Profilo di verosimiglianza in σ_E a μ_E = 3.0 fisso (c)

Somma su eventi di `loglik_marginal_En_hierarchical`, N=300 generati.

- **Direzione stimata dallo stadio 1** (errore angolare 53.05°, 97 eventi tenuti): log-verosimiglianza **monotona crescente** in σ_E, da −8030 a σ=0.05 fino a −2530 a σ=50, senza massimo interno.
- **Direzione vera** (136 eventi tenuti): massimo interno a σ ≈ 0.34–0.43 (487.7 a σ=0.336, 486.5 a σ=0.427) e discesa lenta fino a 405.9 a σ=50. Il picco cade sul valore vero σ_E = 0.4.

Con una direzione errata, θ_p_hat è quasi scorrelato da θ_p vero; il pattern E_p ∝ cos²θ_p svanisce, i dati assomigliano a rumore e il modello più flessibile (σ_E grande, limite Caso B) vince: è il comportamento bayesiano atteso, non un difetto dello stadio 2.

### 3.5 Errore di direzione dello stadio 1 (c)

- L'errore non dipende dalla risoluzione della griglia: 53.05° con 800 pixel, 52.32° con 3000 pixel.
- Ripetizioni MC (8 per ogni N, σ_E vero 0.4), errore angolare in gradi:
  - N=60: 4.8, 4.8, 4.8, 2.9, **53.1**, 9.7, 13.1, **53.1**
  - N=150: **53.1**, **53.1**, 2.9, **53.1**, **53.1**, 4.8, 2.9, **53.1**
  - N=300: **53.1** in tutti e 8
- Lo stesso pixel sbagliato (53.1°) vince sempre più spesso al crescere di N: è un massimo sistematico, non rumore statistico.
- Il test `test_posterior_C_end_to_end_direction_and_hyperparams` (N=60, σ_E=0.3, soglia 15°) passa, ma in questo regime è vicino al bordo della frequenza di fallimento osservata: la sua robustezza è limitata.

### 3.6 Memoria (nota operativa)

`posterior_B.single_event_posterior` alloca un array `(n_eventi, n_pixel, n_En=500)` in float64. Con ~136 eventi e 6000 pixel sono ~3 GB per array, più temporanei: si esaurisce la RAM (processo terminato). Per diagnostici oltre ~3000 pixel, processare a blocchi di eventi (10 alla volta è sufficiente).

## 4. Ipotesi sulla causa (d)

Con prior piatto su E_n, la verosimiglianza marginale in E_p è ∫ π(E_n) N(E_p; E_n cos²θ_p, σ_Ep) dE_n ≈ π(E_p/cos²θ_p) / cos²θ_p. Il fattore 1/cos²θ_p premia le direzioni che rendono θ_p grande per molti eventi. Se le energie vere sono strette attorno a μ_E e il prior è largo, il modello è mal specificato e il massimo si sposta in modo sistematico, con un bias che non decade con N.

Predizioni verificabili di questa ipotesi:
1. Il pixel a 53° dovrebbe avere θ_p mediamente più grande del valore vero sugli eventi tenuti.
2. Con `energy_prior` stretto attorno a μ_E vero il fallimento dovrebbe sparire.
3. Con E_n vero distribuito in modo piatto sul dominio del prior (modello ben specificato), l'errore dovrebbe contrarsi come 1/√N.

## 5. Possibili passi futuri

1. **Verificare l'ipotesi del §4** con i tre controlli elencati, prima di modificare codice.
2. **Stadio 1 iterativo** (opzione preferita): Caso B → stima (μ_E, σ_E) → direzione ricalcolata con prior gerarchico N(μ_E, σ_E) → iterare fino a convergenza. Resta una griglia a due stadi, senza griglia 4D. Da valutare: convergenza, dipendenza dall'inizializzazione, costo.
3. **Prior stretto nello stadio 1** con larghezza fissata: più semplice, ma introduce una scelta arbitraria; da testare la sensibilità al valore scelto.
4. **Stima congiunta** (Ω_n, μ_E, σ_E) con ottimizzazione locale (MLE + Laplace, come da ordine previsto per il Caso C) partendo dal MAP del Caso B: evita la griglia 4D e permette di controllare i massimi multipli.
5. **Marginalizzare Ω_n invece di fissarlo** nello stadio 2 (somma pesata sul posterior di Ω_n, su un sottoinsieme di pixel): propagherebbe l'incertezza di direzione a σ_E ed eviterebbe il collasso a una stima puntuale sbagliata.
6. **Riformulare il fattore di volume**: verificare se una parametrizzazione in cui il prior è piatto in una variabile per cui il Jacobiano non favorisce grandi θ_p (o un prior su E_n non uniforme ma proprio) rimuove il bias senza informazione aggiuntiva.
7. **Rendere robusto il test di integrazione** esistente: aumentare le ripetizioni MC o dichiararne esplicitamente il regime di validità, e aggiungere un test di regressione che fallisce se lo stadio 1 mostra il massimo sistematico a N alto.
8. **Completare la riga 14** una volta risolto lo stadio 1: bias, risoluzione, pull, coverage, contrazione ≈1/√N e robustezza al prior per (Ω_n, μ_E, σ_E), con stimatori puntuali scelti in modo coerente per σ_E (la media pesata è sensibile alle code; valutare il MAP marginale o la mediana in log σ).
9. **Prestazioni**: ridurre la memoria di `posterior_B.single_event_posterior` con elaborazione a blocchi di eventi, poi profilare (riga prestazioni ancora aperta).
10. **Aggiornare `docs/roadmap.md`** con le deviazioni e l'errata relative allo stadio 1 quando la soluzione è scelta.

## 6. Esito (2026-09-25)

### 6.1 Causa vera: combinato −∞ e pixel 0 (c)

Le predizioni 2 e 3 del §4 **falliscono**: con un prior stretto su E_n e con un modello ben specificato (E_n vero piatto) lo stadio 1 finisce sullo stesso pixel sbagliato. L'ipotesi del §4 è quindi **smentita (c)**.

La causa è il taglio netto `θ_p > π/2 ⇒ −∞` combinato con tracce generate senza risoluzione angolare. Ogni evento ammette solo i pixel del proprio emisfero anteriore, per cui l'insieme ammesso dopo N eventi è l'intersezione di N emisferi. Questo insieme si restringe con N: con 3000 pixel a N=300 restano {4,1,1,0,1,0,0,1} pixel e a N=1000 non ne resta nessuno. Quando il combinato è −∞ ovunque, `np.argmax` restituisce il **pixel 0**, cioè il polo nord della griglia di Fibonacci. Il polo dista 53.05° dalla verità con 800 pixel e 52.32° con 3000, esattamente i valori del §3.5. Il σ_E "gonfiato" del §3.3 ne è la conseguenza diretta (§3.4).

### 6.2 Correzioni

1. **Guardia**: `posterior_C` solleva un errore esplicito se il combinato è −∞ su tutti i candidati, invece di restituire il pixel 0.
2. **Generatore fisico**: `kinematics.sample_cm_angle` è isotropo in angolo solido; `sample_recoil_events` genera tracce con densità cosθ_p/π e `smear_direction` aggiunge la risoluzione angolare SIGMA_THETA.
3. **Verosimiglianza con termine di traccia e risoluzione angolare**: `forward_model.loglik_marginal_En_theta` e la sua variante gerarchica marginalizzano θ_p vero con un kernel gaussiano 1D polare, per cui il taglio diventa morbido **(b)**. Sono usate in `posterior_B` e `posterior_C`.
4. **Raffinamento locale**: `refine_shared_direction` (calotta Fibonacci attorno al MAP di Ω_n) e `refine_hyperparameters` (finestra fine su (μ_E, log σ_E)). Le griglie globali erano più larghe del posterior.

### 6.3 Checklist Cap. 40 dopo le correzioni (riga 14) (c)

Script `scripts/caso_C_checklist.py`, con M esperimenti per N, μ_E ~ U(2.5, 4), σ_E log-U(0.2, 0.6) e Ω_n isotropa. Bias ± errore standard sulla media.

| N | M | bias μ_E (MeV) | bias log σ_E | errore Ω medio |
|---|---|---|---|---|
| 50 | 200 | −0.0182 ± 0.0056 | −0.143 ± 0.032 | 2.66° |
| 150 | 200 | −0.0201 ± 0.0036 | −0.050 ± 0.018 | 1.52° |
| 300 | 100 | −0.0212 ± 0.0039 | −0.015 ± 0.016 | 1.09° |
| 1000 | 40 | −0.0191 ± 0.0031 | +0.010 ± 0.008 | 0.58° |

Coverage ai livelli 68/90/95%:

| N | μ_E | log σ_E | Ω (HPD) |
|---|---|---|---|
| 50 | .69/.90/.96 | .67/.88/.94 | .85/.96/1.0 |
| 150 | .66/.88/.94 | .61/.87/.94 | .87/.96/.98 |
| 300 | .55/.81/.86 | .64/.87/.94 | .84/.97/1.0 |
| 1000 | .48/.70/.72 | .72/.95/.98 | .85/.98/.98 |

- **Ω_n**: l'errore si contrae come 1/√N (pendenza −0.505, rms·√N ≈ 0.37 costante), senza più il pixel a 53°. Pull rms ≈ 0.79 a N=50/150/300, quindi le incertezze sono leggermente conservative. A N=1000 il pull rms esplode (≈8·10⁴) perché in alcuni esperimenti la risoluzione dichiarata è ≈0. La causa probabile è un posterior concentrato su un solo pixel della calotta, per cui si tratta di un artefatto di discretizzazione **(d)**.
- **σ_E**: bias verso 0 con N (log σ da −0.14 a +0.01), pull ≈ N(0,1), coverage nominale. La pendenza di contrazione è −0.735, più ripida di −0.5 perché il bias iniziale a N piccolo decade. Il problema del §3.3 è risolto.
- **μ_E**: bias **costante** di ≈ −0.02 MeV (≈0.6%), indipendente da N e significativo a più di 5σ a ogni N. La risoluzione scende (rms 0.082 → 0.028 MeV) ma la pendenza di contrazione è solo −0.36. Il pull medio passa da −0.25 a −1.05 e la coverage al 68% da 0.69 a 0.48. È il plateau di un sistematico condiviso (Cap. 39) e domina da N ≳ 300.
- **Robustezza al prior** (N=150, prior uniforme in σ_E invece che in log σ_E): lo spostamento di μ_E è trascurabile (medio −0.009 σ, massimo 0.15 σ). Quello di log σ_E è moderato (medio +0.15 σ, massimo 0.68 σ), con coverage [0.59, 0.89, 0.95]. SIGMA_E_MAX non entra nel risultato.

### 6.4 Aperto: bias di μ_E (d)

Cause candidate, tutte da testare:

1. **Mismatch generatore/verosimiglianza**: il generatore sparpaglia la traccia nel piano tangente in 2D, mentre la verosimiglianza usa un kernel gaussiano 1D sul meridiano, un'approssimazione di ordine O(σ_θ²) **(b)**. Poiché E_p ∝ cos²θ_p, un errore sistematico su θ_p si traduce in uno spostamento di scala su E_n.
2. **Discretizzazione** della griglia locale in θ o della griglia in E_n.
3. **Ω_n plug-in** nello stadio 2: poco probabile, perché l'errore su Ω si contrae mentre il bias di μ_E no.

Diagnostica proposta: stadio 2 con Ω_n **vera** e dati generati con smearing polare 1D (coerente con la verosimiglianza). Se il bias sparisce la causa è la 1; altrimenti si passa alla griglia fine (causa 2).

## 7. Causa del bias di μ_E: curvatura della sfera nel kernel di traccia (2026-09-26)

### 7.1 Meccanismo (a)

Il kernel 1D del §6.2 punto 3 tratta θ_obs − θ come gaussiano sul meridiano e trascura la geometria della sfera. Uno spostamento t perpendicolare al meridiano **aumenta** sempre l'angolo dal polo, perché cos θ' = cos θ cos t, quindi θ' ≈ θ + t²/(2 tan θ). Il kernel piatto non vede questo effetto. Assegna quindi a θ_p vero un valore medio più piccolo di σ_θ² cot θ / 2.

Poiché E_p = E_n cos²θ_p, vale d ln E_p/dθ = −2 tan θ. L'errore relativo su E_n è quindi

  ΔE_n / E_n ≈ −σ_θ²,

indipendente da θ. Con σ_θ = 0.08 e μ_E ≈ 3.25 MeV si ottiene ≈ −0.021 MeV, lo stesso valore del plateau del §6.3.

Il kernel esatto per un errore von Mises–Fisher con κ = 1/σ_θ², integrato sull'azimut relativo, è:

  K(θ_obs | θ) = κ / (1 − e^{−2κ}) · sin θ · exp(κ (cos(θ_obs − θ) − 1)) · I₀ₑ(κ sin θ_obs sin θ)

Per θ, θ_obs ≫ σ_θ si riduce al kernel piatto moltiplicato per √(sin θ / sin θ_obs).

Controllo sul primo momento, con traccia di densità cosθ_p/π e σ_θ = 0.08:

| | E[cos θ_obs] |
|---|---|
| vMF esatto, (2/3)(coth κ − 1/κ) | 0.66240 |
| MC di `smear_direction` | 0.66244 ± 0.00017 |
| kernel sferico, numerico | 0.66240 |
| kernel piatto, numerico | 0.66454 (≈12σ fuori) |

Il generatore gaussiano 2D e il vMF differiscono solo a O(σ⁴), per cui il generatore resta invariato.

### 7.2 Diagnosi (c)

La diagnosi usa solo lo stadio 2, con Ω_n **vera** fissata a (0.9, 2.1) rad, così da isolarla dallo stadio 1. Parametri:

- μ_E = 3.0 e σ_E = 0.4 MeV;
- N = 1000 eventi e M = 60 esperimenti, con lo stesso seed per i due kernel;
- finestra 80×30 su (μ_E, σ_E);
- stima: media a posteriori di μ_E.

La predizione del §7.1 è −σ_θ² μ_E.

| σ_θ (rad) | predetto (MeV) | kernel piatto (MeV) | kernel sferico (MeV) |
|---|---|---|---|
| 0.04 | −0.0048 | −0.0029 ± 0.0024 | +0.0018 ± 0.0024 |
| 0.08 | −0.0192 | −0.0165 ± 0.0026 | +0.0023 ± 0.0026 |
| 0.16 | −0.0768 | −0.0681 ± 0.0034 | +0.0027 ± 0.0035 |

- Con il kernel piatto il bias scala come σ_θ²: il rapporto tra 0.16 e 0.08 vale 4.1, contro 4 atteso. Il valore coincide con la predizione entro il 10–15%. A σ_θ = 0.08 riproduce il plateau del §6.3.
- Con il kernel sferico il bias è compatibile con 0 (≤ 1σ) a ogni σ_θ.
- La larghezza del posterior non cambia: σ_post ≈ 0.020 MeV a σ_θ = 0.08.

La causa 1 del §6.4 è **confermata (c)**. Le cause 2 e 3 non servono a spiegare il bias.

**Nota sul disegno della diagnosi.** La variante "smearing 1D polare" del piano (§6.4) *non* è coerente con il kernel piatto. Spostare la traccia lungo il meridiano in θ introduce comunque un fattore Jacobiano sin θ / sin θ_obs sulla densità per angolo solido. Con quella variante il bias è −0.0424 MeV con il kernel piatto e −0.0238 con quello sferico, quindi non chiude la domanda ed è stata scartata come test. La conferma viene dallo scaling in σ_θ e dal confronto tra i due kernel sugli stessi dati.

### 7.3 Correzione

- `forward_model.log_track_kernel_sphere` sostituisce il kernel piatto in `track_energy_table` e `hierarchical_base`. Le usano `posterior_B` e entrambi gli stadi di `posterior_C`.
- Il kernel piatto resta nel codice come confronto.
- Test aggiunti in `tests/test_case_B.py`:
  - normalizzazione sulla sfera;
  - primo momento esatto vMF;
  - confronto con il MC di `smear_direction`;
  - limite asintotico piatto × √(sin θ / sin θ_obs).
- Resta da rimisurare con la checklist completa: bias e coverage di μ_E a N = 1000 e il pull di Ω_n (P2 e P3 del piano).

## 8. Direzione Ω_n: pull a N=1000 e sovra-copertura (2026-09-26)

### 8.1 Pull esploso a N=1000: calotta degenere (c)

La diagnosi rigioca lo stadio 1 sui 40 seed della checklist a N=1000 e registra per ogni esperimento:

- raggio della calotta;
- pixel;
- N_eff = 1/Σp²;
- σ dichiarata;
- errore.

In 39 esperimenti su 40 la calotta è sana: N_eff va da 54 a 1400 (mediana ≈ 120) e σ vale ≈ 3.8 pixel. L'ipotesi del §6.3 (posterior su un solo pixel) è quindi **smentita**.

L'unico esperimento anomalo (il n. 8) ha una calotta di **raggio ≈ 4.5·10⁻⁸ rad**: gli 8000 pixel coincidono, il posterior è piatto su di essi, σ = 0 e il pull vale ≈ 5·10⁵. Il meccanismo è il seguente:

1. `refine_shared_direction` stimava il passo della griglia grossolana come `min(angle[angle > 0])`.
2. Per alcuni pixel `arccos(best @ best)` vale 1.49·10⁻⁸, non 0, perché il prodotto scalare è arrotondato a 1 − ε.
3. Il MAP stesso finiva così nel minimo e il passo diventava 1.5·10⁻⁸.
4. A N=1000 un solo pixel supera la soglia, per cui il raggio si riduceva a due volte quel passo.

Correzione: nuova funzione pura `posterior_C.direction_cap_radius`, che esclude il MAP per indice, più un test di regressione. Dopo la correzione l'esperimento 8 ha pull 0.41 e N_eff 164. Il raffinamento adattivo previsto dal piano non serve.

### 8.2 Sovra-copertura di Ω_n: prior largo su E_n nello stadio 1 (c)

Dopo le correzioni del kernel (§7) e della calotta, a N=1000 (M=40) il pull rms di Ω vale 0.71 e la coverage di Rayleigh al 68% vale 0.90. Le incertezze restano quindi conservative.

Diagnosi: stadio 1 sugli stessi seed a N=150 (M=200), cambiando solo il prior su E_n. Lo stadio 1 con il prior vero è un oracolo (usa μ_E e σ_E veri) e serve solo a isolare la causa.

| prior su E_n | pull rms (MAP) | pull rms (media) | coverage 68% | errore rms | σ dichiarata media |
|---|---|---|---|---|---|
| largo U(0.5, 6) (attuale, Caso B) | 0.78 ± 0.04 | 0.77 | 0.86 | 1.73° | 2.20° |
| vero N(μ_E, σ_E) | 1.04 ± 0.05 | 1.05 | 0.66 | 1.11° | 1.06° |

La sovra-copertura è dovuta al nuisance E_n mal specificato: ogni evento viene marginalizzato su un E_n ammesso in [0.5, 6] MeV, mentre i dati hanno E_n ≈ μ_E ± σ_E. Con il prior giusto Ω è calibrato (pull rms ≈ 1, coverage nominale) ed è più preciso del 36% in errore rms. Questo coincide con l'opzione 2 del §5.

### 8.3 Rimedio: stadio 1 iterato con prior plug-in su E_n (c)

Procedura (empirical Bayes, resta a due stadi, niente griglia 4D):

1. stadio 1 con prior largo su E_n, che dà Ω̂⁽⁰⁾;
2. stadio 2, che dà μ̂_E (media a posteriori) e σ̂_E = exp(E[log σ_E]);
3. stadio 1′ con prior N(μ̂_E, σ̂_E) su E_n (`posterior_C.refine_shared_direction_hierarchical`), che dà Ω̂;
4. stadio 2′ con Ω̂.

Validazione sugli stessi seed della checklist:

| N, M | stadio 1 | pull rms Ω | coverage 68% Ω | errore rms Ω | bias μ_E [MeV] | pull μ_E (media/larghezza) | bias log σ_E |
|---|---|---|---|---|---|---|---|
| 150, 200 | largo | 0.78 | 0.86 | 1.73° | +0.0007 ± 0.0037 | +0.03 / 0.98 | −0.07 |
| 150, 200 | iterato | 1.05 | 0.62 | 1.11° | −0.0001 ± 0.0036 | +0.01 / 0.97 | −0.10 |
| 1000, 40 | largo | 0.71 | 0.90 | 0.62° | +0.0017 ± 0.0032 | +0.07 / 1.00 | +0.01 |
| 1000, 40 | iterato | 0.89 | 0.72 | 0.37° | +0.0016 ± 0.0032 | +0.06 / 1.01 | +0.00 |

- A N=150 l'iterazione riproduce l'oracolo del §8.2 (pull 1.04, coverage 0.66, errore 1.11°) senza conoscere μ_E e σ_E veri.
- A N=1000 l'errore su Ω scende del 40% e il pull torna vicino a 1.
- μ_E e σ_E praticamente non cambiano: lo stadio 2 era già poco sensibile a errori su Ω di ~1°.

Nota: a N=1000 la coverage 68% di μ_E vale 0.55 in entrambe le varianti, pur con larghezza del pull 1.0. Con M=40 l'errore binomiale è ±0.07, quindi lo scarto è di ~1.7σ. Va ricontrollata nella checklist completa (d).

Il prior plug-in usa i dati due volte (stadio 2 → stadio 1′). L'effetto atteso è O(1/N) sull'incertezza di Ω e non è visibile nei pull sopra. Etichetta (b): approssimazione empirical-Bayes, validata MC (c) nei regimi N=150 e N=1000.

## 9. Checklist Cap. 40 completa dopo le correzioni P1–P3 (v0.10, 2026-09-26) (c)

Run completo di `scripts/caso_C_checklist.py`: kernel sferico (§7), raggio della calotta corretto (§8.1), stadio 1 iterato (§8.3). Setup e M come in §6.3; circa 10 minuti con 3 processi. I risultati grezzi sono in `outputs/caso_C_checklist_results.pkl`; tabelle, confronto con la v0.9 e figure sono nel notebook `notebooks/07_checklist_caso_C_v2.ipynb`.

| N (M) | bias μ_E [MeV] | bias log σ_E | pull μ_E (media/larghezza) | pull Ω rms | coverage μ_E 68/90/95 | coverage log σ_E 68/90/95 | coverage Ω (HPD) 68/90/95 |
|---|---|---|---|---|---|---|---|
| 50 (200) | +0.0015 ± 0.0057 | −0.27 ± 0.04 | −0.02 / 1.01 | 1.08 | 0.65 / 0.89 / 0.95 | 0.61 / 0.82 / 0.91 | 0.70 / 0.86 / 0.93 |
| 150 (200) | −0.0001 ± 0.0036 | −0.08 ± 0.02 | +0.01 / 0.97 | 1.05 | 0.70 / 0.92 / 0.95 | 0.60 / 0.87 / 0.94 | 0.65 / 0.88 / 0.93 |
| 300 (100) | −0.0013 ± 0.0039 | −0.03 ± 0.02 | −0.04 / 1.14 | 1.00 | 0.66 / 0.87 / 0.91 | 0.64 / 0.87 / 0.92 | 0.66 / 0.91 / 0.95 |
| 1000 (40) | +0.0016 ± 0.0032 | +0.00 ± 0.01 | +0.06 / 1.01 | 0.89 | 0.57 / 0.95 / 1.00 | 0.70 / 0.97 / 1.00 | 0.78 / 0.93 / 0.97 |

Contrazione (pendenza log-log; rms·√N a N = 50, 150, 300, 1000):

- μ_E: −0.46; 0.57, 0.62, 0.68, 0.64. Nella v0.9 era −0.36 e rms·√N cresceva fino a 0.87: il plateau è sparito.
- Ω_n: −0.56; 0.24, 0.24, 0.23, 0.20 rad (v0.9: circa 0.37).
- log σ_E: −0.81, come nella v0.9, perché a N piccolo domina il prior.

Robustezza al prior a N = 150: invariata rispetto a §6.3.

**Esito.**

- **μ_E**: calibrato a ogni N; P1 chiuso.
- **Ω_n**: calibrato a N ≥ 150, errore rms circa −40%, σ finita a N = 1000; P2 chiuso, P3 chiuso a N ≥ 150.
- **Regressione a N = 50 su log σ_E**: bias −0.27 (v0.9 −0.14) e coverage al 90% 0.82 (−3.8 errori binomiali; v0.9 0.88). Ω_n è lievemente sovra-confidente al 90% e al 95%.
  - Ipotesi (d): uso doppio dei dati nel prior plug-in (§8.3), effetto O(1/N) che svanisce da N ≈ 150.
  - La correlazione per esperimento fra il residuo di log σ_E e il pull di Ω non è significativa (Spearman −0.13, p = 0.07): il test non conferma e non esclude il meccanismo.
  - Test diretto da fare: N = 50 con stadio 1 a prior largo contro iterato.
- **Coverage 68% di μ_E a N = 1000**: 0.57 con M = 40 (−1.5 errori binomiali), ma 0.95 e 1.00 al 90% e al 95% e larghezza del pull 1.01: compatibile con una fluttuazione (d). Da ricontrollare con M ≥ 100.
- I numeri a N = 1000 differiscono di 1–2 esperimenti su 40 da quelli di §8.3 (coverage 68%: μ_E 0.57 contro 0.55, Ω 0.78 contro 0.72). La causa non è stata indagata; fanno fede quelli di questo run.

## 10. Stadio 2 su Ω̂_0 e coverage di μ_E a N = 1000 (v0.11, 2026-09-26)

### 10.1 Test diretto: stadio 1 a prior largo contro iterato (c)

Sono stati usati gli stessi seed della checklist (`spawn_key=(N, i)`) e le stesse definizioni di §9.
Per ogni esperimento si calcolano entrambe le varianti:

- **largo**: stadio 1 a prior largo, da cui Ω̂_0, poi stadio 2 su Ω̂_0;
- **iterato**: la pipeline di §9.

Lo script era una diagnostica nello scratchpad, non inclusa nel repo: 3 processi, 515 s.

N = 50, M = 200 (errore binomiale ±0.033 / 0.021 / 0.015):

| | largo | iterato |
|---|---|---|
| Ω: errore rms / σ dichiarata media | 3.05° / 3.85° | 1.96° / 1.82° |
| Ω: pull rms | 0.79 | 1.08 |
| Ω: coverage 68/90/95 | 0.84 / 0.965 / 0.995 | 0.70 / 0.86 / 0.925 |
| μ_E: bias [MeV] / coverage | +0.002 ± 0.006 / 0.675, 0.91, 0.96 | +0.002 ± 0.006 / 0.65, 0.89, 0.95 |
| log σ_E: bias (mediana) | −0.154 ± 0.033 | −0.271 ± 0.037 |
| log σ_E: pull (media / larghezza) | −0.11 / 0.96 | −0.37 / 0.93 |
| log σ_E: coverage 68/90/95 | 0.665 / 0.885 / 0.93 | 0.615 / 0.825 / 0.905 |

N = 1000, M = 100 (i primi 40 sono gli esperimenti della checklist; errore binomiale ±0.047 / 0.030 / 0.022):

| | largo | iterato |
|---|---|---|
| Ω: pull rms / coverage 68/90/95 | 0.76 / 0.80, 0.99, 1.00 | 1.05 / 0.67, 0.88, 0.93 |
| μ_E: pull (media / larghezza) | +0.01 / 0.95 | −0.01 / 0.95 |
| μ_E: coverage 68/90/95 | 0.65 / 0.95 / 0.99 | 0.66 / 0.95 / 0.98 |
| log σ_E: bias / coverage | −0.001 ± 0.006 / 0.71, 0.92, 0.98 | −0.006 ± 0.006 / 0.71, 0.94, 0.98 |

**Lettura.**

- L'iterazione **migliora Ω_n** (errore circa −35%, calibrata) ma **peggiora σ_E a N = 50**.
- Ipotesi (d): σ̂_E è già sottostimato, e il prior plug-in sceglie una direzione adattata a uno spettro stretto. Lo stadio 2 condizionato su quella direzione rinforza la sottostima.
- **Coverage 68% di μ_E a N = 1000:**
  - con M = 100 vale 0.65, nominale;
  - nei 40 esperimenti della checklist vale 0.55, nei 60 nuovi 0.72;
  - lo 0.57 di §9 era quindi una **fluttuazione di M = 40 (c)**.

### 10.2 Correzione (v0.11)

In `scripts/caso_C_checklist.py` (commit `819f2ce`) le due stime vengono ora da stadi diversi:

- (μ̂_E, σ̂_E) si prendono dallo stadio 2 su Ω̂_0;
- lo stadio 1 iterato, con prior plug-in N(μ̂_E, σ̂_E), serve solo a stimare Ω_n.

Il costo è di uno stadio 2 in meno per esperimento. `src/` non cambia. Ω̂_n e (μ̂_E, σ̂_E) non vengono più da un'unica stima congiunta: va detto quando si riportano insieme.

### 10.3 Checklist v0.11 (c)

La checklist completa è stata rilanciata con 3 processi. La v0.10 è conservata in `outputs/caso_C_checklist_results_v0.10.pkl`; il confronto completo è nel notebook `notebooks/08_checklist_caso_C_v0.11.ipynb`. I risultati su Ω_n sono identici alla v0.10, esperimento per esperimento.

| N (M) | bias μ_E [MeV] | bias log σ_E | pull log σ_E (media/larghezza) | coverage μ_E 68/90/95 | coverage log σ_E 68/90/95 | coverage Ω 68/90/95 |
|---|---|---|---|---|---|---|
| 50 (200) | +0.0023 ± 0.0057 | −0.15 ± 0.03 | −0.11 / 0.96 | 0.68 / 0.91 / 0.96 | 0.67 / 0.89 / 0.93 | 0.70 / 0.86 / 0.93 |
| 150 (200) | +0.0007 ± 0.0036 | −0.055 ± 0.019 | −0.02 / 1.05 | 0.69 / 0.92 / 0.95 | 0.60 / 0.87 / 0.94 | 0.65 / 0.88 / 0.93 |
| 300 (100) | −0.0008 ± 0.0040 | −0.017 ± 0.017 | +0.00 / 1.06 | 0.66 / 0.84 / 0.91 | 0.64 / 0.87 / 0.94 | 0.66 / 0.91 / 0.95 |
| 1000 (40) | +0.0017 ± 0.0031 | +0.008 ± 0.008 | +0.15 / 0.87 | 0.55 / 0.95 / 1.00 | 0.70 / 0.93 / 1.00 | 0.78 / 0.93 / 0.97 |

Contrazione, robustezza al prior e pull di Ω sono invariati rispetto a §9:

- pendenza di μ_E −0.46;
- pendenza di Ω −0.56;
- pull di Ω 1.08 / 1.05 / 1.00 / 0.89.

**Esito.**

- **Chiuso (c):**
  - regressione di log σ_E a N = 50: coverage di nuovo entro gli errori binomiali, bias come nella v0.9;
  - coverage di μ_E a N = 1000.
- **Aperto (d):**
  - bias residuo di log σ_E a N piccolo: −0.15 a N = 50, −0.055 a N = 150. Ipotesi: l'incertezza di Ω̂_0, circa 3° a N = 50, non è propagata nello stadio 2. Rimedio da provare: marginalizzare lo stadio 2 su alcuni pixel della calotta, pesati con la posterior di Ω;
  - rms di log σ_E maggiore della σ dichiarata a N = 150–300 (0.27 contro 0.19; 0.17 contro 0.11), con pull e coverage nominali, già presente nella v0.9;
  - a N = 300 la coverage al 90% di μ_E vale 0.84 (−2 errori binomiali; v0.10 0.87), con larghezza del pull 1.14 in entrambe le versioni. Da tenere d'occhio.

## 11. Bias di log σ_E a N piccolo: effetto del prior largo (v0.12, 2026-09-26)

Il bias residuo di log σ_E lasciato aperto in §10.3 (−0.15 a N = 50, −0.055 a N = 150) è stato messo alla prova con due test. Entrambi usano gli stessi seed della checklist (`spawn_key=(N, i)`) e lo stadio 1 a prior largo della v0.11. Gli script sono diagnostiche nello scratchpad, non incluse nel repo, girate con 3 processi.

### 11.1 Correzione 2: stadio 2 marginalizzato su Ω_n (c, scartata)

`posterior_C.refine_hyperparameters_marginal_direction` (commit `9cecac3`, funzione additiva) calcola il marginale esatto del posterior congiunto:

log p(μ_E, σ_E | D) = log π(μ_E, σ_E) + log Σ_j exp Σ_k log L_k(μ_E, σ_E, Ω_j) + cost.

- La somma su Ω_j è una quadratura su `N_DIRECTION_MARGINAL = 100` pixel ad area uguale. I pixel coprono la calotta dello stadio 1 in cui il log-posterior supera max − `WINDOW_DELTA_LOG`.
- Il prior su Ω_n è uniforme, quindi il peso di ogni pixel è costante.
- La finestra fine su (μ_E, σ_E) è quella dello stadio 2 su Ω̂_0 ed è comune a tutti i pixel.
- Test in `tests/test_case_C.py`:
  - con una calotta puntiforme il marginale coincide con il condizionato;
  - il marginale non è più stretto del condizionato.

| | N = 50, condizionato su Ω̂_0 | N = 50, marginalizzato | N = 150, condizionato | N = 150, marginalizzato |
|---|---|---|---|---|
| bias log σ_E (mediana) | −0.154 ± 0.033 | −0.234 ± 0.036 | −0.055 ± 0.019 | −0.073 ± 0.019 |
| rms log σ_E | 0.485 | 0.562 | 0.272 | 0.281 |
| coverage log σ_E 68/90/95 | 0.665 / 0.885 / 0.93 | 0.64 / 0.86 / 0.91 | 0.60 / 0.87 / 0.945 | 0.61 / 0.875 / 0.94 |
| bias μ_E [MeV] | +0.002 ± 0.006 | +0.002 ± 0.006 | +0.001 ± 0.004 | +0.000 ± 0.004 |
| tempo dello stadio 2 per esperimento | 0.5 s | 15.7 s | 1.1 s | 43 s |

**Esito.**

- Marginalizzare su Ω_n **peggiora** il bias. La mediana di log σ_E scende in circa 2/3 degli esperimenti, e il costo sale di circa 30–40 volte. L'ipotesi di §10.3, "incertezza di Ω non propagata", è falsificata (c).
- Spiegazione possibile (d): la verosimiglianza congiunta favorisce le direzioni in cui le E_n ricostruite sono più concentrate. Marginalizzare sposta quindi massa verso σ_E piccola, e questo è compatibile con un modello corretto.
- La funzione resta nel codice come documentazione del test, ma **non è usata** dalla checklist.

### 11.2 Test della causa: prior uguale al generatore (c)

**Test.** Lo stadio 2 su Ω̂_0 è stato ricalcolato con due prior:

- il prior di default (`priors.hyperparameter_prior`): uniforme in (μ_E, log σ_E) su tutta la griglia, cioè μ_E ∈ [`EN_MIN`, `EN_MAX`] e σ_E ∈ [`SIGMA_E_MIN`, `SIGMA_E_MAX`] = [0.01, 50] MeV;
- un prior uguale al generatore della checklist: μ_E ~ U(2.5, 4.0), σ_E log-uniforme in [0.2, 0.6] MeV.

**Perché è un test decisivo.** Con il prior del generatore, le verità sono estratte dal prior stesso. Se il modello è giusto, E[media a posteriori − verità] = 0 esattamente. Con il prior di default si ritrovano esattamente i numeri della checklist v0.11.

| N (M) | prior | bias log σ_E (mediana) | bias log σ_E (media) | rms / σ dichiarata media | coverage log σ_E 68/90/95 | bias μ_E [MeV] | coverage μ_E 68/90/95 |
|---|---|---|---|---|---|---|---|
| 50 (200) | default | −0.154 ± 0.033 | −0.214 ± 0.036 | 0.551 / 0.416 | 0.665 / 0.885 / 0.93 | +0.002 ± 0.006 | 0.675 / 0.91 / 0.96 |
| | generatore | **+0.018 ± 0.014** | **+0.020 ± 0.014** | 0.195 / 0.190 | 0.73 / 0.905 / 0.96 | −0.005 ± 0.005 | 0.715 / 0.94 / 0.975 |
| 150 (200) | default | −0.055 ± 0.019 | −0.072 ± 0.021 | 0.308 / 0.190 | 0.60 / 0.87 / 0.945 | +0.001 ± 0.004 | 0.695 / 0.915 / 0.955 |
| | generatore | **+0.006 ± 0.010** | **+0.007 ± 0.010** | 0.135 / 0.118 | 0.60 / 0.92 / 0.975 | −0.001 ± 0.004 | 0.705 / 0.92 / 0.95 |
| 300 (100) | default | −0.017 ± 0.017 | −0.021 ± 0.018 | 0.185 / 0.111 | 0.64 / 0.87 / 0.94 | −0.001 ± 0.004 | 0.66 / 0.84 / 0.91 |
| | generatore | **+0.004 ± 0.010** | **+0.005 ± 0.010** | 0.102 / 0.091 | 0.60 / 0.84 / 0.94 | −0.002 ± 0.004 | 0.66 / 0.84 / 0.92 |

**Meccanismo (c).**

- A N piccolo σ_E spesso non è risolta dal basso. Nel 30% degli esperimenti a N = 50 (61 su 200) il posterior ha un plateau verso σ_E → 0 che arriva fino a `SIGMA_E_MIN = 0.01` MeV; il MAP resta interno. A N = 150 succede nell'8% dei casi.
- Il prior log-uniforme su [0.01, 50] mette in quel plateau molta massa, che il generatore (σ_E ∈ [0.2, 0.6]) non usa mai. La mediana di log σ_E viene quindi trascinata verso il basso, soprattutto quando σ_E vera è piccola. Col prior di default il bias medio per terzile di σ_E vera a N = 50 vale −0.35 / −0.20 / −0.09.
- Il bias scende con N come ci si aspetta da un effetto del prior: −0.15 → −0.055 → −0.017.
- Col prior del generatore, il bias per terzile vale +0.18 / −0.01 / −0.11. È il restringimento verso il centro del prior, che in media si compensa.
- Anche l'"rms di log σ_E maggiore della σ dichiarata" di §10.3 era un effetto del prior: col prior del generatore rms e σ dichiarata coincidono entro circa il 15% (0.135 contro 0.118 a N = 150).

### 11.3 Decisione ed esito

**Decisione:** si **tiene il prior largo** come default. Il bias di log σ_E a N piccolo viene dichiarato come **effetto del prior (c)**, non come errore del modello. Le ragioni:

- è proprio e non informativo sulla scala;
- l'effetto sparisce con N;
- la coverage di log σ_E resta entro gli errori binomiali a N = 50;
- la robustezza al prior è già un punto della checklist (Cap. 40).

Chi riporta σ̂_E a N ≲ 150 deve dichiarare il prior usato.

**Chiuso (c):**

- bias residuo di log σ_E a N piccolo;
- rms di log σ_E maggiore della σ dichiarata.

**Ancora aperto (d)**, invariato con entrambi i prior:

- coverage al 68% di log σ_E bassa a N = 150–300 (0.60–0.64, tra 1 e 2.4 errori binomiali), mentre 90% e 95% sono nominali;
- coverage al 90% di μ_E a N = 300 pari a 0.84 (circa −1.6 errori binomiali con M = 100).

**Aggiornamento (v0.13):** il primo punto aperto è chiuso come fluttuazione statistica (§12). Il secondo resta al limite (d).

## 12. Coverage al 68% di log σ_E a N = 150–300: fluttuazione statistica (v0.13, 2026-09-26) (c)

Il punto aperto di §11.3 è stato esaminato con `scripts/caso_C_cov68.py` e nel notebook 09.

**Dati esistenti** (seed della checklist, senza nuovi run):

- **Forma.** La semiampiezza dell'intervallo a code uguali al 68% vale in mediana 1.00 std, e al 95% 1.01 × 1.96 std. Il posterior di log σ_E è quindi gaussiano nel nucleo. Fanno eccezione i rari esperimenti col plateau di §11.2: a N = 150 il 5° percentile del rapporto al 68% è 0.83.
- **Asimmetria.** I mancati al 68% sono bilanciati: 35 sotto e 45 sopra a N = 150, 20 e 16 a N = 300.
- **Discretizzazione.** Il passo della griglia fine 40×40 in log σ_E vale circa 0.27 std. Con una griglia 80×80, su 14 esperimenti gli estremi dell'intervallo al 68% si spostano al massimo di 0.036 std, tipicamente di 0.01 std.
- Tutte e tre le cause sono escluse (c). Restava da capire se l'sd dei pull, 1.05–1.06, indicasse un posterior troppo stretto del 5–10%: la coverage attesa al 68% sarebbe circa 0.65.

**Seed nuovi.** Il test usa `spawn_key=(N, i, 68)`, disgiunti da quelli della checklist, e il prior di default. Lo stadio 2 è calcolato su Ω̂_0 (pipeline v0.11) e sulla Ω_n vera. Il run ha usato 3 processi, circa 11 min.

| N (M) | stadio 2 su | coverage log σ_E 68/90/95 | err. binomiale 68/90/95 | pull media/sd | mancati 68% sotto/sopra |
|---|---|---|---|---|---|
| 150 (200), seed della checklist | Ω̂_0 | 0.60 / 0.87 / 0.945 | 0.033 / 0.021 / 0.015 | −0.02 / 1.05 | 35 / 45 |
| 150 (600), seed nuovi | Ω̂_0 | **0.658** / 0.895 / 0.955 | 0.019 / 0.012 / 0.009 | +0.07 / 0.99 | 103 / 102 |
| 150 (600), seed nuovi | Ω vera | 0.650 / 0.902 / 0.957 | | −0.06 / 0.97 | 87 / 123 |
| 300 (100), seed della checklist | Ω̂_0 | 0.64 / 0.87 / 0.94 | 0.047 / 0.030 / 0.022 | +0.00 / 1.06 | 20 / 16 |
| 300 (300), seed nuovi | Ω̂_0 | **0.723** / 0.917 / 0.963 | 0.027 / 0.017 / 0.013 | +0.01 / 0.92 | 42 / 41 |
| 300 (300), seed nuovi | Ω vera | 0.737 / 0.917 / 0.970 | | −0.07 / 0.91 | 35 / 44 |

**Esito (c).**

- Sui seed nuovi la coverage al 68% vale −1.2 errori binomiali a N = 150 e +1.6 a N = 300. Sui 900 esperimenti nuovi combinati vale esattamente 0.680. L'sd dei pull è 0.92–0.99.
- Il posterior troppo stretto è falsificato, e lo stadio 1 non contribuisce: con Ω̂_0 e con la Ω vera le coverage sono compatibili.
- Il deficit dei seed della checklist (0.613 combinata su M = 300, z = −2.5) è una **fluttuazione statistica**. La checklist ha circa 24 coverage (4 N × 3 livelli × 2 parametri scalari), e la probabilità di almeno uno scarto con |z| ≥ 2.4 è circa 0.33.
- log σ_E è quindi calibrato a N = 150–300 col prior di default.

**Ancora al limite (d):** la coverage al 90% di μ_E a N = 300. Vale 0.84 ± 0.037 sui seed della checklist e 0.877 ± 0.017 sui seed nuovi, cioè −1.4 errori binomiali. La sd dei pull di μ_E è 1.03 sui seed nuovi e le coverage al 68% e al 95% sono nominali: non c'è un segnale netto e non si interviene.
