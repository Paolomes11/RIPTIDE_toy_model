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
