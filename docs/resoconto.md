# Resoconto dei risultati — riptide-toy

v0.26 — 2026-11-14

Questa è la sintesi dei risultati del toy model. Il dettaglio tecnico è in `docs/roadmap.md` (stato e
scelte di implementazione) e in `docs/report_caso_C_stadio1.md` (diario del Caso C). I passaggi
numerici sono rifatti nei notebook `notebooks/01`–`05`; il notebook `06_modifiche_10_20_MeV` spiega
cosa è cambiato col passaggio a 10–20 MeV e come è stata chiusa la calibrazione del Caso C.

Etichette: **(a)** fatto consolidato · **(b)** stima toy con assunzioni esplicite · **(c)** verificato
con Monte Carlo · **(d)** ipotesi da testare.

## 1. Modello e assunzioni

Un neutrone di energia E_n diffonde elasticamente su un protone; si misura il protone di rinculo
(energia E_p, direzione della traccia). Unità: MeV, rad (gradi nelle tabelle angolari).

| Caso | Parametro condiviso | Nuisance per evento | Assunzioni usate |
|---|---|---|---|
| A | E_n | θ_p (angolo fra neutrone e protone) | 1 sorgente unica, monoenergetica, direzione nota |
| B | Ω_n (direzione del neutrone) | E_n (prior largo) | 1 sorgente unica + 2 campo lontano |
| C | Ω_n e (μ_E, σ_E) | E_n^(k) ~ N(μ_E, σ_E) | 1 + 2 + 3 energie simili |

Dominio di energia 10–20 MeV (dalla v0.19; fino alla v0.18 era 0.5–6 MeV). Risoluzioni del
rivelatore: σ_Ep = 0.10 MeV, σ_θ = 0.08 rad. Prior sempre propri (E_n piatto su [10, 20] MeV;
σ_E log-uniforme su [0.01, 50] MeV; θ_p con densità sin 2θ_p nel Caso A). I casi di riferimento
R1–R3 restano alle energie con cui sono definiti (R1 su una griglia 0.5–6 MeV).

## 2. Cinematica (notebooks 01, 06)

| Risultato | Valore | |
|---|---|---|
| Energia del protone (relativistica, masse uguali m) | E_p = 2m E_n cos²θ_p / (2m + E_n sin²θ_p), θ_p ≤ 90° | (a) |
| Limite non relativistico (E_n ≪ 2m) | E_p = E_n cos²θ_p; a 20 MeV e 45° lo scarto è 0.05 MeV = 0.5 σ_Ep | (a) |
| Isotropia in CM ⇒ spettro del protone | E_p ~ U(0, E_n) esattamente, anche relativistico: ⟨E_p⟩ = E_n/2, Var = E_n²/12 | (a) |
| Isotropia in CM ⇒ angolo di rinculo | cos²θ_p uniforme nel limite non relativistico: p(θ_p) = sin 2θ_p | (a) |
| Densità delle tracce attorno a Ω_n | cosθ_p/π per steradiante nel limite non relativistico; correzione relativistica ±1% a 20 MeV | (a) |
| Isotropia in CM a 10–20 MeV (ENDF/B-VIII.0, H-1) | anisotropia presente ma senza effetto sulla ricostruzione (§6.5) | (c) |
| Tracce misurate oltre 90° per lo smearing | σ_θ²/2 = 0.0032 | (b) |
| Esempio: E_n = 20 MeV, θ_p = 45° | E_p = 9.947 MeV (non relativistico: 10.000) | (a) |

## 3. Caso A — energia condivisa (notebook 02)

| Verifica | Risultato | |
|---|---|---|
| R1, un evento (Ê_p = 1.40, θ̂_p = 0.50) | E_n = 1.82 ± 0.21 MeV (codice: σ pesata 0.219) | (c) |
| R2, due eventi 2.70 ± 0.45 e 2.35 ± 0.20 | E_n = 2.40 ± 0.18 MeV (codice: 2.408 ± 0.183) | (c) |
| R3, ricostruzione difettosa (+3%, +0.05 MeV, σ dichiarate −30%) | pull medio 0.90, larghezza 1.44, coverage 68% → 0.433 | (c) |
| Checklist, 20 000 eventi, E_n ~ U(10, 20) (= prior), θ_p ed E_p dalla cinematica in CM | pull −0.000 / 1.001; coverage 0.680 / 0.898 / 0.950 (nominale) | (c) |
| Stessa checklist, verità E_n ~ U(11, 19) | pull +0.031 / 0.929; coverage 0.755 / 0.943 / 0.975 | (c) |
| Bias per bin di E_n (verità dal prior) | +1.63, +0.83, +0.11, −0.70, −1.83 MeV: attrazione verso il centro del prior | (c) |
| E_n combinato, N = 1000 a 15 MeV, prior piatto su θ_p (20 ripetizioni) | bias −0.101 ± 0.009 MeV, pull medio −3.9: plateau | (c) |
| Stesso, prior fisico sin 2θ_p (default dalla v0.19) | bias −0.006 ± 0.008 MeV, pull medio −0.17 | (c) |
| E_n combinato a N = 10 / 100 / 1000, E_n ~ U(11, 19) (M = 400 / 200 / 40) | rms 0.45 / 0.094 / 0.029 MeV, std post. 0.44 / 0.098 / 0.029; pull +0.15/0.96, +0.07/0.94, −0.00/0.97 | (c) |
| Contrazione con N eventi | std·√N = 1.38 / 0.98 / 0.93 MeV: ≈ 1/√N da N ≈ 100, nessun plateau; a N = 10 il posterior non è ancora gaussiano | (c) |
| Shift sistematico del 2% su E_p, 15 MeV | l'offset sale verso 0.02·E_n = 0.3 MeV (0.17 a N = 100, 0.25 a N = 1000, una realizzazione); σ scende a 0.03 MeV: plateau | (c) |

Lettura: il posterior del Caso A è calibrato per evento e, dalla v0.19, anche combinato. La
sovra-copertura col generatore U(11, 19) non è un difetto: il generatore è più stretto del prior, e
in media la verità sta più vicino al centro del prior di quanto il prior preveda. A 10–20 MeV la
risoluzione per evento su E_n (~1.3 MeV, dominata da σ_θ) è larga rispetto al dominio, per cui
l'attrazione verso il centro è più forte che a bassa energia.

Il prior piatto su θ_p usato fino alla v0.18 era incoerente con la cinematica a ogni energia: con
scattering isotropo in CM cos²θ_p è uniforme, quindi p(θ_p) = sin 2θ_p (a); l'entità del bias a
0.5–6 MeV non è stata misurata. Le checklist precedenti non lo
vedevano perché generavano θ_p ~ U(0, π/2), coerente con quel prior. Generando dalla cinematica
fisica, il prior piatto dà un bias comune a tutti gli eventi che non si riduce con N. Il caso di
riferimento R1 resta definito col prior piatto e il test lo passa esplicitamente.

## 4. Caso B — direzione condivisa (notebook 03)

Verità: Ω_n isotropa; E_n di ogni evento ~ U(10, 20) MeV, cioè distribuita come il suo prior.
Stima = MAP sulla calotta raffinata. Errori in gradi; M = numero di esperimenti.

| N (M) | errore medio | rms errore | risoluzione dichiarata | pull rms | coverage HPD 68/90/95 |
|---|---|---|---|---|---|
| 10 (400) | 4.30 | 4.94 | 4.76 | 1.02 | 0.68 / 0.90 / 0.95 |
| 30 (300) | 2.31 | 2.60 | 2.57 | 1.00 | 0.67 / 0.90 / 0.94 |
| 100 (200) | 1.21 | 1.36 | 1.37 | 0.99 | 0.66 / 0.92 / 0.95 |
| 300 (100) | 0.76 | 0.84 | 0.79 | 1.07 | 0.63 / 0.84 / 0.92 |
| 1000 (40) | 0.35 | 0.37 | 0.43 | 0.85 | 0.80 / 0.98 / 1.00 |

- Calibrato a ogni N entro gli errori binomiali (±0.02 … ±0.07 al 68%) (c). La sovra-copertura a
  N = 1000 con M = 40 è una fluttuazione: con 200 esperimenti in più su seed disgiunti
  (`scripts/n1000_extra.py`, M = 240) errore medio 0.38°, rms 0.42° contro 0.43° dichiarata, pull
  rms 0.98, coverage 0.70 / 0.90 / 0.95 (±0.03 al 68%) (c).
- Contrazione con M = 240 a N = 1000: rms·√N = 0.27 / 0.25 / 0.24 / 0.25 / 0.23 rad per
  N = 10 / 30 / 100 / 300 / 1000, pendenza log-log −0.53 su 10–1000 e −0.51 su 30–1000 (attesa
  −0.5) (c). Il −0.55 della tabella veniva dal punto a M = 40.
- A 10–20 MeV la direzione è ricostruita circa due volte meglio che a 0.5–6 MeV (rms·√N
  0.24 contro 0.38 rad a N = 100) (c): E_p è più grande rispetto a σ_Ep, quindi l'energia per
  evento è meno incerta e pesa meno sulla ricostruzione angolare (interpretazione, d).
- A N = 10 (M = 400) il pull·√2 ha mediana 1.17 (Rayleigh 1.18) e frazione oltre 3 pari a 0.018
  contro 0.011 di Rayleigh: code un po' più pesanti di una gaussiana 2D, meno che a 0.5–6 MeV
  (0.023) (c). Il regime a pochi eventi resta non gaussiano.
- Un evento singolo riproduce la verosimiglianza analitica L ∝ π(E_n*)/cos θ (c).

## 5. Caso C — direzione ed energia gerarchica (notebooks 04, 05)

Ricostruzione a due stadi, senza griglia 4D. Stadio 1: Ω_n come nel Caso B. Stadio 2:
(μ_E, σ_E) su una griglia 2D con Ω_n fissata. Lo stadio 1 viene poi ripetuto con il prior
predittivo, solo per raffinare Ω_n. La verità è generata con μ_E ~ U(13, 17) MeV e σ_E
log-uniforme in (0.8, 2.4) MeV (stessa σ/μ della v0.18, che usava U(2.5, 4) e (0.2, 0.6)).

Le energie per evento sono estratte dalla gaussiana N(μ_E, σ_E) **troncata a [10, 20] MeV**
(`kinematics.sample_truncated_normal_energy`), la stessa densità che il modello gerarchico assume
(v0.21; prima la gaussiana intera, vedi sotto).

**Stato a 10–20 MeV: Caso C calibrato a N ≥ 150; a N = 50 la coverage di log σ_E resta bassa (c).**
Checklist `scripts/caso_C_checklist.py`, v0.21:

| N (M) | bias μ_E [MeV] | rms μ_E (std post.) | pull μ_E | bias log σ_E | pull log σ_E | rms Ω_n [°] (dichiarata) | coverage 68% μ / log σ / Ω |
|---|---|---|---|---|---|---|---|
| 50 (200) | +0.08 ± 0.04 | 0.55 (0.78) | +0.09 / 0.88 | −0.06 ± 0.06 | +0.07 / 0.97 | 1.85 (1.75) | 0.72 / 0.61 / 0.64 |
| 150 (200) | +0.03 ± 0.02 | 0.35 (0.33) | +0.05 / 1.01 | +0.00 ± 0.02 | +0.11 / 0.95 | 0.88 (0.95) | 0.68 / 0.67 / 0.73 |
| 300 (100) | +0.01 ± 0.02 | 0.21 (0.18) | +0.18 / 0.92 | +0.01 ± 0.01 | +0.11 / 0.89 | 0.69 (0.66) | 0.68 / 0.69 / 0.63 |
| 1000 (40) | −0.01 ± 0.01 | 0.092 (0.086) | −0.14 / 1.04 | −0.00 ± 0.01 | +0.00 / 1.00 | 0.34 (0.36) | 0.55 / 0.65 / 0.65 |

Pull: media / larghezza; errore binomiale al 68% ±0.03 (M = 200), ±0.05 (M = 100), ±0.07 (M = 40).

- **Ω_n calibrata** a ogni N: pull rms 0.93–1.07, coverage HPD nei limiti binomiali, pendenza di
  contrazione −0.55 (c).
- **μ_E calibrato**: bias compatibile con zero, larghezza del pull 0.88–1.04, rms e std posteriore
  media vicine a N ≥ 150, coverage 90/95% 0.90–0.91/0.92–0.96. Il 0.55 al 68% a N = 1000 (−1.8
  errori binomiali con M = 40) è una fluttuazione: con M = 240 (`scripts/n1000_extra.py`) coverage
  0.65 / 0.89 / 0.95, pull −0.01 / 1.02, bias −0.002 ± 0.006 MeV, rms 0.096 MeV contro std
  posteriore media 0.088; p-value globale 0.93 sulle quattro statistiche con le loro correlazioni,
  nessuna struttura per terzili di σ_E o μ_E (`scripts/mu_N1000_diag.py`, previsioni scritte prima)
  (c). Con M = 240 anche log σ_E (pull −0.01 / 1.04, coverage 0.66 / 0.88 / 0.94) e Ω_n (pull rms
  1.01, coverage 0.65 / 0.90 / 0.95) sono nominali a N = 1000 (c). Contrazione:
  pendenza −0.61, rms·√N 3.9 / 4.3 / 3.6 / 2.9 MeV, nessun plateau (c).
- **log σ_E calibrato a N ≥ 150**: bias ≤ 0.012, pull medio +0.00 … +0.11 con larghezza 0.89–1.00,
  coverage 0.67–0.69 / 0.87–0.95 / 0.94–0.98 (c). Fra N = 300 e 1000 rms·√N è costante
  (1.96 → 2.10); la pendenza complessiva −0.85 viene da N = 50, dove il posterior non è gaussiano.
- **Residuo a N = 50**: coverage di log σ_E 0.61 / 0.81 / 0.87 (−2.1 / −4.3 / −3.8 errori binomiali)
  con pull medio +0.07 e larghezza 0.97; l'rms dell'errore (0.81) supera la std posteriore media
  (0.66). Gli esperimenti con spettro vicino a un bordo (p_out ≥ 1%, sotto) hanno pull di log σ_E
  +0.52 a N = 50, che scende a +0.26 / +0.21 / +0.00 a N = 150 / 300 / 1000 (c). Andamento opposto a
  quello senza troncatura, dove cresceva con N: compatibile con un effetto di prior a N piccolo,
  come quello visto a 0.5–6 MeV (v0.11) (d). Punto aperto minore.
- **Ricontrolli su seed nuovi** (v0.24, previsioni scritte prima, tutte confermate salvo una).
  `scripts/caso_C_cov68.py` (stadio 2 su Ω̂_n e sulla Ω_n vera): coverage 68 / 90 / 95% di μ_E a
  N = 150 (M = 600) 0.678 / 0.885 / 0.938, a N = 300 (M = 300) 0.683 / 0.890 / 0.933; con la Ω_n
  vera 0.680 / 0.892 / 0.938 e 0.667 / 0.893 / 0.920 (c). `scripts/caso_C_cov90_mu.py` (N = 300,
  M = 600): coverage di μ_E 0.665 / 0.893 / 0.950, pull +0.02 / 1.01; il +0.18 della checklist era
  una fluttuazione (c). `scripts/caso_C_diag_omega_N50.py` (Ω_n a N = 50, tag 50 + 51, M = 1200,
  errore binomiale ×2 ±0.027 / 0.017 / 0.013): prior predittivo su E_n 0.699 / 0.908 / 0.958,
  oracolo N(μ_true, σ_true) 0.673 / 0.898 / 0.955, nominali; plug-in delle stime puntuali
  0.658 / 0.885 / 0.937, sotto il predittivo di 0.04 / 0.02 / 0.02 (5σ sulle differenze appaiate),
  quindi il predittivo resta la scelta giusta (c). Smentita la previsione che il plug-in con σ_E
  allargata di 1 std stesse fra i due: è a 0.704 / 0.916 / 0.958, pari al predittivo. Verità sempre
  dentro la calotta, massa al bordo ≤ 8·10⁻⁹. Sui 200 seed della checklist tutte le varianti con
  prior informativo sottocoprono (predittivo 0.64 / 0.86 / 0.92, oracolo 0.605 / 0.865 / 0.885),
  mentre lo stadio 1 a prior largo è nominale (0.71 / 0.89 / 0.96): fluttuazione comune di quel
  campione, non ripresa su 1200 seed nuovi (c; interpretazione d).
- Robustezza al prior su σ_E (N = 150, uniforme in σ_E contro log σ_E): spostamento medio di μ_E
  +0.01 std (max 0.32), di log σ_E +0.21 std (max 1.08); coverage di log σ_E col prior uniforme
  0.62 / 0.86 / 0.92 (c).

**Prima della correzione (v0.19–v0.20, gaussiana non troncata).** μ_E con larghezza del pull
1.24–1.30 e coverage 68% 0.51–0.55 a N ≥ 300; log σ_E con pull medio crescente con N (+0.10 →
+0.61) e coverage 68% 0.55–0.60 (c). Causa: la frazione dello spettro vero fuori da [10, 20] MeV,
p_out = Φ((10 − μ_E)/σ_E) + 1 − Φ((20 − μ_E)/σ_E), arrivava a Φ(−1.25) + 1 − Φ(2.92) = 0.107 nel
caso peggiore (μ_E = 13, σ_E = 2.4) e superava l'1% in circa un terzo degli esperimenti (a); a
0.5–6 MeV il margine di 3σ_E c'era (μ_E ± 3σ_E ⊂ [0.7, 5.8] ⊂ [0.5, 6]). Con p_out ≥ 1% il pull
medio di log σ_E era +0.55 / +0.79 / +0.84 / +1.36 a N = 50 / 150 / 300 / 1000, con p_out < 1%
−0.08 / −0.07 / +0.12 / +0.20 (c). Troncando solo il generatore la calibrazione torna (tabella
sopra): la causa è confermata (c). Alternative scartate: σ_E ≤ 1 MeV nel generatore (σ_E sotto la
risoluzione per evento su E_n, ~1.3 MeV, quindi un regime meno vincolato) e griglia di E_n più
larga (modifica di stadio 1 e 2, costo doppio). Limite: un rivelatore reale non può troncare
sull'energia vera; uno spettro che sborda dal dominio è un errore di modello, da trattare come
stress test (come gli spettri non gaussiani di §6.2), non come calibrazione.

La tabella e le verifiche della v0.18 (0.5–6 MeV) restano valide per quel dominio e sono nella
storia git di questo file.

## 6. Stress test delle assunzioni

Stessi dataset delle checklist (stessi seed): nella configurazione di riferimento (prior di
default, spettro gaussiano, nessuna contaminazione) si ritrovano esattamente i loro numeri, e
questo fa da controllo di consistenza. Si cambia una cosa per volta; la ricostruzione resta
quella standard.

### 6.1 Robustezza al prior (`scripts/robustezza_prior.py`)

Metrica: spostamento della stima in unità della std posteriore del prior di default, e coverage
col prior alternativo. Prior alternativi, tutti propri: E_n piatto su [10, 30] MeV ("largo"),
log-uniforme su [10, 20] e su [10, 30] MeV; Ω_n von Mises–Fisher con κ = 2 (larga ~40°) e asse a
90° dalla verità; σ_E uniforme in σ invece che in log σ. Numeri a 10–20 MeV, rilancio v0.25 col
Caso C generato dalla gaussiana troncata al dominio (A e B uguali alla v0.19); nel Caso A la
verità è generata dalla cinematica in CM e il prior su θ_p è quello fisico.

| Caso, parametro | Prior alternativo | Spostamento medio / std (N crescente) | Coverage 68% | |
|---|---|---|---|---|
| A, E_n | piatto largo | +0.79 / +0.36 / +0.09 / +0.05 / +0.03 (N = 1 / 3 / 10 / 30 / 100) | 0.66–0.70 | (c) |
| A, E_n | log-uniforme | −0.12 / −0.06 / −0.03 / −0.01 / −0.01 | 0.65–0.70 | (c) |
| B, Ω_n | von Mises–Fisher a 90° | 0.11 / 0.05 / 0.03 / 0.03 (N = 10 / 30 / 100 / 300) | 0.60–0.70 | (c) |
| B, Ω_n | E_n piatto largo | 0.78 / 0.74 / 0.72 / 0.75 | 0.62–0.73 | (c) |
| B, Ω_n | E_n log-uniforme | 0.13 / 0.11 / 0.12 / 0.08 | 0.56–0.69 | (c) |
| C, μ_E | σ_E uniforme | −0.04 / +0.01 / −0.01 / −0.00 (N = 50 / 150 / 300 / 1000) | 0.76 / 0.69 / 0.68 / 0.55 (default 0.72 / 0.68 / 0.68 / 0.55) | (c) |
| C, log σ_E | σ_E uniforme | +0.76 / +0.26 / +0.13 / +0.07 | 0.59 / 0.63 / 0.68 / 0.63 (default 0.61 / 0.67 / 0.69 / 0.65) | (c) |

- Il prior sui **parametri condivisi** viene dimenticato con N, come a 0.5–6 MeV (c). Nel Caso A
  col prior largo restano spostamenti massimi di 2–4 σ a N = 10–30, negli esperimenti con verità
  vicina a 20 MeV, dove il prior di default tronca il posterior; la risoluzione per evento (~1.3 MeV)
  è larga rispetto al dominio, quindi la troncatura conta più che a bassa energia.
- Il prior sul **nuisance per evento** (E_n nel Caso B) non viene dimenticato: Ω̂_n si sposta di
  ~0.75 σ (prior largo) o ~0.1 σ (log-uniforme) a ogni N (c), meno che a 0.5–6 MeV (~1.1 e ~0.3 σ).
  Col prior largo l'errore rms cresce (7.1° contro 4.8° a N = 10, 1.05° contro 0.87° a N = 300),
  pull rms 0.94–1.00: costa precisione, non calibrazione (c).
- **Analisi tipo Hausman** (`scripts/prior_B_rumore.py`, v0.25, previsioni scritte prima). Lo
  spostamento d = stima_alt − stima_def è rumore a media nulla: ⟨d⟩ nel piano tangente alla verità
  è compatibile con zero entro 2σ in tutte le 16 componenti (c). Col prior largo la relazione
  E|err_alt|² = E|err_def|² + E|d|² regge già da N = 10: rms|d|/σ osservato contro previsto
  1.00 ± 0.08 / 1.08 ± 0.06, 0.86 ± 0.04 / 0.93 ± 0.07, 0.83 ± 0.05 / 0.82 ± 0.08,
  0.83 ± 0.05 / 0.75 ± 0.13 a N = 10 / 30 / 100 / 300, correlazione err_def·d compatibile con
  zero (c; a 0.5–6 MeV reggeva da N = 30). Col prior log-uniforme lo spostamento (0.14–0.16 σ) è
  troppo piccolo perché il test abbia potere (rapporto delle rms 1.00–1.02, previsto con errore
  0.05–0.08): compatibile a N = 10, 100, 300, fuori di 3σ a N = 30 (0.143 ± 0.006 contro
  0.000 ± 0.047) (c). Smentita la previsione che fallisse a N = 10 con correlazione negativa
  (−0.06 ± 0.06).
- Col prior largo il rapporto rms(err_alt)/rms(err_def) **scende** con N, 1.46 / 1.37 / 1.32 / 1.21
  (c), mentre a 0.5–6 MeV restava a 1.6–1.75: qui lo spostamento ~costante non viene da un rapporto
  costante. Lo spostamento osservato scende da 1.00 a 0.83 σ fra N = 10 e 30 e poi resta fermo,
  mentre Hausman col rapporto calante lo darebbe a 0.75 σ a N = 300; i due sono compatibili
  (0.6σ, M = 50). Se resti costante a N più grande è aperto (d).
- σ_E: lo spostamento di log σ_E col prior uniforme scende da +0.76 σ (N = 50) a +0.07 σ
  (N = 1000) (c), più forte che a 0.5–6 MeV (0.32 σ a N = 50) e uguale al run senza troncatura:
  dipende dal prior, non dalla calibrazione del generatore. Coverage 68% col prior uniforme sotto
  il default di 0.01–0.05, oltre l'errore binomiale a N = 150 (0.625 contro 0.67); il pull medio
  col prior uniforme è +0.54 / +0.27 / +0.23 / +0.07 (c).
- **Rilancio v0.25** col generatore troncato, previsioni scritte prima. Controllo di consistenza:
  B coincide bit per bit col run precedente, A entro 1.8·10⁻⁶ relativo (arrotondamento float32
  dell'ottimizzazione di `loglik`, successiva al run v0.19; uguale alle cifre stampate); il Caso C
  col prior log-uniforme coincide bit per bit con la checklist v0.21 sui 540 esperimenti (c).
  Confermati: spostamento di μ_E < 0.05 std a ogni N, coverage di μ_E col prior uniforme uguale al
  default a N ≥ 150, spostamento di log σ_E positivo e calante, < 0.1 std a N = 1000. Smentiti:
  (i) a N = 50 il prior uniforme sovra-copre μ_E (0.76 contro 0.715; std 1.50 volte quella del
  default, larghezza del pull 0.71), perché allarga σ_E e con essa l'incertezza su μ_E (c);
  (ii) la troncatura non riduce lo spostamento di log σ_E a N = 50 (+0.757 contro +0.76). Il +0.21
  di §5 a N = 150 è lo spostamento della **mediana**; qui si usa la media (+0.26) sugli stessi
  posterior: la previsione confrontava due metriche diverse.

### 6.2 Spettro non gaussiano, Caso C (`scripts/stress_spettro.py`)

Valori a 10–20 MeV (v0.22). Energie vere da spettri con la stessa media e sd nominali di
N(μ_E, σ_E) ma forma uniforme, bimodale (due righe a ±0.9 sd), lognormale, Laplace o Student-t con
5 gradi di libertà, tutte troncate a [10, 20] MeV; il modello resta gaussiano. Bersagli: per le
forme non gaussiane i parametri della gaussiana troncata con la stessa media e sd effettive dello
spettro (pseudo-vero). La gaussiana troncata è una famiglia esponenziale con statistiche x e x²,
quindi senza rumore di misura il massimo di verosimiglianza sotto spettro errato eguaglia i primi
due momenti (a); con il rumore è un'approssimazione (b). I momenti effettivi vengono da un campione
ausiliario di 2·10⁶ eventi. A pari seed la `"gauss"` coincide con la checklist del Caso C (`mu_true`,
errore e sigma di Ω_n identici esperimento per esperimento).

| Forma | μ_E: pull medio, tutti gli N (M = 540) | Differenza da gauss a pari verità | log σ_E: larghezza del pull e coverage 68% a N = 150 / 300 | |
|---|---|---|---|---|
| gaussiana | +0.074 ± 0.041 | — | 0.95 / 0.89; 0.67 / 0.69 | (c) |
| lognormale | −0.018 ± 0.040 | −0.092 ± 0.055 | 1.02 / 1.00; 0.64 / 0.66 | (c) |
| uniforme | +0.056 ± 0.042 | −0.017 ± 0.015 | 0.84 / 0.84; 0.70 / 0.76 | (c) |
| bimodale | +0.012 ± 0.039 | −0.062 ± 0.052 | 0.81 / 0.92; 0.75 / 0.63 | (c) |
| Laplace | +0.119 ± 0.043 | +0.045 ± 0.041 | 1.15 / 0.99; 0.55 / 0.67 | (c) |
| Student-t (ν = 5) | +0.073 ± 0.040 | −0.001 ± 0.056 | 1.04 / 1.06; 0.64 / 0.64 | (c) |

- Ω_n non risente della forma dello spettro (pull rms 0.92–1.14, c).
- μ_E è robusto alla forma: nessuna forma si distingue dalla gaussiana a pari verità (c). Il pull
  +0.16 ± 0.04 della Student-t misurato a 0.5–6 MeV senza troncatura non si ripresenta; quel run
  differiva per generatore, bersaglio e dominio, quindi la causa non è attribuita a uno solo dei tre.
- Resta un piccolo pull positivo comune: +0.053 ± 0.024 in media sulle sei forme (2.2σ), sotto la
  soglia di 0.2σ. Origine non indagata (d).
- log σ_E: la varianza intrinseca segue (κ − 1)/(4N) con la curtosi effettiva κ (`scripts/curtosi_log_sigma.py`,
  c). La troncatura abbassa κ: le code pesanti sono meno pesanti di quanto dicano i valori nominali
  6 e 9. La Laplace resta sotto-coperta e con pull più largo del previsto dal solo termine intrinseco:
  1.15 ± 0.05 contro 0.98 a N = 150, 1.24 ± 0.15 contro 1.05 a N = 1000 (c); la Student-t 1.17 ± 0.12
  contro 1.04 a N = 1000 (c, M = 40). Causa dell'eccesso non indagata (d).
- Uniforme e bimodale a N = 150: bias di log σ_E +0.046 ± 0.016 e +0.053 ± 0.016, con pull stretti.
  Coerente con l'approssimazione dello pseudo-vero in presenza di rumore, non verificato (d).
- Coverage di log σ_E a N = 50 per la gaussiana: 0.61 / 0.81 / 0.87, come nella checklist; ipotesi
  dell'effetto del prior ancora da testare (d).

**Sconfinamento dal dominio (`scripts/stress_sconfinamento.py`, v0.22).** Stessi esperimenti, ma
energie non troncate (forme con eventi fuori da [10, 20] MeV), fit invariato, bersagli media e sd
nominali. Il fit integra E_n solo sul dominio, quindi questi eventi sono descritti male.

| Forma | μ_E: rms troncato → sconfinante a N = 1000 (MeV) | coverage 68% di μ_E | pull medio di log σ_E | |
|---|---|---|---|---|
| gaussiana | 0.092 → 0.178 | 0.55 | +0.61 | (c) |
| lognormale | 0.084 → 0.169 | 0.48 | +0.59 | (c) |
| uniforme | 0.123 → 0.237 | 0.38 | +0.82 | (c) |
| bimodale | 0.129 → 0.318 | 0.42 | +0.99 | (c) |
| Laplace | 0.098 → 0.143 | 0.55 | −0.02 | (c) |
| Student-t | 0.074 → 0.197 | 0.50 | −0.15 | (c) |

- M = 40 a N = 1000: errore binomiale sulla coverage ±0.074.
- Il danno cresce con N: a N = 50 è quasi invisibile (gaussiana, rms 0.553 → 0.595), a N = 150 è già
  0.350 → 0.516 e a N = 300 0.207 → 0.470 (c). L'errore non si riassorbe con la statistica: la std
  posteriore si restringe, l'errore molto meno. L'assunzione sul dominio va quindi trattata come
  sistematico, e **pesa di più proprio dove il campione è grande**.
- μ_E non acquista un bias medio significativo: cresce la dispersione. log σ_E invece sì, per le forme
  a code leggere. Ω_n resta calibrato (pull rms 0.94–0.98 a N = 1000), tranne la Student-t (1.14).
- Perché la dispersione di μ_E raddoppi non è stato indagato. Ipotesi: pesano pochi esperimenti con
  μ_E vicino ai bordi o σ_E grande (d).

### 6.3 Sorgente unica, Casi B e C (`scripts/stress_sorgente.py`)

Rifatto a 10–20 MeV (v0.26), col Caso C generato dalla gaussiana troncata. Una frazione f di
eventi viene da una seconda sorgente puntiforme a separazione Δ (10°, 30°, 90°) o da un fondo
isotropo; la ricostruzione resta a sorgente unica. Diagnostica: test KS fra gli angoli osservati
delle tracce rispetto a Ω̂_n e la loro distribuzione attesa con sorgente unica, con soglia al 95°
percentile della statistica a f = 0 (Ω̂_n viene dagli stessi dati). M = 100 per configurazione,
200 a f = 0. Le righe a f = 0 coincidono esperimento per esperimento con le checklist B
(N = 30, 300) e C (N = 50, 300) (c); gli aggregati differiscono di poco perché M è diverso.

Caso B, N = 300 (σ posteriore 0.79°, contro 1.23° a 0.5–6 MeV); fra parentesi il valore a 0.5–6 MeV:

| Contaminazione | Spostamento verso la 2ª sorgente | pull rms | Coverage 68% | Potenza KS | |
|---|---|---|---|---|---|
| nessuna | — | 1.05 | 0.66 | 0.05 (= soglia) | (c) |
| f = 0.05, 10° | 0.5° (0.5°) | 1.28 (1.15) | 0.44 (0.53) | 0.06 | (c) |
| f = 0.10, 10° | 1.0° | 1.70 | 0.28 | 0.05 | (c) |
| f = 0.30, 10° | 3.0° (3.0°) | 4.12 (2.74) | 0.00 (0.03) | 0.05 | (c) |
| f = 0.05, 30° | 1.8° (2.1°) | 2.71 (2.28) | 0.07 (0.16) | 0.03 | (c) |
| f = 0.10, 30° | 3.6° (4.0°) | 4.88 (3.76) | 0.00 (0.01) | 0.06 | (c) |
| f = 0.30, 30° | 9.8° (10.1°) | 13.5 | 0.00 | 0.05 | (c) |
| f = 0.05, 90° | 5.6° (10.3°) | 7.5 (9.5) | 0.00 (0.00) | 0.10 (0.17) | (c) |
| f = 0.10, 90° | 10.6° (16.9°) | 14.2 (16.7) | 0.00 (0.00) | 0.62 (0.80) | (c) |
| f = 0.30, 90° | 27.5° (33.1°) | 38.5 | 0.00 | 1.00 | (c) |
| f = 0.05 isotropo | rms 2.6° (5.0°) | 3.2 (4.1) | 0.08 (0.03) | 0.10 (0.14) | (c) |
| f = 0.10 isotropo | rms 3.5° (7.1°) | 4.3 (5.8) | 0.04 (0.01) | 0.40 (0.48) | (c) |
| f = 0.30 isotropo | rms 7.0° (12.5°) | 7.9 | 0.02 | 1.00 | (c) |

Caso B, N = 30 (σ posteriore 2.6°): f = 0.05 a 30° sposta Ω̂_n di 2.3°, coverage 68% 0.28 (0.41 a
0.5–6 MeV); f = 0.05 a 90° di 7.1°, coverage 0.01; KS ≤ 0.12 fino a f = 0.1, 0.55 / 0.49 a f = 0.3
(90° / isotropo) (c).

Previsioni scritte prima del run (docstring dello script), confrontate col risultato:

- **Consistenza: confermata (c).** Righe a f = 0 identiche bit per bit alle checklist.
- **Meccanismo invariato in gradi: confermato a 10° e 30°, smentito a 90° e col fondo isotropo (c).**
  Lo spostamento è ≈ f·Δ a 10° (0.5° / 1.0° / 3.0°) e un po' di più a 30° (1.8° / 3.6° / 9.8°,
  leggermente meno che a 0.5–6 MeV). A 90° e col fondo isotropo è invece molto più piccolo che a
  0.5–6 MeV: 5.6° contro 10.3° a f = 0.05, rms del fondo isotropo 2.6° contro 5.0°. Ipotesi (d): a
  10–20 MeV l'intervallo relativo di E_n è più stretto (fattore 2 contro 12), quindi un evento da
  una direzione molto diversa è meno compatibile con l'asse ricostruito e pesa meno nella media.
- **σ più piccola ⇒ pull più grandi e coverage più bassa: confermata a 10° e 30° (c).** f = 0.05 a 30°
  porta la coverage 68% a 0.07 (prevista sotto 0.16). A 90° i pull sono più piccoli che a 0.5–6 MeV,
  perché lo spostamento cala più della σ.
- **KS cieco per Δ ≤ 30°: confermata (c).** Potenza ≤ 0.09 nel Caso B a ogni f e N; nel Caso C
  0.11 a N = 50 con f = 0.3 a 30° (al margine, errore binomiale ±0.03). A 90° f = 0.05 resta
  invisibile (0.07–0.10); la potenza diventa utile da f = 0.1 a N = 300 (0.52–0.62) e a f = 0.3.
  Il motivo proposto (d) è che la distribuzione degli angoli delle tracce è larga (θ_p fino a
  90°): spostare l'asse di pochi gradi la cambia poco.
- **Caso C, σ_E il più sensibile e μ_E spinto in alto: confermata (c), con un effetto nuovo.**
  A N = 300 con f = 0.05 a 30°: bias di log σ_E +0.42, pull +2.88, coverage 68% 0.05 (prima +3.95 /
  0.06); Ω_n coverage 0.04; μ_E +0.11 MeV (pull +0.49). A 10° μ_E quasi non si muove (≤ +0.05 MeV).
  Per Δ = 90° e fondo isotropo μ_E sale più che a 0.5–6 MeV: +0.82 MeV a f = 0.05 e 90° (prima
  +0.32), +1.14 MeV a f = 0.10 isotropo (prima +0.36), coverage 68% 0.26; +3.9 / +3.7 MeV a f = 0.3.
  A N = 50 lo stesso andamento con numeri più piccoli: log σ_E +0.55 a f = 0.05 e 30° (coverage
  0.39), μ_E +0.56 MeV a f = 0.05 e 90°. Gli eventi da un'altra direzione vengono ricostruiti con
  l'angolo sbagliato, quindi con energie disperse e in media più alte (d). Un σ_E inatteso è un
  sintomo utile solo se si ha un'aspettativa indipendente sulla larghezza dello spettro (d).
- **Effetto nuovo, collasso al bordo del dominio (c, causa d).** Con contaminazione forte μ_E viene
  spinto verso il bordo superiore, 20 MeV (μ̂_E > 19.5 MeV in 60 esperimenti su 100 a N = 300 con
  f = 0.3 a 90° o isotropo). In 12 esperimenti su 2800 del Caso C il posterior collassa sullo
  spigolo della griglia (μ_E = 20 MeV, σ_E = 0.01 MeV) con std ~10⁻⁹–10⁻⁴ MeV: 6 a N = 300 e 4 a
  N = 50 col fondo isotropo a f = 0.3, 2 a N = 50 con Δ = 90°. Quei pochi rendono privi di senso
  la media e la larghezza del pull stampate (fino a 10⁸), mentre la coverage resta valida. Il pull
  mediano di μ_E è 0.37 / 0.75 / 1.06 / 12.4 a (N = 50, f = 0.1, 90°) / (50, 0.3, 90°) /
  (50, 0.3, isotropo) / (300, 0.3, isotropo). A 0.5–6 MeV non c'erano collassi: il dominio
  stretto accumula al bordo le energie alte delle tracce ricostruite male (d).
- Conclusione: l'assunzione 1 resta necessaria e, per separazioni ≲ 30°, non è verificabile con
  questa diagnostica. A 10–20 MeV il danno a pari f e Δ ≤ 30° è maggiore in σ (σ più piccola),
  minore per sorgenti lontane. Un modello a mistura che stimi f è il passo naturale successivo
  (fuori scope).

### 6.4 Due stadi contro stima congiunta, Caso C (`scripts/correlazione_stadi.py`)

Sugli stessi dataset della checklist del Caso C (M = 100 per N, 10–20 MeV, generatore troncato), si
parte dalle stime dei due stadi e si calcola l'approssimazione di Laplace del posterior congiunto 4D
(Ω_n nel piano tangente, μ_E, log σ_E): stencil 3^4 di differenze centrali, passi pari alle
larghezze dei due stadi, due passi di Newton, nessuna griglia 4D. L'Hessiana è definita negativa in
279 esperimenti su 300: 100 a N = 300 e 150, solo 79 a N = 50 (a 0.5–6 MeV erano 98). Gli altri
sono esclusi.

| N | Correlazione canonica Ω–(μ, log σ): mediana / 90% / max | Allargamento marg./cond. medio (max): μ / log σ / Ω | Laplace / griglia: μ / log σ / Ω | Modo congiunto − due stadi, rms in σ: μ / log σ / Ω |
|---|---|---|---|---|
| 50 | 0.17 / 0.28 / 0.39 | 1.015 (1.082) / 1.008 (1.062) / 1.011 (1.048) | 0.71 / 0.91 / 0.93 | 0.27 / 0.49 / 0.20 |
| 150 | 0.10 / 0.17 / 0.24 | 1.004 (1.019) / 1.004 (1.023) / 1.004 (1.015) | 0.83 / 0.89 / 0.98 | 0.20 / 0.27 / 0.11 |
| 300 | 0.08 / 0.12 / 0.16 | 1.003 (1.011) / 1.002 (1.013) / 1.002 (1.008) | 0.93 / 0.95 / 0.99 | 0.13 / 0.19 / 0.12 |

Pull (media / larghezza; Ω: rms):

| N | Due stadi: μ / log σ / Ω | Modo congiunto: μ / log σ / Ω |
|---|---|---|
| 50 | +0.01/1.02 · −0.18/0.82 · 1.10 | −0.11/1.27 · −0.56/0.89 · 1.18 |
| 150 | +0.17/1.00 · +0.05/0.86 · 0.97 | +0.16/1.09 · −0.11/0.93 · 0.99 |
| 300 | +0.18/0.92 · +0.11/0.89 · 1.04 | +0.17/0.98 · −0.03/0.91 · 1.03 |

- La correlazione posteriore fra la direzione e gli iperparametri è debole e scende con N: la
  mediana passa da 0.17 a 0.08, come a 0.5–6 MeV (c). Ignorarla, cioè fissare un blocco invece di
  marginalizzarlo, stringe le incertezze in media dell'1.5% a N = 50 e dello 0.2–0.3% a N = 300; il
  caso peggiore è l'8% (μ_E a N = 50) (c). **La fattorizzazione a due stadi resta giustificata** (c).
- Le larghezze di Laplace concordano con le griglie dei due stadi entro il 5–7% a N = 300, ma non a
  N piccolo: su μ_E il rapporto è 0.71 a N = 50 e 0.83 a N = 150 (c). A N = 50 la std posteriore
  di log σ_E (0.54 in media) è circa metà della larghezza del prior (log 3 ≈ 1.1): il posterior non
  è gaussiano e la Laplace locale non lo descrive. Le stime dei due stadi, su griglia, non hanno
  questo problema.
- Il modo congiunto si discosta dalle stime dei due stadi soprattutto su log σ_E (0.5 σ a N = 50,
  0.2 σ a N = 300) e ha pull medio più basso della media dei due stadi: −0.56 / −0.11 / −0.03 contro
  −0.18 / +0.05 / +0.11 (c). La relazione di Pearson moda ≈ media − 3(media − mediana)
  (`scripts/asimmetria_log_sigma.py`) ha qui **lo stesso segno** dell'osservato, al contrario di
  0.5–6 MeV: il marginale ha la coda verso σ_E grandi. Ma ne spiega solo una parte: −0.04 / −0.09 /
  −0.05 σ previsti contro −0.35 / −0.22 / −0.16 osservati a N = 50 / 150 / 300, con correlazione per
  esperimento 0.46 / 0.65 / 0.70 (c). La previsione scritta prima (segno opposto, correlazione < 0.5)
  è smentita. La scomposizione esatta (`scripts/modo_log_sigma.py`, stessi dataset, in unità della
  std posteriore s; M = 77 / 100 / 100) dà:

  | N | asimmetria (modo marg. − media) | volume di μ_E (modo 2D − modo marg.) | direzione (modo 4D − modo 2D) | numerica (Laplace d'uso − convergente) | totale |
  |---|---|---|---|---|---|
  | 50 | −0.05 ± 0.04 | −0.17 ± 0.03 | −0.18 ± 0.06 | +0.05 ± 0.05 | −0.35 ± 0.04 |
  | 150 | −0.06 ± 0.02 | −0.11 | −0.06 ± 0.01 | +0.00 ± 0.01 | −0.22 ± 0.02 |
  | 300 | −0.04 ± 0.01 | −0.08 | −0.05 ± 0.01 | +0.01 | −0.16 ± 0.01 |

  Il totale coincide con la somma dei termini. A differenza di 0.5–6 MeV, dove asimmetria (+) e
  volume di μ_E (−) quasi si cancellavano e dominava la direzione, qui i tre termini fisici sono
  tutti negativi e a N ≥ 150 domina il **volume di μ_E** (c); la previsione scritta prima (segni
  del dominio precedente, direzione dominante) è smentita. Da N = 150 a 300 lo scarto/s² resta
  costante entro un fattore 1.3 (totale −1.13 → −1.32), cioè i termini scalano circa come s², come
  un effetto di volume (c); a N = 50 non più (−0.76), dove il posterior tocca i bordi del prior
  (interpretazione, d). Il gradiente numerico di `joint_laplace` è trascurabile (≤ 0.05 s).
  Pull medio delle stime puntuali di log σ_E: media due stadi −0.17 / +0.05 / +0.11, Laplace
  convergente −0.57 / −0.17 / −0.06. A N ≤ 150 la stima a due stadi (media) è migliore; a N = 300
  le due sono equivalenti entro l'errore (±0.09) (c).

### 6.5 Isotropia in CM, Casi A, B e C (`scripts/stress_anisotropia.py`)

Generazione con la distribuzione angolare n–p vera (coefficienti di Legendre di H-1, ENDF/B-VIII.0,
10–20 MeV), ricostruzione isotropa. Esperimenti appaiati: stesso seme e stesso flusso RNG con
generazione isotropa e anisotropa, quindi la differenza isola l'effetto dell'anisotropia.
Criterio fissato prima: trascurabile se il bias aggiunto è < 0.2 std posteriore al N massimo e la
coverage resta nei limiti binomiali.

| Caso, parametro | Bias aggiunto (aniso − iso), in std post. | Effetto su pull / coverage | |
|---|---|---|---|
| A, E_n | 0.01 / 0.04 / 0.12 (N = 10 / 100 / 1000) | nessuno (pull −0.00 → +0.11 a N = 1000, larghezza 0.97) | (c) |
| B, Ω_n | — (simmetria azimutale) | pull rms e coverage invariati a ogni N | (c) |
| C, μ_E | 0.02 / 0.04 / 0.08 / 0.15 (N = 50 / 150 / 300 / 1000) | invariati entro gli errori | (c) |
| C, log σ_E | 0.02 / 0.00 / 0.00 / 0.01 | invariati | (c) |
| C, Ω_n | — | invariati | (c) |

Verdetto: l'anisotropia vera a 10–20 MeV non cambia la ricostruzione al livello del toy (c), e il
modello resta isotropo. Il bias su μ_E cresce con N (0.15 σ a N = 1000, vicino alla soglia): oltre
N ≈ 1000 va ricontrollato o messo nella verosimiglianza (a N = 1000 il bias aggiunto è
0.0128 ± 0.0008 MeV).

Rilancio v0.26 con i tre casi e il Caso C generato dalla gaussiana troncata; le previsioni scritte
prima (docstring dello script) sono tutte confermate (c). Il Caso A riproduce il run precedente
entro 10⁻⁶ MeV; le righe isotrope di B e C coincidono bit per bit con le checklist (stessi seed),
quindi la ricostruzione isotropa resta calibrata e la differenza appaiata isola l'anisotropia.
Il bias aggiunto su μ_E (0.08 σ a N = 300 contro 0.07 previsto) è quello del run col generatore
non troncato: la troncatura non interagisce con l'anisotropia.

## 7. Prestazioni

Tempi di `loglik` del Caso A misurati a 10–20 MeV (cinematica relativistica, buffer unico riusato
in place); gli altri a 0.5–6 MeV, salvo la checklist del Caso C (~7 min con 3 processi) e gli
script marcati 10–20 MeV.

| Operazione | Tempo | |
|---|---|---|
| `loglik` Caso A, 1 evento | 0.67 ms (target < 1 ms) | (c) |
| `loglik` Caso A, 1000 eventi | 0.71 s (target ridefinito a 2 s) | (c) |
| Esperimento Caso C a N = 300 | ~2 s (prima 19 s) | (c) |
| Checklist Caso C completa | ~8 min con 3 processi | (c) |
| `n1000_extra.py` (B e C, 400 esperimenti a N = 1000), 10–20 MeV | ~14 min con 3 processi | (c) |
| `correlazione_stadi.py` / `modo_log_sigma.py` (300 esperimenti), 10–20 MeV | ~19 min ciascuno con 3 processi | (c) |
| `caso_C_cov68.py` (900) / `caso_C_cov90_mu.py` (600), 10–20 MeV | ~12 / ~11 min con 3 processi | (c) |
| `caso_C_diag_omega_N50.py`, M = 200 / 600, 10–20 MeV | ~1.5 / ~4.5 min con 3 processi | (c) |
| `robustezza_prior.py` (A, B, C), 10–20 MeV | ~11 min con 3 processi | (c) |
| `stress_anisotropia.py` (A, B, C, esperimenti appaiati), 10–20 MeV | ~18 min con 3 processi | (c) |
| `stress_sorgente.py` (B e C, 5600 esperimenti), 10–20 MeV | ~40 min con 3 processi | (c) |

## 8. Limiti dichiarati

- **Assunzione 1 (sorgente unica): necessaria e non verificabile per sorgenti vicine (c).** Una
  contaminazione del 5% a 30° porta la coverage 68% di Ω_n da 0.66 a 0.07 a N = 300 e peggiora con
  N; la diagnostica KS sugli angoli delle tracce non la vede per Δ ≲ 30° (§6.3). Nel Caso C una
  contaminazione forte spinge μ_E al bordo superiore del dominio, fino al collasso del posterior
  sullo spigolo della griglia (12 esperimenti su 2800, c). Un modello a mistura è fuori scope.
- **Assunzione 3 (spettro gaussiano):** μ_E e Ω_n robusti alla forma (c); la coverage di log σ_E
  dipende dalla curtosi κ dello spettro (c, κ da 1.7 a 9, §6.2): sovra-coperto per κ < 3,
  sotto-coperto per code pesanti (coverage 68% 0.55–0.67, larghezza del pull fino a 1.24 a
  N = 1000). Pull medio comune di μ_E +0.05 σ, sotto soglia (2.2σ, causa d).
- **Dominio di energia:** se lo spettro vero esce da [10, 20] MeV la calibrazione di μ_E e log σ_E
  si degrada con N (coverage 68% di μ_E 0.38–0.55 a N = 1000, §6.2, c). Ω_n resta calibrato.
- **Prior sul nuisance per evento (Caso B):** il prior su E_n non viene dimenticato al crescere di N
  (spostamento di Ω̂_n ~0.75 σ a ogni N con un prior largo, c). Lo spostamento è rumore a media
  nulla che si somma in quadratura all'errore (relazione tipo Hausman, c da N = 10, §6.1): costa
  precisione (errore rms ×1.46 a N = 10, ×1.21 a N = 300), non calibrazione. Va scelto sullo
  spettro fisico atteso.
  Nel Caso C log σ_E dipende dal prior di 0.76 σ a N = 50, 0.13 σ a N = 300 e 0.07 σ a N = 1000
  (c, §6.1).
- **Caso C, due stadi:** chiuso come limite. La correlazione fra Ω_n e (μ_E, σ_E) trascurata dai due
  stadi costa ≤ 1.5% sulle incertezze in media (≤ 8% nel caso peggiore) a N ≥ 50 (c, §6.4).
- **Nuisance z** (profondità d'interazione) del Caso A fuori scope: nessun osservabile definito.
- **Isotropia in CM** a 10–20 MeV: la distribuzione angolare vera (ENDF/B-VIII.0) non cambia la
  ricostruzione (c, §6.5); il modello resta isotropo.
- **Calibrazione del Caso C a 10–20 MeV:** raggiunta a N ≥ 150 col generatore troncato al dominio
  (c, §5); a N = 50 la coverage di log σ_E resta bassa (0.61 / 0.81 / 0.87), probabile effetto di
  prior (d). Uno spettro vero che sborda da [10, 20] MeV (fino al 10.7% senza troncatura) è un
  errore di modello che sposta log σ_E verso l'alto, crescente con N (c, §5).
- **Fuori scope:** sezione d'urto n–p decrescente con E_n (spettro rivelato ≠ spettro incidente),
  soglia su E_p basse.
- I limiti sulle assunzioni 1 e 3 e sui due stadi sono quantificati a 10–20 MeV (§6.2–§6.4).
- **Target di prestazione** di `loglik` per N = 1000 ridefinito da 50 ms a 2 s: il limite è il numpy denso.
