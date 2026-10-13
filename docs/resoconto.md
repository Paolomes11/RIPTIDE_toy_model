# Resoconto dei risultati — riptide-toy

v0.21 — 2026-10-13

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

- Calibrato a ogni N entro gli errori binomiali (±0.02 … ±0.07 al 68%) (c). A N = 1000 con M = 40
  pull e coverage sono quelli già visti a 0.5–6 MeV con lo stesso M, rientrati al nominale con
  M = 240; il controllo con più esperimenti (`scripts/n1000_extra.py`) non è stato rifatto a
  10–20 MeV.
- Contrazione: rms·√N = 0.27 / 0.25 / 0.24 / 0.25 / 0.20 rad per N = 10 / 30 / 100 / 300 / 1000,
  pendenza log-log −0.55 (attesa −0.5) (c).
- A 10–20 MeV la direzione è ricostruita circa due volte meglio che a 0.5–6 MeV (rms·√N
  0.24 contro 0.38 rad a N = 100) (c): E_p è più grande rispetto a σ_Ep, quindi l'energia per
  evento è meno incerta e pesa meno sulla ricostruzione angolare (interpretazione, d).
- Le code pesanti del pull a N = 10 (frazione oltre 3) sono state misurate a 0.5–6 MeV e non
  rifatte.
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
  media vicine a N ≥ 150, coverage 90/95% 0.90–0.91/0.92–0.96. Il 0.55 al 68% a N = 1000 è a
  −1.8 errori binomiali con M = 40, con pull di larghezza 1.04: fluttuazione (c). Contrazione:
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
90° dalla verità; σ_E uniforme in σ invece che in log σ. Numeri v0.19, 10–20 MeV; nel Caso A la
verità è generata dalla cinematica in CM e il prior su θ_p è quello fisico.

| Caso, parametro | Prior alternativo | Spostamento medio / std (N crescente) | Coverage 68% | |
|---|---|---|---|---|
| A, E_n | piatto largo | +0.79 / +0.36 / +0.09 / +0.05 / +0.03 (N = 1 / 3 / 10 / 30 / 100) | 0.66–0.70 | (c) |
| A, E_n | log-uniforme | −0.12 / −0.06 / −0.03 / −0.01 / −0.01 | 0.65–0.70 | (c) |
| B, Ω_n | von Mises–Fisher a 90° | 0.11 / 0.05 / 0.03 / 0.03 (N = 10 / 30 / 100 / 300) | 0.60–0.70 | (c) |
| B, Ω_n | E_n piatto largo | 0.78 / 0.74 / 0.72 / 0.75 | 0.62–0.73 | (c) |
| B, Ω_n | E_n log-uniforme | 0.13 / 0.11 / 0.12 / 0.08 | 0.56–0.69 | (c) |
| C, μ_E | σ_E uniforme | −0.02 / +0.01 / −0.00 / +0.00 (N = 50 / 150 / 300 / 1000) | 0.50–0.75, uguale al default a N ≥ 150 | (c) |
| C, log σ_E | σ_E uniforme | +0.76 / +0.28 / +0.14 / +0.07 | 0.50–0.60 | (c) |

- Il prior sui **parametri condivisi** viene dimenticato con N, come a 0.5–6 MeV (c). Nel Caso A
  col prior largo restano spostamenti massimi di 2–4 σ a N = 10–30, negli esperimenti con verità
  vicina a 20 MeV, dove il prior di default tronca il posterior; la risoluzione per evento (~1.3 MeV)
  è larga rispetto al dominio, quindi la troncatura conta più che a bassa energia.
- Il prior sul **nuisance per evento** (E_n nel Caso B) non viene dimenticato: Ω̂_n si sposta di
  ~0.75 σ (prior largo) o ~0.1 σ (log-uniforme) a ogni N (c), meno che a 0.5–6 MeV (~1.1 e ~0.3 σ).
  Col prior largo l'errore rms cresce (7.1° contro 4.8° a N = 10, 1.05° contro 0.87° a N = 300),
  pull rms 0.94–1.00: costa precisione, non calibrazione (c). L'analisi tipo Hausman
  (`scripts/prior_B_rumore.py`) è stata fatta a 0.5–6 MeV e non rifatta.
- σ_E: la dipendenza dal prior di log σ_E scende da 0.76 σ (N = 50) a 0.07 σ (N = 1000) (c), più forte
  che a 0.5–6 MeV (0.32 σ a N = 50): coerente con un σ_E meno vincolato dai dati (§5, d).
- Le righe del Caso C vengono dal generatore senza troncatura (v0.19), con la calibrazione mancata
  di §5 per entrambi i prior; vanno rilanciate col generatore troncato. Il confronto dei prior a
  N = 150 della checklist v0.21 dà spostamenti simili (§5).

### 6.2 Spettro non gaussiano, Caso C (`scripts/stress_spettro.py`)

> **Valori a 0.5–6 MeV (v0.18), non rifatti a 10–20 MeV.** Restano come riferimento per il
> metodo; vanno rilanciati sul nuovo dominio, ora che la calibrazione del Caso C è chiusa (§5).

Energie vere da spettri con la stessa media e sd di N(μ_E, σ_E) ma forma uniforme, bimodale (due
righe a ±0.9 sd), lognormale, Laplace (curtosi 6) o Student-t con 5 gradi di libertà (curtosi 9);
il modello resta gaussiano. Bersagli: media e sd vere dello spettro.

| Forma | μ_E | Ω_n | log σ_E: pull (larghezza), coverage 68% a N = 150 / 300 | |
|---|---|---|---|---|
| gaussiana | calibrato | calibrato | 1.05 / 1.06; 0.60 / 0.64 (= checklist) | (c) |
| lognormale | calibrato | calibrato | 1.10 / 1.07; 0.57 / 0.62 | (c) |
| uniforme | calibrato | calibrato | 0.84 / 0.77; 0.75 / 0.81 | (c) |
| bimodale | calibrato | calibrato | 0.80 / 0.91; 0.79 / 0.76 | (c) |
| Laplace | calibrato | calibrato | 1.23 / 1.20; 0.56 / 0.65 | (c) |
| Student-t (ν = 5) | pull medio +0.16 ± 0.04 | calibrato | 1.17 / 1.39; 0.62 / 0.56 | (c) |

- Ω_n non risente della forma dello spettro, a ogni N (c). μ_E nemmeno, salvo la Student-t: pull
  medio +0.16 ± 0.04 su tutti gli N (M = 540), contro 0.00–0.07 per le altre forme. Coverage e
  larghezza del pull restano nominali; il bias non viene dagli esperimenti con energie fuori da
  [0.5, 6] MeV (c). Causa non indagata (d).
- log σ_E non ha bias peggiori del caso gaussiano, ma la sua coverage dipende dalla curtosi κ dello
  spettro. La sd campionaria fluttua con var(log s) ≈ (κ − 1)/(4N), mentre il modello gaussiano ne
  mette nel posterior 2/(4N): verificato per uniforme (1.8), bimodale (≈ 1.7), gaussiana e Laplace
  (`scripts/curtosi_log_sigma.py`, c). Per la Student-t a ν = 5 il valore misurato è più basso
  (5–6 contro 8). È atteso: l'ottavo momento è infinito, quindi la varianza campionaria di log s
  converge lentamente e di solito sottostima (b).
  Con curtosi bassa log σ_E è sovra-coperto, con code pesanti sotto-coperto (coverage 68% 0.56–0.65).
- Previsione scritta prima del run, larghezza del pull a N = 150 / 300, con il solo termine
  intrinseco aggiunto alla larghezza gaussiana: Laplace 1.17 / 1.19, Student-t 1.28 / 1.30.
  Osservato: 1.23 ± 0.06 / 1.20 ± 0.08 e 1.17 ± 0.06 / 1.39 ± 0.10 (c). L'alternativa con il termine
  amplificato di 3.6 volte (misurato su uniforme e bimodale) prevedeva 1.44–1.78 ed è esclusa a
  > 3σ. A N = 50 la previsione regge (1.07 / 1.17 contro 1.20 / 1.20). A N = 1000 le larghezze
  osservate (1.41 ± 0.18, 1.58 ± 0.17, M = 40) superano la previsione (1.01 / 1.14) di ~2.3σ; lì
  si allarga anche la parte di ricostruzione. Non indagato (d).
- La lognormale usata ha curtosi di poco sopra 3 (al più ~4 per i σ_E/μ_E della checklist) e si
  comporta come la gaussiana, con coverage appena più bassa (c).

### 6.3 Sorgente unica, Casi B e C (`scripts/stress_sorgente.py`)

> **Valori a 0.5–6 MeV (v0.18), non rifatti a 10–20 MeV.** Restano come riferimento per il
> metodo; vanno rilanciati sul nuovo dominio, ora che la calibrazione del Caso C è chiusa (§5).

Una frazione f di eventi viene da una seconda sorgente puntiforme a separazione Δ (10°, 30°, 90°)
o da un fondo isotropo; la ricostruzione resta a sorgente unica. Diagnostica: test KS fra gli
angoli osservati delle tracce rispetto a Ω̂_n e la loro distribuzione attesa con sorgente unica,
con soglia al 95° percentile della statistica a f = 0 (Ω̂_n viene dagli stessi dati). M = 100 per
configurazione, 200 a f = 0.

Caso B, N = 300 (σ posteriore ≈ 1.2°):

| Contaminazione | Spostamento verso la 2ª sorgente | pull rms | Coverage 68% | Potenza KS | |
|---|---|---|---|---|---|
| nessuna | — | 1.02 | 0.66 | 0.05 (= soglia) | (c) |
| f = 0.05, 10° | 0.5° | 1.15 | 0.53 | 0.05 | (c) |
| f = 0.30, 10° | 3.0° | 2.74 | 0.03 | 0.03 | (c) |
| f = 0.05, 30° | 2.1° | 2.28 | 0.16 | 0.02 | (c) |
| f = 0.10, 30° | 4.0° | 3.76 | 0.01 | 0.05 | (c) |
| f = 0.05, 90° | 10.3° | 9.5 | 0.00 | 0.17 | (c) |
| f = 0.10, 90° | 16.9° | 16.7 | 0.00 | 0.80 | (c) |
| f = 0.05 isotropo | rms 5.0° | 4.1 | 0.03 | 0.14 | (c) |
| f = 0.10 isotropo | rms 7.1° | 5.8 | 0.01 | 0.48 | (c) |

- Ω̂_n si sposta verso la seconda sorgente di circa f·Δ per Δ = 10°, un po' di più a 30° (4.0°
  contro 3° a f = 0.1) e molto di più a 90° (16.9° contro 9°) (c); la
  lettura "media pesata delle direzioni" è (d). La std posteriore non cresce, quindi a bias
  fissato il pull cresce come √N e la coverage crolla tanto più quanto N è grande: per f = 0.05 a
  30° la coverage 68% è 0.41 a N = 30 e 0.16 a N = 300 (c).
- **La diagnostica KS non vede le contaminazioni vicine** (c): per Δ = 10°–30° la potenza resta
  al livello del falso allarme (≤ 0.08) a ogni f e N, proprio dove Ω̂_n è già fuori di molte σ.
  Diventa utile solo per Δ = 90° o fondo isotropo: con f = 0.1 e N = 300 la potenza è 0.48–0.88,
  con f = 0.3 è 1.00. A Δ = 30° anche f = 0.3 resta invisibile (≤ 0.08).
  Il motivo proposto (d) è che la distribuzione degli angoli delle tracce è larga (θ_p fino a
  90°): spostare l'asse di pochi gradi la cambia poco.
- Caso C: anche (μ_E, σ_E) ne risentono (c). σ_E è il più sensibile: a N = 300 con f = 0.05 a 30°,
  pull di log σ_E +3.95 e coverage 68% 0.06. μ_E si sposta verso l'alto: poco per Δ ≤ 30° con
  f ≤ 0.1 (pull ≤ +0.8), molto per Δ = 90° o fondo isotropo (+0.32 MeV a f = 0.05 e 90°,
  +0.36 MeV a f = 0.10 isotropo; pull +2.5 / +3.0). Gli eventi da un'altra direzione vengono
  ricostruiti con l'angolo sbagliato, quindi con energie disperse e in media più alte (d). Un σ_E
  inatteso è un sintomo utile solo se si ha un'aspettativa indipendente sulla larghezza dello
  spettro (d).
- Conclusione: l'assunzione 1 resta necessaria e, per separazioni ≲ 30°, non è verificabile con
  questa diagnostica. Un modello a mistura che stimi f è il passo naturale successivo (fuori
  scope).

### 6.4 Due stadi contro stima congiunta, Caso C (`scripts/correlazione_stadi.py`)

> **Valori a 0.5–6 MeV (v0.18), non rifatti a 10–20 MeV.** Restano come riferimento per il
> metodo; vanno rilanciati sul nuovo dominio, ora che la calibrazione del Caso C è chiusa (§5).

Sugli stessi dataset della checklist del Caso C (M = 100 per N), si parte dalle stime dei due
stadi e si calcola l'approssimazione di Laplace del posterior congiunto 4D (Ω_n nel piano tangente,
μ_E, log σ_E): stencil 3^4 di differenze centrali, passi pari alle larghezze dei due stadi, due
passi di Newton, nessuna griglia 4D. L'Hessiana è definita negativa in 297 esperimenti su 300;
gli altri 3 sono esclusi.

| N | Correlazione canonica Ω–(μ, log σ): mediana / 90% / max | Allargamento marg./cond. medio (max): μ / log σ / Ω | Laplace / griglia: μ / log σ / Ω | Modo congiunto − due stadi, rms in σ: μ / log σ / Ω |
|---|---|---|---|---|
| 50 | 0.17 / 0.27 / 0.41 | 1.013 (1.059) / 1.008 (1.042) / 1.011 (1.052) | 0.93 / 0.99 / 0.97 | 0.21 / 0.51 / 0.12 |
| 150 | 0.10 / 0.15 / 0.20 | 1.004 (1.019) / 1.003 (1.012) / 1.003 (1.011) | 0.98 / 0.95 / 0.99 | 0.15 / 0.33 / 0.09 |
| 300 | 0.08 / 0.12 / 0.19 | 1.003 (1.012) / 1.002 (1.016) / 1.002 (1.010) | 0.99 / 0.99 / 0.99 | 0.09 / 0.22 / 0.11 |

Pull (media / larghezza; Ω: rms):

| N | Due stadi: μ / log σ / Ω | Modo congiunto: μ / log σ / Ω |
|---|---|---|
| 50 | −0.08/0.91 · +0.03/1.01 · 1.10 | −0.18/0.97 · −0.27/0.93 · 1.11 |
| 150 | −0.04/1.01 · +0.05/1.04 · 1.01 | −0.09/1.01 · −0.20/1.04 · 1.03 |
| 300 | −0.05/1.12 · +0.02/1.05 · 0.99 | −0.08/1.12 · −0.13/1.06 · 0.99 |

- La correlazione posteriore fra la direzione e gli iperparametri è debole e scende con N: la
  mediana passa da 0.17 a 0.08 (c). Ignorarla, cioè fissare un blocco invece di marginalizzarlo,
  stringe le incertezze in media dell'1% a N = 50 e dello 0.2–0.3% a N = 300; il caso peggiore è il
  6% (c). **La fattorizzazione a due stadi è giustificata** in questo toy (c).
- Le larghezze di Laplace concordano con le griglie dei due stadi entro l'1–7% (c).
- Il modo congiunto si discosta dalle stime dei due stadi soprattutto su log σ_E (0.5 σ a N = 50,
  0.2 σ a N = 300). Ha un pull medio negativo (−0.27 → −0.13), mentre la media posteriore dei due
  stadi è centrata. **Non è l'asimmetria del marginale** (`scripts/asimmetria_log_sigma.py`): il
  marginale ha la coda verso σ_E piccoli (media < mediana), quindi la relazione di Pearson
  moda ≈ media − 3(media − mediana) prevede un modo *sopra* la media (+0.25 / +0.14 / +0.06 σ a
  N = 50 / 150 / 300), il segno opposto all'osservato (c). La scomposizione esatta
  (`scripts/modo_log_sigma.py`, stessi dataset, in unità della std posteriore s) dà:

  | N | asimmetria (modo marg. − media) | volume di μ_E (modo 2D − modo marg.) | direzione (modo 4D − modo 2D) | numerica (Laplace d'uso − convergente) | totale |
  |---|---|---|---|---|---|
  | 50 | +0.16 ± 0.03 | −0.10 | −0.28 ± 0.05 | −0.06 ± 0.02 | −0.29 ± 0.04 |
  | 150 | +0.11 ± 0.01 | −0.07 | −0.20 ± 0.02 | −0.04 ± 0.01 | −0.20 ± 0.03 |
  | 300 | +0.06 ± 0.01 | −0.05 | −0.14 ± 0.02 | −0.02 | −0.15 ± 0.02 |

  Asimmetria e volume di μ_E quasi si cancellano. Il termine dominante viene dal lasciare libera
  Ω_n nel massimo congiunto invece di fissarla a ω_0 (c). Tutti i termini scalano circa come s²
  (rapporto scarto/s² costante entro un fattore 1.5), come un effetto di volume: il modo di una
  densità a più dimensioni non conta il volume delle direzioni massimizzate. Il modo profilato su Ω_n preferirebbe i σ_E piccoli, dove Ω_n è meglio
  determinata (interpretazione, d). Il gradiente numerico di `joint_laplace` aggiunge −0.02/−0.06 s.
  Pull medio delle stime puntuali: media due stadi +0.07 / +0.07 / +0.02, Laplace convergente
  −0.16 / −0.09 / −0.11. Da questo lato la stima a due stadi (media) è migliore (c).

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
| C, μ_E | 0.02 / 0.04 / 0.07 / 0.15 (N = 50 / 150 / 300 / 1000) | invariati entro gli errori | (c) |
| C, log σ_E | ≤ 0.02 | invariati | (c) |
| C, Ω_n | — | invariati | (c) |

Verdetto: l'anisotropia vera a 10–20 MeV non cambia la ricostruzione al livello del toy (c), e il
modello resta isotropo. Il bias su μ_E cresce con N (0.15 σ a N = 1000, vicino alla soglia): oltre
N ≈ 1000 va ricontrollato o messo nella verosimiglianza. Le righe del Caso C vengono dal generatore
senza troncatura (v0.19): la calibrazione mancata era la stessa con le due generazioni, quindi non
veniva dall'isotropia (§5); vanno rilanciate col generatore troncato.

Nota: il pkl in `outputs/` contiene oggi solo il Caso A, rilanciato dopo il cambio del prior su
θ_p (`stress_anisotropia.py 3 A`); le righe B e C della tabella vengono dal run precedente con i
tre casi e non sono rilette dal notebook 06.

## 7. Prestazioni

Tempi di `loglik` del Caso A misurati a 10–20 MeV (cinematica relativistica, buffer unico riusato
in place); gli altri a 0.5–6 MeV, salvo la checklist del Caso C a 10–20 MeV (~7 min con 3 processi).

| Operazione | Tempo | |
|---|---|---|
| `loglik` Caso A, 1 evento | 0.67 ms (target < 1 ms) | (c) |
| `loglik` Caso A, 1000 eventi | 0.71 s (target ridefinito a 2 s) | (c) |
| Esperimento Caso C a N = 300 | ~2 s (prima 19 s) | (c) |
| Checklist Caso C completa | ~8 min con 3 processi | (c) |

## 8. Limiti dichiarati

- **Assunzione 1 (sorgente unica): necessaria e non verificabile per sorgenti vicine (c).** Una
  contaminazione del 5% a 30° porta la coverage 68% di Ω_n da 0.66 a 0.16 a N = 300 e peggiora con
  N; la diagnostica KS sugli angoli delle tracce non la vede per Δ ≲ 30° (§6.3). Un modello a
  mistura è fuori scope.
- **Assunzione 3 (spettro gaussiano):** μ_E e Ω_n robusti alla forma (c); la coverage di log σ_E
  dipende dalla curtosi κ dello spettro (c, κ da 1.7 a 9, §6.2): sovra-coperto per κ < 3,
  sotto-coperto per code pesanti (coverage 68% ~0.6, larghezza del pull ~1.2 a N ≤ 300, fino a
  1.6 a N = 1000). Con la Student-t μ_E ha un bias di +0.16 σ (c, causa d).
- **Prior sul nuisance per evento (Caso B):** il prior su E_n non viene dimenticato al crescere di N
  (spostamento di Ω̂_n ~1 σ costante con un prior largo, c). Lo spostamento è rumore a media nulla
  che si somma in quadratura all'errore (relazione tipo Hausman, c da N ≈ 30, §6.1): costa
  precisione, non calibrazione. Va scelto sullo spettro fisico atteso.
  Nel Caso C log σ_E dipende dal prior di 0.3 σ a N = 50 e di meno di 0.1 σ da N ≈ 300 (c, §6.1).
- **Caso C, due stadi:** chiuso come limite. La correlazione fra Ω_n e (μ_E, σ_E) trascurata dai due
  stadi costa ≤ 1% sulle incertezze in media (≤ 6% nel caso peggiore) a N ≥ 50 (c, §6.4).
- **Nuisance z** (profondità d'interazione) del Caso A fuori scope: nessun osservabile definito.
- **Isotropia in CM** a 10–20 MeV: la distribuzione angolare vera (ENDF/B-VIII.0) non cambia la
  ricostruzione (c, §6.5); il modello resta isotropo.
- **Calibrazione del Caso C a 10–20 MeV:** raggiunta a N ≥ 150 col generatore troncato al dominio
  (c, §5); a N = 50 la coverage di log σ_E resta bassa (0.61 / 0.81 / 0.87), probabile effetto di
  prior (d). Uno spettro vero che sborda da [10, 20] MeV (fino al 10.7% senza troncatura) è un
  errore di modello che sposta log σ_E verso l'alto, crescente con N (c, §5).
- **Fuori scope:** sezione d'urto n–p decrescente con E_n (spettro rivelato ≠ spettro incidente),
  soglia su E_p basse.
- I limiti su assunzione 1, assunzione 3 e due stadi sono quantificati a 0.5–6 MeV (§6.2–6.4).
- **Target di prestazione** di `loglik` per N = 1000 ridefinito da 50 ms a 2 s: il limite è il numpy denso.
