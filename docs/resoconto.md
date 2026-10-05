# Resoconto dei risultati — riptide-toy

v0.18 — 2026-10-05

Questa è la sintesi dei risultati del toy model. Il dettaglio tecnico è in `docs/roadmap.md` (stato e
scelte di implementazione) e in `docs/report_caso_C_stadio1.md` (diario del Caso C). I passaggi
numerici sono rifatti nei notebook `notebooks/01`–`05`.

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

Risoluzioni del rivelatore: σ_Ep = 0.10 MeV, σ_θ = 0.08 rad. Prior sempre propri (E_n piatto su
[0.5, 6] MeV; σ_E log-uniforme su [0.01, 50] MeV).

## 2. Cinematica (notebook 01)

| Risultato | Valore | |
|---|---|---|
| Energia del protone | E_p = E_n cos²θ_p, θ_p ≤ 90° | (a) |
| Isotropia in CM ⇒ spettro del protone | E_p ~ U(0, E_n): ⟨E_p⟩ = E_n/2, Var = E_n²/12 | (a) |
| Densità delle tracce attorno a Ω_n | cosθ_p/π per steradiante | (a) |
| Tracce misurate oltre 90° per lo smearing | σ_θ²/2 = 0.0032 | (b) |
| Esempio: E_n = 3 MeV, θ_p = 30° | E_p = 2.25 MeV | (a) |

## 3. Caso A — energia condivisa (notebook 02)

| Verifica | Risultato | |
|---|---|---|
| R1, un evento (Ê_p = 1.40, θ̂_p = 0.50) | E_n = 1.82 ± 0.21 MeV (codice: σ pesata 0.219) | (c) |
| R2, due eventi 2.70 ± 0.45 e 2.35 ± 0.20 | E_n = 2.40 ± 0.18 MeV (codice: 2.408 ± 0.183) | (c) |
| R3, ricostruzione difettosa (+3%, +0.05 MeV, σ dichiarate −30%) | pull medio 0.90, larghezza 1.44, coverage 68% → 0.433 | (c) |
| Checklist, 20 000 eventi, verità estratta dal prior | pull +0.003 / 0.998; coverage 0.683 / 0.899 / 0.949 (nominale) | (c) |
| Stessa checklist, verità E_n ~ U(1, 5) | pull +0.115 / 0.927; coverage 0.735 / 0.925 / 0.964 | (c) |
| Bias per bin di E_n (verità dal prior) | +0.51, +0.30, +0.08, −0.20, −0.69 MeV: attrazione verso il centro del prior | (c) |
| Contrazione con N eventi | σ_N ≈ σ_1/√N | (c) |
| Shift sistematico del 2% su E_p | l'offset si ferma a ≈ 0.05 MeV (= 0.02·E_n); σ continua a scendere: plateau | (c) |

Lettura: il posterior del Caso A è calibrato. La sovra-copertura vista prima (vecchio punto aperto 1)
non è un difetto. Nasce dal generatore U(1, 5), più stretto del prior: in media la verità sta più
vicino al centro del prior di quanto il prior preveda.

## 4. Caso B — direzione condivisa (notebook 03)

Verità: Ω_n isotropa; E_n di ogni evento ~ U(0.5, 6) MeV, cioè distribuita come il suo prior.
Stima = MAP sulla calotta raffinata. Errori in gradi; M = numero di esperimenti.

| N (M) | errore medio | rms errore | risoluzione dichiarata | pull rms | coverage HPD 68/90/95 |
|---|---|---|---|---|---|
| 10 (400) | 7.80 | 9.49 | 8.99 | 1.01 | 0.66 / 0.90 / 0.95 |
| 30 (300) | 3.71 | 4.24 | 4.20 | 0.99 | 0.68 / 0.88 / 0.95 |
| 100 (200) | 1.93 | 2.15 | 2.16 | 1.00 | 0.69 / 0.90 / 0.95 |
| 300 (100) | 1.15 | 1.25 | 1.23 | 1.03 | 0.66 / 0.89 / 0.96 |
| 1000 (40) | 0.53 | 0.59 | 0.68 | 0.88 | 0.78 / 0.95 / 1.00 |
| 1000 (240) | — | 0.65 | 0.67 | 0.96 | 0.70 / 0.91 / 0.95 |

- Calibrato a ogni N (c). A N = 1000 la riga con M = 40 è quella della checklist; quella con
  M = 240 aggiunge 200 esperimenti su seed disgiunti (`scripts/n1000_extra.py`, errore binomiale
  ±0.03 al 68%) e riporta pull e coverage al nominale.
- Contrazione: rms·√N = 0.52 / 0.41 / 0.38 / 0.38 / 0.36 rad per N = 10 / 30 / 100 / 300 / 1000.
  Pendenza log-log −0.53 su N = 30–1000 e −0.57 su N = 10–1000 (attesa −0.5) (c). La pendenza più
  ripida della v0.16 (−0.59) veniva dal punto a N = 1000 con M = 40.
- A N = 10 il regime non è ancora gaussiano (c): la frazione di pull angolari oltre 3 è 0.023
  contro 0.011 attesa per una Rayleigh, con mediana vicina all'attesa (1.12 contro 1.18). Le code
  pesanti alzano rms·√N, non la larghezza tipica.
- Un evento singolo riproduce la verosimiglianza analitica L ∝ π(E_n*)/cos θ (c).

## 5. Caso C — direzione ed energia gerarchica (notebooks 04, 05)

Ricostruzione a due stadi, senza griglia 4D. Stadio 1: Ω_n come nel Caso B. Stadio 2:
(μ_E, σ_E) su una griglia 2D con Ω_n fissata. Lo stadio 1 viene poi ripetuto con il prior
predittivo, solo per raffinare Ω_n. La verità è generata con μ_E ~ U(2.5, 4) MeV e σ_E
log-uniforme in (0.2, 0.6) MeV.

| N (M) | bias μ_E [MeV] | bias log σ_E | rms Ω_n [°] (dichiarata) | coverage 68% μ / log σ / Ω |
|---|---|---|---|---|
| 50 (200) | +0.002 | −0.154 ± 0.033 | 1.97 (1.87) | 0.68 / 0.66 / 0.70 |
| 150 (200) | +0.001 | −0.055 ± 0.019 | 1.11 (1.06) | 0.70 / 0.60 / 0.65 |
| 300 (100) | −0.001 | −0.017 ± 0.017 | 0.75 (0.75) | 0.66 / 0.64 / 0.66 |
| 1000 (40) | +0.002 | +0.008 ± 0.008 | 0.37 (0.41) | 0.55 / 0.70 / 0.78 |
| 1000 (240) | +0.0002 ± 0.0012 | −0.006 ± 0.004 | 0.42 (0.41) | 0.69 / 0.71 / 0.67 |

| Verifica | Risultato | |
|---|---|---|
| Esempio guidato: μ = 3, σ = 0.4, θ = 30° | E_p ~ N(2.25, 0.316) MeV | (a) |
| Limiti σ_E → 0 e σ_E → ∞ | si ritrovano il Caso A e il Caso B | (c) |
| μ_E | bias ≈ 0 e pull ≈ N(0, 1) a ogni N | (c) |
| Ω_n | pull rms 1.00–1.05 per N ≤ 300 | (c) |
| Bias di log σ_E a N piccolo | effetto del prior largo su σ_E; nullo col prior del generatore | (c) |
| Coverage 68% di log σ_E a N = 150–300, ripetuta su seed nuovi | 0.680 ± 0.016 (era una fluttuazione) | (c) |
| Coverage 90% di μ_E a N = 300, ripetuta su seed nuovi | 0.897 ± 0.012 (M = 600); 0.890 ± 0.010 con anche i seed della ripetizione al 68% (M = 900) | (c) |
| Ω_n a N = 50 col prior predittivo (M = 800) | coverage 0.695 / 0.899 / 0.945, pull rms 1.00 | (c) |
| Contrazione (pendenza log-log) | μ −0.46 · Ω −0.56 · log σ −0.74 (il bias va a zero) | (c) |
| N = 1000 con M = 240 (200 esperimenti aggiunti) | coverage 90% μ / log σ / Ω 0.95 / 0.91 / 0.90; pull μ +0.01/0.92, log σ −0.09/0.94, Ω rms 1.01. μ leggermente sovra-coperto al 90% (0.95 ± 0.019): la std posteriore è ~8% più larga della dispersione; causa non indagata | (c) |
| Robustezza al prior, spettro non gaussiano, seconda sorgente, correlazione fra stadi | §6 | |

## 6. Stress test delle assunzioni

Stessi dataset delle checklist (stessi seed): nella configurazione di riferimento (prior di
default, spettro gaussiano, nessuna contaminazione) si ritrovano esattamente i loro numeri, e
questo fa da controllo di consistenza. Si cambia una cosa per volta; la ricostruzione resta
quella standard.

### 6.1 Robustezza al prior (`scripts/robustezza_prior.py`)

Metrica: spostamento della stima in unità della std posteriore del prior di default, e coverage
col prior alternativo. Prior alternativi, tutti propri: E_n piatto su [0.5, 10] MeV ("largo"),
log-uniforme su [0.5, 6] e su [0.5, 10] MeV; Ω_n von Mises–Fisher con κ = 2 (larga ~40°) e asse a
90° dalla verità; σ_E uniforme in σ invece che in log σ.

| Caso, parametro | Prior alternativo | Spostamento medio / std (N crescente) | Coverage 68% | |
|---|---|---|---|---|
| A, E_n | piatto largo | +0.49 / +0.18 / +0.02 / +0.01 / +0.01 (N = 1 / 3 / 10 / 30 / 100) | 0.67–0.71 | (c) |
| A, E_n | log-uniforme | −0.24 / −0.09 / −0.03 / −0.02 / −0.01 | 0.65–0.71 | (c) |
| B, Ω_n | von Mises–Fisher a 90° | 0.21 / 0.09 / 0.04 / 0.02 (N = 10 / 30 / 100 / 300) | 0.60–0.72 | (c) |
| B, Ω_n | E_n piatto largo | 1.12 / 1.18 / 1.04 / 1.22 | 0.68–0.75 | (c) |
| B, Ω_n | E_n log-uniforme | 0.46 / 0.34 / 0.32 / 0.29 | 0.62–0.67 | (c) |
| C, μ_E | σ_E uniforme | −0.02 / −0.01 / −0.00 / −0.00 (N = 50 / 150 / 300 / 1000) | 0.66–0.71 (0.55 a N = 1000, M = 40, uguale al default) | (c) |
| C, log σ_E | σ_E uniforme | +0.32 / +0.17 / +0.11 / +0.06 | 0.59–0.70 | (c) |

- Il prior sui **parametri condivisi** (E_n nel Caso A, Ω_n nel Caso B, (μ_E, σ_E) nel Caso C) viene
  dimenticato come atteso: lo spostamento scende con N (c). Nel Caso A col prior largo restano
  spostamenti massimi di 1.5–2 σ anche a N = 30–100, nei pochi esperimenti con verità vicina a
  6 MeV, dove il prior di default tronca il posterior.
- Il prior sul **nuisance per evento** (E_n nel Caso B) non viene dimenticato: lo spostamento di Ω̂_n
  resta ~1 σ (prior largo) o ~0.3 σ (log-uniforme) a ogni N (c). Col prior largo l'errore rms
  cresce (17.0° contro 9.8° a N = 10, 2.0° contro 1.3° a N = 300) ma la std posteriore cresce
  con esso: pull rms 0.90–0.99, coverage nominale o sopra. Il meccanismo è stato verificato
  (`scripts/prior_B_rumore.py`, sugli stessi esperimenti). Lo spostamento d = Ω̂_alt − Ω̂_def nel
  piano tangente è **rumore, non bias**: il vettore medio è compatibile con zero a ogni N e per
  entrambi i prior (c). Col default uguale al prior del generatore la stima di default è
  efficiente, quindi vale la relazione tipo Hausman rms|d|² ≈ rms(err_alt)² − rms(err_def)². Col
  prior largo regge a N ≥ 30: rms|d|/σ osservato contro previsto 1.38/1.43 (N = 30), 1.18/1.20
  (100), 1.40/1.29 ± 0.1 (300) (c). Il rapporto rms(err_alt)/rms(err_def) è ~costante con N
  (1.74 / 1.75 / 1.60 / 1.58): ogni evento spreca la stessa frazione d'informazione su Ω_n, ed è
  questo che tiene lo spostamento ~1 σ a ogni N (c). Fanno eccezione N = 10 col prior largo
  (correlazione err_def·d = +0.26 ± 0.05; lì il posterior non è gaussiano e il MAP non è la
  media) e il log-uniforme a N ≤ 30, dove non è meno efficiente del default (rapporto 0.98 /
  1.03): lo spostamento di ~0.3 σ è rumore fra due stime equivalenti. In pratica il prior su
  E_n del Caso B va scelto sullo spettro fisico atteso: un prior sbagliato costa precisione,
  non calibrazione (c).
- σ_E: la dipendenza dal prior di log σ_E scende da 0.32 σ (N = 50) a 0.06 σ (N = 1000) (c).

### 6.2 Spettro non gaussiano, Caso C (`scripts/stress_spettro.py`)

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

## 7. Prestazioni

| Operazione | Tempo | |
|---|---|---|
| `loglik` Caso A, 1 evento | 0.64 ms (target < 1 ms) | (c) |
| `loglik` Caso A, 1000 eventi | 1.0–1.2 s (target ridefinito a 2 s) | (c) |
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
- **Isotropia in CM** assunta nel range 0.5–6 MeV; oltre ~10 MeV va verificata (d).
- **Target di prestazione** di `loglik` per N = 1000 ridefinito da 50 ms a 2 s: il limite è il numpy denso.
