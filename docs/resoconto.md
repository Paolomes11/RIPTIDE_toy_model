# Resoconto dei risultati — riptide-toy

v0.16 — 2026-10-01

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

- Calibrato fino a N = 300 (c). A N = 1000 ci sono solo 40 esperimenti (errore binomiale ±0.07 al
  68%) e i numeri sono compatibili con il nominale entro ~1.5σ.
- Contrazione: pendenza log-log di −0.59 (attesa −0.5). Fra N = 30 e N = 300, rms·√N resta a
  0.38–0.41 rad (c). Agli estremi (N = 10 e N = 1000) il valore si scosta, e questo rende la
  pendenza più ripida. L'interpretazione proposta è un regime non ancora gaussiano a N = 10 e la
  statistica bassa a N = 1000 (d).
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
| Robustezza al prior (N = 150, U in σ_E contro U in log σ_E) | μ si sposta di −0.01 σ_post; log σ di +0.16 σ_post (max 0.74) | (c) |

## 6. Prestazioni

| Operazione | Tempo | |
|---|---|---|
| `loglik` Caso A, 1 evento | 0.64 ms (target < 1 ms) | (c) |
| `loglik` Caso A, 1000 eventi | 1.0–1.2 s (target ridefinito a 2 s) | (c) |
| Esperimento Caso C a N = 300 | ~2 s (prima 19 s) | (c) |
| Checklist Caso C completa | ~8 min con 3 processi | (c) |

## 7. Limiti dichiarati

- **Assunzioni 1 e 3 non messe alla prova (d)**: nessun test con una seconda sorgente, con uno spettro
  non gaussiano o con prior diversi su E_n e Ω_n. La robustezza al prior copre solo σ_E a N = 150.
- **N = 1000** misurato con soli M = 40 esperimenti: la coverage ha un errore di ±0.07 (b).
- **Nuisance z** (profondità d'interazione) del Caso A fuori scope: nessun osservabile definito.
- **Isotropia in CM** assunta nel range 0.5–6 MeV; oltre ~10 MeV va verificata (d).
- **Target di prestazione** di `loglik` per N = 1000 ridefinito da 50 ms a 2 s: il limite è il numpy denso.
- **Caso C**: Ω̂_n e (μ̂_E, σ̂_E) vengono da stadi diversi, non da una stima congiunta (b). Fino a
  N ≲ 150, log σ_E dipende dal prior.
