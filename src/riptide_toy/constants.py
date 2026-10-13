"""Costanti numeriche del toy model: seme, masse, dominio di energia, risoluzioni,
griglie e soglie. Unita': MeV, rad. Nessun altro modulo contiene numeri magici."""
SEED = 20260907

# massa del protone (CODATA), MeV; la cinematica usa masse uguali n = p: la
# differenza di massa cambia l'energia massima di rinculo di un fattore
# 1 - ((m_n - m_p) / (m_n + m_p))^2 ~ 1 - 5e-7, trascurabile
PROTON_MASS_MEV = 938.272

# dominio di lavoro del rivelatore per l'energia del neutrone incidente, MeV;
# a queste energie T / (2 m) ~ 1% e la cinematica va trattata in modo relativistico
EN_MIN, EN_MAX = 10.0, 20.0
# estremo superiore del dominio allargato, usato solo come prior alternativo nei
# test di robustezza al prior su E_n (la verita' resta in [EN_MIN, EN_MAX])
EN_MAX_WIDE = 30.0

# spettro bimodale degli stress test sull'assunzione 3 (solo generazione): due righe
# a media +- BIMODAL_HALF_SEPARATION * sd, ciascuna larga sd * sqrt(1 - a^2), cosi'
# media e sd coincidono con la gaussiana di riferimento; a = 0.9 > sqrt(1 - a^2) ~ 0.44
# tiene le due righe separate (due picchi distinti, non una gaussiana appiattita)
BIMODAL_HALF_SEPARATION = 0.9

# spettro Student-t degli stessi stress test (code pesanti): gradi di liberta' > 4
# perche' la curtosi 3 + 6 / (dof - 4) sia finita; con 5 vale 9, ben oltre il 6
# della Laplace, l'altra forma a code pesanti provata
STUDENT_T_DOF = 5

# risoluzioni toy del detector, usate nel caso di riferimento R1
SIGMA_EP = 0.10     # MeV
SIGMA_THETA = 0.08  # rad

# griglia iperparametri Caso C: range di sigma_E (MeV) per coprire i due test
# di limite: sigma_E->0 deve ridare il Caso A, sigma_E->infinito il Caso B.
# SIGMA_E_MIN << SIGMA_EP: lo scatter fra eventi e' indistinguibile dal rumore
# di misura. SIGMA_E_MAX >> (EN_MAX - EN_MIN): la gaussiana troncata sul
# dominio di energy_grid e' gia' ~piatta.
SIGMA_E_MIN, SIGMA_E_MAX = 0.01, 50.0
# griglie per marginalizzare theta_p vero con risoluzione angolare (Caso B/C):
# theta in [0, pi/2] e theta_obs in [0, pi], passo ~pi/2/400 ~ 0.004 rad,
# cioe' ~SIGMA_THETA/20 (la gaussiana di risoluzione e' ben campionata).
N_THETA_TRACK = 400
N_THETA_OBS = 801
# raffinamento locale del Caso C (riga 14, R6): le griglie globali hanno passo
# piu' largo del posterior (a N=150: passo mu 0.093 vs dispersione 0.046 MeV;
# pixel ~3.7 gradi vs errore 1-3 gradi, (c) R5). Si tengono le celle della
# griglia grossolana con log-posterior > max - WINDOW_DELTA_LOG
# (exp(-10) ~ 5e-5: massa trascurabile fuori) e si ricampiona la finestra.
WINDOW_DELTA_LOG = 10.0
N_MU_FINE, N_SIGMA_FINE = 40, 40
N_DIRECTION_CAP = 8000
# stadio 2 marginalizzato su Omega_n (correzione 2, report §11): pixel ad area
# uguale nella calotta tenuta (raggio ~4.5 sigma_Omega con WINDOW_DELTA_LOG),
# passo ~0.8 sigma_Omega; ciascuno costa uno stadio 2 (hierarchical_base).
N_DIRECTION_MARGINAL = 100
# stima congiunta locale del Caso C (Laplace su (Omega_n, mu_E, log sigma_E)):
# passi di Newton dal punto dei due stadi; il secondo ricentra lo stencil sul
# modo, cosi' l'Hessiana e' valutata a ~0 passi da esso.
N_LAPLACE_NEWTON = 2

# distribuzione angolare dello scattering elastico n-p nel CM (angolo del neutrone),
# dalla valutazione ENDF/B-VIII.0 dell'idrogeno-1 (MAT 125, MF4/MT2, coefficienti di
# Legendre nel CM, interpolazione lineare in energia): f(mu) = sum_l (2l+1)/2 a_l P_l(mu),
# a_0 = 1, mu = cos(theta_CM). NP_LEGENDRE_COEFFS[i] = (a_1, ..., a_6) all'energia
# NP_LEGENDRE_ENERGIES[i] (MeV). Lo scostamento dall'isotropia cresce con l'energia:
# f(-1)/f(0) = 1.034 a 10 MeV, 1.061 a 14 MeV, 1.117 a 20 MeV (eccesso di neutroni
# all'indietro, cioe' di protoni di alta energia).
NP_LEGENDRE_ENERGIES = tuple(10.0 + 0.5 * i for i in range(21))
NP_LEGENDRE_COEFFS = (
    (-1.064356e-2, 1.473377e-4, -1.442281e-4, 2.423436e-5, 5.413201e-7, 8.242175e-8),
    (-1.121995e-2, 2.876282e-4, -1.625681e-4, 2.980216e-5, 6.884312e-7, 1.130710e-7),
    (-1.177142e-2, 4.506402e-4, -1.820425e-4, 3.624842e-5, 8.632534e-7, 1.527899e-7),
    (-1.229564e-2, 6.372463e-4, -2.026620e-4, 4.364756e-5, 1.068460e-6, 2.036315e-7),
    (-1.279057e-2, 8.482258e-4, -2.244438e-4, 5.207324e-5, 1.306554e-6, 2.679803e-7),
    (-1.325446e-2, 1.084264e-3, -2.474127e-4, 6.159778e-5, 1.579782e-6, 3.485844e-7),
    (-1.368587e-2, 1.345951e-3, -2.716015e-4, 7.229158e-5, 1.890050e-6, 4.485896e-7),
    (-1.408360e-2, 1.633782e-3, -2.970518e-4, 8.422266e-5, 2.238830e-6, 5.715733e-7),
    (-1.444675e-2, 1.948155e-3, -3.238143e-4, 9.745618e-5, 2.627058e-6, 7.215796e-7),
    (-1.477470e-2, 2.289372e-3, -3.519488e-4, 1.120541e-4, 3.055044e-6, 9.031553e-7),
    (-1.506708e-2, 2.657640e-3, -3.815239e-4, 1.280747e-4, 3.522365e-6, 1.121386e-6),
    (-1.532378e-2, 3.053070e-3, -4.126174e-4, 1.455727e-4, 4.027769e-6, 1.381933e-6),
    (-1.554496e-2, 3.475677e-3, -4.453150e-4, 1.645984e-4, 4.569070e-6, 1.691070e-6),
    (-1.573101e-2, 3.925383e-3, -4.797104e-4, 1.851985e-4, 5.143058e-6, 2.055720e-6),
    (-1.588255e-2, 4.402021e-3, -5.159036e-4, 2.074152e-4, 5.745397e-6, 2.483493e-6),
    (-1.600045e-2, 4.905330e-3, -5.540007e-4, 2.312868e-4, 6.370542e-6, 2.982727e-6),
    (-1.608578e-2, 5.434964e-3, -5.941119e-4, 2.568478e-4, 7.011651e-6, 3.562518e-6),
    (-1.613981e-2, 5.990493e-3, -6.363509e-4, 2.841290e-4, 7.660509e-6, 4.232764e-6),
    (-1.616401e-2, 6.571401e-3, -6.808325e-4, 3.131579e-4, 8.307460e-6, 5.004201e-6),
    (-1.616003e-2, 7.177100e-3, -7.276719e-4, 3.439591e-4, 8.941345e-6, 5.888439e-6),
    (-1.612968e-2, 7.806923e-3, -7.769820e-4, 3.765550e-4, 9.549454e-6, 6.898001e-6),
)
# iterazioni di Newton per invertire la CDF angolare: partendo dalla soluzione
# isotropa l'errore iniziale e' <~ 0.05 in mu e la convergenza e' quadratica
N_CM_NEWTON = 5
