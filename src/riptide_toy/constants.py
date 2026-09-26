SEED = 20260907

# equal masses, natural units of the toy model
M_NEUTRON = 1.0
M_PROTON = 1.0

# energy range of work for the incident particle, MeV
EN_MIN, EN_MAX = 0.5, 6.0

# risoluzioni toy del detector, usate nel caso di riferimento R1
SIGMA_EP = 0.10     # MeV
SIGMA_THETA = 0.08  # rad

# griglia iperparametri Caso C: range di sigma_E (MeV) per coprire i due test
# di limite (CLAUDE.md Sez. 5): sigma_E->0 ~ Caso A, sigma_E->infinito ~ Caso B.
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
