SEED = 20260907

# equal masses, natural units of the toy model
M_NEUTRON = 1.0
M_PROTON = 1.0

# energy range of work for the incident particle, MeV
EN_MIN, EN_MAX = 0.5, 6.0

# risoluzioni toy del detector, usate nell'Esempio 38.1 del libro (Cap. 38)
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
