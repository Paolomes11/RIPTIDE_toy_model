import numpy as np
from scipy.special import logsumexp

from riptide_toy import kinematics

def measure(Ep_true: np.ndarray, theta_p_true: np.ndarray,
            sigma_Ep: float, sigma_theta: float,
            rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return: (Ep_hat, theta_p_hat), same dimension of the inputs,
    with added gaussian noise - they're the observables D."""
    Ep_hat = rng.normal(Ep_true, sigma_Ep, Ep_true.shape)
    theta_p_hat = rng.normal(theta_p_true, sigma_theta, theta_p_true.shape)
    return (Ep_hat, theta_p_hat)


def loglik(D, En_grid, theta_p_grid, sigma_Ep, sigma_theta,
           log_prior_theta, chunk_size=500):
    """Return: array (n_events, n_En) — log-likelihood for every
    event in every En, with theta_p already marginalized (weighted
    from log_prior_theta). DOESN'T give the grid (n_eventi, n_En,
    n_theta) anymore: the realistic size doesn't fit in the RAM."""
    Ep_hat, theta_p_hat = D
    Ep_hat = Ep_hat.astype(np.float32)
    theta_p_hat = theta_p_hat.astype(np.float32)
    n_events, n_En = Ep_hat.shape[0], En_grid.shape[0]

    theta = theta_p_grid[None, :].astype(np.float32)
    En = En_grid[:, None].astype(np.float32)
    Ep_pred = (En * np.cos(theta) ** 2)[None]

    out = np.empty((n_events, n_En), dtype=np.float32)
    for start in range(0, n_events, chunk_size):
        end = min(start + chunk_size, n_events)
        first = -0.5 * ((Ep_hat[start:end, None, None] - Ep_pred) / sigma_Ep) ** 2
        second = -0.5 * ((theta_p_hat[start:end, None, None] - theta[None]) / sigma_theta) ** 2
        joint = first + second + log_prior_theta[None, None, :]   # (chunk, n_En, n_theta)
        out[start:end] = logsumexp(joint, axis=2)                  # marginalizza theta, un lotto alla volta
    return out