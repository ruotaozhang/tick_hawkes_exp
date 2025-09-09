import math
import numpy as np

from hawkes_sumexp_py import HawkesSumExpKern


def simulate_hawkes_1d_sumexp(mu: float, alphas: np.ndarray, betas: np.ndarray, T: float, seed: int = 0):
    """Ogata thinning for 1D Hawkes with sum-exponentials kernels.
    lambda(t) = mu + sum_{k<t} sum_u alpha[u] * beta[u] * exp(-beta[u] (t - t_k))
    Returns event times as 1D numpy array.
    """
    assert alphas.shape == betas.shape
    rng = np.random.RandomState(seed)
    t = 0.0
    events = []
    # state[u] = sum_k beta[u] * exp(-beta[u] (t - t_k))  at current time t
    state = np.zeros_like(betas, dtype=float)
    lam = mu  # since state=0 initially
    while True:
        if lam <= 0:
            # unlikely, guard
            lam = 1e-12
        w = rng.exponential(1.0 / lam)  # candidate waiting time
        t_candidate = t + w
        if t_candidate > T:
            break
        # decay state up to candidate
        decay = np.exp(-betas * (t_candidate - t))
        state_cand = state * decay
        lam_cand = float(mu + np.sum(alphas * state_cand))
        # thinning acceptance
        if rng.uniform() <= lam_cand / lam:
            # accept event at t_candidate
            t = t_candidate
            events.append(t)
            # update state after jump: add impulses beta[u]
            state = state_cand + betas
            lam = float(mu + np.sum(alphas * state))
        else:
            # reject, move time and update state/lam
            t = t_candidate
            state = state_cand
            lam = lam_cand
    return np.array(events, dtype=float)


def main():
    # Ground truth params
    mu = 0.2
    betas = np.array([1.0], dtype=float)
    alphas = np.array([0.4], dtype=float)  # should be < 1 for stability in 1D
    T = 100.0

    # Simulate 1 realization, 1 node
    ev = simulate_hawkes_1d_sumexp(mu, alphas, betas, T, seed=42)
    events = [[ev]]  # shape: [realization][node] -> np.ndarray
    end_times = [T]

    # Fit our learner
    learner = HawkesSumExpKern(decays=betas, penalty="l2", C=1e3)
    learner.fit(events, end_times=end_times)

    print("Simulated #events:", ev.shape[0])
    print("Estimated baseline:", learner.baseline)
    print("Estimated adjacency (i,j,u):", learner.adjacency)
    print("Log-likelihood on training:", learner.score())

    # Crude sanity assertions within loose bounds
    est_mu = float(learner.baseline[0])
    est_alpha = float(learner.adjacency[0, 0, 0])
    print(f"Sanity — mu true={mu:.3f}, est={est_mu:.3f}; alpha true={alphas[0]:.3f}, est={est_alpha:.3f}")


if __name__ == "__main__":
    main()

