import time
import numpy as np

from hawkes_sumexp_py.hawkes_sumexp import HawkesSumExpKern
from hawkes_sumexp_py.simulation import simulate_hawkes_multid_sumexp, scale_adjacency_spectral_radius


def main():
    rng = np.random.RandomState(123)
    D = 3
    U = 2
    betas = np.array([1.0, 3.0], dtype=float)
    mu = rng.uniform(0.05, 0.2, size=D)
    A = rng.uniform(0.0, 0.8, size=(D, D, U))
    A = scale_adjacency_spectral_radius(A, target_rho=0.7)
    T = 2000.0

    print("=== Simulating multi-node Hawkes (sum-exp) ===")
    t0 = time.time()
    events = simulate_hawkes_multid_sumexp(mu=mu, adjacency=A, betas=betas, T=T, seed=7)
    sim_time = time.time() - t0
    sizes = [len(ev) for ev in events]
    print(f"Sim time: {sim_time:.3f}s; events per node: {sizes}")

    # Fit learner
    learner = HawkesSumExpKern(decays=betas, penalty="elasticnet", C=1e3, elastic_net_ratio=0.5)
    t0 = time.time()
    learner.fit([events], end_times=[T])
    fit_time = time.time() - t0
    print(f"Fit time: {fit_time:.3f}s; loglik={learner.score():.3f}")

    print("True mu:", mu)
    print("Est. mu:", learner.baseline)
    # Summarize adjacency as branching matrix norms (sum over decays)
    B_true = A.sum(axis=2)
    B_est = learner.adjacency.sum(axis=2)
    print("True branching matrix (sum α):\n", np.round(B_true, 3))
    print("Est  branching matrix (sum α):\n", np.round(B_est, 3))


if __name__ == "__main__":
    main()

