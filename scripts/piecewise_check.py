import numpy as np

from hawkes_sumexp_py.hawkes_sumexp import HawkesSumExpKern
from hawkes_sumexp_py.simulation import simulate_hawkes_multid_sumexp, scale_adjacency_spectral_radius


def main():
    D, U = 2, 1
    betas = np.array([2.0])
    mu = np.array([0.15, 0.12])
    A = np.random.RandomState(0).uniform(0.0, 0.6, size=(D, D, U))
    A = scale_adjacency_spectral_radius(A, 0.6)
    T = 500.0

    events = simulate_hawkes_multid_sumexp(mu=mu, adjacency=A, betas=betas, T=T, seed=0)

    # Fit with piecewise baseline (4 segments per period), period 10
    learner = HawkesSumExpKern(decays=betas, penalty="l2", C=1e3, n_baselines=4, period_length=10.0)
    learner.fit([events], end_times=[T])
    print("Baseline shape (n_nodes, n_baselines):", learner.baseline.shape)
    print("Baseline values per node:")
    print(learner.baseline)
    print("Adjacency sum over decays:")
    print(learner.adjacency.sum(axis=2))


if __name__ == "__main__":
    main()

