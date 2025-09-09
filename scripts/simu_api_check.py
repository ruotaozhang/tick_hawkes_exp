import numpy as np

from hawkes_sumexp_py import SimuHawkesSumExpKernels, SimuHawkesMulti


def main():
    D, U = 2, 2
    betas = np.array([1.0, 3.0])
    A = np.array([
        [[0.2, 0.1], [0.05, 0.0]],
        [[0.0, 0.1], [0.25, 0.0]],
    ])
    mu = np.array([0.1, 0.2])
    T = 200.0

    # constant baseline
    simu = SimuHawkesSumExpKernels(adjacency=A, decays=betas, baseline=mu, end_time=T, seed=1)
    simu.track_intensity(0.5)
    simu.simulate()
    print("Simulated jumps per node:", [len(x) for x in simu.timestamps])
    print("Tracked intensity shapes:", [arr.shape for arr in simu.tracked_intensity])

    # piecewise baseline
    mu_pw = np.vstack([mu, mu]).T  # (D,2) alternating
    simu2 = SimuHawkesSumExpKernels(adjacency=A, decays=betas, baseline=mu_pw, end_time=T, period_length=10.0, seed=2)
    simu2.track_intensity(1.0)
    simu2.simulate()
    print("PW baseline, jumps per node:", [len(x) for x in simu2.timestamps])

    # Multi simulations
    multi = SimuHawkesMulti(simu, n_simulations=3, n_threads=1)
    multi.simulate()
    print("Multi timestamps lens:", [[len(x) for x in ts] for ts in multi.timestamps])


if __name__ == "__main__":
    main()

