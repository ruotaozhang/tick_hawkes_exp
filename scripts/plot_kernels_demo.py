import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hawkes_sumexp_py import (
    SimuHawkesSumExpKernels, SimuHawkesMulti, HawkesSumExpKern
)
from hawkes_sumexp_py.plotting import plot_hawkes_kernels


def main():
    end_time = 1000
    n_realizations = 10

    decays = np.array([.5, 2., 6.])
    baseline = np.array([0.12, 0.07])
    adjacency = np.array([[[0, .1, .4], [.2, 0., .2]],
                          [[0, 0, 0], [.6, .3, 0]]], dtype=float)

    hawkes_exp_kernels = SimuHawkesSumExpKernels(
        adjacency=adjacency, decays=decays, baseline=baseline,
        end_time=end_time, verbose=False, seed=1039)

    multi = SimuHawkesMulti(hawkes_exp_kernels, n_simulations=n_realizations)

    multi.end_time = [(i + 1) / 10 * end_time for i in range(n_realizations)]
    multi.simulate()

    # Use stronger L1-heavy elastic-net to shrink spurious cross-kernels
    learner = HawkesSumExpKern(decays, penalty='elasticnet', elastic_net_ratio=0.95, C=1e6)
    learner.fit(multi.timestamps)

    fig = plot_hawkes_kernels(learner, hawkes=hawkes_exp_kernels, show=False)
    for ax in fig.axes:
        ax.set_ylim([0., 1.])
    fig.savefig('scripts/hawkes_kernels_demo.png', dpi=150)
    print('Saved figure to scripts/hawkes_kernels_demo.png')

    # Diagnostic print for (i=1, j=0): coefficients and small-t amplitude
    true_a = hawkes_exp_kernels.adjacency[1, 0, :]
    est_a = learner.adjacency[1, 0, :]
    betas = decays
    phi0_true = float((true_a * betas).sum())
    phi0_est = float((est_a * betas).sum())
    print('True  alpha[1,0,:] =', true_a)
    print('Est   alpha[1,0,:] =', est_a)
    print('phi_1,0(0+) true = %.6f, est = %.6f' % (phi0_true, phi0_est))


if __name__ == '__main__':
    main()
