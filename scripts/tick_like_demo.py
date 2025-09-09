import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hawkes_sumexp_py import (SimuHawkesSumExpKernels, SimuHawkesMulti,
                              HawkesSumExpKern)
from hawkes_sumexp_py.plotting import plot_hawkes_baseline_and_kernels


def main():
    period_length = 300
    baselines = [[0.3, 0.5, 0.6, 0.4, 0.2, 0.0],
                 [0.8, 0.5, 0.2, 0.3, 0.3, 0.4]]
    n_baselines = len(baselines[0])
    decays = np.array([0.5, 2.0, 6.0], dtype=float)
    adjacency = [[[0.0, 0.1, 0.4], [0.2, 0.0, 0.2]],
                 [[0.0, 0.0, 0.0], [0.6, 0.3, 0.0]]]

    # simulation
    hawkes = SimuHawkesSumExpKernels(baseline=np.array(baselines, float),
                                     period_length=period_length, decays=decays,
                                     adjacency=np.array(adjacency, float), seed=2093, verbose=False,
                                     end_time=1000.0)
    hawkes.adjust_spectral_radius(0.5)

    multi = SimuHawkesMulti(hawkes, n_simulations=4)
    multi.simulate()

    # estimation
    # Use no regularization to avoid shrinkage that can distort baselines
    learner = HawkesSumExpKern(decays=decays, n_baselines=n_baselines,
                               period_length=period_length, penalty='none')

    learner.fit(multi.timestamps)
    print('Estimated baseline:', learner.baseline)
    print('True baseline first row:', np.array(baselines, float)[0])

    # plot (using Tick's plot API clone)
    fig = plot_hawkes_baseline_and_kernels(learner, hawkes=hawkes, show=False)
    fig.tight_layout()
    out = 'scripts/hawkes_demo.png'
    fig.savefig(out, dpi=150)
    print('Saved figure to', out)


if __name__ == '__main__':
    main()
