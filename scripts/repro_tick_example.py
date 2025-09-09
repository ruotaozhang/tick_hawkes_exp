import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hawkes_sumexp_py import (SimuHawkesSumExpKernels, SimuHawkesMulti,
                              HawkesSumExpKern)
from hawkes_sumexp_py.plotting import plot_hawkes_baseline_and_kernels


def main():
    period_length = 300
    baselines = [[0.3, 0.5, 0.6, 0.4, 0.2, 0], [0.8, 0.5, 0.2, 0.3, 0.3, 0.4]]
    n_baselines = len(baselines[0])
    decays = [.5, 2., 6.]
    adjacency = [[[0, .1, .4], [.2, 0., .2]], [[0, 0, 0], [.6, .3, 0]]]

    # simulation
    hawkes = SimuHawkesSumExpKernels(baseline=np.array(baselines, float),
                                     period_length=period_length, decays=np.array(decays, float),
                                     adjacency=np.array(adjacency, float), seed=2093, verbose=False)
    hawkes.end_time = 1000
    hawkes.adjust_spectral_radius(0.5)

    multi = SimuHawkesMulti(hawkes, n_simulations=4)
    multi.simulate()

    # estimation (use exact defaults: penalty='l2', C=1e3)
    learner = HawkesSumExpKern(decays=np.array(decays, float), n_baselines=n_baselines,
                               period_length=period_length)

    learner.fit(multi.timestamps)

    # plot
    fig = plot_hawkes_baseline_and_kernels(learner, hawkes=hawkes, show=False)
    fig.tight_layout()
    fig.savefig('scripts/hawkes_repro.png', dpi=150)
    print('Saved figure to scripts/hawkes_repro.png')


if __name__ == '__main__':
    main()

