import os
import sys
import time
import numpy as np

# Ensure project root on path when running from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import hawkes_sumexp_py.hawkes_sumexp as hmod
from hawkes_sumexp_py import (
    SimuHawkesSumExpKernels, SimuHawkesMulti, HawkesSumExpKern
)


def define_naive_H():
    njit = hmod.njit
    math = hmod.math

    @njit(cache=True)
    def _sumexp_H_for_pair_naive(events_l, events_j, beta_v, beta_u, T):
        if events_l.shape[0] == 0 or events_j.shape[0] == 0:
            return 0.0
        s = 0.0
        bsum = beta_v + beta_u
        for a in range(events_l.shape[0]):
            s_l = events_l[a]
            for b in range(events_j.shape[0]):
                r_j = events_j[b]
                if s_l <= r_j:
                    contrib = beta_v * beta_u * math.exp(beta_v * (s_l - r_j))
                    contrib *= (1.0 - math.exp(-bsum * (T - r_j))) / bsum
                else:
                    contrib = beta_v * beta_u * math.exp(beta_u * (r_j - s_l))
                    contrib *= (1.0 - math.exp(-bsum * (T - s_l))) / bsum
                s += contrib
        return s

    return _sumexp_H_for_pair_naive


def make_data(end_time: float, n_simulations: int = 4, seed: int = 4321):
    period_length = 300.0
    baselines = [[0.3, 0.5, 0.6, 0.4, 0.2, 0.0], [0.8, 0.5, 0.2, 0.3, 0.3, 0.4]]
    decays = [0.5, 2.0, 6.0]
    adjacency = [[[0.0, 0.1, 0.4], [0.2, 0.0, 0.2]], [[0.0, 0.0, 0.0], [0.6, 0.3, 0.0]]]
    hawkes = SimuHawkesSumExpKernels(
        baseline=baselines,
        period_length=period_length,
        decays=decays,
        adjacency=adjacency,
        seed=seed,
        verbose=False,
    )
    hawkes.end_time = end_time
    hawkes.adjust_spectral_radius(0.5)
    multi = SimuHawkesMulti(hawkes, n_simulations=n_simulations)
    multi.simulate()
    return multi.timestamps, decays, len(baselines[0]), period_length


def time_fit(events, decays, n_baselines, period_length):
    t0 = time.perf_counter()
    HawkesSumExpKern(decays=decays, n_baselines=n_baselines, period_length=period_length).fit(events)
    t1 = time.perf_counter()
    return t1 - t0


def main():
    print('Simulating 100x dataset (end_time=100000, 4 simulations)...', flush=True)
    events, decays, n_baselines, period_length = make_data(end_time=100000.0, n_simulations=4)

    # Warm JIT (small synthetic data)
    _ = time_fit([ [np.array([0.1, 0.2]), np.array([0.15]) ] ], decays, n_baselines, period_length)

    print('Timing optimized fit...', flush=True)
    t_fast = time_fit(events, decays, n_baselines, period_length)
    print(f'Optimized fit time (100x): {t_fast:.3f} s', flush=True)

    # Prepare naive H and warm it
    naive_H = define_naive_H()
    orig_H = hmod._sumexp_H_for_pair
    hmod._sumexp_H_for_pair = naive_H
    _ = time_fit([ [np.array([0.1, 0.2]), np.array([0.15]) ] ], decays, n_baselines, period_length)

    print('Timing naive fit (this may take a long time)...', flush=True)
    t_naive = time_fit(events, decays, n_baselines, period_length)
    hmod._sumexp_H_for_pair = orig_H

    print(f'Naive fit time (100x): {t_naive:.3f} s', flush=True)
    if t_fast > 0:
        print(f'Speedup: {t_naive / t_fast:.2f}x', flush=True)


if __name__ == '__main__':
    main()


