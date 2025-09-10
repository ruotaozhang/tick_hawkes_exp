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


def make_params(n_nodes: int, n_decays: int, n_baselines: int, seed: int = 123):
    rng = np.random.default_rng(seed)
    period_length = 300.0
    baselines = (0.02 + 0.08 * rng.random((n_nodes, n_baselines))).tolist()
    adjacency = np.zeros((n_nodes, n_nodes, n_decays), dtype=float)
    mask = rng.random((n_nodes, n_nodes)) < 0.05
    for u in range(n_decays):
        W = 0.3 * rng.random((n_nodes, n_nodes)) * mask
        np.fill_diagonal(W, 0.0)
        adjacency[:, :, u] = W
    decays = [0.5, 2.0, 6.0][:n_decays]
    return baselines, period_length, decays, adjacency.tolist()


def simulate_dataset(n_nodes=40, n_simulations=10000, end_time=100.0, seed=2025):
    baselines, period_length, decays, adjacency = make_params(n_nodes, n_decays=3, n_baselines=6, seed=seed)
    hawkes = SimuHawkesSumExpKernels(
        baseline=baselines,
        period_length=period_length,
        decays=decays,
        adjacency=adjacency,
        seed=seed,
        verbose=False,
    )
    hawkes.end_time = end_time
    hawkes.adjust_spectral_radius(0.2)
    multi = SimuHawkesMulti(hawkes, n_simulations=n_simulations)
    multi.simulate()
    return multi.timestamps, decays, len(baselines[0]), period_length


def time_naive_full():
    print('Simulating 40-node dataset with 10,000 realizations...', flush=True)
    events, decays, n_baselines, period_length = simulate_dataset(n_nodes=40, n_simulations=10000, end_time=100.0)

    # Warm JIT for both optimized and naive to avoid counting compile overhead
    _ = HawkesSumExpKern(decays=decays, n_baselines=n_baselines, period_length=period_length).fit([[np.array([0.1]), np.array([0.2])]])

    naive_H = define_naive_H()
    orig_H = hmod._sumexp_H_for_pair
    hmod._sumexp_H_for_pair = naive_H
    _ = HawkesSumExpKern(decays=decays, n_baselines=n_baselines, period_length=period_length).fit([[np.array([0.1]), np.array([0.2])]])

    print('Timing naive fit on full dataset (this may take a long time)...', flush=True)
    t0 = time.perf_counter()
    HawkesSumExpKern(decays=decays, n_baselines=n_baselines, period_length=period_length).fit(events)
    t1 = time.perf_counter()
    hmod._sumexp_H_for_pair = orig_H
    print(f'Naive fit time (full 10k): {t1 - t0:.3f} s', flush=True)


if __name__ == '__main__':
    time_naive_full()


