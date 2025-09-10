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
    # Piecewise baseline values per node, small to keep event counts moderate
    baselines = (0.02 + 0.08 * rng.random((n_nodes, n_baselines))).tolist()
    # Random sparse adjacency per decay; values small, will rescale via spectral radius
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


def time_fit(events, decays, n_baselines, period_length):
    t0 = time.perf_counter()
    HawkesSumExpKern(decays=decays, n_baselines=n_baselines, period_length=period_length).fit(events)
    t1 = time.perf_counter()
    return t1 - t0


def main():
    print('Simulating 40-node dataset with 10,000 realizations...', flush=True)
    events, decays, n_baselines, period_length = simulate_dataset(n_nodes=40, n_simulations=10000, end_time=100.0)

    # Warm JIT quickly on tiny toy data to avoid counting compile time
    _ = time_fit([[np.array([0.1]), np.array([0.2])]], decays, n_baselines, period_length)

    # Optimized on full dataset
    print('Timing optimized fit on full dataset...', flush=True)
    t_fast_full = time_fit(events, decays, n_baselines, period_length)
    print(f'Optimized fit time (full 10k): {t_fast_full:.3f} s', flush=True)

    # Build a manageable subset for naive vs optimized comparison
    subset_n = 50
    print(f'Comparing optimized vs naive on subset of {subset_n} realizations...', flush=True)
    subset_events = events[:subset_n]

    # Optimized timing on subset
    t_fast_sub = time_fit(subset_events, decays, n_baselines, period_length)

    # Swap in naive H, warm, and time
    naive_H = define_naive_H()
    orig_H = hmod._sumexp_H_for_pair
    hmod._sumexp_H_for_pair = naive_H
    _ = time_fit([[np.array([0.1]), np.array([0.2])]], decays, n_baselines, period_length)
    t_naive_sub = time_fit(subset_events, decays, n_baselines, period_length)
    hmod._sumexp_H_for_pair = orig_H

    print(f'Optimized fit time (subset {subset_n}): {t_fast_sub:.3f} s', flush=True)
    print(f'Naive     fit time (subset {subset_n}): {t_naive_sub:.3f} s', flush=True)
    if t_fast_sub > 0:
        print(f'Speedup on subset: {t_naive_sub / t_fast_sub:.2f}x', flush=True)

    # Equality check on subset: stats and fitted params
    E_norm, _, _ = hmod._ensure_events_format(subset_events)
    T_arr = hmod._prepare_end_times(E_norm, None)
    T_sum1, G1, H1, S1, n1 = hmod._aggregate_statistics(E_norm, T_arr, np.asarray(decays, float))
    hmod._sumexp_H_for_pair = naive_H
    T_sum2, G2, H2, S2, n2 = hmod._aggregate_statistics(E_norm, T_arr, np.asarray(decays, float))
    hmod._sumexp_H_for_pair = orig_H

    print('max|G diff| =', float(np.max(np.abs(G1 - G2))))
    print('max|S diff| =', float(np.max(np.abs(S1 - S2))))
    print('max|H diff| =', float(np.max(np.abs(H1 - H2))))

    lf = HawkesSumExpKern(decays=decays, n_baselines=n_baselines, period_length=period_length)
    lf.fit(subset_events)
    hmod._sumexp_H_for_pair = naive_H
    ln = HawkesSumExpKern(decays=decays, n_baselines=n_baselines, period_length=period_length)
    ln.fit(subset_events)
    hmod._sumexp_H_for_pair = orig_H
    b1 = np.asarray(lf.baseline)
    b2 = np.asarray(ln.baseline)
    A1 = lf.adjacency
    A2 = ln.adjacency
    print('max|baseline diff| =', float(np.max(np.abs(b1 - b2))))
    print('max|adjacency diff| =', float(np.max(np.abs(A1 - A2))))


if __name__ == '__main__':
    main()


