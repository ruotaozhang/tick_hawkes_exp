import time
import os
import sys
import numpy as np

# Ensure project root on path when running from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from hawkes_sumexp_py import (
    SimuHawkesSumExpKernels, SimuHawkesMulti, HawkesSumExpKern
)
import hawkes_sumexp_py.hawkes_sumexp as hmod


def make_model_and_data(end_time: float, n_simulations: int = 4, seed: int = 2093):
    period_length = 300.0
    baselines = [[0.3, 0.5, 0.6, 0.4, 0.2, 0.0], [0.8, 0.5, 0.2, 0.3, 0.3, 0.4]]
    n_baselines = len(baselines[0])
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
    return decays, n_baselines, period_length, multi.timestamps


def build_learner(decays, n_baselines, period_length):
    return HawkesSumExpKern(decays=decays, n_baselines=n_baselines, period_length=period_length)


def define_naive_H():
    # Import njit and math from the module to match types
    njit = hmod.njit
    math = hmod.math
    np_mod = hmod.np

    @njit(cache=True)
    def _sumexp_H_for_pair_naive(events_l, events_j, beta_v, beta_u, T):
        # Integral via double sum with closed-form per event pair
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


def time_fit(events, decays, n_baselines, period_length):
    learner = build_learner(decays, n_baselines, period_length)
    t0 = time.perf_counter()
    learner.fit(events)
    t1 = time.perf_counter()
    return t1 - t0


def main():
    # Build big dataset: 10x longer time horizon compared to tick_like_demo.py
    decays, n_baselines, period_length, big_events = make_model_and_data(end_time=10000.0, n_simulations=4)

    # Warm-up JIT on a tiny dataset for both fast and naive implementations
    _, _, _, small_events = make_model_and_data(end_time=100.0, n_simulations=1, seed=2094)

    # Warm fast
    _ = time_fit(small_events, decays, n_baselines, period_length)

    # Prepare naive and warm it
    naive_fn = define_naive_H()
    orig_fn = hmod._sumexp_H_for_pair
    hmod._sumexp_H_for_pair = naive_fn
    _ = time_fit(small_events, decays, n_baselines, period_length)

    # Benchmark naive on big dataset
    t_naive = time_fit(big_events, decays, n_baselines, period_length)

    # Restore optimized and ensure JIT warm
    hmod._sumexp_H_for_pair = orig_fn
    _ = time_fit(small_events, decays, n_baselines, period_length)

    # Benchmark optimized on big dataset
    t_fast = time_fit(big_events, decays, n_baselines, period_length)

    print(f"Naive H time (10x data): {t_naive:.3f} s")
    print(f"Fast  H time (10x data): {t_fast:.3f} s")
    if t_fast > 0:
        print(f"Speedup: {t_naive / t_fast:.2f}x")


if __name__ == "__main__":
    main()


