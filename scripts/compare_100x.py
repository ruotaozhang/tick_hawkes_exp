import os
import sys
import numpy as np

# Ensure project root on path when running from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import hawkes_sumexp_py.hawkes_sumexp as hmod
from hawkes_sumexp_py import SimuHawkesSumExpKernels, SimuHawkesMulti, HawkesSumExpKern


def define_naive_H():
    njit = hmod.njit
    math = hmod.math

    @njit(cache=False)
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


def simulate_data():
    period_length = 300.0
    baselines = [[0.3, 0.5, 0.6, 0.4, 0.2, 0.0], [0.8, 0.5, 0.2, 0.3, 0.3, 0.4]]
    n_baselines = len(baselines[0])
    decays = np.array([0.5, 2.0, 6.0], float)
    adjacency = [[[0.0, 0.1, 0.4], [0.2, 0.0, 0.2]], [[0.0, 0.0, 0.0], [0.6, 0.3, 0.0]]]

    hawkes = SimuHawkesSumExpKernels(
        baseline=baselines,
        period_length=period_length,
        decays=decays,
        adjacency=adjacency,
        seed=4321,
        verbose=False,
    )
    # 100x bigger than tick_like_demo.py (which uses 1000)
    hawkes.end_time = 100000.0
    hawkes.adjust_spectral_radius(0.5)
    multi = SimuHawkesMulti(hawkes, n_simulations=4)
    multi.simulate()
    return decays, n_baselines, period_length, multi.timestamps


def warmup_jit():
    # Small warmup to reduce first-call Numba overhead
    period_length = 300.0
    baselines = [[0.3, 0.5, 0.6, 0.4, 0.2, 0.0], [0.8, 0.5, 0.2, 0.3, 0.3, 0.4]]
    decays = np.array([0.5, 2.0, 6.0], float)
    adjacency = [[[0.0, 0.1, 0.4], [0.2, 0.0, 0.2]], [[0.0, 0.0, 0.0], [0.6, 0.3, 0.0]]]
    hawkes = SimuHawkesSumExpKernels(
        baseline=baselines,
        period_length=period_length,
        decays=decays,
        adjacency=adjacency,
        seed=111,
        verbose=False,
    )
    hawkes.end_time = 100.0
    hawkes.adjust_spectral_radius(0.5)
    multi = SimuHawkesMulti(hawkes, n_simulations=1)
    multi.simulate()
    # One quick fit to trigger jitting
    learner = HawkesSumExpKern(decays=decays, n_baselines=len(baselines[0]), period_length=period_length)
    learner.fit(multi.timestamps)


def main():
    warmup_jit()
    decays, n_baselines, period_length, events = simulate_data()

    # Normalize inputs for stats
    E_norm, _, _ = hmod._ensure_events_format(events)
    T_arr = hmod._prepare_end_times(E_norm, None)

    # Stats with optimized H
    T_sum1, G1, H1, S1, n1 = hmod._aggregate_statistics(E_norm, T_arr, decays)

    # Swap to naive and compute stats
    naive_H = define_naive_H()
    orig_H = hmod._sumexp_H_for_pair
    hmod._sumexp_H_for_pair = naive_H
    T_sum2, G2, H2, S2, n2 = hmod._aggregate_statistics(E_norm, T_arr, decays)
    hmod._sumexp_H_for_pair = orig_H

    # Print diffs for stats
    print('max|G diff| =', float(np.max(np.abs(G1 - G2))))
    print('max|S diff| =', float(np.max(np.abs(S1 - S2))))
    print('max|H diff| =', float(np.max(np.abs(H1 - H2))))

    # Fit learners and compare parameters
    lf = HawkesSumExpKern(decays=decays, n_baselines=n_baselines, period_length=period_length)
    lf.fit(events)

    hmod._sumexp_H_for_pair = naive_H
    ln = HawkesSumExpKern(decays=decays, n_baselines=n_baselines, period_length=period_length)
    ln.fit(events)
    hmod._sumexp_H_for_pair = orig_H

    b1 = np.asarray(lf.baseline)
    b2 = np.asarray(ln.baseline)
    A1 = lf.adjacency
    A2 = ln.adjacency

    print('max|baseline diff| =', float(np.max(np.abs(b1 - b2))))
    print('max|adjacency diff| =', float(np.max(np.abs(A1 - A2))))


if __name__ == '__main__':
    main()


