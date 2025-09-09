import numpy as np
from hawkes_sumexp_py.hawkes_sumexp import (
    _aggregate_statistics, _segment_durations, _counts_per_segment, _J_segments_for_node,
)
from hawkes_sumexp_py import SimuHawkesSumExpKernels, SimuHawkesMulti


def main():
    period_length = 300.0
    baselines = np.array([[0.3, 0.5, 0.6, 0.4, 0.2, 0.0],
                          [0.8, 0.5, 0.2, 0.3, 0.3, 0.4]], float)
    K = baselines.shape[1]
    decays = np.array([0.5, 2.0, 6.0])
    adjacency = np.array([[[0, 0.1, 0.4], [0.2, 0.0, 0.2]],
                          [[0.0, 0.0, 0.0], [0.6, 0.3, 0.0]]], dtype=float)

    hawkes = SimuHawkesSumExpKernels(baseline=baselines, period_length=period_length, decays=decays,
                                     adjacency=adjacency, seed=2093, verbose=False, end_time=1000.0)
    hawkes.adjust_spectral_radius(0.5)

    multi = SimuHawkesMulti(hawkes, n_simulations=4)
    multi.simulate()
    events = multi.timestamps
    end_times = np.array([max([ev[-1] if len(ev)>0 else 0.0 for ev in r]) for r in events], float)
    Tsum, G, H, S, n_counts = _aggregate_statistics(events, end_times, decays)
    print('T_sum', Tsum)
    print('n_counts', n_counts)
    print('sum G', G.sum())

    seg_durs = np.zeros(K)
    seg_counts = np.zeros((2, K))
    J = np.zeros((K, 2, len(decays)))
    for r in range(len(events)):
        T = float(end_times[r])
        durs = _segment_durations(T, period_length, K)
        seg_durs += durs
        for i in range(2):
            seg_counts[i] += _counts_per_segment(events[r][i], period_length, K)
        for j in range(2):
            for u in range(len(decays)):
                J[:, j, u] += _J_segments_for_node(events[r][j], decays[u], T, period_length, K)
    print('seg_durs', seg_durs)
    print('seg_counts per i', seg_counts)
    print('J sum over k (per j,u):')
    print(J.sum(axis=0))

    # Closed-form baseline using true adjacency
    A = hawkes.adjacency
    mu_cf = np.zeros((2, K))
    for i in range(2):
        cross = np.tensordot(A[i], J, axes=((0, 1), (1, 2)))  # (K,)
        mu_cf[i] = np.maximum(0.0, (seg_counts[i] - cross) / np.maximum(1e-12, seg_durs))
    print('mu closed-form (true A)', mu_cf)


if __name__ == '__main__':
    main()

