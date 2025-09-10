import os
import sys
import time
import math
import numpy as np

# Ensure project root on path when running from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import hawkes_sumexp_py.hawkes_sumexp as hmod
from hawkes_sumexp_py.hawkes_sumexp import (
    _ensure_events_format, _prepare_end_times,
)


def make_synthetic_events(n_nodes=40, n_real=128, T=500.0, rate=0.1, seed=42):
    rng = np.random.default_rng(seed)
    events = []
    for r in range(n_real):
        per_nodes = []
        for j in range(n_nodes):
            # Poisson number of events and sorted uniform times in [0, T]
            n = rng.poisson(rate * T)
            if n > 0:
                t = np.sort(rng.uniform(0.0, T, size=n).astype(np.float64))
            else:
                t = np.empty(0, dtype=np.float64)
            per_nodes.append(t)
        events.append(per_nodes)
    return events, np.array([T] * n_real, dtype=np.float64)


def old_aggregator_stats(events, end_times, decays):
    # Re-implement the pre-optimization parallel pattern: S and G per-decay, H per (v,u)
    njit = hmod.njit
    prange = hmod.prange
    _sumexp_G_for_node = hmod._sumexp_G_for_node
    _sumexp_S_for_pair = hmod._sumexp_S_for_pair
    _sumexp_H_for_pair = hmod._sumexp_H_for_pair

    @njit(parallel=True, cache=False)
    def G_kernel(E, T, D):
        n_real = len(E)
        n_nodes = len(E[0])
        U = D.shape[0]
        G = np.zeros((n_nodes, U))
        total = n_nodes * U
        for p in prange(total):
            j = p // U
            u = p % U
            beta = float(D[u])
            acc = 0.0
            for r in range(n_real):
                acc += _sumexp_G_for_node(E[r][j], beta, float(T[r]))
            G[j, u] = acc
        return G

    @njit(parallel=True, cache=False)
    def S_kernel(E, D):
        n_real = len(E)
        n_nodes = len(E[0])
        U = D.shape[0]
        S = np.zeros((n_nodes, n_nodes, U))
        total = n_nodes * n_nodes * U
        for q in prange(total):
            i = q // (n_nodes * U)
            rem = q % (n_nodes * U)
            j = rem // U
            u = rem % U
            beta = float(D[u])
            acc = 0.0
            for r in range(n_real):
                acc += _sumexp_S_for_pair(E[r][i], E[r][j], beta)
            S[i, j, u] = acc
        return S

    @njit(parallel=True, cache=False)
    def H_kernel(E, T, D):
        n_real = len(E)
        n_nodes = len(E[0])
        U = D.shape[0]
        DU = n_nodes * U
        H = np.zeros((DU, DU))
        total_upper = DU * (DU + 1) // 2
        for idx in prange(total_upper):
            # Map linear index to (a,b) in upper triangle using formula
            a = int((math.sqrt(8.0 * idx + 1.0) - 1.0) // 2)
            prev = a * (a + 1) // 2
            b = int(idx - prev)
            l = a // U
            v = a % U
            j = b // U
            u = b % U
            beta_v = float(D[v])
            beta_u = float(D[u])
            acc = 0.0
            for r in range(n_real):
                acc += _sumexp_H_for_pair(E[r][l], E[r][j], beta_v, beta_u, float(T[r]))
            H[a, b] = acc
            if a != b:
                H[b, a] = acc
        return H

    # typed-list conversion for numba
    nbE = hmod.NumbaList()
    for r in range(len(events)):
        row = hmod.NumbaList()
        for j in range(len(events[0])):
            row.append(events[r][j])
        nbE.append(row)
    G = G_kernel(nbE, end_times, decays)
    S = S_kernel(nbE, decays)
    H = H_kernel(nbE, end_times, decays)
    # counts
    n_nodes = len(events[0])
    counts = np.zeros(n_nodes)
    for i in range(n_nodes):
        s = 0.0
        for r in range(len(events)):
            s += float(events[r][i].shape[0])
        counts[i] = s
    return float(np.sum(end_times)), G, H, S, counts


def run_benchmark():
    # Larger synthetic dataset, but kept manageable for CI/runtime limits
    n_nodes = 40
    n_real = 128
    T = 500.0
    rate = 0.1
    decays = np.array([0.5, 2.0, 6.0], dtype=np.float64)

    events, T_arr = make_synthetic_events(n_nodes=n_nodes, n_real=n_real, T=T, rate=rate)
    E_norm, _, _ = _ensure_events_format(events)
    end_times = _prepare_end_times(E_norm, T_arr)

    # New aggregator timing
    t0 = time.perf_counter()
    T_sum1, G1, H1, S1, n1 = hmod._aggregate_statistics(E_norm, end_times, decays)
    t1 = time.perf_counter()

    # Old aggregator timing
    t2 = time.perf_counter()
    T_sum2, G2, H2, S2, n2 = old_aggregator_stats(E_norm, end_times, decays)
    t3 = time.perf_counter()

    # Equality checks
    print('Equality checks:')
    print('  T_sum equal:', T_sum1 == T_sum2)
    print('  max|G diff| =', float(np.max(np.abs(G1 - G2))))
    print('  max|S diff| =', float(np.max(np.abs(S1 - S2))))
    print('  max|H diff| =', float(np.max(np.abs(H1 - H2))))
    print('  max|n diff| =', float(np.max(np.abs(n1 - n2))))

    print('Timings:')
    print(f'  New aggregator: {t1 - t0:.3f} s')
    print(f'  Old aggregator: {t3 - t2:.3f} s')
    if (t1 - t0) > 0:
        print(f'  Speedup (old/new): {(t3 - t2) / (t1 - t0):.2f}x')

    # Fit parameter equality using the stats
    learner = hmod.HawkesSumExpKern(decays=decays, n_baselines=1)
    # Solve with new stats
    b_new, A_new = learner._solve_by_prox(T_sum=T_sum1, G=G1, H=H1, S=S1, n_counts=n1,
                                          seg_durs=None, seg_counts=None, J=None)
    # Closed-form baseline refinement (same as fit)
    mu_new = np.zeros(n_nodes)
    cross_new = G1.reshape(-1)
    for i in range(n_nodes):
        mu_new[i] = max(0.0, (n1[i] - float(np.sum(A_new[i].reshape(-1) * cross_new))) / max(T_sum1, 1e-12))

    # Solve with old stats
    b_old, A_old = learner._solve_by_prox(T_sum=T_sum2, G=G2, H=H2, S=S2, n_counts=n2,
                                          seg_durs=None, seg_counts=None, J=None)
    mu_old = np.zeros(n_nodes)
    cross_old = G2.reshape(-1)
    for i in range(n_nodes):
        mu_old[i] = max(0.0, (n2[i] - float(np.sum(A_old[i].reshape(-1) * cross_old))) / max(T_sum2, 1e-12))

    print('Fit param diffs:')
    print('  max|baseline diff| =', float(np.max(np.abs(mu_new - mu_old))))
    print('  max|adjacency diff| =', float(np.max(np.abs(A_new - A_old))))


if __name__ == '__main__':
    run_benchmark()

