import os
import sys
import numpy as np
import time

# Ensure project root on path when running from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import hawkes_sumexp_py.hawkes_sumexp as hmod
from hawkes_sumexp_py import SimuHawkesSumExpKernels, SimuHawkesMulti, HawkesSumExpKern


def simulate_small(seed=123):
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
        seed=seed,
        verbose=False,
    )
    hawkes.end_time = 500.0
    hawkes.adjust_spectral_radius(0.5)
    multi = SimuHawkesMulti(hawkes, n_simulations=3)
    multi.simulate()
    return decays, n_baselines, period_length, multi.timestamps


def _compute_G_parallel_old(events, end_times, decays):
    njit = hmod.njit
    _sumexp_G_for_node = hmod._sumexp_G_for_node

    @njit(parallel=True, cache=False)
    def kernel(E, T, D):
        n_real = len(E)
        n_nodes = len(E[0])
        U = D.shape[0]
        G = np.zeros((n_nodes, U))
        total = n_nodes * U
        for p in hmod.prange(total):
            j = p // U
            u = p % U
            beta = float(D[u])
            acc = 0.0
            for r in range(n_real):
                acc += _sumexp_G_for_node(E[r][j], beta, float(T[r]))
            G[j, u] = acc
        return G

    # typed list wrapping
    nbE = hmod.NumbaList()
    for r in range(len(events)):
        row = hmod.NumbaList()
        for j in range(len(events[0])):
            row.append(events[r][j])
        nbE.append(row)
    return kernel(nbE, end_times, decays)


def _compute_S_parallel_old(events, decays):
    njit = hmod.njit
    _sumexp_S_for_pair = hmod._sumexp_S_for_pair

    @njit(parallel=True, cache=False)
    def kernel(E, D):
        n_real = len(E)
        n_nodes = len(E[0])
        U = D.shape[0]
        S = np.zeros((n_nodes, n_nodes, U))
        total = n_nodes * n_nodes * U
        for q in hmod.prange(total):
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

    nbE = hmod.NumbaList()
    for r in range(len(events)):
        row = hmod.NumbaList()
        for j in range(len(events[0])):
            row.append(events[r][j])
        nbE.append(row)
    return kernel(nbE, decays)


def _compute_H_parallel_old(events, end_times, decays):
    njit = hmod.njit
    _sumexp_H_for_pair = hmod._sumexp_H_for_pair

    @njit(parallel=True, cache=False)
    def kernel(E, T, D):
        n_real = len(E)
        n_nodes = len(E[0])
        U = D.shape[0]
        DU = n_nodes * U
        H = np.zeros((DU, DU))
        total_upper = DU * (DU + 1) // 2
        for idx in hmod.prange(total_upper):
            # map idx to (a,b) in upper triangle using sqrt formula
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

    import math
    nbE = hmod.NumbaList()
    for r in range(len(events)):
        row = hmod.NumbaList()
        for j in range(len(events[0])):
            row.append(events[r][j])
        nbE.append(row)
    return kernel(nbE, end_times, decays)


def aggregate_old(events, end_times, decays):
    n_real = len(events)
    n_nodes = len(events[0])
    T_sum = float(np.sum(end_times))
    G = _compute_G_parallel_old(events, end_times, decays)
    S = _compute_S_parallel_old(events, decays)
    H = _compute_H_parallel_old(events, end_times, decays)
    counts = np.zeros(n_nodes)
    for i in range(n_nodes):
        s = 0.0
        for r in range(n_real):
            s += float(events[r][i].shape[0])
        counts[i] = s
    return T_sum, G, H, S, counts


def main():
    # Simulate and normalize
    decays, n_baselines, period_length, events = simulate_small()
    E_norm, _, _ = hmod._ensure_events_format(events)
    T_arr = hmod._prepare_end_times(E_norm, None)

    # New stats
    t0 = time.perf_counter()
    T_sum1, G1, H1, S1, n1 = hmod._aggregate_statistics(E_norm, T_arr, decays)
    t1 = time.perf_counter()

    # Old stats
    t2 = time.perf_counter()
    T_sum2, G2, H2, S2, n2 = aggregate_old(E_norm, T_arr, decays)
    t3 = time.perf_counter()

    print('T_sum equal:', T_sum1 == T_sum2)
    print('max|G diff| =', float(np.max(np.abs(G1 - G2))))
    print('max|S diff| =', float(np.max(np.abs(S1 - S2))))
    print('max|H diff| =', float(np.max(np.abs(H1 - H2))))
    print('max|n diff| =', float(np.max(np.abs(n1 - n2))))
    print(f'new time: {t1 - t0:.4f}s  old time: {t3 - t2:.4f}s')


if __name__ == '__main__':
    main()

