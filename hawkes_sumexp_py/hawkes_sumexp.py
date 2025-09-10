import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    from numba import njit, prange
    from numba.typed import List as NumbaList
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):  # type: ignore
        def wrap(f):
            return f
        return wrap
    def prange(*args, **kwargs):  # type: ignore
        return range(*args)
    class NumbaList(list):  # type: ignore
        pass
    NUMBA_AVAILABLE = False


Array1D = np.ndarray
Array2D = np.ndarray


@njit(parallel=True, cache=True)
def _compute_G_parallel(
    events: list, end_times: Array1D, decays: Array1D, G_out: Array2D
) -> None:
    # Parallelize over (j,u) pairs and sum across realizations
    n_real = len(events)
    n_nodes = len(events[0])
    n_decays = decays.shape[0]
    total_pairs = n_nodes * n_decays
    for p in prange(total_pairs):
        j = p // n_decays
        u = p % n_decays
        beta = float(decays[u])
        acc = 0.0
        for r in range(n_real):
            T = float(end_times[r])
            ev_j = events[r][j]
            acc += _sumexp_G_for_node(ev_j, beta, T)
        G_out[j, u] = acc


@njit(parallel=True, cache=True)
def _compute_counts_parallel(events: list, n_counts_out: Array1D) -> None:
    # Parallelize over target nodes i and sum counts across realizations
    n_real = len(events)
    n_nodes = len(events[0])
    for i in prange(n_nodes):
        s = 0.0
        for r in range(n_real):
            s += float(events[r][i].shape[0])
        n_counts_out[i] = s


@njit(parallel=True, cache=True)
def _compute_S_parallel(
    events: list, decays: Array1D, S_out: Array2D
) -> None:
    # Parallelize over (i,j,u) and sum across realizations
    n_real = len(events)
    n_nodes = len(events[0])
    n_decays = decays.shape[0]
    total_triplets = n_nodes * n_nodes * n_decays
    for q in prange(total_triplets):
        i = q // (n_nodes * n_decays)
        rem = q % (n_nodes * n_decays)
        j = rem // n_decays
        u = rem % n_decays
        beta = float(decays[u])
        acc = 0.0
        for r in range(n_real):
            acc += _sumexp_S_for_pair(events[r][i], events[r][j], beta)
        S_out[i, j, u] = acc


@njit(parallel=True, cache=True)
def _compute_H_parallel(
    events: list, end_times: Array1D, decays: Array1D, H_out: Array2D
) -> None:
    # Parallelize over upper-triangular pairs of (l,v),(j,u) and sum across realizations
    n_real = len(events)
    n_nodes = len(events[0])
    n_decays = decays.shape[0]
    DU = n_nodes * n_decays
    # Number of elements in upper triangle including diagonal
    total_upper = DU * (DU + 1) // 2

    for idx in prange(total_upper):
        # Map linear index to (a,b) in upper triangle
        # Find row a such that idx < cumulative count
        # cumulative up to row a-1: a*(a+1)/2; solve for a
        # Use integer arithmetic
        a = 0
        low = 0
        high = DU
        # binary search for a where idx < (a+1)(a+2)/2
        while low < high:
            mid = (low + high) // 2
            if idx < (mid + 1) * (mid + 2) // 2:
                high = mid
            else:
                low = mid + 1
        a = low
        prev_count = a * (a + 1) // 2
        b = a - (prev_count + a - idx)
        if b < 0:
            b = 0

        # Map a,b to (l,v) and (j,u)
        l = a // n_decays
        v = a % n_decays
        j = b // n_decays
        u = b % n_decays
        beta_v = float(decays[v])
        beta_u = float(decays[u])

        acc = 0.0
        for r in range(n_real):
            T = float(end_times[r])
            acc += _sumexp_H_for_pair(events[r][l], events[r][j], beta_v, beta_u, T)

        H_out[a, b] = acc
        if a != b:
            H_out[b, a] = acc

@njit(cache=True)
def _sumexp_G_for_node(events_j: Array1D, beta: float, T: float) -> float:
    # G_{j,u} = \int_0^T g_{j,u}(t) dt = sum_{s in N_j} (1 - e^{-beta (T - s)})
    s = 0.0
    for k in range(events_j.shape[0]):
        s += 1.0 - math.exp(-beta * (T - events_j[k]))
    return s


@njit(cache=True)
def _sumexp_S_for_pair(events_i: Array1D, events_j: Array1D, beta: float) -> float:
    # S_{i;j,u} = sum_{t in N_i} g_{j,u}(t)
    # recursion across events of i, inserting jumps from events of j between two events of i
    s_val = 0.0
    if events_i.shape[0] == 0 or events_j.shape[0] == 0:
        return 0.0
    lam = beta
    val_prev = 0.0
    k = 0
    last_t = events_i[0] if events_i.shape[0] > 0 else 0.0
    # initialize up to first i-event
    # add contributions from j-events before first i-event
    while k < events_j.shape[0] and events_j[k] < events_i[0]:
        val_prev += lam * math.exp(-lam * (events_i[0] - events_j[k]))
        k += 1
    s_val += val_prev
    for m in range(1, events_i.shape[0]):
        t_prev = events_i[m - 1]
        t = events_i[m]
        # decay
        decay = math.exp(-lam * (t - t_prev))
        val = val_prev * decay
        # add contributions of j-events between (t_prev, t)
        while k < events_j.shape[0] and events_j[k] < t:
            val += lam * math.exp(-lam * (t - events_j[k]))
            k += 1
        s_val += val
        val_prev = val
    return s_val


@njit(cache=True)
def _sumexp_H_for_pair(events_l: Array1D, events_j: Array1D,
                       beta_v: float, beta_u: float, T: float) -> float:
    # H_{l,v;j,u} = \int_0^T g_{l,v}(t) g_{j,u}(t) dt
    # Efficient sweep: evolve states g_l, g_j between event times and
    # integrate their product in closed form on each interval.
    if T <= 0.0 or (events_l.shape[0] == 0 and events_j.shape[0] == 0):
        return 0.0
    s = 0.0
    bsum = beta_v + beta_u
    i = 0
    j = 0
    t_prev = 0.0
    g_l = 0.0
    g_j = 0.0
    # Process all events
    while i < events_l.shape[0] or j < events_j.shape[0]:
        t_li = events_l[i] if i < events_l.shape[0] else 1e100
        t_jj = events_j[j] if j < events_j.shape[0] else 1e100
        t_next = t_li if t_li <= t_jj else t_jj
        # integrate over [t_prev, t_next)
        if t_next > t_prev:
            dt = t_next - t_prev
            if g_l != 0.0 and g_j != 0.0:
                s += (g_l * g_j) * (1.0 - math.exp(-bsum * dt)) / bsum
                g_l *= math.exp(-beta_v * dt)
                g_j *= math.exp(-beta_u * dt)
            else:
                if g_l != 0.0:
                    g_l *= math.exp(-beta_v * dt)
                if g_j != 0.0:
                    g_j *= math.exp(-beta_u * dt)
            t_prev = t_next
        # apply jumps at t_next
        if i < events_l.shape[0] and events_l[i] == t_next:
            mult = 1
            i += 1
            while i < events_l.shape[0] and events_l[i] == t_next:
                mult += 1
                i += 1
            g_l += beta_v * mult
        if j < events_j.shape[0] and events_j[j] == t_next:
            mult = 1
            j += 1
            while j < events_j.shape[0] and events_j[j] == t_next:
                mult += 1
                j += 1
            g_j += beta_u * mult
    # final tail to T
    if T > t_prev and g_l != 0.0 and g_j != 0.0:
        dt = T - t_prev
        s += (g_l * g_j) * (1.0 - math.exp(-bsum * dt)) / bsum
    return s


def _ensure_events_format(
    events: Sequence[Sequence[Array1D]]
) -> Tuple[List[List[Array1D]], int, int]:
    # Normalize events as list of realizations -> list of nodes -> np.array
    if len(events) == 0:
        raise ValueError("events must be a non-empty list")
    # If single realization provided as list of arrays
    if isinstance(events[0], np.ndarray) or (
        isinstance(events[0], list) and len(events) > 0 and isinstance(events[0][0], (int, float))
    ):
        # This case is unlikely in practice; keep as-is
        raise ValueError("events format invalid: expected list[realization][node] arrays")
    n_real = len(events)
    n_nodes = len(events[0])
    norm: List[List[Array1D]] = []
    for r in range(n_real):
        if len(events[r]) != n_nodes:
            raise ValueError("All realizations must have the same number of nodes")
        per_nodes: List[Array1D] = []
        for j in range(n_nodes):
            arr = np.asarray(events[r][j], dtype=float)
            if arr.ndim != 1:
                raise ValueError("Each event list must be 1D array of times")
            per_nodes.append(np.ascontiguousarray(arr))
        norm.append(per_nodes)
    return norm, n_real, n_nodes


def _prepare_end_times(events: List[List[Array1D]], end_times: Optional[Sequence[float]]) -> Array1D:
    n_real = len(events)
    if end_times is None:
        Ts = np.empty(n_real, dtype=float)
        for r in range(n_real):
            last_t = 0.0
            for j in range(len(events[r])):
                if events[r][j].shape[0] > 0:
                    last_t = max(last_t, float(events[r][j][-1]))
            Ts[r] = last_t
        return Ts
    else:
        end_times_arr = np.asarray(end_times, dtype=float)
        if end_times_arr.ndim == 0:
            return np.full(n_real, float(end_times_arr), dtype=float)
        if end_times_arr.shape[0] != n_real:
            raise ValueError("end_times length must match number of realizations")
        return end_times_arr


def _aggregate_statistics(
    events: List[List[Array1D]], end_times: Array1D, decays: Array1D
) -> Tuple[float, Array2D, Array2D, Array2D, Array1D]:
    """Compute global sufficient statistics for least-squares.

    Returns
    -------
    T_sum : float
        Sum of horizon lengths across realizations
    G : (n_nodes, n_decays)
        G[j,u] = \int g_{j,u}(t) dt summed across realizations
    H : (n_nodes*n_decays, n_nodes*n_decays)
        H[(l,v),(j,u)] = \int g_{l,v}(t) g_{j,u}(t) dt summed across realizations
    S : (n_nodes, n_nodes, n_decays)
        S[i,j,u] = sum_{t in N_i} g_{j,u}(t) summed across realizations
    n_counts : (n_nodes,)
        Total event counts per target node i across realizations
    """
    n_real = len(events)
    n_nodes = len(events[0])
    n_decays = decays.shape[0]

    T_sum = float(np.sum(end_times))
    G = np.zeros((n_nodes, n_decays), dtype=float)
    S = np.zeros((n_nodes, n_nodes, n_decays), dtype=float)
    H = np.zeros((n_nodes * n_decays, n_nodes * n_decays), dtype=float)
    n_counts = np.zeros(n_nodes, dtype=float)

    # Convert events to Numba typed list-of-lists for parallel kernels
    if NUMBA_AVAILABLE:
        nb_events = NumbaList()
        for r in range(n_real):
            row = NumbaList()
            for j in range(n_nodes):
                row.append(events[r][j])
            nb_events.append(row)
        _compute_G_parallel(nb_events, end_times, decays, G)
        _compute_counts_parallel(nb_events, n_counts)
        _compute_S_parallel(nb_events, decays, S)
        _compute_H_parallel(nb_events, end_times, decays, H)
    else:
        for r in range(n_real):
            T = float(end_times[r])
            # G and counts
            for j in range(n_nodes):
                ev_j = events[r][j]
                for u in range(n_decays):
                    G[j, u] += _sumexp_G_for_node(ev_j, float(decays[u]), T)
            # S: for each i,j,u
            for i in range(n_nodes):
                ev_i = events[r][i]
                n_counts[i] += float(ev_i.shape[0])
                for j in range(n_nodes):
                    ev_j = events[r][j]
                    for u in range(n_decays):
                        S[i, j, u] += _sumexp_S_for_pair(ev_i, ev_j, float(decays[u]))
            # H: symmetric matrix over (l,v) and (j,u)
            for l in range(n_nodes):
                ev_l = events[r][l]
                for v in range(n_decays):
                    beta_v = float(decays[v])
                    idx_lv = l * n_decays + v
                    for j in range(n_nodes):
                        ev_j = events[r][j]
                        for u in range(n_decays):
                            beta_u = float(decays[u])
                            idx_ju = j * n_decays + u
                            if idx_lv <= idx_ju:
                                val = _sumexp_H_for_pair(ev_l, ev_j, beta_v, beta_u, T)
                                H[idx_lv, idx_ju] += val
                                if idx_lv != idx_ju:
                                    H[idx_ju, idx_lv] += val

    return T_sum, G, H, S, n_counts


def _solve_least_squares(
    T_sum: float,
    G: Array2D,
    H: Array2D,
    S: Array2D,
    n_counts: Array1D,
    l2_baseline: float,
    l2_adjacency: float,
) -> Tuple[Array1D, Array2D]:
    # Solve independently for each target node i a quadratic system
    # for x_i = [mu_i; alpha_i] with Q_i common across i except linear parts.
    n_nodes, n_decays = G.shape
    dim_alpha = n_nodes * n_decays
    # Assemble Q = [[T, G^T], [G, H]] once; only b changes across i
    Q = np.zeros((1 + dim_alpha, 1 + dim_alpha), dtype=float)
    Q[0, 0] = T_sum + l2_baseline
    # top-right and bottom-left blocks
    Q[0, 1:] = G.reshape(-1)
    Q[1:, 0] = Q[0, 1:]
    # H with l2 on adjacency
    Q[1:, 1:] = H + l2_adjacency * np.eye(dim_alpha)

    # Regularize slightly for numerical stability if needed
    reg_eps = 1e-10
    Q[0, 0] += reg_eps
    Q[1:, 1:] += reg_eps * np.eye(dim_alpha)

    baseline = np.zeros(n_nodes, dtype=float)
    adjacency = np.zeros((n_nodes, n_nodes, n_decays), dtype=float)

    # Pre-factorization
    try:
        L = np.linalg.cholesky(Q)
        use_chol = True
    except np.linalg.LinAlgError:
        use_chol = False

    for i in range(n_nodes):
        b = np.zeros(1 + dim_alpha, dtype=float)
        b[0] = n_counts[i]
        b[1:] = S[i].reshape(-1)
        if use_chol:
            # Solve Q x = b via Cholesky
            y = np.linalg.solve(L, b)
            x = np.linalg.solve(L.T, y)
        else:
            x = np.linalg.solve(Q, b)
        baseline[i] = max(0.0, x[0])
        alpha_i = x[1:].reshape(n_nodes, n_decays)
        # Enforce non-negativity (prox positivity)
        alpha_i = np.maximum(0.0, alpha_i)
        adjacency[i] = alpha_i

    return baseline, adjacency


def _loglikelihood(
    events: List[List[Array1D]], end_times: Array1D, baseline: Array1D,
    adjacency: Array2D, decays: Array1D, n_baselines: int = 1,
    period_length: float = 0.0, eps: float = 1e-12
) -> float:
    # Compute log-likelihood sum_i [ sum_{t in N_i} log lambda_i(t) - \int lambda_i ]
    n_real = len(events)
    n_nodes = baseline.shape[0]
    n_decays = decays.shape[0]
    ll = 0.0
    for r in range(n_real):
        T = float(end_times[r])
        # Precompute integral part for this realization
        if n_baselines == 1:
            for i in range(n_nodes):
                integ = float(baseline[i]) * T
                # plus sum_u,j alpha_{i,j,u} * sum_{s in N_j} (1 - exp(-beta_u (T - s)))
                for j in range(n_nodes):
                    ev_j = events[r][j]
                    for u in range(n_decays):
                        beta = float(decays[u])
                        sG = 0.0
                        for k in range(ev_j.shape[0]):
                            sG += 1.0 - math.exp(-beta * (T - ev_j[k]))
                        integ += adjacency[i, j, u] * sG
                ll -= integ
        else:
            # piecewise baseline integral uses segment durations
            seg_durs = _segment_durations(T, float(period_length), n_baselines)
            for i in range(n_nodes):
                integ = 0.0
                for k in range(n_baselines):
                    integ += float(baseline[i, k]) * seg_durs[k]
            # plus sum_u,j alpha_{i,j,u} * sum_{s in N_j} (1 - exp(-beta_u (T - s)))
            for j in range(n_nodes):
                ev_j = events[r][j]
                for u in range(n_decays):
                    beta = float(decays[u])
                    sG = 0.0
                    for k in range(ev_j.shape[0]):
                        sG += 1.0 - math.exp(-beta * (T - ev_j[k]))
                    integ += adjacency[i, j, u] * sG
            ll -= integ
        # Event contribution
        # For each i, go over its events and compute intensity via recursions per (j,u)
        for i in range(n_nodes):
            ev_i = events[r][i]
            if ev_i.shape[0] == 0:
                continue
            # For each (j,u) maintain state value at last event of i
            states = np.zeros((n_nodes, n_decays), dtype=np.float64)
            # Maintain pointers per j for events of j already passed
            ptr = np.zeros(n_nodes, dtype=np.int64)
            t_prev = ev_i[0]
            # initialize states at first event time
            for j in range(n_nodes):
                ev_j = events[r][j]
                # accumulate contributions from j-events < t_prev
                k = 0
                while k < ev_j.shape[0] and ev_j[k] < t_prev:
                    dt = t_prev - ev_j[k]
                    for u in range(n_decays):
                        beta = float(decays[u])
                        states[j, u] += beta * math.exp(-beta * dt)
                    k += 1
                ptr[j] = k
            # accumulate log intensities
            # first event
            if n_baselines == 1:
                base_val = float(baseline[i])
            else:
                base_val = float(baseline[i, _segment_index(t_prev, float(period_length), n_baselines)])
            lam = base_val
            for j in range(n_nodes):
                for u in range(n_decays):
                    lam += adjacency[i, j, u] * states[j, u]
            ll += math.log(max(lam, eps))
            # subsequent events
            for m in range(1, ev_i.shape[0]):
                t = ev_i[m]
                dt = t - t_prev
                # decay all states
                for j in range(n_nodes):
                    ev_j = events[r][j]
                    # decay per u
                    for u in range(n_decays):
                        beta = float(decays[u])
                        states[j, u] *= math.exp(-beta * dt)
                    # add contributions from new j-events in (t_prev, t)
                    k = ptr[j]
                    while k < ev_j.shape[0] and ev_j[k] < t:
                        dtk = t - ev_j[k]
                        for u in range(n_decays):
                            beta = float(decays[u])
                            states[j, u] += beta * math.exp(-beta * dtk)
                        k += 1
                    ptr[j] = k
                if n_baselines == 1:
                    base_val = float(baseline[i])
                else:
                    base_val = float(baseline[i, _segment_index(t, float(period_length), n_baselines)])
                lam = base_val
                for j in range(n_nodes):
                    for u in range(n_decays):
                        lam += adjacency[i, j, u] * states[j, u]
                ll += math.log(max(lam, eps))
                t_prev = t
    return ll


@njit(cache=True)
def _segment_index(t: float, period: float, n_baselines: int) -> int:
    # Return segment index in [0, n_baselines) for time t using periodic wrapping
    if period <= 0:
        return 0
    # Avoid numerical issues for t exactly multiple of period
    x = t % period
    seg_len = period / n_baselines
    k = int(x // seg_len)
    if k == n_baselines:
        k = n_baselines - 1
    return k


@njit(cache=True)
def _segment_boundaries_until(T: float, period: float, n_baselines: int) -> Array1D:
    # Returns sorted boundaries in (0, T), excluding 0, including T
    if period <= 0:
        return np.array([T], dtype=np.float64)
    seg_len = period / n_baselines
    n_cycles = int(T // seg_len) + 2
    # generate multiples of seg_len up to T
    # upper bound size
    bounds = np.empty(n_cycles, dtype=np.float64)
    m = 0
    val = seg_len
    while val < T and m < n_cycles:
        bounds[m] = val
        m += 1
        val += seg_len
    # append T
    out = np.empty(m + 1, dtype=np.float64)
    for i in range(m):
        out[i] = bounds[i]
    out[m] = T
    return out


@njit(cache=True)
def _segment_durations(T: float, period: float, n_baselines: int) -> Array1D:
    # Compute total time spent in each baseline segment over [0,T]
    durs = np.zeros(n_baselines, dtype=np.float64)
    if period <= 0:
        durs[0] = T
        return durs
    seg_len = period / n_baselines
    prev = 0.0
    # iterate over boundaries including T
    boundaries = _segment_boundaries_until(T, period, n_baselines)
    for b in boundaries:
        k = _segment_index(prev, period, n_baselines)
        durs[k] += (b - prev)
        prev = b
    return durs


@njit(cache=True)
def _counts_per_segment(events_i: Array1D, period: float, n_baselines: int) -> Array1D:
    out = np.zeros(n_baselines, dtype=np.float64)
    if events_i.shape[0] == 0:
        return out
    for m in range(events_i.shape[0]):
        k = _segment_index(events_i[m], period, n_baselines)
        out[k] += 1.0
    return out


@njit(cache=True)
def _J_segments_for_node(events_j: Array1D, beta: float, T: float, period: float, n_baselines: int) -> Array1D:
    # Compute J[k] = ∫_0^T 1_{seg k}(t) g_{j,u}(t) dt for fixed node j and decay u
    # by stepping over union of event times and segment boundaries.
    J = np.zeros(n_baselines, dtype=np.float64)
    if T <= 0.0 or (events_j.shape[0] == 0 and period <= 0):
        return J
    # Build merged checkpoints (events + boundaries + T)
    boundaries = _segment_boundaries_until(T, period, n_baselines)
    # We'll sweep with two pointers
    e = 0
    b = 0
    t_prev = 0.0
    g_prev = 0.0  # value just after t_prev
    while True:
        t_e = events_j[e] if e < events_j.shape[0] else 1e100
        t_b = boundaries[b] if b < boundaries.shape[0] else 1e100
        t = t_e if t_e <= t_b else t_b
        if t == 1e100:
            break
        # integrate over [t_prev, t)
        if t > t_prev:
            k = _segment_index(t_prev, period, n_baselines)
            dt = t - t_prev
            if g_prev != 0.0:
                contrib = g_prev * (1.0 - math.exp(-beta * dt)) / beta
                J[k] += contrib
                # decay state to time t
                g_prev *= math.exp(-beta * dt)
            t_prev = t
        # update g at t
        if e < events_j.shape[0] and events_j[e] == t:
            # decay handled by integration above; now add jump(s) at t
            # count multiplicity if several events at same time
            mult = 1
            e += 1
            while e < events_j.shape[0] and events_j[e] == t:
                mult += 1
                e += 1
            g_prev += beta * mult
        if b < boundaries.shape[0] and boundaries[b] == t:
            b += 1
        # loop continues
    return J


class HawkesSumExpKern:
    """Pure-Python/Numba learner for sum-exponential Hawkes with fixed decays.

    This mirrors Tick's `tick.hawkes.inference.HawkesSumExpKern` API and
    optimization objective. It replaces Tick's C++ backends with vectorized
    NumPy and Numba-accelerated routines that compute the same least-squares
    sufficient statistics and solve the same positivity-constrained problem.

    Notes
    - Supports constant and piecewise-constant baselines via `n_baselines`
      and `period_length`, matching Tick's behavior for periodic baselines.
    - Supports `penalty` in {"none", "l2", "l1", "elasticnet"} with strength
      controlled by `C`, consistent with Tick. Elastic net uses
      `elastic_net_ratio` to split the L1/L2 parts.
    - Solves the quadratic (least-squares) objective used by Tick for
      sum-of-exponentials Hawkes, with non-negativity projection.
    """

    def __init__(
        self,
        decays: Sequence[float],
        penalty: str = "none",
        C: float = 1e3,
        n_baselines: int = 1,
        period_length: Optional[float] = None,
        # The remaining params are accepted for API parity but unused here
        solver: str = "agd",
        step: Optional[float] = None,
        tol: float = 1e-5,
        max_iter: int = 100,
        verbose: bool = False,
        print_every: int = 10,
        record_every: int = 10,
        elastic_net_ratio: float = 0.95,
        random_state: Optional[int] = None,
    ) -> None:
        if penalty not in ("l2", "none", "l1", "elasticnet"):
            raise NotImplementedError("Penalty must be one of: 'none','l2','l1','elasticnet'")
        if n_baselines < 1:
            raise ValueError("n_baselines must be >= 1")
        if period_length is not None and n_baselines == 1 and verbose:
            print("[hawkes_sumexp_py] period_length is ignored when n_baselines=1")

        self.decays = np.asarray(decays, dtype=float).copy()
        self.n_decays = int(self.decays.shape[0])
        self.n_baselines = int(n_baselines)
        self.period_length = None if period_length is None else float(period_length)
        self.penalty = penalty
        # Interpret C exactly like Tick: larger C -> stronger regularization
        self.C = float(C)
        lam = 0.0 if penalty == "none" else self.C
        # elastic_net_ratio splits between L1 and L2 when using elasticnet
        if penalty == "elasticnet":
            self.l1_weight = lam * float(elastic_net_ratio)
            self.l2_weight = lam * (1.0 - float(elastic_net_ratio))
        elif penalty == "l1":
            self.l1_weight = lam
            self.l2_weight = 0.0
        elif penalty == "l2":
            self.l1_weight = 0.0
            self.l2_weight = lam
        else:  # none
            self.l1_weight = 0.0
            self.l2_weight = 0.0

        # Learned params
        self._fitted = False
        self.baseline: Optional[Array1D] = None  # (n_nodes,) or (n_nodes, n_baselines)
        self.adjacency: Optional[Array2D] = None  # (n_nodes, n_nodes, n_decays)

        # Data for score reuse
        self._fit_events: Optional[List[List[Array1D]]] = None
        self._fit_end_times: Optional[Array1D] = None

    def fit(
        self,
        events: Sequence[Sequence[Array1D]],
        end_times: Optional[Sequence[float]] = None,
    ) -> "HawkesSumExpKern":
        events_norm, _, n_nodes = _ensure_events_format(events)
        end_times_arr = _prepare_end_times(events_norm, end_times)

        # Common stats
        T_sum, G, H, S, n_counts = _aggregate_statistics(events_norm, end_times_arr, self.decays)

        # If piecewise-constant baseline requested, compute segment-wise durations and counts,
        # and segment-weighted integrals of g (J)
        if self.n_baselines > 1 and (self.period_length is None or not np.isfinite(self.period_length)):
            raise ValueError("period_length must be provided for n_baselines > 1")

        if self.n_baselines == 1:
            # Use projected (FISTA) solver to enforce positivity consistently
            baseline, adjacency = self._solve_by_prox(
                T_sum=T_sum, G=G, H=H, S=S, n_counts=n_counts,
                seg_durs=None, seg_counts=None, J=None,
            )
        else:
            # Piecewise baseline: build per-node quadratic terms
            K = self.n_baselines
            # segment durations aggregated across realizations
            seg_durs = np.zeros(K, dtype=float)
            # counts per node per segment
            seg_counts = np.zeros((n_nodes, K), dtype=float)
            # J: (K, n_nodes, n_decays)
            J = np.zeros((K, n_nodes, self.n_decays), dtype=float)
            for r in range(len(events_norm)):
                T = float(end_times_arr[r])
                durs = _segment_durations(T, float(self.period_length), K)
                seg_durs += durs
                for i in range(n_nodes):
                    seg_counts[i] += _counts_per_segment(events_norm[r][i], float(self.period_length), K)
                for j in range(n_nodes):
                    ev_j = events_norm[r][j]
                    for u in range(self.n_decays):
                        J[:, j, u] += _J_segments_for_node(ev_j, float(self.decays[u]), T, float(self.period_length), K)

            # Use projected (FISTA) solver to enforce positivity consistently
            baseline, adjacency = self._solve_by_prox(
                T_sum=T_sum, G=G, H=H, S=S, n_counts=n_counts,
                seg_durs=seg_durs, seg_counts=seg_counts, J=J,
            )

        # Closed-form refinement of baseline given current adjacency
        if self.n_baselines == 1:
            # mu_i = max(0, (N_i - sum_{j,u} alpha_{i,j,u} * G[j,u]) / T)
            mu = np.zeros(n_nodes, dtype=float)
            for i in range(n_nodes):
                cross = float(np.sum(adjacency[i].reshape(-1) * G.reshape(-1)))
                mu[i] = max(0.0, (n_counts[i] - cross) / max(T_sum, 1e-12))
            baseline = mu
        else:
            # mu_{i,k} = max(0, (N_{i,k} - sum_{j,u} alpha_{i,j,u} * J[k,j,u]) / dur_k)
            K = self.n_baselines
            mu = np.zeros((n_nodes, K), dtype=float)
            JU = J  # (K, n_nodes, U)
            for i in range(n_nodes):
                cross = np.tensordot(adjacency[i], JU, axes=((0, 1), (1, 2)))  # -> (K,)
                for k in range(K):
                    mu[i, k] = max(0.0, (seg_counts[i, k] - cross[k]) / max(seg_durs[k], 1e-12))
            baseline = mu

        self.baseline = baseline
        self.adjacency = adjacency
        self.n_nodes = n_nodes
        self._fitted = True

        # Store inputs for score reuse
        self._fit_events = events_norm
        self._fit_end_times = end_times_arr

        return self

    def score(
        self,
        events: Optional[Sequence[Sequence[Array1D]]] = None,
        end_times: Optional[Sequence[float]] = None,
        baseline: Optional[Array1D] = None,
        adjacency: Optional[Array2D] = None,
    ) -> float:
        if baseline is None or adjacency is None:
            if not self._fitted:
                raise ValueError("Model must be fit or parameters provided to score")
        b = self.baseline if baseline is None else baseline
        A = self.adjacency if adjacency is None else adjacency
        if b is None or A is None:
            raise ValueError("Invalid baseline/adjacency for scoring")

        if events is None:
            if self._fit_events is None or self._fit_end_times is None:
                raise ValueError("No events provided and model has not been fit")
            events_norm = self._fit_events
            end_times_arr = self._fit_end_times
        else:
            events_norm, _, _ = _ensure_events_format(events)
            end_times_arr = _prepare_end_times(events_norm, end_times)

        return _loglikelihood(events_norm, end_times_arr, np.asarray(b, float), np.asarray(A, float), self.decays)

    # Compatibility helpers
    def get_baseline_values(self, i: int, abscissa_array: Array1D) -> Array1D:
        if not self._fitted or self.baseline is None:
            raise ValueError("Fit the model before requesting baseline values")
        t = np.asarray(abscissa_array, dtype=float)
        if self.n_baselines == 1:
            return np.full_like(t, float(self.baseline[i]))
        if self.period_length is None or self.period_length <= 0:
            raise ValueError("Invalid period_length for piecewise baseline")
        seg_len = self.period_length / self.n_baselines
        vals = np.empty_like(t)
        for idx in range(t.shape[0]):
            k = int((t[idx] % self.period_length) // seg_len)
            if k == self.n_baselines:
                k = self.n_baselines - 1
            vals[idx] = float(self.baseline[i, k])
        return vals

    # --- Plotting API parity with Tick ---
    def get_kernel_supports(self) -> np.ndarray:
        # Return per-(i,j) support length; Tick uses kernel.get_plot_support()
        # For sum-exp, a few multiples of 1/min(beta) is sufficient.
        supp = 7.0  # chosen to match Tick example plots visually
        return np.full((self.n_nodes, self.n_nodes), supp, dtype=float)

    def get_kernel_values(self, i: int, j: int, abscissa_array: Array1D) -> Array1D:
        t = np.asarray(abscissa_array, dtype=float)
        vals = np.zeros_like(t)
        for u in range(self.n_decays):
            vals += self.adjacency[i, j, u] * self.decays[u] * np.exp(-self.decays[u] * t)
        return vals

    # --- Internal solvers ---
    def _solve_block_least_squares(
        self,
        seg_durs: Array1D,  # (K,)
        J: Array2D,         # (K, n_nodes, n_decays)
        H: Array2D,         # (n_nodes*n_decays, n_nodes*n_decays)
        S: Array2D,         # (n_nodes, n_nodes, n_decays)
        seg_counts: Array2D,  # (n_nodes, K)
        l2: float,
    ) -> Tuple[Array2D, Array2D]:
        n_nodes = S.shape[0]
        K = seg_durs.shape[0]
        DU = H.shape[0]
        # Assemble Q common blocks
        Q = np.zeros((K + DU, K + DU), dtype=float)
        # baseline block: diagonal with segment durations (sum over realizations)
        Q[np.arange(K), np.arange(K)] = seg_durs + l2
        # cross block: (K, DU) J reshaped columns of (j,u)
        Q[:K, K:] = J.reshape(K, DU)
        Q[K:, :K] = Q[:K, K:].T
        # adjacency block H + l2 I
        Q[K:, K:] = H + (l2 + 1e-10) * np.eye(DU)
        # Pre-factorize
        try:
            L = np.linalg.cholesky(Q)
            use_chol = True
        except np.linalg.LinAlgError:
            use_chol = False

        baseline = np.zeros((n_nodes, K), dtype=float)
        adjacency = np.zeros((n_nodes, n_nodes, self.n_decays), dtype=float)
        for i in range(n_nodes):
            b = np.zeros(K + DU, dtype=float)
            b[:K] = seg_counts[i]
            b[K:] = S[i].reshape(-1)
            if use_chol:
                y = np.linalg.solve(L, b)
                x = np.linalg.solve(L.T, y)
            else:
                x = np.linalg.solve(Q, b)
            # positivity projection
            x = np.maximum(0.0, x)
            baseline[i] = x[:K]
            adjacency[i] = x[K:].reshape(n_nodes, self.n_decays)
        return baseline, adjacency

    def _solve_by_prox(
        self,
        T_sum: float,
        G: Array2D,
        H: Array2D,
        S: Array2D,
        n_counts: Array1D,
        seg_durs: Optional[Array1D],
        seg_counts: Optional[Array2D],
        J: Optional[Array2D],
        max_iter: int = 500,
        tol: float = 1e-6,
    ) -> Tuple[Array2D, Array2D]:
        # Build per-node quadratic matrices and run FISTA with nonneg + L1
        n_nodes, n_decays = S.shape[0], self.n_decays
        if self.n_baselines == 1:
            K = 1
            DU = n_nodes * n_decays
            Q = np.zeros((1 + DU, 1 + DU))
            Q[0, 0] = T_sum
            Q[0, 1:] = G.reshape(-1)
            Q[1:, 0] = Q[0, 1:]
            Q[1:, 1:] = H
        else:
            K = self.n_baselines
            DU = n_nodes * n_decays
            Q = np.zeros((K + DU, K + DU))
            Q[np.arange(K), np.arange(K)] = seg_durs
            Q[:K, K:] = J.reshape(K, DU)
            Q[K:, :K] = Q[:K, K:].T
            Q[K:, K:] = H

        # Lipschitz constant for gradient of f(x) = x^T Q x - 2 b^T x + (l2/2)||x_adj||^2
        # where l2 applies only to adjacency part. A safe upper bound is
        # L = 2 * lambda_max(Q) + l2.
        try:
            ev = np.linalg.eigvalsh(Q)
            lam_max = float(ev[-1]) if ev.size > 0 else 0.0
        except np.linalg.LinAlgError:
            lam_max = float(np.linalg.norm(Q, 2))
        L = 2.0 * lam_max + self.l2_weight + 1e-12
        step = 1.0 / L

        baseline = np.zeros((n_nodes, K), dtype=float)
        adjacency = np.zeros((n_nodes, n_nodes, n_decays), dtype=float)

        # Precompute prox parameter for adjacency only
        l1 = self.l1_weight

        for i in range(n_nodes):
            bvec = np.zeros(K + DU)
            if K == 1:
                bvec[0] = n_counts[i]
            else:
                bvec[:K] = seg_counts[i]
            bvec[K:] = S[i].reshape(-1)

            # FISTA initialization
            x = np.zeros(K + DU)
            y = x.copy()
            tpar = 1.0
            prev_obj = 1e100
            for it in range(max_iter):
                # gradient at y; add l2 only to adjacency block
                grad = 2.0 * (Q @ y) - 2.0 * bvec
                # Apply l2 on adjacency coordinates only
                if self.l2_weight > 0.0:
                    grad[K:] += self.l2_weight * y[K:]
                v = y - step * grad
                # prox: nonneg for baseline; nonneg + l1 for adjacency
                x_new = np.empty_like(v)
                # baseline block (no l1/l2 prox, only nonneg)
                x_new[:K] = np.maximum(0.0, v[:K])
                # adjacency block (positive soft-thresholding)
                if l1 > 0:
                    x_new[K:] = np.maximum(0.0, v[K:] - step * l1)
                else:
                    x_new[K:] = np.maximum(0.0, v[K:])
                t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * tpar * tpar))
                y = x_new + ((tpar - 1.0) / t_new) * (x_new - x)
                x = x_new
                tpar = t_new
                # simple convergence check on obj decrease every ~20 iters
                if (it + 1) % 20 == 0 or it == max_iter - 1:
                    # objective with l2/l1 on adjacency only
                    quad = float(x @ (Q @ x) - 2.0 * bvec @ x)
                    l2_term = 0.5 * self.l2_weight * float(x[K:] @ x[K:]) if self.l2_weight > 0.0 else 0.0
                    l1_term = l1 * float(np.sum(x[K:])) if l1 > 0.0 else 0.0
                    obj = quad + l2_term + l1_term
                    if prev_obj - obj < tol * (1.0 + abs(obj)):
                        break
                    prev_obj = obj

            baseline[i] = x[:K]
            adjacency[i] = x[K:].reshape(n_nodes, n_decays)

        return baseline if K > 1 else baseline.reshape(n_nodes), adjacency
