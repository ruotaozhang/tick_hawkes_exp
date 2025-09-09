import math
import copy
import multiprocessing
from multiprocessing import Pool
from typing import List, Tuple, Optional

import numpy as np

from .hawkes_sumexp import _segment_index, _segment_durations


class HawkesKernel0Py:
    def get_values(self, t_values: np.ndarray) -> np.ndarray:
        t = np.asarray(t_values, dtype=float)
        return np.zeros_like(t)


class HawkesKernelSumExpPy:
    def __init__(self, intensities: np.ndarray, decays: np.ndarray):
        self.intensities = np.asarray(intensities, dtype=float)
        self.decays = np.asarray(decays, dtype=float)

    def get_values(self, t_values: np.ndarray) -> np.ndarray:
        t = np.asarray(t_values, dtype=float)
        y = np.zeros_like(t)
        for u in range(self.decays.shape[0]):
            y += self.intensities[u] * self.decays[u] * np.exp(-self.decays[u] * t)
        return y


def scale_adjacency_spectral_radius(adjacency: np.ndarray, target_rho: float) -> np.ndarray:
    """Scale sum_u adjacency[:,:,u] to have spectral radius `target_rho`.

    Parameters
    ----------
    adjacency : np.ndarray, shape (D, D, U)
        Nonnegative adjacency tensor with kernel norms α_{i,j,u}.
    target_rho : float
        Desired spectral radius (<1 for stationarity).
    """
    D, _, U = adjacency.shape
    B = adjacency.sum(axis=2)  # branching matrix (kernel norms)
    # compute spectral radius
    ev = np.linalg.eigvals(B)
    rho = float(max(ev.real)) if ev.size > 0 else 0.0
    if rho == 0.0:
        return adjacency
    scale = target_rho / rho
    return adjacency * scale


def simulate_hawkes_multid_sumexp(
    mu: np.ndarray,  # (D,)
    adjacency: np.ndarray,  # (D,D,U) kernel norms α
    betas: np.ndarray,  # (U,)
    T: float,
    seed: int = 0,
) -> List[np.ndarray]:
    """Multi-dimensional Hawkes simulation with sum-exponential kernels via thinning.

    Intensity: λ_i(t) = μ_i + Σ_j Σ_u α_{i,j,u} g_{j,u}(t), with g_{j,u} obeying
    g' = -β_u g + Σ_{s∈N_j} β_u δ(t-s).
    """
    rng = np.random.RandomState(seed)
    D = int(mu.shape[0])
    U = int(betas.shape[0])
    assert adjacency.shape == (D, D, U)
    t = 0.0
    events: List[list] = [[] for _ in range(D)]
    states = np.zeros((D, U), dtype=float)  # g_{j,u}

    def intensities_from_states(states_arr: np.ndarray) -> np.ndarray:
        # λ_i = μ_i + Σ_{j,u} α_{i,j,u} * g_{j,u}
        # compute via tensordot: (D,D,U) • (D,U) -> (D,)
        # tensordot over axes ((1,2),(0,1)) => sum_{j,u}
        return mu + np.tensordot(adjacency, states_arr, axes=((1, 2), (0, 1)))

    lam_vec = intensities_from_states(states)
    lam_sum = float(np.sum(lam_vec))
    while True:
        if lam_sum <= 0:
            # no events will occur
            break
        w = rng.exponential(1.0 / lam_sum)
        t_cand = t + w
        if t_cand > T:
            break
        # decay states to t_cand
        decay = np.exp(-betas * (t_cand - t))  # (U,)
        states *= decay  # broadcast over columns
        # intensities at candidate
        lam_vec = intensities_from_states(states)
        lam_sum_cand = float(np.sum(lam_vec))
        # thinning acceptance
        if rng.uniform() <= lam_sum_cand / lam_sum:
            # accept: choose dimension i with probabilities lam_vec / lam_sum_cand
            if lam_sum_cand <= 0:
                # extremely unlikely numerical issue
                lam_sum_cand = 1e-12
                lam_vec = lam_vec + 1e-12
            probs = lam_vec / lam_sum_cand
            i = rng.choice(np.arange(D), p=probs)
            # register event
            t = t_cand
            events[i].append(t)
            # jump on source j = i
            states[i, :] += betas
            # recompute intensities after jump
            lam_vec = intensities_from_states(states)
            lam_sum = float(np.sum(lam_vec))
        else:
            # reject: move time and update lam_sum to cand without jump
            t = t_cand
            lam_sum = lam_sum_cand
    return [np.array(ev, dtype=float) for ev in events]


class SimuHawkesSumExpKernels:
    """Pure-Python/Numba simulation of Hawkes with sum-exponential kernels.

    Mirrors tick.hawkes.simulation.SimuHawkesSumExpKernels (subset):
    - adjacency: (D,D,U) kernel intensities α_{i,j,u}
    - decays: (U,) β values shared across all kernels
    - baseline: array (D,) constant, or (D,K) piecewise-constant with period_length
    - end_time or max_jumps to stop
    - force_simulation: allow spectral radius >= 1
    - track_intensity(step): records intensities on a uniform grid after simulate()
    """

    def __init__(
        self,
        adjacency: np.ndarray,
        decays: np.ndarray,
        baseline: Optional[np.ndarray] = None,
        end_time: Optional[float] = None,
        period_length: Optional[float] = None,
        max_jumps: Optional[int] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
        force_simulation: bool = False,
    ) -> None:
        adjacency = np.asarray(adjacency, dtype=float)
        decays = np.asarray(decays, dtype=float)
        if adjacency.ndim != 3:
            raise ValueError("adjacency must have shape (D,D,U)")
        D = adjacency.shape[0]
        U = adjacency.shape[2]
        if adjacency.shape != (D, D, U):
            raise ValueError(f"adjacency shape must be (D,D,U); got {adjacency.shape}")
        if decays.ndim != 1 or decays.shape[0] != U:
            raise ValueError("decays must have shape (U,) matching adjacency")

        if baseline is None:
            baseline_arr = np.zeros(D, dtype=float)
            n_baselines = 1
        else:
            baseline_arr = np.asarray(baseline, dtype=float)
            if baseline_arr.ndim == 1:
                if baseline_arr.shape[0] != D:
                    raise ValueError("baseline shape (D,) must match number of nodes")
                n_baselines = 1
            elif baseline_arr.ndim == 2:
                if baseline_arr.shape[0] != D:
                    raise ValueError("baseline shape (D,K) must match number of nodes")
                if period_length is None:
                    raise ValueError("period_length must be provided when baseline has multiple segments")
                n_baselines = baseline_arr.shape[1]
            else:
                raise ValueError("baseline must be (D,) or (D,K) array")

        self.adjacency = adjacency
        self.decays = decays
        self.baseline = baseline_arr
        self._n_baselines = n_baselines
        self.period_length = None if period_length is None else float(period_length)
        self.end_time = None if end_time is None else float(end_time)
        self.max_jumps = None if max_jumps is None else int(max_jumps)
        self.seed = None if seed is None else int(seed)
        self.verbose = bool(verbose)
        self.force_simulation = bool(force_simulation)

        # outputs
        self.timestamps: List[np.ndarray] = [np.empty(0, dtype=float) for _ in range(D)]
        self.simulation_time: float = 0.0
        self.n_total_jumps: int = 0

        # tracking
        self._track_step: Optional[float] = None
        self.tracked_intensity: Optional[List[np.ndarray]] = None
        self.intensity_tracked_times: Optional[np.ndarray] = None

        # RNG
        self._rng = np.random.RandomState(self.seed if self.seed is not None else None)

        # stability check
        if not self.force_simulation and self.spectral_radius() >= 1.0:
            raise ValueError("Unstable process (spectral radius >= 1); set force_simulation=True to proceed")

        # Build kernel objects for plotting parity with Tick
        self._build_sumexp_kernels()

    @property
    def n_nodes(self) -> int:
        return self.adjacency.shape[0]

    @property
    def n_decays(self) -> int:
        return self.decays.shape[0]

    def track_intensity(self, step: float) -> None:
        self._track_step = float(step)

    def spectral_radius(self) -> float:
        B = self.adjacency.sum(axis=2)
        if B.size == 0:
            return 0.0
        vals = np.linalg.eigvals(B)
        return float(np.max(vals.real))

    def adjust_spectral_radius(self, spectral_radius: float) -> None:
        cur = self.spectral_radius()
        if cur == 0:
            return
        self.adjacency = self.adjacency * (float(spectral_radius) / cur)
        # Rebuild kernels to reflect change
        self._build_sumexp_kernels()

    def _baseline_at_time(self, t: float) -> np.ndarray:
        if self._n_baselines == 1:
            return self.baseline
        if self.period_length is None or self.period_length <= 0:
            raise ValueError("Invalid period_length for piecewise baseline")
        k = _segment_index(t, float(self.period_length), int(self._n_baselines))
        return self.baseline[:, k]

    def simulate(self) -> "SimuHawkesSumExpKernels":
        D = self.n_nodes
        U = self.n_decays
        if self.end_time is None and self.max_jumps is None:
            raise ValueError("Provide end_time and/or max_jumps for simulation")

        T = self.end_time if self.end_time is not None else np.inf
        max_jumps = self.max_jumps if self.max_jumps is not None else np.inf

        # state g_{j,u}
        states = np.zeros((D, U), dtype=float)
        t = 0.0
        events: List[list] = [[] for _ in range(D)]
        n_jumps = 0

        # next baseline boundary time
        if self._n_baselines == 1:
            next_bndry = np.inf
            seg_len = np.inf
        else:
            seg_len = float(self.period_length) / float(self._n_baselines)
            # next multiple of seg_len strictly greater than 0
            next_bndry = seg_len

        # initial intensities
        base = self._baseline_at_time(0.0)
        lam_vec = base + np.tensordot(self.adjacency, states, axes=((1, 2), (0, 1)))
        lam_sum = float(np.sum(lam_vec))

        while t < T and n_jumps < max_jumps:
            if lam_sum <= 0:
                # no more events
                break
            # propose candidate interval, cap to next boundary if piecewise baseline
            w = self._rng.exponential(1.0 / lam_sum)
            t_cand = t + w
            if t_cand >= next_bndry:
                # jump to boundary without event; decay states and refresh baselines
                if next_bndry > T:
                    t = T
                    break
                decay = np.exp(-self.decays * (next_bndry - t))
                states *= decay  # broadcast on columns
                t = next_bndry
                # update boundary to next
                next_bndry += seg_len
                # recompute intensities
                base = self._baseline_at_time(t)
                lam_vec = base + np.tensordot(self.adjacency, states, axes=((1, 2), (0, 1)))
                lam_sum = float(np.sum(lam_vec))
                continue

            # decay to candidate time
            decay = np.exp(-self.decays * (t_cand - t))
            states_cand = states * decay
            base_cand = self._baseline_at_time(t_cand)
            lam_vec_cand = base_cand + np.tensordot(self.adjacency, states_cand, axes=((1, 2), (0, 1)))
            lam_sum_cand = float(np.sum(lam_vec_cand))

            if self._rng.uniform() <= lam_sum_cand / lam_sum:
                # accept event at t_cand; choose component i
                probs = lam_vec_cand / (lam_sum_cand if lam_sum_cand > 0 else 1.0)
                i = int(self._rng.choice(np.arange(D), p=probs))
                # register event
                t = t_cand
                events[i].append(t)
                # jump: source j=i
                states = states_cand
                states[i, :] += self.decays
                base = base_cand
                lam_vec = base + np.tensordot(self.adjacency, states, axes=((1, 2), (0, 1)))
                lam_sum = float(np.sum(lam_vec))
                n_jumps += 1
            else:
                # reject; move to t_cand without jump
                t = t_cand
                states = states_cand
                base = base_cand
                lam_vec = lam_vec_cand
                lam_sum = lam_sum_cand

        # finalize
        self.timestamps = [np.array(ev, dtype=float) for ev in events]
        self.simulation_time = min(t, T)
        self.n_total_jumps = int(sum(len(ev) for ev in self.timestamps))

        # Tracking intensities on grid (post-simulation)
        if self._track_step is not None and np.isfinite(T):
            grid = np.arange(0.0, T + 1e-12, self._track_step, dtype=float)
            lam_grid = self._compute_intensity_grid(grid)
            self.tracked_intensity = [lam_grid[i].copy() for i in range(D)]
            self.intensity_tracked_times = grid

        return self

    def _compute_intensity_grid(self, t_grid: np.ndarray) -> np.ndarray:
        D = self.n_nodes
        U = self.n_decays
        M = t_grid.shape[0]
        # baseline on grid
        if self._n_baselines == 1:
            base_grid = np.repeat(self.baseline.reshape(D, 1), M, axis=1)
        else:
            # piecewise periodic
            if self.period_length is None or self.period_length <= 0:
                raise ValueError("Invalid period_length for piecewise baseline")
            base_grid = np.zeros((D, M), dtype=float)
            seg_len = float(self.period_length) / float(self._n_baselines)
            for m in range(M):
                k = int((t_grid[m] % self.period_length) // seg_len)
                if k == self._n_baselines:
                    k = self._n_baselines - 1
                base_grid[:, m] = self.baseline[:, k]

        # states g_{j,u}(t_grid)
        g_grid = np.zeros((D, U, M), dtype=float)
        for j in range(D):
            ev = self.timestamps[j]
            for ui in range(U):
                beta = float(self.decays[ui])
                val = 0.0
                pos = 0
                prev_t = t_grid[0] if M > 0 else 0.0
                # handle events before first grid time
                while pos < ev.shape[0] and ev[pos] <= prev_t:
                    val += beta * math.exp(-beta * (prev_t - ev[pos]))
                    pos += 1
                if M > 0:
                    g_grid[j, ui, 0] = val
                for m in range(1, M):
                    r = t_grid[m]
                    # decay
                    val *= math.exp(-beta * (r - prev_t))
                    # add new events between prev_t and r
                    while pos < ev.shape[0] and ev[pos] <= r:
                        val += beta * math.exp(-beta * (r - ev[pos]))
                        pos += 1
                    g_grid[j, ui, m] = val
                    prev_t = r

        # intensity = baseline + Σ_{j,u} α_{i,j,u} g_{j,u}
        lam_grid = np.zeros((D, M), dtype=float)
        for i in range(D):
            lam_grid[i, :] = base_grid[i, :]
            for j in range(D):
                for ui in range(U):
                    lam_grid[i, :] += self.adjacency[i, j, ui] * g_grid[j, ui, :]
        return lam_grid

    def mean_intensity(self) -> np.ndarray:
        T = self.simulation_time
        if T <= 0:
            return np.zeros(self.n_nodes, dtype=float)
        return np.array([len(ev) / T for ev in self.timestamps], dtype=float)

    def _build_sumexp_kernels(self) -> None:
        D = self.n_nodes
        kernels = np.empty((D, D), dtype=object)
        k0 = HawkesKernel0Py()
        for i in range(D):
            for j in range(D):
                intensities = self.adjacency[i, j, :]
                if np.all(intensities == 0):
                    kernels[i, j] = k0
                else:
                    kernels[i, j] = HawkesKernelSumExpPy(intensities, self.decays)
        self.kernels = kernels


def _simulate_single(simulation: "SimuHawkesSumExpKernels") -> "SimuHawkesSumExpKernels":
    simulation = copy.deepcopy(simulation)
    simulation.simulate()
    return simulation


class SimuHawkesMulti:
    """Parallel simulations of a single Hawkes simulation (pure Python).

    Replicates a base SimuHawkesSumExpKernels `n_simulations` times and runs
    them on `n_threads` processes (or sequentially if n_threads == 1).
    """

    def __init__(self, hawkes_simu: SimuHawkesSumExpKernels, n_simulations: int, n_threads: int = 1):
        if n_simulations <= 0:
            raise ValueError("n_simulations must be >= 1")
        self.hawkes_simu = hawkes_simu
        self.n_simulations = int(n_simulations)
        if n_threads <= 0:
            n_threads = multiprocessing.cpu_count()
        self.n_threads = int(n_threads)
        self._simulations: List[SimuHawkesSumExpKernels] = [copy.deepcopy(hawkes_simu) for _ in range(self.n_simulations)]

        # seed handling: generate unique seeds if base has a non-negative seed
        base_seed = hawkes_simu.seed if hawkes_simu.seed is not None else -1
        self.reseed_simulations(base_seed)

    @property
    def timestamps(self) -> List[List[np.ndarray]]:
        return [simu.timestamps for simu in self._simulations]

    @property
    def n_total_jumps(self) -> List[int]:
        return [simu.n_total_jumps for simu in self._simulations]

    @property
    def simulation_time(self) -> List[float]:
        return [simu.simulation_time for simu in self._simulations]

    def reseed_simulations(self, seed: int) -> None:
        if seed is None or seed < 0:
            # leave seeds unchanged if negative; to preserve randomness
            return
        rng = np.random.RandomState(seed)
        new_seeds = rng.randint(0, 2 ** 31 - 1, size=self.n_simulations).astype(int)
        for simu, s in zip(self._simulations, new_seeds):
            simu.seed = int(s)
            simu._rng = np.random.RandomState(simu.seed)

    def get_single_simulation(self, i: int) -> SimuHawkesSumExpKernels:
        return self._simulations[i]

    def simulate(self) -> "SimuHawkesMulti":
        if self.n_threads == 1:
            self._simulations = [ _simulate_single(s) for s in self._simulations ]
        else:
            with Pool(self.n_threads) as p:
                self._simulations = p.map(_simulate_single, self._simulations)
        return self
