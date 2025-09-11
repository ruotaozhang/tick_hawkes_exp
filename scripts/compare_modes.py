import os
import sys
import time
import json
import subprocess
import numpy as np

# Ensure project root on path when running from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from hawkes_sumexp_py import SimuHawkesSumExpKernels, SimuHawkesMulti, HawkesSumExpKern


def simulate_small(seed=123):
    period_length = 300.0
    baselines = [[0.3, 0.5, 0.6, 0.4, 0.2, 0.0], [0.8, 0.5, 0.2, 0.3, 0.3, 0.4]]
    n_baselines = len(baselines[0])
    decays = np.array([0.5, 2.0, 6.0], float)
    adjacency = np.array(
        [[[0.0, 0.1, 0.4], [0.2, 0.0, 0.2]], [[0.0, 0.0, 0.0], [0.6, 0.3, 0.0]]],
        dtype=float,
    )

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


def compare_outputs():
    decays, n_baselines, period_length, events = simulate_small()

    # Fit using matrix mode
    t0 = time.perf_counter()
    learner_m = HawkesSumExpKern(
        decays=decays,
        n_baselines=n_baselines,
        period_length=period_length,
        fit_mode="matrix",
    ).fit(events)
    t1 = time.perf_counter()

    # Fit using operator (memory-efficient) mode
    t2 = time.perf_counter()
    learner_o = HawkesSumExpKern(
        decays=decays,
        n_baselines=n_baselines,
        period_length=period_length,
        fit_mode="operator",
    ).fit(events)
    t3 = time.perf_counter()

    res = {
        "t_matrix": t1 - t0,
        "t_operator": t3 - t2,
        "baseline_equal": np.allclose(learner_m.baseline, learner_o.baseline, rtol=1e-7, atol=1e-9),
        "adjacency_equal": np.allclose(learner_m.adjacency, learner_o.adjacency, rtol=1e-7, atol=1e-9),
        "max_abs_baseline_diff": float(np.max(np.abs(learner_m.baseline - learner_o.baseline))),
        "max_abs_adj_diff": float(np.max(np.abs(learner_m.adjacency - learner_o.adjacency))),
    }
    print(json.dumps(res, indent=2))


def measure_peak_rss(D=300, U=5, R=1):
    # Use separate subprocesses to measure peak RSS for each mode with empty events
    # Build args: mode, D, U, R
    here = os.path.dirname(__file__)
    entry = os.path.join(here, "run_fit_once.py")

    def run_one(mode):
        cmd = [sys.executable, entry, "--mode", mode, "--D", str(D), "--U", str(U), "--R", str(R)]
        out = subprocess.check_output(cmd, text=True)
        line = out.strip().splitlines()[-1]
        return json.loads(line)

    r_matrix = run_one("matrix")
    r_operator = run_one("operator")
    print(json.dumps({"matrix": r_matrix, "operator": r_operator}, indent=2))


if __name__ == "__main__":
    # 1) Compare outputs and runtime on a small realistic dataset
    compare_outputs()
    # 2) Compare peak memory on a large synthetic, event-free dataset
    # Increase D and U moderately to observe memory differences without long runtimes
    measure_peak_rss(D=450, U=6, R=1)

