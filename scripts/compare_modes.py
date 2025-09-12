import os
import sys
import time
import json
import argparse
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

    def run_one(mode, memory_mode):
        cmd = [sys.executable, entry, "--mode", mode, "--D", str(D), "--U", str(U), "--R", str(R), "--memory_mode", memory_mode]
        out = subprocess.check_output(cmd, text=True)
        line = out.strip().splitlines()[-1]
        return json.loads(line)

    r_matrix_mem = run_one("matrix", "in_memory")
    r_operator_mem = run_one("operator", "in_memory")
    r_operator_packed = run_one("operator", "packed")
    r_operator_mm = run_one("operator", "memmap")
    print(json.dumps({
        "matrix_in_memory": r_matrix_mem,
        "operator_in_memory": r_operator_mem,
        "operator_packed": r_operator_packed,
        "operator_memmap": r_operator_mm,
    }, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--memtest", action="store_true", help="Run peak memory benchmark as a separate step")
    ap.add_argument("--D", type=int, default=300)
    ap.add_argument("--U", type=int, default=5)
    ap.add_argument("--R", type=int, default=1)
    args = ap.parse_args()

    # 1) Compare outputs and runtime on a small realistic dataset (quick)
    compare_outputs()

    # 2) Peak memory benchmark (optional; can take longer due to JIT + subprocess)
    if args.memtest:
        measure_peak_rss(D=args.D, U=args.U, R=args.R)
