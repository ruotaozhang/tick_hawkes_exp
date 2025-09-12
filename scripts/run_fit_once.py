import os
import sys
import time
import json
import argparse
import resource
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from hawkes_sumexp_py import HawkesSumExpKern


def build_empty_events(D: int, R: int):
    # R realizations, D nodes, empty arrays
    return [[np.empty(0, dtype=float) for _ in range(D)] for _ in range(R)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["matrix", "operator"], required=True)
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--U", type=int, required=True)
    ap.add_argument("--R", type=int, default=1)
    ap.add_argument("--memory_mode", choices=["in_memory", "memmap", "packed"], default="in_memory")
    args = ap.parse_args()

    D = int(args.D)
    U = int(args.U)
    R = int(args.R)
    mode = args.mode

    # Construct decays and dummy period
    decays = np.linspace(0.5, 6.0, U, dtype=float)
    events = build_empty_events(D, R)
    # End times zero to short-circuit H integrals
    end_times = [0.0] * R

    # Fit
    t0 = time.perf_counter()
    HawkesSumExpKern(decays=decays, n_baselines=1, fit_mode=mode, memory_mode=args.memory_mode).fit(events, end_times=end_times)
    t1 = time.perf_counter()

    usage = resource.getrusage(resource.RUSAGE_SELF)
    # On macOS (darwin), ru_maxrss is in bytes; on Linux, it's in kilobytes
    ru = usage.ru_maxrss
    if sys.platform == 'darwin':
        peak_mb = ru / (1024**2)
    else:
        peak_mb = ru / 1024.0

    print(json.dumps({
        "D": D,
        "U": U,
        "R": R,
        "mode": mode,
        "time_sec": t1 - t0,
        "peak_rss_mb": peak_mb,
    }))


if __name__ == "__main__":
    main()
