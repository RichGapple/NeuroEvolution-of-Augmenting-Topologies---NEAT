#!/usr/bin/env python3
"""Pilot run (roadmap Section 11): measure runtime and between-seed variance,
then compute how many seeds the main sweep actually needs.

The roadmap says "approximately 30 or more seeds ... depending on computational
cost and pilot variance".  This script turns that into a number instead of a
guess: for each primary metric it reports the observed effect between the
extreme conditions and the n-per-group required to detect it.

Usage
-----
    python scripts/03_pilot.py --workers 4
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from neat_sparsity.config import pilot_config
from neat_sparsity.runner import run_sweep
from neat_sparsity.analysis import load_summary, groups_from, RUN_METRICS
from neat_sparsity.stats import cliffs_delta, cliffs_magnitude


def n_required(a, b, power: float = 0.80, alpha: float = 0.05) -> float:
    """Approximate n per group for a two-sample test at the observed effect.

    Uses the standardised mean difference with a 15% inflation for the
    efficiency loss of the rank test (Pitman ARE of Mann-Whitney vs t is
    3/pi ~ 0.955 at normality; 1.15 is a conservative allowance for the
    non-normal, often bimodal, case).
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    s = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                  / (len(a) + len(b) - 2))
    if s == 0:
        return float("nan")
    d = abs(a.mean() - b.mean()) / s
    if d == 0:
        return float("inf")
    z_a, z_b = 1.959964, 0.8416212 if power == 0.80 else 1.281552
    return math.ceil(1.15 * 2 * ((z_a + z_b) / d) ** 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/pilot")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=6)
    args = ap.parse_args()

    cfg = pilot_config()
    cfg.seeds = list(range(1000, 1000 + args.seeds))
    print(f"pilot: {len(cfg.etas)} conditions x {len(cfg.seeds)} seeds "
          f"= {len(cfg.etas) * len(cfg.seeds)} runs, {cfg.generations} generations, "
          f"pop {cfg.neat.pop_size}\nconfig hash {cfg.config_hash()}\n")

    results = run_sweep(cfg, args.out, workers=args.workers)
    df = load_summary(args.out)

    tot = df["wallclock_s"].sum()
    per = df["wallclock_s"].mean()
    print(f"\n--- runtime ---")
    print(f"mean {per:.1f}s per run, {tot:.0f}s total on this machine")

    print(f"\n--- variance and required seeds "
          f"(extreme conditions eta={min(cfg.etas)} vs eta={max(cfg.etas)}) ---")
    print(f"{'metric':38} {'cv_low':>7} {'cv_high':>8} {'cliff_d':>8} "
          f"{'magnitude':>11} {'n/group':>8}")
    n_needed = []
    for col, label, _h in RUN_METRICS:
        g = groups_from(df, col)
        if len(g) < 2:
            continue
        lo, hi = g[min(g)], g[max(g)]
        a = np.asarray(lo, float); b = np.asarray(hi, float)
        a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
        if len(a) < 2 or len(b) < 2:
            continue
        cv_lo = a.std(ddof=1) / abs(a.mean()) if a.mean() else float("nan")
        cv_hi = b.std(ddof=1) / abs(b.mean()) if b.mean() else float("nan")
        d = cliffs_delta(lo, hi)
        n = n_required(lo, hi)
        if n == n and n != float("inf"):
            n_needed.append(n)
        print(f"{label[:38]:38} {cv_lo:7.2f} {cv_hi:8.2f} {d:8.2f} "
              f"{cliffs_magnitude(d):>11} {n:8.0f}")

    if n_needed:
        rec = int(min(120, max(30, np.percentile(n_needed, 75))))
        print(f"\nRECOMMENDED seeds per condition: {rec}")
        print(f"  (75th percentile of the per-metric requirement, floored at the "
              f"roadmap's 30)")
        est = rec * len(cfg.etas) * per
        print(f"  estimated main-sweep cost at pilot settings: "
              f"{est/3600:.1f} core-hours; scale by "
              f"(generations x pop) if you change them.")


if __name__ == "__main__":
    main()
