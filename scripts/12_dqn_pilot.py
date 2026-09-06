#!/usr/bin/env python3
"""Pilot: measure runtime and between-seed variance, then size the sweep.

Mirrors the NEAT arm's `03_pilot.py`. Two jobs:

  1. Tell you how long the full sweep will take before you start it.
  2. Convert observed between-seed variance into a required seed count, so the
     number of seeds is a measurement rather than a guess.

The seed count is derived from the smallest effect you care about (`--mde`,
minimum detectable effect, in true-objective units). Declare that number before
running, not after.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dqn_sparsity import mazes
from dqn_sparsity.config import core_config
from dqn_sparsity.runner import run_single


def seeds_needed(sd: float, mde: float, alpha: float = 0.05,
                 power: float = 0.80) -> int:
    """Two-sample comparison of means, inflated 15% for rank-based tests.

    Rank tests (Mann-Whitney) are what will actually be used, and they cost
    roughly 15% more samples than the parametric equivalent for normal-ish data
    -- the standard asymptotic relative efficiency adjustment.
    """
    if sd <= 0 or mde <= 0:
        return 0
    z_a, z_b = 1.959964, 0.8416212       # two-sided 0.05, power 0.80
    n = 2 * ((z_a + z_b) * sd / mde) ** 2
    return int(math.ceil(n * 1.15))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--budget", type=int, default=200_000)
    ap.add_argument("--etas", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    ap.add_argument("--mde", type=float, default=0.15,
                    help="smallest difference in true objective worth detecting")
    ap.add_argument("--outdir", default="results/dqn_pilot")
    ap.add_argument("--maze", choices=["A", "B"], default=None,
                    help="A = deceptive (original), B = non-deceptive")
    args = ap.parse_args()

    cfg = core_config()
    if args.maze:
        cfg.env = mazes.apply_maze(cfg.env, args.maze)
    cfg.dqn.budget_steps = args.budget
    cfg.dqn.eval_points = 40

    print(f"pilot: {len(args.etas)} etas x {args.seeds} seeds, "
          f"{args.budget:,} steps each")
    print(f"config hash {cfg.config_hash()}\n")

    os.makedirs(args.outdir, exist_ok=True)
    rows, t0 = [], time.time()
    for eta in args.etas:
        for i in range(args.seeds):
            seed = 4000 + i
            r = run_single(cfg, eta, seed, args.outdir)
            rows.append(r)
            print(f"  eta={eta:<6g} seed={seed}  true={r['final_true_best']:.3f}  "
                  f"{r['wallclock_s']:.1f}s", flush=True)
    elapsed = time.time() - t0

    per_run = elapsed / len(rows)
    print(f"\n{len(rows)} runs in {elapsed:.0f}s  ->  {per_run:.1f}s per run "
          f"at {args.budget:,} steps")

    full = core_config()
    scale = full.dqn.budget_steps / args.budget
    n_full = len(full.etas) * len(full.seeds)
    est = per_run * scale * n_full
    print(f"\nProjected full sweep ({len(full.etas)} etas x {len(full.seeds)} "
          f"seeds at {full.dqn.budget_steps:,} steps):")
    print(f"  {est / 3600:.1f} core-hours")
    for w in (4, 8, 16):
        print(f"  {est / 3600 / w:.1f} h on {w} workers")

    print("\nBetween-seed variability of final true objective:")
    need = {}
    for eta in args.etas:
        v = np.array([r["final_true_best"] for r in rows if r["eta"] == eta])
        sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        n = seeds_needed(sd, args.mde)
        need[eta] = n
        print(f"  eta={eta:<6g} mean={v.mean():.3f} sd={sd:.3f}  ->  "
              f"{n} seeds for MDE={args.mde}")

    rec = max(need.values()) if need else 0
    print(f"\nRecommended seeds per condition: {rec}")
    print(f"Configured: {len(full.seeds)}")
    if rec > len(full.seeds):
        print(f"  -> UNDERPOWERED for MDE={args.mde}. Either raise the seed "
              f"count to {rec}, or declare a larger MDE and justify it.")
    else:
        print("  -> adequate.")
    print("\nNote: the pilot budget is smaller than the sweep budget, so these "
          "variances are an upper bound (longer training usually reduces "
          "between-seed spread). Treat the seed count as conservative.")

    with open(os.path.join(args.outdir, "pilot.json"), "w") as fh:
        json.dump({"per_run_s": per_run, "pilot_budget": args.budget,
                   "projected_core_hours": est / 3600, "mde": args.mde,
                   "seeds_needed": need, "recommended": rec}, fh, indent=2)


if __name__ == "__main__":
    main()
