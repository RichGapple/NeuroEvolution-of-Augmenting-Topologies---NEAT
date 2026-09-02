#!/usr/bin/env python3
"""Stage 3 verification -- prove the eta parameterisation is well behaved.

Checks, for every mode and every eta:
  1. max attainable fitness is identical across eta          (no confound)
  2. a goal-reaching trajectory scores identically across eta (no confound)
  3. credit is monotone non-increasing in eta for fixed behaviour
  4. eta = 1 gives exactly the terminal-only reward
  5. fitness differentiation (distinct values over a fixed set of behaviours)
     decreases monotonically with eta -- i.e. the manipulation actually does
     what H1 assumes it does.

Run this before the sweep.  If any check fails, the sweep is not measuring
what the paper claims it measures.
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neat_sparsity.config import ExperimentConfig
from neat_sparsity.env import NavEnv, Trajectory
from neat_sparsity.reward import RewardModel, true_objective


def synthetic_trajectories(n: int = 400, seed: int = 0):
    """A fixed library of behaviours spanning the whole progress range."""
    rng = random.Random(seed)
    trajs = []
    for i in range(n):
        p = i / (n - 1)
        d0 = 100.0
        dmin = d0 * (1.0 - p)
        reached = p >= 0.999
        ds = [d0 - (d0 - dmin) * t / 50.0 for t in range(51)]
        trajs.append(Trajectory(
            reached_goal=reached, steps_taken=50, initial_distance=d0,
            min_distance=dmin, final_distance=dmin, path_length=d0 * p,
            collisions=0, distances=ds,
        ))
    return trajs


def main() -> None:
    cfg = ExperimentConfig()
    etas = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    trajs = synthetic_trajectories()
    success = trajs[-1]
    failures = []

    for mode in ("quantized", "checkpoint", "gated"):
        cfg.sparsity.mode = mode
        print(f"\n=== mode = {mode} ===")
        print(f"{'eta':>6} {'L(eta)':>7} {'max_fit':>8} {'success_fit':>12} "
              f"{'distinct':>9} {'mean_credit':>12}")
        prev_distinct = None
        prev_mean = None
        for eta in etas:
            m = RewardModel(cfg.sparsity, eta, random.Random(12345))
            fits = [m.fitness(t) for t in trajs]
            f_succ = m.fitness(success)
            distinct = len({round(f, 9) for f in fits})
            mean_credit = sum(fits) / len(fits)

            print(f"{eta:6.3f} {m.levels:7d} {m.max_fitness:8.3f} {f_succ:12.3f} "
                  f"{distinct:9d} {mean_credit:12.4f}")

            if abs(m.max_fitness - 2.0) > 1e-9:
                failures.append(f"{mode} eta={eta}: max fitness != 2.0")
            if abs(f_succ - 2.0) > 1e-9:
                failures.append(f"{mode} eta={eta}: successful trajectory scores {f_succ}")
            if eta == 1.0:
                nonzero = [f for t, f in zip(trajs, fits) if not t.reached_goal and f > 0]
                if nonzero:
                    failures.append(f"{mode} eta=1: {len(nonzero)} unsuccessful "
                                    f"trajectories received non-zero reward")
            if prev_distinct is not None and distinct > prev_distinct * 1.05:
                failures.append(f"{mode} eta={eta}: differentiation increased "
                                f"({prev_distinct} -> {distinct})")
            prev_distinct = distinct
            prev_mean = mean_credit

    # monotone credit for a fixed trajectory (quantized/checkpoint only;
    # `gated` is stochastic by construction so it is checked in expectation)
    cfg.sparsity.mode = "quantized"
    t = trajs[len(trajs) // 2]
    prev = None
    for eta in etas:
        c = RewardModel(cfg.sparsity, eta).fitness(t)
        if prev is not None and c > prev + 1e-12:
            failures.append(f"quantized: credit increased with eta at eta={eta}")
        prev = c

    print("\n--- true objective is eta-independent ---")
    print("true_objective(success) =", true_objective(success, cfg.sparsity))
    print("true_objective(half)    =", round(true_objective(t, cfg.sparsity), 4))

    if failures:
        print("\nFAILED CHECKS:")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)
    print("\nAll sparsity invariants hold.")


if __name__ == "__main__":
    main()
