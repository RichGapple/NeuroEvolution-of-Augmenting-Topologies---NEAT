#!/usr/bin/env python3
"""Calibration -- run this BEFORE the main sweep, and freeze the result.

A sweep can be perfectly executed and still be uninterpretable, because the
dependent variables have no room to move.  Four preconditions are checked here.
All of them concern *performance and range*, never the topology-vs-sparsity
relationship itself, so running this cannot bias the result the paper reports.

  C1  the dense endpoint solves the task in a reasonable fraction of runs
      -> otherwise every condition is at the floor and the comparison is
         between failure modes, not between levels of reward information
  C2  the dense endpoint does not solve it immediately (generation 0-2)
      -> otherwise the task is at the ceiling and evolution has nothing to do
  C3  topology actually changes over a run under dense reward
      -> otherwise the dependent variable of H2/H3 has no range and a null
         result would be an artefact of the setup, not a finding
  C4  the sparse endpoint measurably reduces fitness differentiation
      -> otherwise the independent variable is not doing anything

If a check fails, the printed suggestions say which knob to turn.  Turn it,
re-run this script, and only then run the sweep.

Usage
-----
    python scripts/00_calibrate.py --seeds 6 --generations 80
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from neat_sparsity.config import ExperimentConfig
from neat_sparsity.env import render_ascii
from neat_sparsity.runner import run_sweep
from neat_sparsity.analysis import load_summary, load_generations


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/calibration")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--generations", type=int, default=100)
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    cfg = ExperimentConfig(name="calibration")
    cfg.generations = args.generations
    cfg.neat.pop_size = args.pop
    cfg.etas = [0.0, 1.0]
    cfg.seeds = list(range(2000, 2000 + args.seeds))

    print(render_ascii(cfg.env))
    print(f"\ncalibration: eta in {{0, 1}} x {args.seeds} seeds, "
          f"{cfg.generations} generations, pop {cfg.neat.pop_size}")
    print(f"config hash {cfg.config_hash()}\n")

    run_sweep(cfg, args.out, workers=args.workers)
    df = load_summary(args.out)
    gens = load_generations(args.out)

    dense = df[df["eta"] == 0.0]
    sparse = df[df["eta"] == 1.0]
    dg = gens[gens["eta"] == 0.0]

    solve_rate = dense["ever_solved"].mean()
    first = dense["gen_first_solution"].dropna()
    med_first = first.median() if len(first) else float("nan")

    # topology range under dense reward, per run
    growth = []
    for seed, sub in dg.groupby("seed"):
        sub = sub.sort_values("generation")
        growth.append(sub["mean_nodes"].iloc[-1] - sub["mean_nodes"].iloc[0])
    growth = np.array(growth) if growth else np.array([0.0])
    conn_growth = []
    for seed, sub in dg.groupby("seed"):
        sub = sub.sort_values("generation")
        conn_growth.append(sub["mean_connections"].iloc[-1] - sub["mean_connections"].iloc[0])
    conn_growth = np.array(conn_growth) if conn_growth else np.array([0.0])

    diff_dense = dense["mean_distinct_fitness"].median()
    diff_sparse = sparse["mean_distinct_fitness"].median()

    checks = []

    ok1 = 0.3 <= solve_rate <= 0.95
    checks.append(("C1 dense endpoint solvable but not trivial",
                   ok1, f"{solve_rate:.0%} of dense runs solved "
                        f"(target 30-95%)",
                   "raise --generations / --pop, or move the goal closer / "
                   "widen goal_radius in EnvConfig"))

    ok2 = not (med_first == med_first and med_first <= 2)
    checks.append(("C2 not solved at initialisation",
                   ok2, f"median first solution at generation {med_first}",
                   "make the route harder: add obstacles or move the goal"))

    ok3 = float(np.median(np.abs(growth))) > 0.05 or float(np.median(np.abs(conn_growth))) > 0.5
    checks.append(("C3 topology has range under dense reward",
                   ok3, f"median change over the run: {np.median(growth):+.2f} nodes, "
                        f"{np.median(conn_growth):+.2f} connections",
                   "raise --generations, raise add_node_prob/add_conn_prob, or "
                   "use a task that rewards memory/nonlinearity"))

    ok4 = diff_sparse < 0.25 * diff_dense
    checks.append(("C4 sparsity reduces fitness differentiation",
                   ok4, f"distinct fitness values per generation: "
                        f"{diff_dense:.1f} (dense) vs {diff_sparse:.1f} (sparse)",
                   "check SparsityConfig.levels_max and the sparsity mode"))

    print("\n" + "=" * 74)
    for name, ok, detail, fix in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       {detail}")
        if not ok:
            print(f"       fix: {fix}")
    print("=" * 74)

    if all(ok for _, ok, _, _ in checks):
        print("\nAll preconditions met. Freeze this configuration, record the hash "
              f"({cfg.config_hash()}), and run the sweep.")
    else:
        print("\nDo NOT run the main sweep yet. A sweep on an uncalibrated setup "
              "produces results that cannot be interpreted either way.")
        sys.exit(1)


if __name__ == "__main__":
    main()
