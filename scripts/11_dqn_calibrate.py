#!/usr/bin/env python3
"""Calibration gates for the DQN arm.

Same discipline as the NEAT arm's `00_calibrate.py`, and for the same reason:
these check that the *range* is usable, never that the hypothesis is true.

  D1  Dense endpoint reaches the maze's greedy-descent optimum.
      REVISED. The original form required solving the maze in 30-95% of runs,
      on the reasoning that a learner which never solves must be broken. That
      reasoning fails on a deceptive maze: the intermediate credit term is a
      monotone function of distance-to-goal, so a correctly functioning
      gradient follower converges to the point where distance stops decreasing,
      and on a deceptive maze that point is not the goal.
      D1 now asks the honest version of the same question -- does the learner
      extract what its reward signal actually offers? On a NON-deceptive maze
      the greedy optimum IS the goal, so this reduces to the original gate.
      The deceptiveness is computed, not assumed: see maze_analysis.py.
  D2  Not solved immediately.  If a random policy solves it, the task carries
      no information.
  D3  Learning actually happens.  Final performance must exceed the performance
      of the untrained network by a clear margin. Guards against a run that
      "solves" the maze by luck during exploration.
  D4  Sparse endpoint degrades.  At eta = 1 the agent must do measurably worse
      than at eta = 0, otherwise the manipulation is not reaching this arm.

  D5  Headroom exists.  The dense endpoint must sit clear of both the untrained
      floor and the theoretical ceiling, so degradation is detectable at all.
      This is what D1 was really guarding in its original form.

A failure here is a configuration problem, not a result. Fix the learner, do
not reinterpret the gate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dqn_sparsity.config import core_config
from dqn_sparsity import mazes, maze_analysis
from dqn_sparsity.runner import run_single, evaluate_greedy
from dqn_sparsity.step_env import StepNavEnv
from dqn_sparsity.agent import DQNAgent


def untrained_baseline(cfg, seeds) -> float:
    """Median true objective of an untrained network. D3 compares against this."""
    vals = []
    for s in seeds:
        env = StepNavEnv(cfg.env)
        agent = DQNAgent(env.obs_dim, env.n_actions, cfg.dqn, seed=s)
        agent.step_count = 10 ** 9        # force greedy, skip warmup randomness
        vals.append(evaluate_greedy(env, agent, cfg, 1)["true_best"])
    return float(np.median(vals))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--budget", type=int, default=None,
                    help="env steps per run (default: the configured budget)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--outdir", default="results/dqn_calibration")
    ap.add_argument("--maze", choices=["A", "B"], default=None,
                    help="A = deceptive (original), B = non-deceptive")
    args = ap.parse_args()

    cfg = core_config()
    if args.maze:
        cfg.env = mazes.apply_maze(cfg.env, args.maze)
    if args.budget:
        cfg.dqn.budget_steps = args.budget
    cfg.etas = [0.0, 1.0]
    cfg.seeds = list(range(3000, 3000 + args.seeds))

    maze_id = mazes.identify(cfg.env)
    print(f"maze {maze_id}: {mazes.DESCRIPTIONS.get(maze_id, 'custom layout')}")
    print(maze_analysis.describe(cfg.env))
    print()
    print(f"DQN calibration -- {args.seeds} seeds x {{eta=0, eta=1}}")
    print(f"{cfg.budget_note()}")
    print(f"config hash {cfg.config_hash()}\n")

    os.makedirs(args.outdir, exist_ok=True)
    from dqn_sparsity.runner import run_sweep
    rows = run_sweep(cfg, args.outdir, workers=args.workers)

    dense = [r for r in rows if r["eta"] == 0.0]
    sparse = [r for r in rows if r["eta"] == 1.0]

    n_solved = sum(r["ever_solved"] for r in dense)
    solve_rate = n_solved / len(dense)
    dense_true = np.median([r["final_true_best"] for r in dense])
    sparse_true = np.median([r["final_true_best"] for r in sparse])

    first_blocks = [r["first_solution_block"] for r in dense
                    if r["first_solution_block"] is not None]
    earliest = min(first_blocks) if first_blocks else None
    baseline = untrained_baseline(cfg, cfg.seeds)

    print("\n" + "=" * 66)
    print("CALIBRATION GATES")
    print("=" * 66)

    greedy = maze_analysis.greedy_descent_optimum(cfg.env)
    target = greedy["true_objective"]
    deceptive = not greedy["reaches_goal"]

    d1 = dense_true >= 0.95 * target
    print(f"D1  dense endpoint reaches the greedy-descent optimum")
    print(f"    maze is {'DECEPTIVE' if deceptive else 'non-deceptive'}; "
          f"greedy optimum = {target:.3f} at "
          f"({greedy['x']:.0f}, {greedy['y']:.0f})")
    print(f"    achieved {dense_true:.3f} = {100 * dense_true / max(1e-9, target):.0f}% "
          f"of it   {'PASS' if d1 else 'FAIL'}")
    print(f"    (solve rate {n_solved}/{len(dense)} = {solve_rate:.0%}"
          + (" -- expected to be 0 on a deceptive maze; the reward gradient "
             "does not point at the goal)" if deceptive else ")"))
    if not d1:
        print("    -> the learner is NOT extracting what its signal offers. "
              "Raise the budget, lower the learning rate, or lengthen "
              "exploration before touching anything else.")
    if deceptive:
        print("    -> NOTE: on this maze, reward RESOLUTION (eta) and reward "
              "DECEPTIVENESS are confounded. Run maze B as well before "
              "attributing any DQN result to sparsity.")

    d2 = earliest is None or earliest >= 2
    print(f"\nD2  not solved immediately (first solution at block >= 2)")
    print(f"    earliest solution block = {earliest}   "
          f"{'PASS' if d2 else 'FAIL'}")

    d3 = dense_true > baseline + 0.20
    print(f"\nD3  learning happens (final > untrained + 0.20)")
    print(f"    untrained median = {baseline:.3f}, trained median = "
          f"{dense_true:.3f}   {'PASS' if d3 else 'FAIL'}")

    d4 = sparse_true < dense_true - 0.05
    print(f"\nD4  sparse endpoint degrades")
    print(f"    eta=0 median = {dense_true:.3f}, eta=1 median = "
          f"{sparse_true:.3f}   {'PASS' if d4 else 'FAIL'}")
    if not d4:
        print("    -> if eta=1 matches eta=0, the reward manipulation is not "
              "reaching the learner. Check shaping.py and the zero-reward "
              "fraction in blocks.csv before proceeding.")

    ceiling = 2.0
    d5 = (dense_true > baseline + 0.20) and (dense_true < 0.95 * ceiling)
    print(f"\nD5  headroom exists (untrained < dense < ceiling)")
    print(f"    {baseline:.3f} < {dense_true:.3f} < {ceiling:.3f}   "
          f"{'PASS' if d5 else 'FAIL'}")
    print(f"    dynamic range across eta: {dense_true:.3f} -> {sparse_true:.3f} "
          f"({dense_true / max(1e-9, sparse_true):.1f}x)")

    zr0 = np.median([r["mean_zero_reward_fraction"] for r in dense])
    zr1 = np.median([r["mean_zero_reward_fraction"] for r in sparse])
    print(f"\n    (diagnostic) zero-reward transition fraction: "
          f"eta=0 {zr0:.3f} -> eta=1 {zr1:.3f}")

    passed = all([d1, d2, d3, d4, d5])
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config_hash": cfg.config_hash(),
        "budget_steps": cfg.dqn.budget_steps,
        "seeds": cfg.seeds,
        "D1_solve_rate": solve_rate,
        "D2_earliest_block": earliest,
        "D3_untrained": baseline, "D3_trained": dense_true,
        "D4_eta0": dense_true, "D4_eta1": sparse_true,
        "D5_headroom": bool(d5),
        "maze": maze_id,
        "maze_deceptive": bool(deceptive),
        "greedy_optimum": target,
        "greedy_endpoint": [greedy["x"], greedy["y"]],
        "zero_reward_eta0": zr0, "zero_reward_eta1": zr1,
        "passed": passed,
    }
    with open(os.path.join(args.outdir, "calibration.json"), "w") as fh:
        json.dump(record, fh, indent=2)

    print("\n" + "=" * 66)
    if passed:
        print(f"CALIBRATION PASSED -- freeze config hash {cfg.config_hash()} "
              f"and run the sweep.")
        print("Record the hash in your pre-registration NOW, before looking at "
              "any sweep results.")
        sys.exit(0)
    print("CALIBRATION FAILED -- fix the configuration. Do not run the sweep "
          "and do not reinterpret a failed gate as a finding.")
    sys.exit(1)


if __name__ == "__main__":
    main()
