#!/usr/bin/env python3
"""Honest compute accounting for both arms.

Why this script exists
----------------------
"NEAT is computationally cheaper than RL" is the easiest sentence in this
project to get wrong, and the easiest for a reviewer to disprove. NEAT consumes
*more* environment interaction than DQN on this task. What it needs less of is
infrastructure: no automatic differentiation, no replay buffer, no GPU, and
near-linear parallel scaling.

So report three numbers, never one:

    environment steps   sample efficiency        (DQN usually wins)
    wall-clock seconds  practical speed          (depends on parallelism)
    peak RSS            hardware footprint       (NEAT usually wins)

plus two categorical facts: whether autodiff is required, and whether a GPU is.

A single "compute" figure would hide exactly the trade the paper is about.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dqn_sparsity import compare as C
from dqn_sparsity import plots as P


def measure_throughput(n_steps: int = 20_000) -> dict:
    """Time each arm's inner loop on this machine, so the table is not copied
    from someone else's hardware."""
    from neat_sparsity.config import core_config as neat_core
    from neat_sparsity.env import NavEnv
    from dqn_sparsity.config import core_config as dqn_core
    from dqn_sparsity.step_env import StepNavEnv
    from dqn_sparsity.agent import DQNAgent

    ncfg = neat_core()

    class _Rand:
        def __init__(self, seed):
            self.rng = np.random.default_rng(seed)
        def activate(self, obs):
            return list(self.rng.random(3))

    env = NavEnv(ncfg.env)
    t0 = time.perf_counter()
    steps = 0
    i = 0
    while steps < n_steps:
        traj = env.rollout(_Rand(i))
        steps += traj.steps_taken
        i += 1
    neat_sps = steps / (time.perf_counter() - t0)

    dcfg = dqn_core()
    senv = StepNavEnv(dcfg.env)
    agent = DQNAgent(senv.obs_dim, senv.n_actions, dcfg.dqn, seed=0)
    obs = senv.reset()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        a = agent.act(obs)
        nobs, done, info = senv.step(a)
        agent.observe(obs, a, 0.0, nobs, False)
        obs = senv.reset() if done else nobs
    dqn_sps = n_steps / (time.perf_counter() - t0)

    return {"neat_steps_per_s": round(neat_sps), "dqn_steps_per_s": round(dqn_sps),
            "ratio_neat_over_dqn": round(neat_sps / max(1e-9, dqn_sps), 2)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--neat", required=True)
    ap.add_argument("--dqn", required=True)
    ap.add_argument("--outdir", default="results/comparison")
    ap.add_argument("--skip-throughput", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    figdir = os.path.join(args.outdir, "figures")
    os.makedirs(figdir, exist_ok=True)

    neat_df = C.load_arm(args.neat, "neat")
    dqn_df = C.load_arm(args.dqn, "dqn")
    costs = C.cost_table(neat_df, dqn_df)

    # NEAT does not log env_steps directly; derive its upper bound from the config.
    if "env_steps" not in neat_df.columns or neat_df["env_steps"].isna().all():
        from neat_sparsity.config import core_config as neat_core
        ncfg = neat_core()
        gens = int(neat_df["generations"].iloc[0]) if "generations" in neat_df else ncfg.generations
        ub = gens * ncfg.neat.pop_size * ncfg.env.max_steps
        costs.loc[costs["arm"] == "NEAT", "env_steps_median"] = ub
        costs.loc[costs["arm"] == "NEAT", "env_steps_total"] = ub * len(neat_df)
        note = (f"NEAT env steps are an UPPER BOUND "
                f"({gens} gen x {ncfg.neat.pop_size} pop x {ncfg.env.max_steps} "
                f"steps = {ub:,}); episodes that reach the goal terminate early, "
                f"so actual usage is lower. Instrument NavEnv.rollout to count "
                f"steps if you need the exact figure for the paper.")
    else:
        note = ""

    from dqn_sparsity.sysinfo import describe_host
    env_info = describe_host()

    thr = {} if args.skip_throughput else measure_throughput()

    P.cost_plot(neat_df.to_dict("records") + dqn_df.to_dict("records"), figdir)

    payload = {"environment": env_info, "throughput": thr,
               "note": note, "cost_table": costs.to_dict("records")}
    with open(os.path.join(args.outdir, "cost_report.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    costs.to_csv(os.path.join(args.outdir, "cost_table.csv"), index=False)

    print("=" * 70)
    print("COST ACCOUNTING")
    print("=" * 70)
    print(f"{env_info.get('platform', '?')}  |  "
          f"{env_info.get('cpu_count', '?')} CPUs  |  GPU: no\n")
    for _, r in costs.iterrows():
        print(f"{r['arm']}")
        print(f"  env steps / run       {r['env_steps_median']:>15,.0f}")
        print(f"  wall-clock s / run    {r['wallclock_s_median']:>15,.1f}")
        print(f"  peak RSS MB           {r['peak_rss_mb_median']:>15,.1f}")
        print(f"  gradient updates      {r['gradient_updates_median']:>15,.0f}")
        print(f"  learner params        {r['n_params_median']:>15,.0f}")
        print(f"  autodiff required     {r['autodiff_required']:>15s}")
        print(f"  GPU required          {r['gpu_required']:>15s}\n")
    if thr:
        print(f"Measured on this machine: NEAT {thr['neat_steps_per_s']:,} env "
              f"steps/s, DQN {thr['dqn_steps_per_s']:,} env steps/s "
              f"({thr['ratio_neat_over_dqn']}x)")
        print("DQN is slower per step because every step also runs a forward "
              "pass and, every `train_every` steps, a gradient update.\n")
    if note:
        print("NOTE: " + note + "\n")

    print("Sentence you can defend in the paper:")
    print("  \"NEAT required more environment interaction but strictly simpler")
    print("   infrastructure: no automatic differentiation, no replay buffer,")
    print("   no GPU, and near-linear parallel scaling across independent")
    print("   episode evaluations.\"")
    print("\nSentence you cannot defend:")
    print("  \"NEAT is computationally cheaper.\"  It is not, on this task.")
    print("=" * 70)


if __name__ == "__main__":
    main()
