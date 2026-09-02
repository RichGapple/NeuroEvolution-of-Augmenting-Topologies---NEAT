#!/usr/bin/env python3
"""Stage 1 + Stage 5 -- the first coding milestone.

    "Get one NEAT population to successfully evolve an agent through one tiny
     environment with dense reward."

This script does exactly that and nothing else, plus the reproducibility check
that the roadmap asks for before any experimental variable is introduced.

Usage
-----
    python scripts/01_baseline.py                       # one dense run
    python scripts/01_baseline.py --seeds 1000 1001 1002
    python scripts/01_baseline.py --check-reproducibility
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neat_sparsity.config import ExperimentConfig
from neat_sparsity.env import render_ascii
from neat_sparsity.runner import run_single


def build_config(args) -> ExperimentConfig:
    cfg = ExperimentConfig(name="baseline")
    cfg.generations = args.generations
    cfg.neat.pop_size = args.pop
    cfg.env.max_steps = args.max_steps
    cfg.etas = [0.0]
    cfg.seeds = list(args.seeds)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=60)
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1000])
    ap.add_argument("--out", default="results/baseline")
    ap.add_argument("--check-reproducibility", action="store_true",
                    help="run seed 1000 twice and assert identical trajectories")
    args = ap.parse_args()

    cfg = build_config(args)
    print(f"config hash: {cfg.config_hash()}")
    print(render_ascii(cfg.env))
    print(f"\ninputs={cfg.neat.num_inputs}  outputs={cfg.neat.num_outputs}  "
          f"pop={cfg.neat.pop_size}  generations={cfg.generations}  eta=0.0 (dense)\n")

    solved = 0
    for seed in cfg.seeds:
        print(f"--- seed {seed} ---")
        s = run_single(cfg, eta=0.0, seed=seed, outdir=args.out, verbose=True)
        solved += int(s["ever_solved"])
        print(f"  best(true)={s['final_true_best']:.3f}  success_rate="
              f"{s['final_success_rate']:.2f}  first_solution_gen="
              f"{s['gen_first_solution']}  champion: {s['champion_nodes']} nodes / "
              f"{s['champion_connections']} conns  [{s['wallclock_s']}s]\n")

    print(f"BASELINE: {solved}/{len(cfg.seeds)} runs reached the goal at least once.")
    if solved == 0:
        print("\nThe dense baseline did not solve the task. Do NOT proceed to the")
        print("sparsity sweep. Fix this first: raise --generations, raise --pop,")
        print("shorten the route, or enlarge goal_radius in EnvConfig.")

    if args.check_reproducibility:
        print("\n--- reproducibility check ---")
        import pandas as pd
        a = run_single(cfg, 0.0, cfg.seeds[0], outdir=args.out + "_rep_a")
        b = run_single(cfg, 0.0, cfg.seeds[0], outdir=args.out + "_rep_b")
        da = pd.read_csv(os.path.join(args.out + "_rep_a",
                                      f"eta=0", f"seed={cfg.seeds[0]}",
                                      "generations.csv"))
        db = pd.read_csv(os.path.join(args.out + "_rep_b",
                                      f"eta=0", f"seed={cfg.seeds[0]}",
                                      "generations.csv"))
        same = da.drop(columns=[]).equals(db)
        print("identical generation-by-generation logs:", same)
        if not same:
            diff = (da != db).any()
            print("differing columns:", list(diff[diff].index))
            raise SystemExit("REPRODUCIBILITY FAILURE -- fix before continuing.")


if __name__ == "__main__":
    main()
