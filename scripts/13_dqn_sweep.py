#!/usr/bin/env python3
"""Run the DQN sweep across the eta grid.

Budget matching
---------------
By default DQN receives the NEAT arm's *upper bound* on environment
interaction: generations x pop_size x max_steps. NEAT episodes terminate early
on success, so NEAT actually consumes fewer steps than this. Giving the baseline
the larger number is deliberate. If the paper's conclusion is that evolution
wins in sparse conditions, that conclusion is far more persuasive when the
baseline was handed extra budget rather than less.

Use `--match-neat <dir>` to read the budget from an existing NEAT sweep instead
of assuming it, and `--budget` to override manually.

Shaping ablation
----------------
`--shaping terminal` pays the whole episode fitness in one lump at the final
step instead of telescoping it. That is NOT the fair comparison -- it is a cheap
second experiment measuring how much of DQN's performance depends on having
temporal structure in the reward at all. Run both; the gap between them is a
result in its own right.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dqn_sparsity import mazes
from dqn_sparsity.config import core_config, matched_to_neat
from dqn_sparsity.runner import run_sweep


def budget_from_neat_dir(path: str) -> int:
    """Read a NEAT sweep's config to derive the matched interaction budget."""
    import pandas as pd
    from neat_sparsity.config import core_config as neat_core
    ncfg = neat_core()
    summ = os.path.join(path, "summary.csv")
    gens = ncfg.generations
    if os.path.exists(summ):
        df = pd.read_csv(summ)
        if "generations" in df.columns:
            gens = int(df["generations"].iloc[0])
    return gens * ncfg.neat.pop_size * ncfg.env.max_steps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="results/dqn_sweep")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--seed-start", type=int, default=1000)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--match-neat", default=None,
                    help="path to a NEAT sweep; derive the budget from it")
    ap.add_argument("--shaping", choices=["telescoping", "terminal"],
                    default="telescoping")
    ap.add_argument("--etas", type=float, nargs="+", default=None)
    ap.add_argument("--mode", choices=["quantized", "checkpoint", "gated"],
                    default="quantized",
                    help="sparsity mode; must match the NEAT sweep you compare to")
    ap.add_argument("--maze", choices=["A", "B"], default=None,
                    help="A = deceptive (original), B = non-deceptive")
    args = ap.parse_args()

    if args.match_neat:
        from neat_sparsity.config import core_config as neat_core
        ncfg = neat_core()
        ncfg.sparsity.mode = args.mode
        cfg = matched_to_neat(ncfg, budget_from_neat_dir(args.match_neat))
    else:
        cfg = core_config()
        cfg.sparsity.mode = args.mode
        if args.budget:
            cfg.dqn.budget_steps = args.budget

    if args.maze:
        cfg.env = mazes.apply_maze(cfg.env, args.maze)
    if args.budget:
        cfg.dqn.budget_steps = args.budget
    if args.etas:
        cfg.etas = list(args.etas)
    cfg.seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    print("=" * 70)
    print("DQN SWEEP")
    print("=" * 70)
    print(f"maze          : {mazes.identify(cfg.env)} "
          f"({mazes.DESCRIPTIONS.get(mazes.identify(cfg.env), 'custom')})")
    print(f"sparsity mode : {cfg.sparsity.mode}")
    print(f"shaping       : {args.shaping}")
    print(f"etas          : {cfg.etas}")
    print(f"seeds         : {len(cfg.seeds)} ({cfg.seeds[0]}..{cfg.seeds[-1]})")
    print(f"budget        : {cfg.dqn.budget_steps:,} env steps per run")
    print(f"config hash   : {cfg.config_hash()}")
    print(f"total runs    : {len(cfg.etas) * len(cfg.seeds)}")
    print("=" * 70 + "\n")

    if args.shaping == "terminal":
        print("NOTE: --shaping terminal is the ABLATION, not the fair "
              "comparison.\n      Do not report it as 'DQN'.\n")

    os.makedirs(args.outdir, exist_ok=True)
    rows = run_sweep(cfg, args.outdir, workers=args.workers,
                     shaping_mode=args.shaping)

    print(f"\nwrote {len(rows)} runs to {args.outdir}/summary.csv")
    print(f"next: python scripts/14_dqn_analyze.py --outdir {args.outdir}")


if __name__ == "__main__":
    main()
