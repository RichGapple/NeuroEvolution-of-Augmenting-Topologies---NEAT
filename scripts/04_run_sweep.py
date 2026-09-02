#!/usr/bin/env python3
"""Stage 4 -- the main experiment: sweep eta across many independent seeds.

Usage
-----
    python scripts/04_run_sweep.py --workers 8
    python scripts/04_run_sweep.py --seeds 30 --generations 120 --pop 120
    python scripts/04_run_sweep.py --mode gated --out results/robustness_gated
    python scripts/04_run_sweep.py --resume            # skip completed runs
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neat_sparsity.config import core_config, ExperimentConfig
from neat_sparsity.runner import run_sweep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/core_sweep")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=30, help="seeds per condition")
    ap.add_argument("--seed-start", type=int, default=1000)
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--pop", type=int, default=None)
    ap.add_argument("--etas", type=float, nargs="+", default=None)
    ap.add_argument("--mode", choices=["quantized", "checkpoint", "gated"],
                    default=None, help="sparsity operationalisation")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = core_config()
    cfg.seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    if args.generations:
        cfg.generations = args.generations
    if args.pop:
        cfg.neat.pop_size = args.pop
    if args.etas:
        cfg.etas = args.etas
    if args.mode:
        cfg.sparsity.mode = args.mode
        cfg.name = f"sweep_{args.mode}"

    n = len(cfg.etas) * len(cfg.seeds)
    print(f"sweep: {len(cfg.etas)} conditions x {len(cfg.seeds)} seeds = {n} runs")
    print(f"generations={cfg.generations}  pop={cfg.neat.pop_size}  "
          f"mode={cfg.sparsity.mode}")
    print(f"config hash = {cfg.config_hash()}")
    print(f"L(eta): " + ", ".join(f"{e:g}->{cfg.sparsity.levels(e)}" for e in cfg.etas))
    print()

    if args.resume:
        done = set()
        for eta in cfg.etas:
            for seed in cfg.seeds:
                p = os.path.join(args.out, f"eta={eta:g}", f"seed={seed}", "run.json")
                if os.path.exists(p):
                    done.add((eta, seed))
        if done:
            print(f"resume: {len(done)} runs already complete, skipping them")
            remaining_seeds = sorted({s for e in cfg.etas for s in cfg.seeds
                                      if (e, s) not in done})
            # simplest safe resume: rerun only conditions with missing seeds
            cfg.seeds = remaining_seeds
            if not cfg.seeds:
                print("nothing to do")
                return

    run_sweep(cfg, args.out, workers=args.workers)
    print(f"\ndone -> {args.out}/summary.csv")
    print("next:  python scripts/05_ablation.py --out", args.out)


if __name__ == "__main__":
    main()
