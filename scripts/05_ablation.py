#!/usr/bin/env python3
"""Stage 6 -- functional vs. non-functional topology.

Ablates every champion produced by the sweep, under the sparsity-independent
objective, and writes ablation_summary.csv next to summary.csv.

Usage
-----
    python scripts/05_ablation.py --out results/core_sweep
    python scripts/05_ablation.py --out results/core_sweep --tau 0.02 --no-sequential
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neat_sparsity.ablation import run_ablation_sweep, DEFAULT_TAU


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/core_sweep")
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU,
                    help="functional threshold on the true objective (max 2.0)")
    ap.add_argument("--no-sequential", action="store_true",
                    help="skip greedy sequential pruning (much faster)")
    args = ap.parse_args()

    print(f"ablating champions in {args.out} (tau={args.tau})")
    rows = run_ablation_sweep(args.out, tau=args.tau,
                              sequential=not args.no_sequential)
    print(f"\n{len(rows)} champions ablated -> {args.out}/ablation_summary.csv")
    print("next:  python scripts/06_analyze.py --out", args.out)


if __name__ == "__main__":
    main()
