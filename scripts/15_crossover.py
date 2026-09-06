#!/usr/bin/env python3
"""NEAT vs DQN: the crossover analysis. This is the paper's headline.

Produces
--------
    comparison/crossover.json        the crossover point and its bootstrap CI
    comparison/per_condition.csv     Mann-Whitney + Cliff's delta at every eta
    comparison/cost_table.csv        env steps / wall-clock / memory, separately
    comparison/sample_efficiency.csv budget fraction DQN needed to match NEAT
    comparison/report.md             the write-up
    comparison/figures/*.png|pdf     including fig_crossover

What this can and cannot support
--------------------------------
It CAN support: "at eta = X, method A scores higher than method B by this effect
size", and "the two curves cross at eta = X with this CI".

It CANNOT support "NEAT is better than DQN". The result is condition-dependent
by construction -- that is the entire point of the study. The script refuses to
compare arms whose environment, sparsity mode or eta grid differ, because a
crossover computed across mismatched configurations is worse than no crossover.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dqn_sparsity import compare as C
from dqn_sparsity import plots as P
from dqn_sparsity.runner import load_blocks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--neat", required=True, help="NEAT sweep directory")
    ap.add_argument("--dqn", required=True, help="DQN sweep directory")
    ap.add_argument("--outdir", default="results/comparison")
    ap.add_argument("--metric", default="final_true_best",
                    help="primary comparison metric (must exist in both arms)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--allow-seed-mismatch", action="store_true")
    args = ap.parse_args()

    figdir = os.path.join(args.outdir, "figures")
    os.makedirs(figdir, exist_ok=True)

    neat_df = C.load_arm(args.neat, "neat")
    dqn_df = C.load_arm(args.dqn, "dqn")

    print("=" * 70)
    print("NEAT vs DQN")
    print("=" * 70)
    print(f"NEAT: {len(neat_df)} runs from {args.neat}")
    print(f"DQN : {len(dqn_df)} runs from {args.dqn}")

    warnings = C.assert_comparable(neat_df, dqn_df,
                                   strict_seeds=not args.allow_seed_mismatch)
    for w in warnings:
        print(f"  {w}")
    print("  comparability: OK\n")

    neat_g = C.groups(neat_df, args.metric)
    dqn_g = C.groups(dqn_df, args.metric)
    if not neat_g or not dqn_g:
        raise SystemExit(f"metric {args.metric!r} missing from one of the arms")

    # -- per-condition tests ---------------------------------------------------- #
    tests = C.per_condition_tests(neat_g, dqn_g, alpha=args.alpha)
    tests.to_csv(os.path.join(args.outdir, "per_condition.csv"), index=False)
    print("Per-condition medians (true objective):")
    print(f"  {'eta':>6}  {'NEAT':>7}  {'DQN':>7}  {'delta':>7}  "
          f"{'Cliff d':>8}  {'p(Holm)':>9}  winner")
    for _, r in tests.iterrows():
        star = "*" if r["significant"] else " "
        print(f"  {r['eta']:>6g}  {r['neat_median']:>7.3f}  "
              f"{r['dqn_median']:>7.3f}  {r['delta_median']:>+7.3f}  "
              f"{r['cliffs_delta']:>+8.2f}  {r['p_holm']:>9.3g}{star} "
              f"{r['winner']}")

    # -- crossover -------------------------------------------------------------- #
    cross = C.crossover_point(neat_g, dqn_g)
    print("\nCrossover:")
    if cross["eta"] is None:
        print("  none on this grid -- one arm dominates everywhere.")
        print("  That is a legitimate result. Report it as a null crossover")
        print("  rather than extrapolating past the grid.")
    else:
        lo, hi = cross["ci"]
        print(f"  eta* = {cross['eta']:.3f}", end="")
        if lo is not None:
            print(f"   95% CI [{lo:.3f}, {hi:.3f}]  (width {cross['ci_width']:.3f})")
        else:
            print()
        print(f"  sign change present in {100 * cross['p_exists']:.1f}% of "
              f"bootstrap resamples")
        if cross.get("ci_width") and cross["ci_width"] > 0.30:
            print("  WARNING: CI spans >30% of the grid. Report the crossover as")
            print("           located between grid points, not as a point estimate.")
    with open(os.path.join(args.outdir, "crossover.json"), "w") as fh:
        json.dump({k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in cross.items()}, fh, indent=2)

    # -- per-arm breakpoints ---------------------------------------------------- #
    bp = {"neat": C.breakpoint_per_arm(neat_g), "dqn": C.breakpoint_per_arm(dqn_g)}
    with open(os.path.join(args.outdir, "breakpoints.json"), "w") as fh:
        json.dump(bp, fh, indent=2, default=str)
    print("\nWhere each arm falls off individually:")
    for arm, t in bp.items():
        if isinstance(t, dict) and t.get("breakpoint") is not None:
            ci = t.get("breakpoint_ci", (None, None))
            ci_s = (f" CI [{ci[0]:.3g}, {ci[1]:.3g}]"
                    if ci and ci[0] is not None else "")
            print(f"  {arm.upper():5s} breakpoint eta = {t['breakpoint']:.3g}"
                  f"{ci_s}, delta-AIC = {t.get('delta_aic', float('nan')):.1f}")

    # -- sample efficiency ------------------------------------------------------ #
    effic = pd.DataFrame()
    try:
        bdf = load_blocks(args.dqn)
        if not bdf.empty:
            effic = C.sample_efficiency(bdf, neat_g)
            effic.to_csv(os.path.join(args.outdir, "sample_efficiency.csv"),
                         index=False)
            print("\nBudget fraction DQN needed to match NEAT's final median:")
            for _, r in effic.iterrows():
                bf = "never" if not r["matched"] else f"{r['budget_frac_to_match']:.3f}"
                print(f"  eta={r['eta']:<6g} {bf}")
    except Exception as exc:
        print(f"\n(sample efficiency unavailable: {exc})")

    # -- cost ------------------------------------------------------------------- #
    costs = C.cost_table(neat_df, dqn_df)
    costs.to_csv(os.path.join(args.outdir, "cost_table.csv"), index=False)

    # -- figures ---------------------------------------------------------------- #
    P.crossover_plot(neat_g, dqn_g, figdir, crossover=cross,
                     ylabel="final best score (true objective)")
    P.paired_delta_plot(neat_g, dqn_g, figdir)

    rows = (neat_df.to_dict("records") + dqn_df.to_dict("records"))
    P.cost_plot(rows, figdir)

    try:
        if not bdf.empty:
            curves = {}
            for eta, sub in bdf.groupby("eta"):
                piv = sub.pivot_table(index="seed", columns="block",
                                      values="true_best")
                curves[float(eta)] = piv.to_numpy(dtype=float)
            P.learning_curves(curves, figdir, "fig_dqn_learning_all")
            P.budget_curve_plot(curves, neat_g, figdir)
    except Exception:
        pass
    print(f"\nfigures -> {figdir}/")

    # -- report ----------------------------------------------------------------- #
    md = C.render_report(neat_df, dqn_df, tests, cross, costs, effic,
                         warnings, alpha=args.alpha)
    path = os.path.join(args.outdir, "report.md")
    with open(path, "w") as fh:
        fh.write(md)
    print(f"report  -> {path}")
    print("\n" + "=" * 70)
    print("Read comparison/report.md. The 'what this does and does not")
    print("license' section at the bottom is the part to copy into your")
    print("discussion, not to skip.")
    print("=" * 70)


if __name__ == "__main__":
    main()
