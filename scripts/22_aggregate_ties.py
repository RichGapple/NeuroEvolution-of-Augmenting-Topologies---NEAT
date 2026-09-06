#!/usr/bin/env python3
"""Aggregate the selection-tie instrumentation from an existing mechanism sweep.

Why this is separate
--------------------
`arbitrary_selection_fraction` is written per *generation*, into each run's
`generations.csv`. `21_mechanism_sweep.py` only reads the per-*run* summaries
that `run_sweep` returns, so the metric never reached its dataframe and the
script reported it missing. The data was collected correctly; only the
aggregation was wrong.

So no re-running is needed. This walks the generation logs of a completed
sweep and produces what the mechanism claim actually needs.

What it answers
---------------
The claim is that NEAT tolerates coarse reward because truncation selection
reads ranks, not magnitudes, and only breaks when ties blur the cut. The
observed threshold shift is consistent with that but does not prove it. The
direct evidence is whether the share of selection decisions made by genome key
rather than fitness rises with eta, and whether it is higher in the conditions
whose thresholds are lower.

The pre-registered direction was WRONG: a looser cut (s=0.5) was predicted to
tolerate coarser reward, and instead failed earlier. The post-hoc account is
that a cut near the median sits where individuals are densest in fitness and so
most likely to tie, whereas a cut in the tail sits where values are spread
apart. This script tests that account directly: if it is right, the tie
fraction at s=0.5 must exceed that at s=0.2.

If it does not, the account is wrong and the paper reports the threshold shift
alone -- which is still the main result, since "the threshold belongs to the
selection cut, not the algorithm" holds whichever way it moves.

Usage
-----
    python scripts/22_aggregate_ties.py --sweep results/mechanism2
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TIE = "arbitrary_selection_fraction"
AMB = "cut_ambiguous_species_fraction"


def levels(eta: float, k: int = 8) -> int:
    return int(2 ** round(k * (1.0 - eta)))


def load_generations(sweep_dir: str) -> pd.DataFrame:
    """Walk <sweep>/<condition>/eta=*/seed=*/generations.csv."""
    rows = []
    pattern = os.path.join(sweep_dir, "*", "eta=*", "seed=*", "generations.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(
            f"no generations.csv under {sweep_dir}.\n"
            f"Expected {pattern}. Check the sweep directory name.")
    for p in paths:
        parts = p.replace("\\", "/").split("/")
        cond = parts[-4]
        eta = float(parts[-3].split("=")[1])
        seed = int(parts[-2].split("=")[1])
        df = pd.read_csv(p)
        if TIE not in df.columns:
            continue
        rows.append({
            "condition": cond, "eta": eta, "seed": seed,
            "tie_mean": float(df[TIE].mean()),
            "tie_max": float(df[TIE].max()),
            "tie_final": float(df[TIE].iloc[-1]),
            "tie_last25": float(df[TIE].tail(25).mean()),
            "ambiguous_mean": (float(df[AMB].mean()) if AMB in df.columns
                               else float("nan")),
            "n_generations": len(df),
        })
    if not rows:
        raise SystemExit(
            f"found {len(paths)} generation logs but none contain {TIE!r}.\n"
            f"The sweep ran before scripts/patch_population_ties.py --apply.\n"
            f"Apply the patch and re-run the sweep to collect it.")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", default="results/mechanism2")
    ap.add_argument("--metric", default="tie_mean",
                    choices=["tie_mean", "tie_max", "tie_final", "tie_last25"])
    ap.add_argument("--min-ceiling", type=float, default=1.9,
                    help="exclude conditions whose eta=0 median falls below "
                         "this; they never solve the maze so their 'threshold' "
                         "is meaningless")
    args = ap.parse_args()

    gen = load_generations(args.sweep)
    print(f"loaded {len(gen)} runs from {args.sweep}")
    gen.to_csv(os.path.join(args.sweep, "tie_by_run.csv"), index=False)

    # thresholds and ceilings from the sweep's own table
    th_path = os.path.join(args.sweep, "thresholds.csv")
    th = pd.read_csv(th_path) if os.path.exists(th_path) else None

    piv = gen.pivot_table(index="condition", columns="eta",
                          values=args.metric, aggfunc="median")

    print("\n" + "=" * 78)
    print(f"ARBITRARY SELECTION FRACTION  ({args.metric}, median over seeds)")
    print("=" * 78)
    print("Share of the population whose survival was decided by genome key")
    print("rather than fitness, because the truncation cut fell inside a tie.\n")

    etas = sorted(piv.columns)
    hdr = "  ".join(f"{e:>6g}" for e in etas)
    print(f"{'condition':<10} {'ceiling':>8} {'thresh':>7}   {hdr}")
    order = ["s0.1", "s0.2", "s0.3", "s0.5", "s0.7", "pop50", "pop200"]
    conds = [c for c in order if c in piv.index] + \
            [c for c in piv.index if c not in order]

    valid = []
    for c in conds:
        ceil = thr = float("nan")
        if th is not None and c in set(th["name"]):
            r = th[th["name"] == c].iloc[0]
            ceil, thr = float(r["ceiling"]), float(r["threshold"])
        flag = "" if ceil >= args.min_ceiling else "  <- excluded, no ceiling"
        if ceil >= args.min_ceiling:
            valid.append((c, thr))
        vals = "  ".join(f"{piv.loc[c, e]:>6.3f}" if e in piv.columns
                         and piv.loc[c, e] == piv.loc[c, e] else "     -"
                         for e in etas)
        print(f"{c:<10} {ceil:>8.3f} {thr:>7.3f}   {vals}{flag}")

    # -- does the tie fraction explain the direction? -------------------------- #
    print("\n" + "=" * 78)
    print("DOES THE TIE FRACTION EXPLAIN THE THRESHOLD ORDER?")
    print("=" * 78)
    print("Post-hoc account under test: a cut near the median sits where")
    print("individuals are densest in fitness and so most likely to tie; a cut")
    print("in the tail sits where values are spread apart. If right, conditions")
    print("with LOWER thresholds should show HIGHER tie fractions.\n")

    if len(valid) >= 3:
        rows = []
        for c, thr in valid:
            v = piv.loc[c].dropna()
            rows.append({"condition": c, "threshold": thr,
                         "tie_at_low_eta": float(v.iloc[:2].mean()),
                         "tie_overall": float(v.mean())})
        d = pd.DataFrame(rows).sort_values("threshold")
        print(f"{'condition':<10} {'threshold':>10} {'levels':>7} "
              f"{'tie (low eta)':>14} {'tie (all eta)':>14}")
        for _, r in d.iterrows():
            print(f"{r['condition']:<10} {r['threshold']:>10.3f} "
                  f"{levels(r['threshold']):>7} "
                  f"{r['tie_at_low_eta']:>14.3f} {r['tie_overall']:>14.3f}")

        from scipy import stats as sps
        if len(d) >= 3:
            rho, p = sps.spearmanr(d["threshold"], d["tie_overall"])
            print(f"\nSpearman(threshold, tie fraction) = {rho:+.3f}  (p = {p:.3g})")
            if rho < -0.5:
                print("NEGATIVE as the account predicts: conditions that tie")
                print("more fail at finer resolution. The post-hoc explanation")
                print("is supported by direct measurement -- but it remains")
                print("post-hoc and must be labelled so.")
            elif rho > 0.5:
                print("POSITIVE, the opposite of the account. The tie")
                print("explanation is WRONG. Report the threshold shift alone.")
            else:
                print("No clear relationship. The tie fraction does not")
                print("explain the threshold order. Report the shift alone and")
                print("say the mechanism remains open.")
        d.to_csv(os.path.join(args.sweep, "tie_vs_threshold.csv"), index=False)
    else:
        print("fewer than 3 conditions reached ceiling; cannot test.")

    # -- figure ---------------------------------------------------------------- #
    figdir = os.path.join(args.sweep, "figures")
    os.makedirs(figdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    cmap = plt.get_cmap("viridis")
    surv = [c for c in conds if c.startswith("s")]
    for i, c in enumerate(surv):
        v = piv.loc[c].dropna()
        ax.plot(v.index, v.values, "o-", lw=1.9, ms=5,
                color=cmap(i / max(1, len(surv) - 1)),
                label=f"survival = {c[1:]}")
        if th is not None and c in set(th["name"]):
            t = float(th[th["name"] == c].iloc[0]["threshold"])
            ax.axvline(t, color=cmap(i / max(1, len(surv) - 1)), ls=":",
                       lw=1.1, alpha=0.65)
    ax.set_xlabel(r"reward sparsity  $\eta$")
    ax.set_ylabel("arbitrary selection fraction")
    ax.set_title("Selection decided by key rather than fitness\n"
                 "(dotted lines mark each condition's own threshold)",
                 fontsize=10.5)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(figdir, f"fig_arbitrary_selection.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)

    piv.to_csv(os.path.join(args.sweep, "tie_by_condition.csv"))
    print(f"\nfigure -> {figdir}/fig_arbitrary_selection.png")
    print(f"tables -> {args.sweep}/tie_by_condition.csv, tie_vs_threshold.csv")


if __name__ == "__main__":
    main()
