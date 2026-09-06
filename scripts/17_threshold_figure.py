#!/usr/bin/env python3
"""The headline figure: where does each arm fall off its OWN ceiling?

Why this figure exists
----------------------
Comparing raw scores across arms conflates two unrelated things. On maze A,
NEAT reaches 2.0 and DQN is capped at ~0.72 by the maze's deceptive gradient.
That 1.29 gap is *deception*, not reward sparsity, and reporting it as the
headline buries the actual result.

Normalising each arm against its own eta=0 performance removes the confound.
Deception sets each method's ceiling; this figure measures where each method
falls off the ceiling it can actually reach. Both arms are measured on the same
maze, the same interaction budget and the same reward, so the comparison of
*thresholds* is clean even though the comparison of *scores* is not.

Threshold definition
--------------------
    The largest eta, reached contiguously from eta = 0, at which the median
    stays within `--tol` (default 5%) of that arm's own eta = 0 median.

"Contiguously" matters: DQN's curve is non-monotonic (it recovers at eta=0.875),
so a plain "last eta above threshold" rule would report the recovery rather than
the failure. The threshold is where the arm first leaves its ceiling and does
not come back.

This deliberately does NOT use the segmented-regression breakpoints from
`15_crossover.py`. A continuous two-segment fit cannot represent a step
function -- it pulls the breakpoint early -- and on this data it also latches
onto the degenerate eta = 1 endpoint. Those numbers are not reportable.

Usage
-----
    python scripts/17_threshold_figure.py \
        --neat results/core_sweep --dqn results/dqn_A --outdir results/cmp_A
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NEAT_C = "#C44E52"
DQN_C = "#4C72B0"
GREY = "#7f7f7f"


def levels(eta: float, k: int = 8) -> int:
    return int(2 ** round(k * (1.0 - eta)))


def load(outdir: str, metric: str) -> dict:
    path = os.path.join(outdir, "summary.csv")
    if not os.path.exists(path):
        raise SystemExit(f"no summary.csv in {outdir}")
    df = pd.read_csv(path)
    if metric not in df.columns:
        raise SystemExit(f"{outdir}: no column {metric!r}")
    out = {}
    for eta, sub in df.groupby("eta"):
        v = pd.to_numeric(sub[metric], errors="coerce").dropna().to_numpy()
        if len(v):
            out[round(float(eta), 6)] = v
    return out


def threshold(groups: dict, tol: float) -> tuple:
    """Largest eta reached contiguously from 0 while within tol of the ceiling.

    Returns (eta, ceiling, normalised medians keyed by eta).
    """
    keys = sorted(groups)
    ceiling = float(np.median(groups[keys[0]]))
    if ceiling <= 0:
        return None, ceiling, {}
    norm = {k: float(np.median(groups[k])) / ceiling for k in keys}
    last = keys[0]
    for k in keys:
        if norm[k] >= 1.0 - tol:
            last = k
        else:
            break
    return last, ceiling, norm


def bootstrap_threshold(groups: dict, tol: float, n_boot: int, seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    keys = sorted(groups)
    out = []
    for _ in range(n_boot):
        res = {k: rng.choice(groups[k], len(groups[k]), replace=True) for k in keys}
        t, _, _ = threshold(res, tol)
        if t is not None:
            out.append(t)
    if not out:
        return None, None
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--neat", required=True)
    ap.add_argument("--dqn", required=True)
    ap.add_argument("--outdir", default="results/cmp_A")
    ap.add_argument("--metric", default="final_true_best")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="fraction below own ceiling that still counts as 'holding'")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="fig_threshold_normalised")
    args = ap.parse_args()

    figdir = os.path.join(args.outdir, "figures")
    os.makedirs(figdir, exist_ok=True)

    arms = {}
    for label, path, colour in (("NEAT", args.neat, NEAT_C),
                                ("DQN", args.dqn, DQN_C)):
        g = load(path, args.metric)
        t, ceiling, norm = threshold(g, args.tol)
        lo, hi = bootstrap_threshold(g, args.tol, args.boot, args.seed)
        arms[label] = {"groups": g, "threshold": t, "ceiling": ceiling,
                       "norm": norm, "ci": (lo, hi), "colour": colour,
                       "levels": levels(t) if t is not None else None,
                       "n_seeds": int(np.median([len(v) for v in g.values()]))}

    # -- figure ---------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    keys = sorted(set().union(*[set(a["groups"]) for a in arms.values()]))

    rng = np.random.default_rng(args.seed)
    for label, a in arms.items():
        ks = [k for k in keys if k in a["norm"]]
        med = [a["norm"][k] for k in ks]
        los, his = [], []
        for k in ks:
            v = a["groups"][k]
            b = [np.median(rng.choice(v, len(v), replace=True)) / a["ceiling"]
                 for _ in range(1500)]
            los.append(np.percentile(b, 2.5))
            his.append(np.percentile(b, 97.5))
        ax.plot(ks, med, "o-", color=a["colour"], lw=2.2, ms=5.5, zorder=3,
                label=f"{label}  (own ceiling = {a['ceiling']:.3f})")
        ax.fill_between(ks, los, his, color=a["colour"], alpha=0.16, lw=0, zorder=2)

        if a["threshold"] is not None:
            ax.axvline(a["threshold"], color=a["colour"], ls="--", lw=1.3,
                       alpha=0.75, zorder=1)

    ax.axhline(1.0 - args.tol, color=GREY, ls=":", lw=1.2, zorder=1)
    ax.text(0.015, 1.0 - args.tol + 0.015,
            f"{100 * (1 - args.tol):.0f}% of own ceiling",
            fontsize=8.5, color=GREY, va="bottom")

    ax.set_xlabel(r"reward sparsity  $\eta$")
    ax.set_ylabel("performance relative to own $\\eta$=0 ceiling")
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="lower left")

    top = ax.secondary_xaxis("top")
    top.set_xticks(keys)
    top.set_xticklabels([str(levels(k)) for k in keys], fontsize=8.5)
    top.set_xlabel("distinguishable reward levels  $L(\\eta)$", fontsize=9.5)

    n = arms["NEAT"]; d = arms["DQN"]
    if n["levels"] and d["levels"]:
        ratio = d["levels"] / n["levels"]
        ax.set_title(f"DQN needs {ratio:.0f}x finer reward resolution than NEAT "
                     f"({d['levels']} vs {n['levels']} levels)", fontsize=11.5)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(figdir, f"{args.name}.{ext}"), dpi=200,
                    bbox_inches="tight")
    plt.close(fig)

    # -- numbers --------------------------------------------------------------- #
    print("=" * 68)
    print("CEILING-NORMALISED THRESHOLDS")
    print("=" * 68)
    print(f"metric: {args.metric}   tolerance: within {100*args.tol:.0f}% of "
          f"own eta=0 median\n")
    for label, a in arms.items():
        ci = a["ci"]
        ci_s = (f"  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]"
                if ci[0] is not None else "")
        print(f"{label}")
        print(f"  own ceiling        {a['ceiling']:.3f}   ({a['n_seeds']} seeds/cell)")
        print(f"  holds down to      eta = {a['threshold']:.3f}{ci_s}")
        print(f"  i.e.               {a['levels']} levels "
              f"= {int(round(8 * (1 - a['threshold'])))} bits")
        nxt = [k for k in sorted(a["norm"]) if k > a["threshold"]]
        if nxt:
            print(f"  first failure at   eta = {nxt[0]:.3f} "
                  f"({levels(nxt[0])} levels, "
                  f"{100 * a['norm'][nxt[0]]:.0f}% of ceiling)")
        print()

    if n["levels"] and d["levels"]:
        ratio = d["levels"] / n["levels"]
        bits = int(round(8 * (1 - d["threshold"]) - 8 * (1 - n["threshold"])))
        print(f"HEADLINE: DQN requires {ratio:.0f}x finer reward resolution "
              f"than NEAT")
        print(f"          ({d['levels']} levels vs {n['levels']} levels; "
              f"{bits} additional bits)\n")

    rec = {"metric": args.metric, "tolerance": args.tol,
           "arms": {k: {"ceiling": v["ceiling"], "threshold": v["threshold"],
                        "levels": v["levels"], "ci": list(v["ci"]),
                        "normalised_medians": v["norm"]}
                    for k, v in arms.items()}}
    if n["levels"] and d["levels"]:
        rec["resolution_ratio"] = d["levels"] / n["levels"]
    with open(os.path.join(args.outdir, "thresholds.json"), "w") as fh:
        json.dump(rec, fh, indent=2)

    print(f"figure -> {figdir}/{args.name}.png (+.pdf)")
    print(f"numbers -> {args.outdir}/thresholds.json")
    print("\nCaveat for the caption: each arm is normalised against its OWN "
          "eta=0 median.\nOn a deceptive maze the two ceilings differ "
          "(deception, not sparsity), so this\nfigure compares thresholds, not "
          "absolute performance. Report the raw curves\nalongside it.")


if __name__ == "__main__":
    main()
