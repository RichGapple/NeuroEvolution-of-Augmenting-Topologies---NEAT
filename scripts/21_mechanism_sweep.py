#!/usr/bin/env python3
"""W3: does the reward-resolution threshold move as the mechanism predicts?

The claim under test
--------------------
NEAT tolerates coarse reward because parent selection is truncation: sort a
species by fitness, keep the top `survival_threshold` fraction. Truncation reads
only the ORDERING. Quantising the reward is therefore harmless until it creates
enough ties that the cut falls inside a tied group, at which point survival is
decided by genome key rather than merit.

If that account is right, the threshold is not a property of NEAT. It is a
property of *how hard the ordering is to resolve at the cut*, and it must move
when you change the cut. Two falsifiable predictions:

    P1  survival_threshold
        A TIGHTER cut (0.10) must separate the top 10%, which needs finer
        resolution -> threshold moves LEFT (fails at lower eta).
        A LOOSER cut (0.70) tolerates coarser reward -> moves RIGHT.

    P2  pop_size
        More individuals distributed over the same L bins means more per bin,
        so the cut is more likely to land inside a tie -> threshold moves LEFT.

If neither moves, the rank-invariance account is WRONG and the paper must say
so. A mechanism that survives a real attempt to break it is worth far more than
one that was never tested.

Direct evidence as well as indirect
-----------------------------------
The threshold shift is indirect. `patch_population_ties.py` adds the direct
measurement -- `arbitrary_selection_fraction`, the share of the population whose
survival was decided by key rather than fitness. Run that patch first. Then this
script can show the mechanism firing (the fraction rising with eta) AND the
consequence (the threshold moving), which together are much stronger than either
alone.

Cost
----
Default grid: 7 conditions x 6 eta x 20 seeds = 840 runs, ~48 s each
= ~11 core-hours, roughly 1.5 h on 8 workers.

Usage
-----
    python scripts/21_mechanism_sweep.py --workers 8
    python scripts/21_mechanism_sweep.py --workers 8 --seeds 10   # faster, noisier
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neat_sparsity.config import core_config
from neat_sparsity.runner import run_sweep


def levels(eta: float, k: int = 8) -> int:
    return int(2 ** round(k * (1.0 - eta)))


# --------------------------------------------------------------------------- #
def build_conditions(survivals, pops, base_survival, base_pop):
    """One condition per knob setting. The baseline appears once."""
    conds = []
    for s in survivals:
        conds.append({"name": f"s{s:g}", "survival_threshold": s,
                      "pop_size": base_pop, "knob": "survival_threshold",
                      "value": s})
    for p in pops:
        if p == base_pop:
            continue
        conds.append({"name": f"pop{p}", "survival_threshold": base_survival,
                      "pop_size": p, "knob": "pop_size", "value": p})
    return conds


def threshold_of(medians: dict, tol: float):
    """Largest eta, contiguous from the smallest, staying within tol of ceiling."""
    keys = sorted(medians)
    ceiling = medians[keys[0]]
    if ceiling <= 0:
        return None, ceiling
    last = keys[0]
    for k in keys:
        if medians[k] / ceiling >= 1.0 - tol:
            last = k
        else:
            break
    return last, ceiling


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/mechanism")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed-start", type=int, default=5000)
    ap.add_argument("--etas", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.625, 0.75, 0.875])
    ap.add_argument("--survivals", type=float, nargs="+",
                    default=[0.10, 0.20, 0.30, 0.50, 0.70])
    ap.add_argument("--pops", type=int, nargs="+", default=[50, 100, 200])
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--generations", type=int, default=100)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and cost estimate, run nothing")
    args = ap.parse_args()

    base = core_config()
    conds = build_conditions(args.survivals, args.pops,
                             base.neat.survival_threshold, base.neat.pop_size)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    n_runs = len(conds) * len(args.etas) * len(seeds)

    print("=" * 72)
    print("W3 MECHANISM SWEEP -- does the threshold move with the cut?")
    print("=" * 72)
    print(f"conditions : {len(conds)}  ({', '.join(c['name'] for c in conds)})")
    print(f"eta grid   : {args.etas}")
    print(f"seeds      : {len(seeds)}")
    print(f"total runs : {n_runs}")
    print(f"estimate   : ~{n_runs * 48 / 3600:.1f} core-hours, "
          f"~{n_runs * 48 / 3600 / max(1, args.workers):.1f} h on "
          f"{args.workers} workers")
    print("\nPredictions recorded BEFORE running:")
    print("  P1  tighter cut (s=0.10) -> threshold LOWER than baseline s=0.30")
    print("  P1  looser  cut (s=0.70) -> threshold HIGHER than baseline")
    print("  P2  larger population    -> threshold LOWER than baseline")
    print("=" * 72 + "\n")

    if args.dry_run:
        return

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "predictions.json"), "w") as fh:
        json.dump({"P1_tighter_cut": "threshold decreases",
                   "P1_looser_cut": "threshold increases",
                   "P2_larger_pop": "threshold decreases",
                   "recorded_before_running": True,
                   "conditions": conds, "etas": args.etas,
                   "seeds": seeds, "tolerance": args.tol}, fh, indent=2)

    t0 = time.time()
    all_rows = []
    for i, c in enumerate(conds, 1):
        cfg = core_config()
        cfg.name = f"mech_{c['name']}"
        cfg.generations = args.generations
        cfg.etas = list(args.etas)
        cfg.seeds = list(seeds)
        cfg.neat.survival_threshold = c["survival_threshold"]
        cfg.neat.pop_size = c["pop_size"]

        outdir = os.path.join(args.out, c["name"])
        print(f"[{i}/{len(conds)}] {c['name']}: "
              f"survival={c['survival_threshold']:g}, pop={c['pop_size']}, "
              f"hash={cfg.config_hash()}", flush=True)
        rows = run_sweep(cfg, outdir, workers=args.workers)
        for r in rows:
            r["condition"] = c["name"]
            r["knob"] = c["knob"]
            r["knob_value"] = c["value"]
            r["survival_threshold"] = c["survival_threshold"]
            r["pop_size"] = c["pop_size"]
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(args.out, "all_runs.csv"), index=False)
    print(f"\n{len(df)} runs in {(time.time() - t0) / 3600:.2f} h")

    # -- thresholds ------------------------------------------------------------ #
    results = []
    for c in conds:
        sub = df[df["condition"] == c["name"]]
        med = {float(e): float(g["final_true_best"].median())
               for e, g in sub.groupby("eta")}
        th, ceiling = threshold_of(med, args.tol)
        tie = None
        if "arbitrary_selection_fraction" in sub.columns:
            tie = {float(e): float(g["arbitrary_selection_fraction"].median())
                   for e, g in sub.groupby("eta")}
        results.append({**c, "threshold": th, "ceiling": ceiling,
                        "levels": levels(th) if th is not None else None,
                        "medians": med, "tie_fraction": tie})

    res = pd.DataFrame([{k: v for k, v in r.items()
                         if k not in ("medians", "tie_fraction")}
                        for r in results])
    res.to_csv(os.path.join(args.out, "thresholds.csv"), index=False)

    print("\n" + "=" * 72)
    print("THRESHOLDS BY CONDITION")
    print("=" * 72)
    print(f"{'condition':<10} {'knob':<20} {'value':>7} {'ceiling':>8} "
          f"{'threshold':>10} {'levels':>7}")
    for r in results:
        th = f"{r['threshold']:.3f}" if r["threshold"] is not None else "--"
        lv = r["levels"] if r["levels"] else "--"
        print(f"{r['name']:<10} {r['knob']:<20} {r['value']:>7g} "
              f"{r['ceiling']:>8.3f} {th:>10} {lv:>7}")

    # -- verdict --------------------------------------------------------------- #
    by_name = {r["name"]: r for r in results}
    base_s = f"s{base.neat.survival_threshold:g}"
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    verdict = {}
    if base_s in by_name and by_name[base_s]["threshold"] is not None:
        b = by_name[base_s]["threshold"]
        print(f"baseline ({base_s}) threshold = {b:.3f} "
              f"({levels(b)} levels)\n")
        for key, pred, label in ((f"s{min(args.survivals):g}", "lower",
                                  "P1 tighter cut"),
                                 (f"s{max(args.survivals):g}", "higher",
                                  "P1 looser cut"),
                                 (f"pop{max(args.pops)}", "lower",
                                  "P2 larger population")):
            if key not in by_name or by_name[key]["threshold"] is None:
                continue
            t = by_name[key]["threshold"]
            moved = ("lower" if t < b else "higher" if t > b else "unchanged")
            ok = (moved == pred)
            verdict[label] = {"predicted": pred, "observed": moved,
                              "baseline": b, "value": t,
                              "supported": ok}
            print(f"{label:<22} predicted {pred:<8} observed {moved:<10} "
                  f"({b:.3f} -> {t:.3f})   {'SUPPORTED' if ok else 'NOT SUPPORTED'}")

        n_ok = sum(v["supported"] for v in verdict.values())
        print(f"\n{n_ok}/{len(verdict)} predictions supported.")
        if n_ok == len(verdict):
            print("The threshold moves with the selection cut, as the")
            print("rank-invariance account requires. Report as mechanism")
            print("evidence, not just correlation.")
        elif n_ok == 0:
            print("The threshold does NOT move. The rank-invariance account is")
            print("not supported. Report this: a mechanism that fails its own")
            print("test is a finding, and hiding it is worse than not testing.")
        else:
            print("Mixed. Report each prediction separately and do not")
            print("summarise as 'the mechanism is confirmed'.")

    with open(os.path.join(args.out, "verdict.json"), "w") as fh:
        json.dump({"results": results, "verdict": verdict,
                   "tolerance": args.tol}, fh, indent=2, default=str)

    # -- figures --------------------------------------------------------------- #
    figdir = os.path.join(args.out, "figures")
    os.makedirs(figdir, exist_ok=True)

    for knob, xlabel in (("survival_threshold", "survival_threshold (cut fraction)"),
                         ("pop_size", "population size")):
        sel = [r for r in results if r["knob"] == knob and r["threshold"] is not None]
        if len(sel) < 2:
            continue
        sel.sort(key=lambda r: r["value"])
        fig, ax = plt.subplots(figsize=(5.6, 4.0))
        xs = [r["value"] for r in sel]
        ys = [r["threshold"] for r in sel]
        ax.plot(xs, ys, "o-", color="#C44E52", lw=2, ms=7)
        for r in sel:
            ax.annotate(f"{r['levels']}L", (r["value"], r["threshold"]),
                        textcoords="offset points", xytext=(6, 6), fontsize=8.5,
                        color="#5F5E5A")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"reward-resolution threshold  $\eta^*$")
        ax.set_title("Threshold shifts with the selection cut" if
                     knob == "survival_threshold" else
                     "Threshold shifts with population size", fontsize=11)
        ax.grid(alpha=0.25)
        if knob == "pop_size":
            ax.set_xscale("log")
            ax.set_xticks(xs)
            ax.set_xticklabels([str(x) for x in xs])
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(figdir, f"fig_threshold_vs_{knob}.{ext}"),
                        dpi=200, bbox_inches="tight")
        plt.close(fig)

    have_tie = any(r["tie_fraction"] for r in results)
    if have_tie:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        cmap = plt.get_cmap("viridis")
        sel = [r for r in results if r["knob"] == "survival_threshold"
               and r["tie_fraction"]]
        sel.sort(key=lambda r: r["value"])
        for i, r in enumerate(sel):
            ks = sorted(r["tie_fraction"])
            ax.plot(ks, [r["tie_fraction"][k] for k in ks], "o-",
                    color=cmap(i / max(1, len(sel) - 1)), lw=1.8, ms=4.5,
                    label=f"s={r['value']:g}")
            if r["threshold"] is not None:
                ax.axvline(r["threshold"], color=cmap(i / max(1, len(sel) - 1)),
                           ls=":", lw=1.0, alpha=0.7)
        ax.set_xlabel(r"reward sparsity  $\eta$")
        ax.set_ylabel("arbitrary selection fraction")
        ax.set_title("The mechanism firing: selection decided by key, not fitness",
                     fontsize=11)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8.5)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(figdir, f"fig_arbitrary_selection.{ext}"),
                        dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("\nDirect mechanism metric present. If arbitrary_selection_fraction")
        print("rises sharply at each condition's own threshold, that is the")
        print("mechanism caught in the act rather than inferred.")
    else:
        print("\nNOTE: arbitrary_selection_fraction not found in the run records.")
        print("Run scripts/patch_population_ties.py --apply first to get the")
        print("direct measurement alongside the threshold shift.")

    print(f"\nfigures -> {figdir}/")
    print(f"tables  -> {args.out}/thresholds.csv, verdict.json")


if __name__ == "__main__":
    main()
