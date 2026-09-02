#!/usr/bin/env python3
"""Compare two sweeps at matched eta values.

Two uses:
  1. Extension vs. baseline -- does novelty pressure or adaptive parsimony
     change the outcome at a given sparsity level?
  2. Robustness -- does the headline result survive a different
     operationalisation of eta (`quantized` vs `checkpoint` vs `gated`)?
     If the direction and rough magnitude replicate across all three, the
     finding is about reward sparsity; if it flips, the finding is about the
     particular reward function and the paper must say so.

Usage
-----
    python scripts/08_compare.py --a results/core_sweep --b results/ext_novelty
    python scripts/08_compare.py --a results/core_sweep --b results/robust_gated \
        --label-a quantized --label-b gated
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy import stats as sps

from neat_sparsity.analysis import load_summary, RUN_METRICS
from neat_sparsity.stats import cliffs_delta, cliffs_magnitude, bootstrap_ci


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline sweep directory")
    ap.add_argument("--b", required=True, help="comparison sweep directory")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    A = load_summary(args.a)
    B = load_summary(args.b)
    la = args.label_a or os.path.basename(args.a.rstrip("/"))
    lb = args.label_b or os.path.basename(args.b.rstrip("/"))
    out = args.out or os.path.join(args.b, "analysis")
    os.makedirs(out, exist_ok=True)

    shared = sorted(set(A["eta"]).intersection(set(B["eta"])))
    if not shared:
        raise SystemExit("the two sweeps share no eta values")
    print(f"comparing {la} vs {lb} at eta in {shared}\n")

    rows = []
    for col, label, hyp in RUN_METRICS:
        if col not in A.columns or col not in B.columns:
            continue
        for eta in shared:
            a = pd.to_numeric(A[A["eta"] == eta][col], errors="coerce").dropna().values
            b = pd.to_numeric(B[B["eta"] == eta][col], errors="coerce").dropna().values
            if len(a) < 3 or len(b) < 3:
                continue
            try:
                u, p = sps.mannwhitneyu(a, b, alternative="two-sided")
            except ValueError:
                u, p = float("nan"), float("nan")
            d = cliffs_delta(b, a)          # >0 => b exceeds a
            ma, loa, hia = bootstrap_ci(a)
            mb, lob, hib = bootstrap_ci(b)
            rows.append({
                "metric": col, "label": label, "hypothesis": hyp, "eta": eta,
                f"median_{la}": ma, f"ci_{la}": f"[{loa:.4g}, {hia:.4g}]",
                f"median_{lb}": mb, f"ci_{lb}": f"[{lob:.4g}, {hib:.4g}]",
                "mannwhitney_p": p, "cliffs_delta_b_minus_a": d,
                "magnitude": cliffs_magnitude(d),
                "n_a": len(a), "n_b": len(b),
            })

    if not rows:
        raise SystemExit(
            "no metric had at least 3 usable runs per condition in both sweeps; "
            "nothing can be compared. Run more seeds."
        )
    df = pd.DataFrame(rows)
    path = os.path.join(out, f"compare_{la}_vs_{lb}.csv")
    df.to_csv(path, index=False)

    key = ["final_true_best", "final_success_rate", "final_mean_connections",
           "mean_tir", "mean_diversity"]
    show = df[df["metric"].isin(key)]
    if not show.empty:
        with pd.option_context("display.width", 200, "display.max_columns", 30):
            print(show[["label", "eta", f"median_{la}", f"median_{lb}",
                        "mannwhitney_p", "cliffs_delta_b_minus_a",
                        "magnitude"]].to_string(index=False))
    print(f"\nfull table -> {path}")
    print("\nreminder: this is a family of tests across metrics x eta. Adjust "
          "across the family, or restrict interpretation to the metrics you "
          "pre-registered as primary.")


if __name__ == "__main__":
    main()
