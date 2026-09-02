#!/usr/bin/env python3
"""Stages 5, 7 and 9 -- analysis, statistics and figures.

Produces, under <out>/analysis/:
    descriptives.csv        per-condition distribution summary (inspect FIRST)
    tests.csv               omnibus + trend + threshold, one row per metric
    posthoc.csv             Dunn pairwise with Cliff's delta, Holm/BH adjusted
    report.md               a readable write-up keyed to H1-H5
    fig_*.png / .pdf        figures

Usage
-----
    python scripts/06_analyze.py --out results/core_sweep
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from neat_sparsity.analysis import (load_summary, load_generations, load_ablation,
                                    groups_from, curves_from, check_comparability,
                                    RUN_METRICS, ABLATION_METRICS, TRAJECTORY_METRICS)
from neat_sparsity.config import ExperimentConfig
from neat_sparsity.plots import (sparsity_curve, trajectory_plot, threshold_plot,
                                 ablation_plot, env_plot)
from neat_sparsity.stats import (describe_distributions, kruskal, dunn, trend_tests,
                                 threshold_analysis, bootstrap_ci, cliffs_delta,
                                 cliffs_magnitude)


def slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s).strip("_").lower()


def analyse_metric(groups, col, label, hyp, figdir, seed=0):
    desc = describe_distributions(groups)
    for d in desc:
        d["metric"] = col
    omni = kruskal(groups)
    post = dunn(groups)
    for p in post:
        p["metric"] = col
    trend = trend_tests(groups)
    thr = threshold_analysis(groups, seed=seed)

    fig = sparsity_curve(groups, label, figdir, f"fig_{slug(col)}",
                         title=f"{label}  [{hyp}]", seed=seed)
    if "breakpoint" in thr:
        threshold_plot(groups, thr, label, figdir, f"fig_{slug(col)}_threshold")

    row = {
        "metric": col, "label": label, "hypothesis": hyp,
        "kruskal_H": omni.statistic, "kruskal_p": omni.pvalue,
        "epsilon_sq": omni.epsilon_squared, "k": omni.k, "n": omni.n,
        "spearman_rho": trend.get("spearman_rho"),
        "spearman_p": trend.get("spearman_p"),
        "jonckheere_p": trend.get("jonckheere_p"),
        "breakpoint": thr.get("breakpoint"),
        "breakpoint_ci_lo": (thr.get("breakpoint_ci95") or [None, None])[0],
        "breakpoint_ci_hi": (thr.get("breakpoint_ci95") or [None, None])[1],
        "delta_aic_piecewise_vs_linear": thr.get("delta_aic"),
        "piecewise_preferred": thr.get("piecewise_preferred"),
        "slope_before": thr.get("slope_before"),
        "slope_after": thr.get("slope_after"),
    }
    ks = sorted(groups)
    if len(ks) >= 2:
        d = cliffs_delta(groups[ks[0]], groups[ks[-1]])
        row["cliffs_delta_extremes"] = d
        row["magnitude_extremes"] = cliffs_magnitude(d)
        m0, l0, h0 = bootstrap_ci(groups[ks[0]], seed=seed)
        m1, l1, h1 = bootstrap_ci(groups[ks[-1]], seed=seed)
        row["median_eta_min"] = m0
        row["median_eta_min_ci"] = f"[{l0:.4g}, {h0:.4g}]"
        row["median_eta_max"] = m1
        row["median_eta_max_ci"] = f"[{l1:.4g}, {h1:.4g}]"
    return row, desc, post, fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/core_sweep")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    check_comparability(args.out)
    andir = os.path.join(args.out, "analysis")
    figdir = os.path.join(andir, "figures")
    os.makedirs(figdir, exist_ok=True)

    summary = load_summary(args.out)
    gens = load_generations(args.out)
    abl = load_ablation(args.out)

    print(f"{len(summary)} runs, {summary['eta'].nunique()} conditions, "
          f"{summary['seed'].nunique()} seeds")

    cfg_path = os.path.join(args.out, "config.json")
    if os.path.exists(cfg_path):
        cfg = ExperimentConfig.load(cfg_path)
        env_plot(cfg.env, figdir)

    rows, descs, posts, made = [], [], [], []

    # ---- run-level metrics ------------------------------------------------- #
    for col, label, hyp in RUN_METRICS:
        g = groups_from(summary, col)
        g = {k: [v for v in vs] for k, vs in g.items()}
        if not g or all(all(np.isnan(np.asarray(v, float))) for v in g.values()):
            continue
        r, d, p, f = analyse_metric(g, col, label, hyp, figdir, seed=args.seed)
        rows.append(r); descs.extend(d); posts.extend(p); made.append(f)

    # ---- ablation metrics --------------------------------------------------- #
    if not abl.empty:
        for col, label, hyp in ABLATION_METRICS:
            if col not in abl.columns:
                continue
            g = groups_from(abl, col)
            r, d, p, f = analyse_metric(g, col, label, hyp, figdir, seed=args.seed)
            rows.append(r); descs.extend(d); posts.extend(p); made.append(f)

        merged = summary.merge(abl, on=["eta", "seed"], how="inner",
                               suffixes=("", "_abl"))
        if "functional_conn_fraction" in merged:
            ablation_plot(groups_from(merged, "functional_conn_fraction"),
                          groups_from(merged, "champion_enabled_connections"),
                          figdir)

    # ---- longitudinal figures ------------------------------------------------ #
    if not gens.empty:
        for col, label in TRAJECTORY_METRICS:
            c = curves_from(gens, col)
            if c:
                trajectory_plot(c, label, figdir, f"fig_traj_{slug(col)}",
                                title=f"{label} over generations")

    # ---- write tables --------------------------------------------------------- #
    pd.DataFrame(descs).to_csv(os.path.join(andir, "descriptives.csv"), index=False)
    tests = pd.DataFrame(rows)
    tests.to_csv(os.path.join(andir, "tests.csv"), index=False)
    if posts:
        pd.DataFrame(posts).to_csv(os.path.join(andir, "posthoc.csv"), index=False)

    # ---- report ---------------------------------------------------------------- #
    write_report(andir, summary, tests, pd.DataFrame(posts), abl, args.alpha)
    print(f"\nwrote {andir}/report.md and {len(made)} figures")


def write_report(andir, summary, tests, posts, abl, alpha):
    etas = sorted(summary["eta"].unique())
    lines = []
    A = lines.append
    A("# Reward sparsity in NEAT -- results\n")
    A(f"- runs: **{len(summary)}** "
      f"({summary['seed'].nunique()} seeds x {len(etas)} sparsity conditions)")
    A(f"- eta grid: {', '.join(f'{e:g}' for e in etas)}")
    A(f"- sparsity mode: `{summary['sparsity_mode'].iloc[0]}`; "
      f"config hash `{summary['config_hash'].iloc[0]}`")
    A(f"- alpha = {alpha}; p-values below are Kruskal-Wallis omnibus, "
      f"post-hoc Dunn adjusted by Holm.\n")

    A("## Baseline validity\n")
    dense = summary[summary["eta"] == min(etas)]
    A(f"At the dense endpoint (eta={min(etas):g}), "
      f"{int(dense['ever_solved'].sum())}/{len(dense)} runs reached the goal; "
      f"median final success rate {dense['final_success_rate'].median():.2f}. "
      f"If this is near zero the sweep is uninterpretable -- the comparison would "
      f"be between failure modes, not between levels of selection information.\n")

    order = ["H1", "H2", "H3", "H4", "H5", "performance", "dynamics"]
    titles = {
        "H1": "H1 -- selection: does sparsity reduce fitness differentiation?",
        "H2": "H2 -- evolution: does sparsity change topological innovation?",
        "H3": "H3 -- complexity: does sparsity change structural complexity?",
        "H4": "H4 -- functionality: is the extra structure functional?",
        "H5": "H5 -- nonlinearity: are there threshold effects?",
        "performance": "Performance (reported in sparsity-independent units)",
        "dynamics": "Population dynamics",
    }

    for hyp in order:
        sub = tests[tests["hypothesis"] == hyp] if hyp != "H5" else tests
        if hyp == "H5":
            sub = tests[tests["piecewise_preferred"] == True]  # noqa: E712
            A(f"\n## {titles[hyp]}\n")
            if sub.empty:
                A("No metric preferred the two-segment fit over a linear fit "
                  "(all delta-AIC <= 2). There is no evidence for a threshold "
                  "effect in this data; describe the relationship as gradual.\n")
            else:
                for _, r in sub.iterrows():
                    ci = ""
                    if pd.notna(r.get("breakpoint_ci_lo")):
                        ci = (f", bootstrap 95% CI "
                              f"[{r['breakpoint_ci_lo']:.3g}, {r['breakpoint_ci_hi']:.3g}]")
                    A(f"- **{r['label']}**: breakpoint at eta = "
                      f"{r['breakpoint']:.3g}{ci}; slope {r['slope_before']:.3g} "
                      f"-> {r['slope_after']:.3g}; delta-AIC = "
                      f"{r['delta_aic_piecewise_vs_linear']:.1f}.")
                A("\nThis is evidence of a change in slope, not of a phase "
                  "transition. Only use stronger language if the breakpoint CI is "
                  "narrow and the effect replicates under the other sparsity modes.\n")
            continue

        if sub.empty:
            continue
        A(f"\n## {titles[hyp]}\n")
        A("| metric | median @ eta_min | median @ eta_max | Kruskal p | eps^2 | "
          "Spearman rho | Cliff's d (extremes) |")
        A("|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            A(f"| {r['label']} | {r.get('median_eta_min', float('nan')):.4g} "
              f"{r.get('median_eta_min_ci','')} | "
              f"{r.get('median_eta_max', float('nan')):.4g} "
              f"{r.get('median_eta_max_ci','')} | "
              f"{r['kruskal_p']:.2e} | {r['epsilon_sq']:.3f} | "
              f"{r['spearman_rho']:.3f} | {r.get('cliffs_delta_extremes', float('nan')):.2f} "
              f"({r.get('magnitude_extremes','')}) |")

        sig = sub[(sub["kruskal_p"] < alpha)]
        ns = sub[~(sub["kruskal_p"] < alpha)]
        if len(sig):
            A(f"\nSignificant at alpha={alpha}: "
              + ", ".join(f"`{m}`" for m in sig["label"]) + ".")
        if len(ns):
            A(f"\nNot significant: " + ", ".join(f"`{m}`" for m in ns["label"])
              + ". Report these as null results with their effect sizes and CIs; "
                "a non-significant metric with a wide CI means underpowered, not "
                "\"no effect\".")

    if not abl.empty:
        A("\n## Functional topology accounting\n")
        A("| eta | median enabled conns | median functional fraction | "
          "median functional count | median sequentially removable |")
        A("|---|---|---|---|---|")
        for eta, sub in abl.groupby("eta"):
            A(f"| {eta:g} | {sub['n_enabled_connections'].median():.1f} | "
              f"{sub['functional_conn_fraction'].median():.3f} | "
              f"{sub['n_functional_conns'].median():.1f} | "
              f"{sub.get('seq_removable_fraction', pd.Series([float('nan')])).median():.3f} |")
        A("\nA larger network is only *more complex in a useful sense* if the "
          "functional count grows too. Compare column 2 against column 4 before "
          "using the word \"bloat\" anywhere in the manuscript.\n")

    A("\n## Caveats that belong in the manuscript\n")
    A("- `generations to first solution` is defined only for runs that solved; "
      "at high eta it is computed on a small, self-selected subset. Always read "
      "it next to the solve rate, and never report it alone.")
    A("- `best score ever` is a maximum statistic over "
      f"{summary['generations'].iloc[0]} generations x population; it is "
      "comparable across conditions only because the evaluation budget is "
      "identical. `final best score` is the primary performance measure.")
    A("- Fitness differentiation at eta = 1 is zero by construction. That is the "
      "manipulation working, not a finding; the findings are what happens to "
      "topology and dynamics as a consequence.")
    A("- Several metrics are tested here. Decide before writing which are "
      "primary (one per hypothesis is a defensible choice) and treat the rest "
      "as exploratory.\n")
    A("\n## Interpretation checklist\n")
    A("- Only reward sparsity differs between conditions (config hash is shared).")
    A("- Performance is reported on the eta-independent objective, so "
      "\"sparse runs score worse\" is a finding, not a definition.")
    A("- Effect sizes and CIs accompany every p-value.")
    A("- Terms 'bloat', 'stagnation' and 'phase transition' are used only where "
      "the corresponding measurement supports them.")
    A("- The causal chain (Section 5) is supported only if the H1 metrics move "
      "*and* the H2/H3 metrics move *and* the ordering is consistent across "
      "sparsity modes.\n")

    with open(os.path.join(andir, "report.md"), "w") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
