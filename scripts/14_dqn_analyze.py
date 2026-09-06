#!/usr/bin/env python3
"""Analyse a DQN sweep: descriptives, tests, figures, report.md.

Deliberately reuses `neat_sparsity.stats` rather than reimplementing the tests,
so both arms are analysed by identical code. If the statistics ever change, they
change for both arms at once and cannot drift apart.

Order of operations matters and is not cosmetic:
  1. `descriptives.csv` is written FIRST. Look at it before anything else.
     Medians and CIs will tell you whether a result is real long before a
     p-value will.
  2. Then omnibus tests with effect sizes.
  3. Then figures.
  4. Then the report, which states null results as null results.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neat_sparsity.stats import full_report, describe_distributions
from dqn_sparsity import plots as P
from dqn_sparsity.runner import load_blocks


# (column, label, group) -- fixed before looking at results.
RUN_METRICS = [
    ("final_true_best",      "final best score (true objective)", "performance"),
    ("best_true_ever",       "best score ever (true objective)",  "performance"),
    ("final_success_rate",   "final success rate",                "performance"),
    ("success_rate_last10",  "success rate (last 10 blocks)",     "performance"),
    ("first_solution_step",  "env steps to first solution",       "performance"),
    ("mean_zero_reward_fraction",
     "zero-reward transition fraction",                           "manipulation"),
    ("final_distinct_rewards",
     "distinct reward values in buffer",                          "manipulation"),
    ("mean_reward_entropy",  "reward entropy (normalised)",       "manipulation"),
    ("mean_terminal_correction",
     "terminal correction (mean per episode)",                    "manipulation"),
    ("mean_td_loss",         "mean TD loss",                      "learning"),
    ("final_td_loss",        "final TD loss",                     "learning"),
    ("mean_q_spread",        "mean Q-value spread",               "learning"),
    ("final_q_spread",       "final Q-value spread",              "learning"),
    ("wallclock_s",          "wall-clock seconds per run",        "cost"),
    ("gradient_updates",     "gradient updates",                  "cost"),
]

TRAJECTORY_METRICS = [
    ("true_best",            "true objective (greedy policy)"),
    ("td_loss",              "TD loss"),
    ("q_spread",             "Q-value spread"),
    ("zero_reward_fraction", "zero-reward transition fraction"),
    ("reward_entropy",       "reward entropy (normalised)"),
    ("train_return_mean",    "training return (eta units)"),
]

THRESHOLD_METRICS = ["final_true_best", "final_success_rate", "mean_q_spread"]


def groups_from(df, metric):
    if metric not in df.columns:
        return {}
    return {float(k): pd.to_numeric(sub[metric], errors="coerce").tolist()
            for k, sub in df.groupby("eta")}


def curves_from(bdf, metric):
    if metric not in bdf.columns:
        return {}
    out = {}
    for eta, sub in bdf.groupby("eta"):
        piv = sub.pivot_table(index="seed", columns="block", values=metric)
        out[float(eta)] = piv.to_numpy(dtype=float)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="results/dqn_sweep")
    ap.add_argument("--figdir", default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    figdir = args.figdir or os.path.join(args.outdir, "figures")
    os.makedirs(figdir, exist_ok=True)

    df = pd.read_csv(os.path.join(args.outdir, "summary.csv"))
    if df.empty:
        raise SystemExit("empty summary.csv")

    hashes = set(df["config_hash"].astype(str))
    if len(hashes) > 1:
        raise SystemExit(f"runs span multiple config hashes {sorted(hashes)}; "
                         "they are not poolable.")

    bdf = load_blocks(args.outdir)
    print(f"{len(df)} runs, {df['eta'].nunique()} conditions, "
          f"config hash {list(hashes)[0]}")

    # -- 1. descriptives first -------------------------------------------------- #
    rows = []
    for col, label, group in RUN_METRICS:
        g = groups_from(df, col)
        if not g:
            continue
        for d in describe_distributions(g):
            d.update({"metric": col, "label": label, "group": group})
            rows.append(d)
    desc = pd.DataFrame(rows)
    desc.to_csv(os.path.join(args.outdir, "descriptives.csv"), index=False)
    print(f"  wrote descriptives.csv ({len(desc)} rows) -- read this first")

    # -- 2. tests --------------------------------------------------------------- #
    test_rows, posthoc_rows, thresholds = [], [], {}
    for col, label, group in RUN_METRICS:
        g = groups_from(df, col)
        g = {k: [v for v in vs if v == v] for k, vs in g.items()}
        g = {k: vs for k, vs in g.items() if len(vs) >= 3}
        if len(g) < 3:
            continue
        rep = full_report(g, label)
        keys = sorted(g)
        omni = rep["omnibus_kruskal"]
        trend = rep["trend"]
        boots = rep["bootstrap_medians"]
        lo_k, hi_k = str(keys[0]), str(keys[-1])
        # Cliff's delta between the extreme conditions, pulled out of the Dunn
        # table rather than recomputed, so the two arms use identical numbers.
        cd, mag = float("nan"), "?"
        for ph in rep["posthoc_dunn"]:
            if {float(ph["a"]), float(ph["b"])} == {keys[0], keys[-1]}:
                cd, mag = ph["cliffs_delta"], ph["magnitude"]
                if float(ph["a"]) != keys[0]:
                    cd = -cd
                break
        row = {
            "metric": col, "label": label, "group": group,
            "median_min": boots.get(lo_k, {}).get("median", float("nan")),
            "median_max": boots.get(hi_k, {}).get("median", float("nan")),
            "ci_lo_min": boots.get(lo_k, {}).get("lo95", float("nan")),
            "ci_hi_min": boots.get(lo_k, {}).get("hi95", float("nan")),
            "ci_lo_max": boots.get(hi_k, {}).get("lo95", float("nan")),
            "ci_hi_max": boots.get(hi_k, {}).get("hi95", float("nan")),
            "kruskal_h": omni["statistic"], "kruskal_p": omni["pvalue"],
            "epsilon_sq": float(omni["epsilon_squared"]),
            "spearman_rho": trend["spearman_rho"],
            "spearman_p": trend["spearman_p"],
            "jonckheere_p": trend["jonckheere_p"],
            "cliffs_delta": cd, "cliffs_magnitude": mag,
        }
        test_rows.append(row)
        for ph in rep["posthoc_dunn"]:
            posthoc_rows.append({**ph, "metric": col, "label": label})
        th = rep.get("threshold") or {}
        if col in THRESHOLD_METRICS and th.get("breakpoint") is not None:
            thresholds[col] = th

    pd.DataFrame(test_rows).to_csv(os.path.join(args.outdir, "tests.csv"),
                                   index=False)
    if posthoc_rows:
        pd.DataFrame(posthoc_rows).to_csv(
            os.path.join(args.outdir, "posthoc.csv"), index=False)
    print(f"  wrote tests.csv ({len(test_rows)} metrics)")

    # -- 3. figures ------------------------------------------------------------- #
    n_fig = 0
    for col, label, group in RUN_METRICS:
        g = groups_from(df, col)
        g = {k: [v for v in vs if v == v] for k, vs in g.items()}
        if len(g) < 2:
            continue
        P.sparsity_curve(g, label, figdir, f"fig_dqn_{col}",
                         title=f"{label}  [DQN, {group}]")
        n_fig += 1

    for col in THRESHOLD_METRICS:
        if col in thresholds:
            g = groups_from(df, col)
            g = {k: [v for v in vs if v == v] for k, vs in g.items()}
            P.threshold_plot(g, thresholds[col], col, figdir,
                             f"fig_dqn_{col}_threshold")
            n_fig += 1

    if not bdf.empty:
        for col, label in TRAJECTORY_METRICS:
            c = curves_from(bdf, col)
            if c:
                P.learning_curves(c, figdir, f"fig_dqn_traj_{col}", ylabel=label)
                n_fig += 1

    zf = groups_from(df, "mean_zero_reward_fraction")
    re_ = groups_from(df, "mean_reward_entropy")
    if zf and re_:
        P.reward_signal_plot(zf, re_, figdir)
        n_fig += 1
    print(f"  wrote {n_fig} figures to {figdir}/")

    # -- 4. report -------------------------------------------------------------- #
    L = ["# DQN under reward sparsity -- results\n"]
    L.append(f"- runs: **{len(df)}** ({df.groupby('eta')['seed'].nunique().max()} "
             f"seeds x {df['eta'].nunique()} sparsity conditions)")
    L.append(f"- eta grid: {', '.join(f'{e:g}' for e in sorted(df['eta'].unique()))}")
    L.append(f"- sparsity mode: `{df['sparsity_mode'].iloc[0]}`; "
             f"shaping: `{df.get('shaping_mode', pd.Series(['?'])).iloc[0]}`; "
             f"config hash `{list(hashes)[0]}`")
    L.append(f"- interaction budget: "
             f"{int(pd.to_numeric(df['env_steps']).median()):,} env steps per run")
    L.append(f"- alpha = {args.alpha}; Kruskal-Wallis omnibus, Dunn post-hoc, "
             f"Holm-adjusted.\n")

    L.append("## Manipulation check\n")
    L.append("Before reading any performance result, confirm the manipulation "
             "reached the learner. `zero-reward transition fraction` should rise "
             "with eta and `reward entropy` should fall. If they do not, eta is "
             "not doing anything on this arm and nothing below is "
             "interpretable.\n")
    mc = [r for r in test_rows if r["group"] == "manipulation"]
    if mc:
        L.append("| metric | median @ eta_min | median @ eta_max | Spearman rho | "
                 "eps^2 |")
        L.append("|---|---|---|---|---|")
        for r in mc:
            L.append(f"| {r['label']} | {r.get('median_min', float('nan')):.4g} | "
                     f"{r.get('median_max', float('nan')):.4g} | "
                     f"{r.get('spearman_rho', float('nan')):.3f} | "
                     f"{r.get('epsilon_sq', float('nan')):.3f} |")
        L.append("")

    for group, title in (("performance", "Performance (eta-independent units)"),
                         ("learning", "Learning diagnostics"),
                         ("cost", "Cost")):
        rs = [r for r in test_rows if r["group"] == group]
        if not rs:
            continue
        L.append(f"## {title}\n")
        L.append("| metric | median @ eta_min | median @ eta_max | Kruskal p | "
                 "eps^2 | Spearman rho | Cliff's d (extremes) |")
        L.append("|---|---|---|---|---|---|---|")
        for r in rs:
            L.append(f"| {r['label']} | {r.get('median_min', float('nan')):.4g} | "
                     f"{r.get('median_max', float('nan')):.4g} | "
                     f"{r.get('kruskal_p', float('nan')):.3g} | "
                     f"{r.get('epsilon_sq', float('nan')):.3f} | "
                     f"{r.get('spearman_rho', float('nan')):.3f} | "
                     f"{r.get('cliffs_delta', float('nan')):.2f} "
                     f"({r.get('cliffs_magnitude', '?')}) |")
        sig = [r["label"] for r in rs
               if r.get("kruskal_p", 1) < args.alpha]
        ns = [r["label"] for r in rs if r.get("kruskal_p", 1) >= args.alpha]
        L.append("")
        if sig:
            L.append("Significant at alpha=%.2g: %s\n"
                     % (args.alpha, ", ".join(f"`{s}`" for s in sig)))
        if ns:
            L.append("Not significant: %s. Report these as null results with "
                     "their effect sizes and CIs; a non-significant metric with "
                     "a wide CI means underpowered, not \"no effect\".\n"
                     % ", ".join(f"`{s}`" for s in ns))

    if thresholds:
        L.append("## Nonlinearity / threshold check\n")
        for col, t in thresholds.items():
            if not isinstance(t, dict) or t.get("breakpoint") is None:
                continue
            ci = t.get("breakpoint_ci", (None, None))
            L.append(f"- **{col}**: breakpoint at eta = {t['breakpoint']:.3g}"
                     + (f", bootstrap 95% CI [{ci[0]:.3g}, {ci[1]:.3g}]"
                        if ci[0] is not None else "")
                     + f"; delta-AIC = {t.get('delta_aic', float('nan')):.1f}.")
        L.append("\nThis is evidence of a change in slope, not of a phase "
                 "transition. With run-level N in the hundreds, delta-AIC > 2 "
                 "is a very weak bar; treat anything under 10 as suggestive.\n")

    L.append("## Caveats that belong in the manuscript\n")
    L.append("- `first_solution_step` is defined only for runs that solved. At "
             "high eta it is computed on a small, self-selected subset. Always "
             "read it next to the solve rate, and never report it alone.")
    L.append("- Performance is reported on the eta-independent true objective, "
             "so \"sparse runs score worse\" is a finding, not a definition.")
    L.append("- `final_success_rate` here is the greedy policy's success on the "
             "evaluation episode, which is NOT the same quantity as the NEAT "
             "arm's population success fraction. `final_true_best` is the "
             "directly comparable measure; use it for the cross-arm result.")
    L.append("- DQN hyperparameters were tuned once at eta = 0 and frozen. "
             "Re-tuning per condition would turn the sweep into a comparison of "
             "tuning effort.")
    L.append("- `mean_terminal_correction` measures how much of the return "
             "arrives as the end-of-episode correction rather than as shaped "
             "per-step reward. If it dominates at high eta, the agent is "
             "effectively on terminal-only reward there, and that is worth "
             "reporting explicitly rather than hiding.\n")

    path = os.path.join(args.outdir, "report.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L))
    print(f"  wrote {path}")
    print(f"\nnext: python scripts/15_crossover.py --neat <neat_dir> "
          f"--dqn {args.outdir}")


if __name__ == "__main__":
    main()
