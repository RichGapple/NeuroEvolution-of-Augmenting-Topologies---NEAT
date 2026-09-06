"""Figures for the DQN arm and for the NEAT-vs-DQN comparison.

Style is inherited from `neat_sparsity.plots` so the two arms' figures sit
together in a paper without looking like they came from different projects.
The single-arm helpers (`sparsity_curve`, `threshold_plot`, `trajectory_plot`)
are re-exported rather than reimplemented.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neat_sparsity.plots import (_save, sparsity_curve, threshold_plot,
                                 trajectory_plot, env_plot)
from neat_sparsity.stats import bootstrap_ci

NEAT_C = "#C44E52"      # red   -- NEAT
DQN_C = "#4C72B0"       # blue  -- DQN
GREY = "#7f7f7f"

__all__ = ["sparsity_curve", "threshold_plot", "trajectory_plot", "env_plot",
           "crossover_plot", "learning_curves", "cost_plot",
           "reward_signal_plot", "paired_delta_plot", "budget_curve_plot"]


# --------------------------------------------------------------------------- #
def crossover_plot(neat_groups: Dict[float, Sequence[float]],
                   dqn_groups: Dict[float, Sequence[float]],
                   outdir: str, name: str = "fig_crossover",
                   ylabel: str = "final best score (true objective)",
                   crossover: Optional[dict] = None,
                   seed: int = 0) -> str:
    """The headline figure: both arms' performance against eta, on one axis.

    If the two medians cross, the crossing point is the paper's contribution --
    the reward resolution below which evolution becomes the better choice.
    """
    keys = sorted(set(neat_groups) | set(dqn_groups))
    fig, ax = plt.subplots(figsize=(6.8, 4.3))

    for groups, colour, label in ((neat_groups, NEAT_C, "NEAT"),
                                  (dqn_groups, DQN_C, "DQN")):
        ks = [k for k in keys if k in groups and len(groups[k])]
        meds, los, his = [], [], []
        for k in ks:
            m, lo, hi = bootstrap_ci(groups[k], seed=seed)
            meds.append(m); los.append(lo); his.append(hi)
        ax.plot(ks, meds, "o-", color=colour, lw=2.0, ms=5, label=label, zorder=3)
        ax.fill_between(ks, los, his, color=colour, alpha=0.18, lw=0, zorder=2)

    if crossover and crossover.get("eta") is not None:
        x = crossover["eta"]
        ax.axvline(x, color=GREY, ls="--", lw=1.2, zorder=1)
        lo, hi = crossover.get("ci", (None, None))
        if lo is not None and hi is not None and hi > lo:
            ax.axvspan(lo, hi, color=GREY, alpha=0.14, lw=0, zorder=0)
        ax.annotate(f"crossover  $\\eta$ = {x:.3f}",
                    xy=(x, ax.get_ylim()[1]), xytext=(4, -12),
                    textcoords="offset points", fontsize=9, color=GREY)

    ax.set_xlabel(r"reward sparsity  $\eta$")
    ax.set_ylabel(ylabel)
    ax.set_title("NEAT vs DQN under matched interaction budget")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, outdir, name)


# --------------------------------------------------------------------------- #
def paired_delta_plot(neat_groups: Dict[float, Sequence[float]],
                      dqn_groups: Dict[float, Sequence[float]],
                      outdir: str, name: str = "fig_arm_delta",
                      seed: int = 0) -> str:
    """Median(DQN) - Median(NEAT) per eta, with a bootstrap CI.

    Reads more cleanly than two overlaid curves when the effect is small: the
    zero line is where the two methods are indistinguishable.
    """
    keys = sorted(set(neat_groups) & set(dqn_groups))
    deltas, los, his = [], [], []
    rng = np.random.default_rng(seed)
    for k in keys:
        a = np.asarray([v for v in dqn_groups[k] if v == v], float)
        b = np.asarray([v for v in neat_groups[k] if v == v], float)
        deltas.append(np.median(a) - np.median(b))
        boots = [np.median(rng.choice(a, len(a))) - np.median(rng.choice(b, len(b)))
                 for _ in range(4000)]
        los.append(float(np.percentile(boots, 2.5)))
        his.append(float(np.percentile(boots, 97.5)))

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axhline(0, color=GREY, lw=1.2, ls="--")
    ax.errorbar(keys, deltas,
                yerr=[np.array(deltas) - np.array(los),
                      np.array(his) - np.array(deltas)],
                fmt="o-", color="#55A868", capsize=3, lw=1.8, ms=5)
    ax.fill_between(keys, 0, deltas, where=np.array(deltas) > 0,
                    color=DQN_C, alpha=0.12, lw=0)
    ax.fill_between(keys, 0, deltas, where=np.array(deltas) < 0,
                    color=NEAT_C, alpha=0.12, lw=0)
    ax.set_xlabel(r"reward sparsity  $\eta$")
    ax.set_ylabel("median(DQN) $-$ median(NEAT)")
    ax.set_title("Advantage by condition (above 0 = DQN better)")
    ax.grid(alpha=0.25)
    return _save(fig, outdir, name)


# --------------------------------------------------------------------------- #
def learning_curves(curves: Dict[float, np.ndarray], outdir: str,
                    name: str = "fig_dqn_learning",
                    ylabel: str = "true objective (greedy policy)",
                    xlabel: str = "fraction of interaction budget") -> str:
    """Median learning curve per eta with an IQR band."""
    keys = sorted(curves)
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for i, k in enumerate(keys):
        arr = np.asarray(curves[k], float)
        if arr.size == 0:
            continue
        x = np.linspace(0, 1, arr.shape[1])
        med = np.nanmedian(arr, axis=0)
        lo = np.nanpercentile(arr, 25, axis=0)
        hi = np.nanpercentile(arr, 75, axis=0)
        c = cmap(i / max(1, len(keys) - 1))
        ax.plot(x, med, color=c, lw=1.5, label=fr"$\eta$={k:g}")
        ax.fill_between(x, lo, hi, color=c, alpha=0.13, lw=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " over training")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    return _save(fig, outdir, name)


# --------------------------------------------------------------------------- #
def reward_signal_plot(zero_frac: Dict[float, Sequence[float]],
                       entropy: Dict[float, Sequence[float]],
                       outdir: str, name: str = "fig_reward_signal") -> str:
    """Manipulation check for the DQN arm.

    Left: what fraction of transitions carry zero reward.  Right: normalised
    entropy of the reward histogram.  These are the DQN counterparts of the NEAT
    arm's "distinct fitness values" and "selection entropy": they show the
    manipulation is doing what it claims on this side too.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for ax, groups, lab in ((axes[0], zero_frac, "zero-reward transition fraction"),
                            (axes[1], entropy, "reward entropy (normalised)")):
        keys = sorted(groups)
        meds, los, his = [], [], []
        for k in keys:
            m, lo, hi = bootstrap_ci(groups[k])
            meds.append(m); los.append(lo); his.append(hi)
        ax.errorbar(keys, meds,
                    yerr=[np.array(meds) - np.array(los),
                          np.array(his) - np.array(meds)],
                    fmt="o-", color=DQN_C, capsize=3, lw=1.8, ms=5)
        ax.set_xlabel(r"reward sparsity  $\eta$")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.25)
    fig.suptitle("DQN manipulation check: what the agent actually receives")
    fig.tight_layout()
    return _save(fig, outdir, name)


# --------------------------------------------------------------------------- #
def cost_plot(rows: list, outdir: str, name: str = "fig_cost") -> str:
    """Three-panel honest cost accounting: env steps, wall-clock, peak memory.

    Never collapse these into one "compute" number.  NEAT and DQN differ in
    opposite directions on sample efficiency and on hardware footprint, and a
    single number hides exactly the trade the paper is about.
    """
    arms = sorted({r["arm"] for r in rows})
    fields = [("env_steps", "environment steps"),
              ("wallclock_s", "wall-clock seconds / run"),
              ("peak_rss_mb", "peak RSS (MB)")]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    for ax, (f, lab) in zip(axes, fields):
        vals, labels, colours = [], [], []
        for arm in arms:
            v = [r.get(f) for r in rows if r["arm"] == arm and r.get(f) is not None]
            v = [x for x in v if x == x]
            if not v:
                continue
            vals.append(v)
            labels.append(arm.upper())
            colours.append(NEAT_C if arm == "neat" else DQN_C)
        if not vals:
            continue
        bp = ax.boxplot(vals, patch_artist=True, widths=0.5)
        ax.set_xticks(range(1, len(labels) + 1), labels)
        for patch, c in zip(bp["boxes"], colours):
            patch.set_facecolor(c)
            patch.set_alpha(0.45)
        for med in bp["medians"]:
            med.set_color("black")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.25, axis="y")
        if f == "env_steps":
            ax.set_yscale("log")
    fig.suptitle("Cost accounting (report all three; never a single number)")
    fig.tight_layout()
    return _save(fig, outdir, name)


# --------------------------------------------------------------------------- #
def budget_curve_plot(curves: Dict[float, np.ndarray],
                      neat_final: Dict[float, Sequence[float]],
                      outdir: str, name: str = "fig_budget_efficiency",
                      etas: Optional[Sequence[float]] = None) -> str:
    """Sample efficiency: DQN's score as a function of budget consumed, with
    NEAT's *final* score as a horizontal reference per eta.

    Where the DQN curve crosses the NEAT line is the fraction of NEAT's budget
    DQN needed to match it.  That number belongs in the abstract.
    """
    keys = sorted(etas if etas is not None else curves)
    keys = [k for k in keys if k in curves]
    n = len(keys)
    ncol = min(3, max(1, n))
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.0 * nrow),
                             squeeze=False)
    for i, k in enumerate(keys):
        ax = axes[i // ncol][i % ncol]
        arr = np.asarray(curves[k], float)
        x = np.linspace(0, 1, arr.shape[1])
        ax.plot(x, np.nanmedian(arr, axis=0), color=DQN_C, lw=1.8, label="DQN")
        ax.fill_between(x, np.nanpercentile(arr, 25, axis=0),
                        np.nanpercentile(arr, 75, axis=0),
                        color=DQN_C, alpha=0.15, lw=0)
        if k in neat_final and len(neat_final[k]):
            ref = float(np.nanmedian(neat_final[k]))
            ax.axhline(ref, color=NEAT_C, ls="--", lw=1.6,
                       label="NEAT final (median)")
        ax.set_title(fr"$\eta$ = {k:g}", fontsize=10)
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(frameon=False, fontsize=8)
        if i // ncol == nrow - 1:
            ax.set_xlabel("fraction of budget")
        if i % ncol == 0:
            ax.set_ylabel("true objective")
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.tight_layout()
    return _save(fig, outdir, name)
