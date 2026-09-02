"""Figures.

Every summary figure shows run-level uncertainty (median + bootstrap CI +
individual runs), never a single representative run (roadmap Section 15).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .stats import bootstrap_ci

PALETTE = plt.get_cmap("viridis")


def _save(fig, outdir: str, name: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
def sparsity_curve(groups: Dict[float, Sequence[float]], ylabel: str,
                   outdir: str, name: str, title: str = "",
                   seed: int = 0) -> str:
    """Median + 95% bootstrap CI + jittered per-run points against eta."""
    keys = sorted(groups)
    meds, los, his = [], [], []
    for k in keys:
        m, lo, hi = bootstrap_ci(groups[k], seed=seed)
        meds.append(m); los.append(lo); his.append(hi)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    rng = np.random.default_rng(seed)
    for k in keys:
        y = np.asarray(groups[k], float)
        y = y[~np.isnan(y)]
        if len(y):
            jitter = (rng.random(len(y)) - 0.5) * 0.018
            ax.scatter(np.full(len(y), k) + jitter, y, s=12, alpha=0.35,
                       color="#4C72B0", linewidths=0, zorder=2)
    ax.errorbar(keys, meds,
                yerr=[np.array(meds) - np.array(los), np.array(his) - np.array(meds)],
                fmt="o-", color="#C44E52", capsize=3, lw=1.8, ms=5, zorder=3,
                label="median (95% bootstrap CI)")
    ax.set_xlabel(r"reward sparsity  $\eta$")
    ax.set_ylabel(ylabel)
    ax.set_title(title or ylabel)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, outdir, name)


def trajectory_plot(per_eta_curves: Dict[float, np.ndarray], ylabel: str,
                    outdir: str, name: str, title: str = "") -> str:
    """Generation-wise median across seeds, one line per eta, IQR band."""
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    keys = sorted(per_eta_curves)
    for i, k in enumerate(keys):
        arr = per_eta_curves[k]          # (n_seeds, n_generations)
        if arr.size == 0:
            continue
        med = np.nanmedian(arr, axis=0)
        q1 = np.nanpercentile(arr, 25, axis=0)
        q3 = np.nanpercentile(arr, 75, axis=0)
        gens = np.arange(len(med))
        c = PALETTE(i / max(1, len(keys) - 1))
        ax.plot(gens, med, color=c, lw=1.8, label=rf"$\eta$={k:g}")
        ax.fill_between(gens, q1, q3, color=c, alpha=0.15, linewidth=0)
    ax.set_xlabel("generation")
    ax.set_ylabel(ylabel)
    ax.set_title(title or ylabel)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    return _save(fig, outdir, name)


def threshold_plot(groups: Dict[float, Sequence[float]], fit: dict, ylabel: str,
                   outdir: str, name: str) -> Optional[str]:
    if "breakpoint" not in fit:
        return None
    keys = sorted(groups)
    xs, ys = [], []
    for k in keys:
        for v in groups[k]:
            if v == v:
                xs.append(k); ys.append(v)
    x = np.asarray(xs, float); y = np.asarray(ys, float)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.scatter(x, y, s=14, alpha=0.35, color="#4C72B0", linewidths=0)
    bp = fit["breakpoint"]
    grid = np.linspace(x.min(), x.max(), 200)
    A = np.vstack([np.ones_like(grid), grid, np.maximum(0.0, grid - bp)]).T
    Ax = np.vstack([np.ones_like(x), x, np.maximum(0.0, x - bp)]).T
    coef, *_ = np.linalg.lstsq(Ax, y, rcond=None)
    ax.plot(grid, A @ coef, color="#C44E52", lw=2,
            label=f"two-segment fit (bp={bp:.3f}, dAIC={fit['delta_aic']:.1f})")
    lin = np.polyfit(x, y, 1)
    ax.plot(grid, np.polyval(lin, grid), "--", color="grey", lw=1.4, label="linear fit")
    if "breakpoint_ci95" in fit:
        ax.axvspan(*fit["breakpoint_ci95"], color="#C44E52", alpha=0.10,
                   label="breakpoint 95% CI")
    ax.set_xlabel(r"reward sparsity  $\eta$")
    ax.set_ylabel(ylabel)
    ax.set_title("Nonlinearity / threshold check")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, outdir, name)


def ablation_plot(groups_func: Dict[float, Sequence[float]],
                  groups_size: Dict[float, Sequence[float]],
                  outdir: str, name: str = "fig_ablation") -> str:
    """Total structure vs. functional structure per sparsity level."""
    keys = sorted(groups_func)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    sizes = [np.nanmedian(np.asarray(groups_size[k], float)) for k in keys]
    fracs = [np.nanmedian(np.asarray(groups_func[k], float)) for k in keys]
    funcs = [s * f for s, f in zip(sizes, fracs)]
    w = 0.35 * (keys[1] - keys[0]) if len(keys) > 1 else 0.3
    ax.bar([k - w / 2 for k in keys], sizes, width=w, label="total enabled connections",
           color="#B0B0B0")
    ax.bar([k + w / 2 for k in keys], funcs, width=w, label="functional (ablation)",
           color="#4C72B0")
    ax.set_xlabel(r"reward sparsity  $\eta$")
    ax.set_ylabel("median count (champion)")
    ax.set_title("Functional vs. non-functional topology")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, outdir, name)


def env_plot(env_cfg, outdir: str, name: str = "fig_environment",
             trajectory: Optional[np.ndarray] = None) -> str:
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    for (x0, y0, x1, y1) in env_cfg.obstacles:
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   color="#666666", alpha=0.85))
    ax.add_patch(plt.Circle(env_cfg.goal, env_cfg.goal_radius,
                            color="#2CA02C", alpha=0.6))
    ax.plot(*env_cfg.start, "o", color="#C44E52", ms=8)
    ax.text(env_cfg.start[0] + 2, env_cfg.start[1] + 2, "S")
    ax.text(env_cfg.goal[0] + 2, env_cfg.goal[1] + 2, "G")
    if trajectory is not None and len(trajectory):
        ax.plot(trajectory[:, 0], trajectory[:, 1], "-", lw=1.2, color="#1F77B4")
    ax.set_xlim(0, env_cfg.width); ax.set_ylim(0, env_cfg.height)
    ax.set_aspect("equal"); ax.grid(alpha=0.2)
    ax.set_title("Navigation environment (identical for all $\\eta$)")
    return _save(fig, outdir, name)
