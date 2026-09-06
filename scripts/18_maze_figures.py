#!/usr/bin/env python3
"""Figures 2 and 5: the two mazes, and the drift-reachability measurement.

Everything here is drawn from the actual `EnvConfig` obstacle list and from
paths computed by grid BFS -- nothing is hand-placed. Hand-placing walls and
routes separately is how you end up with a figure where the route passes
through a wall, which is both wrong and the kind of error a reviewer spots
instantly.

Two figures
-----------
fig_mazes.png
    Maze A and maze B side by side. Obstacles from `mazes.MAZE_A/B`, the true
    shortest route from BFS, and the greedy distance-descending route from
    `maze_analysis.greedy_descent_optimum`. The trap on maze A is marked at the
    point where greedy descent stops, computed rather than eyeballed.

fig_drift_reachability.png
    What fraction of undirected policies reach the goal on each maze. This is
    the measurement the whole moderator claim rests on, and at present it
    exists only in a console log. Sampling matches the weight scales NEAT
    actually explores, and the second panel converts the rate into expected
    solvers per 10,000-genome run, which is the number that explains why maze B
    is solved with no reward at all.

Usage
-----
    python scripts/18_maze_figures.py --outdir results/figures
    python scripts/18_maze_figures.py --outdir results/figures --samples 8000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dqn_sparsity.config import core_config
from dqn_sparsity import mazes, maze_analysis
from neat_sparsity.env import NavEnv

WALL = "#B4B2A9"
WALL_EDGE = "#888780"
TRUE_C = "#639922"
GREEDY_C = "#D85A30"
START_C = "#5F5E5A"
GOAL_C = "#0F6E56"


# --------------------------------------------------------------------------- #
def _free_grid(cfg, res):
    r = cfg.agent_radius
    W, H = int(cfg.width / res) + 1, int(cfg.height / res) + 1
    free = np.ones((W, H), bool)
    for i in range(W):
        x = i * res
        for j in range(H):
            y = j * res
            if x < r or y < r or x > cfg.width - r or y > cfg.height - r:
                free[i, j] = False
                continue
            for (x0, y0, x1, y1) in cfg.obstacles:
                if x0 - r <= x <= x1 + r and y0 - r <= y <= y1 + r:
                    free[i, j] = False
                    break
    return free, W, H


def shortest_route(cfg, res=0.5):
    """BFS shortest path, returned as world-coordinate waypoints."""
    free, W, H = _free_grid(cfg, res)
    si, sj = int(cfg.start[0] / res), int(cfg.start[1] / res)
    gx, gy = cfg.goal
    par, dist = {}, -np.ones((W, H), int)
    dist[si, sj] = 0
    q = deque([(si, sj)])
    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < W and 0 <= b < H and free[a, b] and dist[a, b] < 0:
                dist[a, b] = dist[i, j] + 1
                par[(a, b)] = (i, j)
                q.append((a, b))
    best = None
    for i in range(W):
        for j in range(H):
            if dist[i, j] >= 0 and math.hypot(gx - i * res, gy - j * res) <= cfg.goal_radius:
                if best is None or dist[i, j] < dist[best]:
                    best = (i, j)
    if best is None:
        return [], None
    path, cur = [], best
    while cur in par:
        path.append((cur[0] * res, cur[1] * res))
        cur = par[cur]
    path.append(tuple(cfg.start))
    path.reverse()
    return path, dist[best] * res


def greedy_route(cfg, res=0.5):
    """The path a strictly distance-reducing policy takes, and where it stops."""
    free, W, H = _free_grid(cfg, res)
    gx, gy = cfg.goal
    ii, jj = np.meshgrid(np.arange(W) * res, np.arange(H) * res, indexing="ij")
    euc = np.hypot(gx - ii, gy - jj)
    i, j = int(cfg.start[0] / res), int(cfg.start[1] / res)
    seen, path = set(), [tuple(cfg.start)]
    while True:
        seen.add((i, j))
        cand = [(euc[a, b], a, b)
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1),
                               (1, 1), (1, -1), (-1, 1), (-1, -1))
                for a, b in [(i + di, j + dj)]
                if 0 <= a < W and 0 <= b < H and free[a, b] and (a, b) not in seen]
        if not cand:
            break
        e, a, b = min(cand)
        if e >= euc[i, j]:
            break
        i, j = a, b
        path.append((i * res, j * res))
    return path, (i * res, j * res), float(euc[i, j])


# --------------------------------------------------------------------------- #
def draw_maze(ax, cfg, title, subtitle, mark_trap):
    for (x0, y0, x1, y1) in cfg.obstacles:
        ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                        facecolor=WALL, edgecolor=WALL_EDGE,
                                        linewidth=0.6, zorder=2))

    sp, sp_len = shortest_route(cfg)
    gp, gend, gdist = greedy_route(cfg)

    if sp:
        a = np.array(sp)
        ax.plot(a[:, 0], a[:, 1], color=TRUE_C, lw=2.2, zorder=4,
                solid_capstyle="round")
    a = np.array(gp)
    ax.plot(a[:, 0], a[:, 1], color=GREEDY_C, lw=2.2, ls=(0, (5, 3)), zorder=5,
            solid_capstyle="round")

    d0 = math.hypot(cfg.goal[0] - cfg.start[0], cfg.goal[1] - cfg.start[1])
    progress = max(0.0, 1.0 - gdist / d0)
    reaches = gdist <= cfg.goal_radius

    if mark_trap and not reaches:
        ax.plot(*gend, "o", ms=13, mfc="none", mec=GREEDY_C, mew=2.2, zorder=6)
        ax.annotate(f"greedy route\nstops here\n({gend[0]:.0f}, {gend[1]:.0f})",
                    xy=gend, xytext=(gend[0] - 30, gend[1] - 22),
                    fontsize=8.5, color=GREEDY_C, ha="center", zorder=7,
                    arrowprops=dict(arrowstyle="-", color=GREEDY_C, lw=0.8))

    ax.plot(*cfg.start, "o", ms=7, color=START_C, zorder=6)
    ax.annotate("start", cfg.start, xytext=(4, -12), textcoords="offset points",
                fontsize=9, color=START_C)
    circ = plt.Circle(cfg.goal, cfg.goal_radius, fill=False, color=GOAL_C,
                      lw=1.4, ls=":", zorder=6)
    ax.add_patch(circ)
    ax.plot(*cfg.goal, "o", ms=6, color=GOAL_C, zorder=6)
    ax.annotate("goal", cfg.goal, xytext=(-8, 10), textcoords="offset points",
                fontsize=9, color=GOAL_C)

    ax.set_xlim(-2, cfg.width + 2)
    ax.set_ylim(-2, cfg.height + 2)
    ax.set_aspect("equal")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.15, zorder=0)
    ax.set_title(title, fontsize=11.5, loc="left", pad=14)
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9,
            color="#5F5E5A", va="bottom")
    return {"shortest_units": sp_len, "greedy_end": gend,
            "greedy_progress": progress, "reaches_goal": bool(reaches)}


def figure_mazes(outdir, res=0.5):
    base = core_config().env
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.2))
    info = {}
    for ax, m in zip(axes, ("A", "B")):
        cfg = mazes.apply_maze(base, m)
        g = maze_analysis.greedy_descent_optimum(cfg)
        if g["reaches_goal"]:
            title = f"Maze B — not deceptive"
            sub = "greedy distance-descent reaches the goal"
        else:
            title = f"Maze A — deceptive"
            sub = (f"greedy distance-descent dead-ends at "
                   f"({g['x']:.0f}, {g['y']:.0f}), score {g['true_objective']:.3f}")
        info[m] = draw_maze(ax, cfg, title, sub, mark_trap=not g["reaches_goal"])

    handles = [
        plt.Line2D([], [], color=TRUE_C, lw=2.2, label="true shortest route"),
        plt.Line2D([], [], color=GREEDY_C, lw=2.2, ls=(0, (5, 3)),
                   label="greedy distance-following route"),
        mpatches.Patch(facecolor=WALL, edgecolor=WALL_EDGE, label="obstacle"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"fig_mazes.{ext}"), dpi=220,
                    bbox_inches="tight")
    plt.close(fig)
    return info


# --------------------------------------------------------------------------- #
class _DriftNet:
    """A minimal genome with drifted weights. No hidden nodes -- matching the
    initial NEAT topology, which is what the population spends most of its time
    at (median final nodes 11.04-12.02 out of a floor of 11)."""

    def __init__(self, rng, scale):
        self.w = rng.normal(scale=scale, size=(8, 3))

    def reset(self):
        pass

    def activate(self, obs):
        return list(np.asarray(obs) @ self.w)


def measure_drift(cfg, n, seed=0):
    """Fraction of undirected policies that reach the goal.

    Scales span what NEAT explores: weights are initialised at sigma=1 and
    perturbed with std 0.5 per gene per reproduction over 100 generations, so
    the population wanders well beyond its initial distribution.
    """
    rng = np.random.default_rng(seed)
    env = NavEnv(cfg)
    scales = [0.5, 1.0, 2.0, 4.0]
    solved = 0
    for i in range(n):
        if env.rollout(_DriftNet(rng, scales[i % len(scales)])).reached_goal:
            solved += 1
    return solved / n, solved


def figure_drift(outdir, samples, pop=100, gens=100, seed=0):
    base = core_config().env
    rates, counts = {}, {}
    for m in ("A", "B"):
        cfg = mazes.apply_maze(base, m)
        r, c = measure_drift(cfg, samples, seed)
        rates[m], counts[m] = r, c
        print(f"  maze {m}: {c}/{samples} = {r:.4%}")

    evaluated = pop * gens
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9))
    labels = ["Maze A\n(deceptive)", "Maze B\n(not deceptive)"]
    colours = ["#C44E52", "#4C72B0"]

    ax = axes[0]
    vals = [rates["A"] * 100, rates["B"] * 100]
    bars = ax.bar(labels, vals, color=colours, alpha=0.75, width=0.55,
                  edgecolor=colours, linewidth=1.2)
    for b, v, c in zip(bars, vals, [counts["A"], counts["B"]]):
        ax.text(b.get_x() + b.get_width() / 2,
                v + max(vals) * 0.04 + 0.004,
                f"{v:.3f}%\n({c}/{samples:,})", ha="center", fontsize=9)
    ax.set_ylabel("undirected policies reaching the goal (%)")
    ax.set_ylim(0, max(max(vals) * 1.45, 0.02))
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    exp = [rates["A"] * evaluated, rates["B"] * evaluated]
    bars = ax.bar(labels, exp, color=colours, alpha=0.75, width=0.55,
                  edgecolor=colours, linewidth=1.2)
    for b, v in zip(bars, exp):
        ax.text(b.get_x() + b.get_width() / 2, v + max(exp) * 0.04 + 0.1,
                f"{v:.0f}", ha="center", fontsize=10)
    ax.axhline(1.0, color="#7f7f7f", ls="--", lw=1.2)
    ax.text(0.02, 1.15, "one solver is enough — elitism keeps it",
            transform=ax.get_yaxis_transform(), fontsize=8.5, color="#5F5E5A")
    ax.set_ylabel(f"expected solvers per run\n({gens} gen x {pop} pop "
                  f"= {evaluated:,} genomes)")
    ax.set_ylim(0, max(max(exp) * 1.4, 2))
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("Drift-reachability: can undirected search find the goal?",
                 fontsize=11.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"fig_drift_reachability.{ext}"),
                    dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {"rates": rates, "counts": counts, "samples": samples,
            "evaluated_per_run": evaluated,
            "expected_solvers": {k: rates[k] * evaluated for k in rates}}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="results/figures")
    ap.add_argument("--samples", type=int, default=4000,
                    help="policies sampled per maze for drift-reachability")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-drift", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rec = {}

    print("fig_mazes: drawing walls and routes from coordinates")
    rec["mazes"] = figure_mazes(args.outdir)
    for m, v in rec["mazes"].items():
        print(f"  maze {m}: shortest {v['shortest_units']:.0f} units, "
              f"greedy ends {v['greedy_end']} "
              f"(progress {v['greedy_progress']:.3f}, "
              f"reaches goal: {v['reaches_goal']})")

    if not args.skip_drift:
        print(f"\nfig_drift_reachability: sampling {args.samples:,} policies per maze")
        rec["drift"] = figure_drift(args.outdir, args.samples, seed=args.seed)

    with open(os.path.join(args.outdir, "maze_figures.json"), "w") as fh:
        json.dump(rec, fh, indent=2, default=str)
    print(f"\nwrote figures + maze_figures.json to {args.outdir}/")


if __name__ == "__main__":
    main()
