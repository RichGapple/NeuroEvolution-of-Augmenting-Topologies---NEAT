"""Analytic properties of a maze, computed by grid search rather than assumed.

Why this module exists
----------------------
Calibration gate D1 originally required the dense endpoint to solve the maze in
30-95% of runs, on the reasoning that a learner which never solves must be
broken. On a *deceptive* maze that reasoning is wrong: a perfectly functioning
learner will converge to the optimum of the signal it was given, and if the
signal is deceptive that optimum is not the goal.

This module makes deceptiveness measurable instead of arguable:

    greedy_descent_optimum()  where a policy that always reduces distance-to-goal
                              ends up, and what progress that scores
    is_deceptive()            True when that endpoint is not the goal
    shortest_path()           BFS path length through free space
    step_feasibility()        can the true path even fit in max_steps?

D1 then becomes "reach the greedy optimum", which on a non-deceptive maze is
exactly the original "solve the maze" and on a deceptive one is the honest
version of the same question.

Everything is grid-based at `res` resolution with the agent radius inflated into
the obstacles, so results are slightly conservative (paths are 4-connected and
therefore over-estimated). That is the safe direction for a feasibility check.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional, Sequence, Tuple

import numpy as np


def _free_grid(cfg, res: float) -> Tuple[np.ndarray, int, int]:
    r = cfg.agent_radius
    W = int(cfg.width / res) + 1
    H = int(cfg.height / res) + 1
    free = np.ones((W, H), dtype=bool)
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


def shortest_path(cfg, res: float = 0.5) -> Optional[float]:
    """BFS path length from start to goal through free space, in world units.

    None means the goal is unreachable -- which would make the whole experiment
    void, so this is worth checking before any sweep.
    """
    free, W, H = _free_grid(cfg, res)
    si, sj = int(cfg.start[0] / res), int(cfg.start[1] / res)
    gi, gj = int(cfg.goal[0] / res), int(cfg.goal[1] / res)
    if not free[si, sj]:
        return None
    dist = -np.ones((W, H), dtype=int)
    dist[si, sj] = 0
    q = deque([(si, sj)])
    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < W and 0 <= b < H and free[a, b] and dist[a, b] < 0:
                dist[a, b] = dist[i, j] + 1
                q.append((a, b))
    # Any cell within the goal radius counts as arrival.
    gx, gy = cfg.goal
    best = None
    for i in range(W):
        for j in range(H):
            if dist[i, j] < 0:
                continue
            if math.hypot(gx - i * res, gy - j * res) <= cfg.goal_radius:
                d = dist[i, j] * res
                best = d if best is None else min(best, d)
    return best


def greedy_descent_optimum(cfg, res: float = 0.5) -> dict:
    """Where a strictly distance-reducing policy ends up.

    This is the ceiling for any learner following the reward gradient alone,
    because the intermediate credit term is a monotone function of
    distance-to-goal. If it is not the goal, the maze is deceptive and no
    amount of exploration-schedule tuning changes that.
    """
    free, W, H = _free_grid(cfg, res)
    gx, gy = cfg.goal
    sx, sy = cfg.start
    d0 = math.hypot(gx - sx, gy - sy)

    ii, jj = np.meshgrid(np.arange(W) * res, np.arange(H) * res, indexing="ij")
    euc = np.hypot(gx - ii, gy - jj)

    i, j = int(sx / res), int(sy / res)
    seen = set()
    while True:
        seen.add((i, j))
        cands = []
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)):
            a, b = i + di, j + dj
            if 0 <= a < W and 0 <= b < H and free[a, b] and (a, b) not in seen:
                cands.append((euc[a, b], a, b))
        if not cands:
            break
        e, a, b = min(cands)
        if e >= euc[i, j]:
            break
        i, j = a, b

    d_end = float(euc[i, j])
    reached = d_end <= cfg.goal_radius
    progress = min(1.0, max(0.0, 1.0 - d_end / d0))
    return {
        "x": i * res, "y": j * res,
        "distance_to_goal": d_end,
        "progress": progress,
        "reaches_goal": bool(reached),
        # The true-objective score a greedy-gradient policy can attain.
        "true_objective": (cfg_progress_score(cfg, progress, reached)),
    }


def cfg_progress_score(cfg, progress: float, reached: bool,
                       success_bonus: float = 1.0,
                       progress_weight: float = 1.0) -> float:
    return progress_weight * progress + (success_bonus if reached else 0.0)


def is_deceptive(cfg, res: float = 0.5) -> bool:
    return not greedy_descent_optimum(cfg, res)["reaches_goal"]


def step_feasibility(cfg, res: float = 0.5, n_turns: int = 4) -> dict:
    """Can the true path fit inside max_steps, allowing for turning?

    Turning consumes a step without moving, so a path that fits on distance
    alone can still be infeasible. `n_turns` is the number of ~90-degree turns
    the route requires; 4 is right for a north-east-south-east-north route.
    """
    path = shortest_path(cfg, res)
    if path is None:
        return {"reachable": False}
    fwd = path / cfg.speed
    per_turn = (math.pi / 2) / cfg.turn_rate
    turn_steps = n_turns * per_turn
    total = fwd + turn_steps
    return {
        "reachable": True,
        "path_units": path,
        "forward_steps": fwd,
        "turn_steps": turn_steps,
        "total_steps": total,
        "max_steps": cfg.max_steps,
        "slack_steps": cfg.max_steps - total,
        "feasible": total <= cfg.max_steps,
        "margin_frac": (cfg.max_steps - total) / cfg.max_steps,
    }


def describe(cfg, res: float = 0.5) -> str:
    """One block of text summarising the maze. Put this in the paper."""
    g = greedy_descent_optimum(cfg, res)
    f = step_feasibility(cfg, res)
    L = []
    L.append(f"start {cfg.start} -> goal {cfg.goal} (radius {cfg.goal_radius:g}), "
             f"{len(cfg.obstacles)} obstacles, {cfg.max_steps} steps")
    if not f["reachable"]:
        L.append("  GOAL UNREACHABLE -- the experiment is void.")
        return "\n".join(L)
    L.append(f"  shortest path        {f['path_units']:.0f} units "
             f"= {f['forward_steps']:.0f} forward steps")
    L.append(f"  + ~{f['turn_steps']:.0f} turn steps -> {f['total_steps']:.0f} "
             f"of {f['max_steps']} ({f['margin_frac']*100:.0f}% slack)  "
             f"{'FEASIBLE' if f['feasible'] else 'INFEASIBLE'}")
    L.append(f"  greedy-descent ends  ({g['x']:.0f}, {g['y']:.0f}), "
             f"progress {g['progress']:.3f}, true objective {g['true_objective']:.3f}")
    L.append(f"  DECEPTIVE: {'yes -- gradient-following cannot reach the goal' if not g['reaches_goal'] else 'no'}")
    return "\n".join(L)
