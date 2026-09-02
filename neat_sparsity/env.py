"""Deterministic 2D navigation environment.

Fixed start, fixed goal, fixed obstacles, no stochastic dynamics.  This is a
deliberate experimental choice (roadmap Section 8): because the environment
contributes zero variance, all between-condition variance can be attributed to
the reward-sparsity manipulation plus the evolutionary algorithm's own
stochasticity.

The environment does *not* know about rewards.  It reports a raw trajectory
summary; `reward.py` turns that into a fitness.  This separation is what
guarantees that changing eta cannot change the task.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .config import EnvConfig


@dataclass
class Trajectory:
    """Everything the reward function is allowed to see."""
    reached_goal: bool
    steps_taken: int
    initial_distance: float
    min_distance: float          # closest approach to the goal centre
    final_distance: float
    path_length: float
    collisions: int
    # per-step distance-to-goal, used by the `gated` sparsity mode
    distances: List[float]
    # behaviour descriptor: final position (used by the novelty-search extension)
    final_x: float = 0.0
    final_y: float = 0.0


class NavEnv:
    def __init__(self, cfg: EnvConfig) -> None:
        self.cfg = cfg
        self._ray_offsets = self._compute_ray_offsets()
        self._bounds = (0.0, 0.0, cfg.width, cfg.height)

    def _compute_ray_offsets(self) -> List[float]:
        n = self.cfg.n_rangefinders
        if n == 1:
            return [0.0]
        half = self.cfg.rangefinder_fov / 2.0
        return [-half + i * (self.cfg.rangefinder_fov / (n - 1)) for i in range(n)]

    # -- geometry ------------------------------------------------------------- #
    def _blocked(self, x: float, y: float) -> bool:
        r = self.cfg.agent_radius
        if x - r < 0 or y - r < 0 or x + r > self.cfg.width or y + r > self.cfg.height:
            return True
        for (x0, y0, x1, y1) in self.cfg.obstacles:
            if (x + r > x0) and (x - r < x1) and (y + r > y0) and (y - r < y1):
                return True
        return False

    def _ray_distance(self, x: float, y: float, angle: float) -> float:
        """Distance to the first obstacle/wall along `angle`, capped at max."""
        dx, dy = math.cos(angle), math.sin(angle)
        best = self.cfg.rangefinder_max

        boxes = list(self.cfg.obstacles)
        for (x0, y0, x1, y1) in boxes:
            t = _ray_box(x, y, dx, dy, x0, y0, x1, y1)
            if t is not None and t < best:
                best = t
        # outer walls (ray leaving the arena)
        t = _ray_walls(x, y, dx, dy, 0.0, 0.0, self.cfg.width, self.cfg.height)
        if t is not None and t < best:
            best = t
        return best

    def _observe(self, x: float, y: float, heading: float) -> List[float]:
        cfg = self.cfg
        obs = [self._ray_distance(x, y, heading + off) / cfg.rangefinder_max
               for off in self._ray_offsets]
        gx, gy = cfg.goal
        dx, dy = gx - x, gy - y
        dist = math.hypot(dx, dy)
        diag = math.hypot(cfg.width, cfg.height)
        bearing = math.atan2(dy, dx) - heading
        obs.append(min(1.0, dist / diag))
        obs.append(math.sin(bearing))
        obs.append(math.cos(bearing))
        return obs

    # -- rollout --------------------------------------------------------------- #
    def rollout(self, net) -> Trajectory:
        cfg = self.cfg
        x, y = cfg.start
        heading = cfg.start_heading
        gx, gy = cfg.goal
        d0 = math.hypot(gx - x, gy - y)
        dmin = d0
        path = 0.0
        collisions = 0
        distances: List[float] = [d0]
        reached = False
        steps = 0

        if hasattr(net, "reset"):
            net.reset()

        for steps in range(1, cfg.max_steps + 1):
            out = net.activate(self._observe(x, y, heading))
            # discrete action = argmax (deterministic tie-break on lowest index)
            a, best = 0, out[0]
            for i in range(1, len(out)):
                if out[i] > best:
                    best, a = out[i], i

            if a == 0:
                heading -= cfg.turn_rate
            elif a == 1:
                heading += cfg.turn_rate
            else:
                nx = x + math.cos(heading) * cfg.speed
                ny = y + math.sin(heading) * cfg.speed
                if self._blocked(nx, ny):
                    collisions += 1
                    if cfg.collision_ends_episode:
                        break
                else:
                    path += math.hypot(nx - x, ny - y)
                    x, y = nx, ny

            d = math.hypot(gx - x, gy - y)
            distances.append(d)
            if d < dmin:
                dmin = d
            if d <= cfg.goal_radius:
                reached = True
                break

        return Trajectory(
            reached_goal=reached,
            steps_taken=steps,
            initial_distance=d0,
            min_distance=dmin,
            final_distance=math.hypot(gx - x, gy - y),
            path_length=path,
            collisions=collisions,
            distances=distances,
            final_x=x,
            final_y=y,
        )


# --------------------------------------------------------------------------- #
def _ray_box(px, py, dx, dy, x0, y0, x1, y1):
    """Slab method: entry distance to an axis-aligned box, or None."""
    tmin, tmax = 0.0, float("inf")
    for p, d, lo, hi in ((px, dx, x0, x1), (py, dy, y0, y1)):
        if abs(d) < 1e-12:
            if p < lo or p > hi:
                return None
        else:
            t1 = (lo - p) / d
            t2 = (hi - p) / d
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return None
    return tmin if tmin >= 0.0 else None


def _ray_walls(px, py, dx, dy, x0, y0, x1, y1):
    """Distance until the ray exits the arena rectangle."""
    tmax = float("inf")
    for p, d, lo, hi in ((px, dx, x0, x1), (py, dy, y0, y1)):
        if abs(d) < 1e-12:
            continue
        t1 = (lo - p) / d
        t2 = (hi - p) / d
        t = max(t1, t2)
        if t >= 0.0:
            tmax = min(tmax, t)
    return None if tmax == float("inf") else tmax


def render_ascii(cfg: EnvConfig, cols: int = 60, rows: int = 24) -> str:
    """Sanity-check drawing of the arena (used by scripts/inspect_env.py)."""
    grid = [[" "] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            x = (c + 0.5) / cols * cfg.width
            y = (1 - (r + 0.5) / rows) * cfg.height
            for (x0, y0, x1, y1) in cfg.obstacles:
                if x0 <= x <= x1 and y0 <= y <= y1:
                    grid[r][c] = "#"
    def put(px, py, ch):
        c = int(px / cfg.width * cols)
        r = int((1 - py / cfg.height) * rows)
        c = min(max(c, 0), cols - 1)
        r = min(max(r, 0), rows - 1)
        grid[r][c] = ch
    put(*cfg.start, "S")
    put(*cfg.goal, "G")
    border = "+" + "-" * cols + "+"
    return "\n".join([border] + ["|" + "".join(row) + "|" for row in grid] + [border])
