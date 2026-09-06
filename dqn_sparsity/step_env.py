"""A reset/step interface over the *same* NavEnv physics used by the NEAT arm.

Why a wrapper and not a reimplementation
----------------------------------------
NEAT evaluates a whole episode at once (`NavEnv.rollout(net)`); DQN needs to act,
observe and learn one step at a time.  The temptation is to write a second maze.
Do not.  A second maze would silently become a second experiment, and any
NEAT-vs-DQN difference could then be blamed on the environment rather than on
reward sparsity.

So this class reuses `NavEnv`'s geometry (`_observe`, `_blocked`) and its
`EnvConfig` verbatim, and only restructures the *loop*.  `scripts/10_dqn_check.py`
then asserts conformance numerically: for any deterministic policy, stepping
through this wrapper must reconstruct a `Trajectory` byte-identical to the one
`NavEnv.rollout` returns.  That test, not this docstring, is the guarantee.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from neat_sparsity.config import EnvConfig
from neat_sparsity.env import NavEnv, Trajectory


class StepNavEnv:
    """Gym-style reset/step over NavEnv physics.  Deterministic, no seeding."""

    n_actions = 3          # 0 = turn left, 1 = turn right, 2 = forward

    def __init__(self, cfg: EnvConfig) -> None:
        self.cfg = cfg
        self._core = NavEnv(cfg)          # geometry + observation model
        self.obs_dim = len(self._core._observe(cfg.start[0], cfg.start[1],
                                               cfg.start_heading))
        self.reset()

    # -- state ---------------------------------------------------------------- #
    def reset(self) -> np.ndarray:
        cfg = self.cfg
        self.x, self.y = cfg.start
        self.heading = cfg.start_heading
        gx, gy = cfg.goal
        self.d0 = math.hypot(gx - self.x, gy - self.y)
        self.dmin = self.d0
        self.path = 0.0
        self.collisions = 0
        self.distances: List[float] = [self.d0]
        self.reached = False
        self.steps = 0
        self.done = False
        return self.observe()

    def observe(self) -> np.ndarray:
        return np.asarray(self._core._observe(self.x, self.y, self.heading),
                          dtype=np.float32)

    # -- dynamics ------------------------------------------------------------- #
    def step(self, action: int) -> Tuple[np.ndarray, bool, dict]:
        """Advance one step.  Returns (obs, done, info).

        Deliberately returns NO reward.  Reward is not the environment's job in
        this study -- `shaping.py` owns it, exactly as `reward.py` owns it on the
        NEAT side.  Keeping the split means eta can never leak into the dynamics.
        """
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset()")

        cfg = self.cfg
        self.steps += 1

        if action == 0:
            self.heading -= cfg.turn_rate
        elif action == 1:
            self.heading += cfg.turn_rate
        else:
            nx = self.x + math.cos(self.heading) * cfg.speed
            ny = self.y + math.sin(self.heading) * cfg.speed
            if self._core._blocked(nx, ny):
                self.collisions += 1
                if cfg.collision_ends_episode:
                    self.done = True
            else:
                self.path += math.hypot(nx - self.x, ny - self.y)
                self.x, self.y = nx, ny

        gx, gy = cfg.goal
        d = math.hypot(gx - self.x, gy - self.y)
        self.distances.append(d)
        if d < self.dmin:
            self.dmin = d
        if d <= cfg.goal_radius:
            self.reached = True
            self.done = True
        if self.steps >= cfg.max_steps:
            self.done = True

        info = {"distance": d, "min_distance": self.dmin,
                "reached": self.reached, "timeout": self.steps >= cfg.max_steps}
        return self.observe(), self.done, info

    # -- reporting ------------------------------------------------------------ #
    def trajectory(self) -> Trajectory:
        """Reconstruct the exact object the NEAT arm's reward function consumes."""
        gx, gy = self.cfg.goal
        return Trajectory(
            reached_goal=self.reached,
            steps_taken=self.steps,
            initial_distance=self.d0,
            min_distance=self.dmin,
            final_distance=math.hypot(gx - self.x, gy - self.y),
            path_length=self.path,
            collisions=self.collisions,
            distances=list(self.distances),
            final_x=self.x,
            final_y=self.y,
        )


# --------------------------------------------------------------------------- #
class _PolicyNet:
    """Adapter so a step-wise policy can be fed to NavEnv.rollout for the
    conformance test.  Replays a fixed action sequence, or wraps a callable."""

    def __init__(self, fn) -> None:
        self.fn = fn
        self.t = 0

    def reset(self) -> None:
        self.t = 0

    def activate(self, obs):
        a = self.fn(np.asarray(obs, dtype=np.float32), self.t)
        self.t += 1
        out = [0.0, 0.0, 0.0]
        out[int(a)] = 1.0
        return out


def rollout_stepwise(cfg: EnvConfig, policy_fn) -> Trajectory:
    """Run an episode through StepNavEnv with `policy_fn(obs, t) -> action`."""
    env = StepNavEnv(cfg)
    obs = env.reset()
    t = 0
    while not env.done:
        a = int(policy_fn(obs, t))
        obs, _, _ = env.step(a)
        t += 1
    return env.trajectory()


def rollout_monolithic(cfg: EnvConfig, policy_fn) -> Trajectory:
    """Run the same policy through the untouched NavEnv.rollout."""
    return NavEnv(cfg).rollout(_PolicyNet(policy_fn))


def trajectories_equal(a: Trajectory, b: Trajectory, tol: float = 0.0) -> bool:
    if a.reached_goal != b.reached_goal or a.steps_taken != b.steps_taken:
        return False
    if a.collisions != b.collisions:
        return False
    for f in ("initial_distance", "min_distance", "final_distance",
              "path_length", "final_x", "final_y"):
        if abs(getattr(a, f) - getattr(b, f)) > tol:
            return False
    if len(a.distances) != len(b.distances):
        return False
    return all(abs(u - v) <= tol for u, v in zip(a.distances, b.distances))
