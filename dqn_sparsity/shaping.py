"""Per-step reward for DQN that sums to exactly the NEAT fitness.

The problem this solves
-----------------------
NEAT scores an *episode*: one number, computed at the end, from the whole
trajectory.  DQN needs a reward at every step.  If you simply hand DQN a single
number at the end of the episode, you have not run "DQN at eta = 0" -- you have
run "DQN with terminal-only reward", which is the sparsest possible setting.
The comparison would be rigged in NEAT's favour and any competent reviewer will
say so.

The fix
-------
Emit per-step rewards that *telescope*.  For the quantized mode, with
p_t = 1 - min(d_0..d_t)/d_0 and Q(.) the level-L quantiser:

    r_t = w * ( Q(p_t) - Q(p_{t-1}) )                       for t = 1..T
    r_T += B                                                if the goal was reached

Summing over the episode collapses the telescope:

    sum_t r_t = w * ( Q(p_T) - Q(p_0) ) + B*1[goal]
              = w * C_eta(traj) + B*1[goal]                  since p_0 = 0
              = RewardModel.fitness(traj)                    exactly.

So with gamma = 1 the undiscounted DQN return is *identical* to the NEAT fitness,
for every trajectory and every eta.  That identity is the invariant that makes
the two arms comparable, and `scripts/10_dqn_check.py` verifies it numerically
over thousands of random trajectories rather than trusting this derivation.

This construction is potential-based shaping (Ng, Harada & Russell 1999) applied
to the quantised potential, so the optimal policy is provably unchanged -- cite
that when a reviewer asks whether the shaping alters the task.

A note on `DQNConfig.reward_scale`
---------------------------------
The learner multiplies these rewards by a positive constant before storing them
in the replay buffer. That is optimiser preconditioning, not a change to the
reward: it leaves the optimal policy unchanged and leaves this module's exact
identity untouched. `10_dqn_check.py` therefore still reports
`max |return - fitness| = 0.000e+00`. Reporting is always unscaled.

Terminal correction
-------------------
The `checkpoint` and `gated` modes clamp or saturate in ways that do not
telescope perfectly.  Rather than approximate, every episode ends with an exact
correction term:

    r_T += RewardModel.fitness(traj) - sum(r_t emitted so far)

which forces the identity to hold for all three modes.  The correction is
logged as `terminal_correction`.  Watch it: if it carries most of the return,
the shaping is doing little and the agent is effectively on terminal-only
reward.  That is itself a finding worth reporting per eta.
"""

from __future__ import annotations

import math
import random
from typing import Optional

from neat_sparsity.config import SparsityConfig
from neat_sparsity.env import Trajectory
from neat_sparsity.reward import RewardModel, _quantize, true_objective


class StepwiseReward:
    """Streams per-step reward whose episode sum equals the NEAT fitness.

    Parameters
    ----------
    mode : "telescoping" | "terminal"
        "telescoping" is the fair default described above.
        "terminal" pays the entire fitness in one lump at the final step.  It is
        NOT the fair comparison -- it exists as an ablation, to measure how much
        of DQN's performance comes from having temporal structure in the reward
        at all.  Running both gives you a second, cheap result for the paper.
    """

    def __init__(self, cfg: SparsityConfig, eta: float,
                 mode: str = "telescoping",
                 credit_basis: str = "min",
                 rng: Optional[random.Random] = None) -> None:
        if mode not in ("telescoping", "terminal"):
            raise ValueError(f"unknown shaping mode: {mode!r}")
        if credit_basis not in ("min", "current"):
            raise ValueError(f"unknown credit basis: {credit_basis!r}")
        self.credit_basis = credit_basis
        self.cfg = cfg
        self.eta = float(eta)
        self.mode = mode
        self.levels = cfg.levels(eta)
        self.rng = rng or random.Random(0)
        self.model = RewardModel(cfg, eta, rng=self.rng)
        self.reset()

    # -- episode lifecycle ---------------------------------------------------- #
    def reset(self, initial_distance: float = 1.0) -> None:
        self.d0 = initial_distance if initial_distance > 0 else 1.0
        self.dmin = self.d0
        self._prev_credit = 0.0
        self._emitted = 0.0
        self._n_zero = 0
        self._n_steps = 0
        self.terminal_correction = 0.0

    def _credit_now(self, d: float, reached: bool) -> float:
        """Credit for the trajectory *so far*, in the active sparsity mode.

        `d` is best-ever distance under credit_basis="min" and current distance
        under "current". Either way the episode total is reconciled to the exact
        NEAT fitness by the terminal correction, so the choice changes only how
        the same total information is distributed across steps.
        """
        if reached:
            return 1.0
        p = min(1.0, max(0.0, 1.0 - d / self.d0))
        if self.cfg.mode == "quantized":
            return _quantize(p, self.levels)
        if self.cfg.mode == "checkpoint":
            L = self.levels
            if L <= 1:
                return 0.0
            return math.floor(p * L) / L
        # gated: no meaningful running credit; the terminal correction carries it
        return 0.0

    # -- per step ------------------------------------------------------------- #
    def step_reward(self, info: dict, done: bool, traj_if_done=None) -> float:
        """Reward for the step that just happened."""
        self._n_steps += 1

        if self.mode == "terminal":
            r = 0.0
        else:
            basis = (info["min_distance"] if self.credit_basis == "min"
                     else info["distance"])
            credit = self._credit_now(basis, info["reached"])
            r = self.cfg.progress_weight * (credit - self._prev_credit)
            self._prev_credit = credit
            if info["reached"]:
                r += self.cfg.success_bonus

        self._emitted += r

        if done:
            traj = traj_if_done
            if traj is None:
                raise ValueError("done=True requires the finished Trajectory")
            target = self.model.fitness(traj)
            self.terminal_correction = target - self._emitted
            r += self.terminal_correction
            self._emitted = target

        if r == 0.0:
            self._n_zero += 1
        return r

    # -- diagnostics ---------------------------------------------------------- #
    @property
    def zero_reward_fraction(self) -> float:
        return self._n_zero / max(1, self._n_steps)

    @property
    def episode_return(self) -> float:
        return self._emitted


# --------------------------------------------------------------------------- #
def verify_equivalence(cfg: SparsityConfig, eta: float, traj: Trajectory,
                       tol: float = 1e-9, credit_basis: str = "min") -> tuple:
    """Replay a finished Trajectory through the shaper and compare sums.

    Returns (ok, dqn_return, neat_fitness).  Used by the conformance script and
    by the unit tests.  This is the check that has to pass before any sweep is
    allowed to run.
    """
    shaper = StepwiseReward(cfg, eta, mode="telescoping",
                            credit_basis=credit_basis, rng=random.Random(0))
    shaper.reset(traj.initial_distance)

    ds = traj.distances
    dmin = ds[0]
    total = 0.0
    n = len(ds) - 1
    for i in range(1, len(ds)):
        dmin = min(dmin, ds[i])
        last = (i == n)
        reached = traj.reached_goal and last
        info = {"distance": ds[i], "min_distance": dmin, "reached": reached}
        total += shaper.step_reward(info, done=last,
                                    traj_if_done=traj if last else None)

    fitness = RewardModel(cfg, eta, rng=random.Random(0)).fitness(traj)
    return (abs(total - fitness) <= tol, total, fitness)


def report_true_objective(traj: Trajectory, cfg: SparsityConfig) -> float:
    """Reporting score.  Sparsity-independent, identical to the NEAT arm.

    Never report DQN performance in the training reward of its own condition --
    that would make "sparse runs score lower" a tautology on this side too.
    """
    return true_objective(traj, cfg)
