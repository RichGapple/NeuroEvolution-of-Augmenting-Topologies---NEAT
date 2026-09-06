#!/usr/bin/env python3
"""Validate the DQN implementation against a published benchmark.

Why this script exists
----------------------
A reviewer looking at `dqn_sparsity/nets.py` sees a hand-written numpy MLP with
hand-written backprop, and reasonably asks: is this DQN actually competitive, or
does the maze result just show that *this particular* implementation is weak?
The gradient check and the calibration gates argue the code is correct, but
neither shows it is *competitive*.

This script closes that objection by running the **same `DQNAgent` class**,
unmodified, on CartPole-v1 -- a task with a published, universally agreed
success criterion. Only hyperparameters change; not one line of the learner.

CartPole-v1
-----------
Dynamics are the standard cart-pole from Barto, Sutton & Anderson (1983), as
used by OpenAI Gym / Gymnasium. Implemented inline so this adds no dependency
and so the exact constants are auditable in one place.

    state       (x, x_dot, theta, theta_dot)
    actions     push left / push right (force +/- 10 N)
    reward      +1 per surviving step
    terminate   |x| > 2.4, or |theta| > 12 degrees
    truncate    500 steps
    SOLVED      mean return >= 475 over 100 consecutive episodes

The 475/500 threshold is the canonical bar. If this agent clears it, "the
baseline might just be bad" is no longer available as a criticism.

Two terminal cases are distinguished, which matters and is easy to get wrong:
falling over is a *true* terminal state (bootstrap 0), while hitting the 500
step limit is a *truncation* (bootstrap normally). Conflating them teaches the
agent that surviving to the time limit is as bad as falling over.

Usage
-----
    python scripts/20_dqn_benchmark.py
    python scripts/20_dqn_benchmark.py --seeds 5 --budget 150000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dqn_sparsity.agent import DQNAgent
from dqn_sparsity.config import DQNConfig


# --------------------------------------------------------------------------- #
class CartPole:
    """CartPole-v1. Constants match Gymnasium exactly."""

    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masspole + masscart          # 1.1
    length = 0.5                              # actually half the pole's length
    polemass_length = masspole * length       # 0.05
    force_mag = 10.0
    tau = 0.02                                # seconds between state updates

    theta_threshold = 12 * 2 * math.pi / 360  # 0.2095 rad
    x_threshold = 2.4
    max_steps = 500

    obs_dim = 4
    n_actions = 2

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> np.ndarray:
        self.state = self.rng.uniform(-0.05, 0.05, size=4)
        self.steps = 0
        return self.state.astype(np.float32)

    def step(self, action: int):
        """Returns (obs, reward, terminated, truncated).

        `terminated` means the pole fell or the cart left the track -- a real
        absorbing state. `truncated` means the 500-step limit was reached, which
        is not a failure and must not be bootstrapped as one.
        """
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta, sintheta = math.cos(theta), math.sin(theta)

        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = ((self.gravity * sintheta - costheta * temp)
                    / (self.length * (4.0 / 3.0
                                      - self.masspole * costheta ** 2 / self.total_mass)))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Euler integration, as in the reference implementation
        x += self.tau * x_dot
        x_dot += self.tau * xacc
        theta += self.tau * theta_dot
        theta_dot += self.tau * thetaacc
        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1

        terminated = bool(abs(x) > self.x_threshold
                          or abs(theta) > self.theta_threshold)
        truncated = bool(self.steps >= self.max_steps)
        return self.state.astype(np.float32), 1.0, terminated, truncated


# --------------------------------------------------------------------------- #
def benchmark_config(budget: int, hidden=(64, 64), lr=1e-3,
                     target_update=500, buffer=50_000,
                     eps_frac=0.15, eps_end=0.05,
                     train_every=1, huber=1.0) -> DQNConfig:
    """Standard CartPole DQN hyperparameters.

    Deliberately conventional -- taken from common reference implementations
    rather than tuned here. The point is to show the learner works under
    ordinary settings, not to show it can be coaxed into working.

    Note `reward_scale = 1.0`: CartPole already pays +1 per step, so the
    preconditioning needed on the maze (where a step earns ~0.012) does not
    apply. That the same class works at both scales is itself informative.
    """
    c = DQNConfig()
    c.hidden = list(hidden)
    c.activation = "relu"
    c.lr = lr
    c.gamma = 0.99
    c.double_dqn = True
    c.huber_delta = huber
    c.grad_clip = 10.0
    c.buffer_size = buffer
    c.batch_size = 64
    c.warmup_steps = 1_000
    c.train_every = train_every
    c.target_update_every = target_update
    c.eps_start = 1.0
    c.eps_end = eps_end
    c.eps_decay_frac = eps_frac
    c.budget_steps = budget
    c.reward_scale = 1.0
    return c


def run_one(seed: int, budget: int, verbose: bool = False, hp: dict = None) -> dict:
    cfg = benchmark_config(budget, **(hp or {}))
    env = CartPole(seed=seed)
    agent = DQNAgent(env.obs_dim, env.n_actions, cfg, seed=seed)

    obs = env.reset()
    ep_return = 0.0
    returns: list = []
    window = deque(maxlen=100)
    solved_at = None
    best_window = 0.0
    t0 = time.time()

    for step in range(1, budget + 1):
        a = agent.act(obs)
        nobs, r, terminated, truncated = env.step(a)
        # Only a genuine fall is terminal for bootstrapping; a 500-step
        # truncation is not a failure and must bootstrap normally.
        agent.observe(obs, a, r * cfg.reward_scale, nobs, terminated)
        ep_return += r
        obs = nobs

        if terminated or truncated:
            returns.append(ep_return)
            window.append(ep_return)
            if len(window) == 100:
                m = float(np.mean(window))
                best_window = max(best_window, m)
                if m >= 475.0 and solved_at is None:
                    solved_at = step
                    if verbose:
                        print(f"    seed {seed}: SOLVED at {step:,} steps "
                              f"(100-ep mean {m:.1f})")
            ep_return = 0.0
            obs = env.reset()
            if verbose and len(returns) % 100 == 0:
                print(f"    seed {seed}: ep {len(returns):>4d}  "
                      f"step {step:>7,}  100-ep mean "
                      f"{np.mean(window):.1f}", flush=True)

    final100 = float(np.mean(list(window))) if window else 0.0
    return {
        "seed": seed,
        "episodes": len(returns),
        "final_100ep_mean": final100,
        "best_100ep_mean": best_window,
        "solved": bool(best_window >= 475.0),
        "solved_at_step": solved_at,
        "max_episode_return": float(max(returns)) if returns else 0.0,
        "wallclock_s": round(time.time() - t0, 1),
        "gradient_updates": agent.updates,
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--budget", type=int, default=150_000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--outdir", default="results/dqn_benchmark")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--target-update", type=int, default=500)
    ap.add_argument("--buffer", type=int, default=50_000)
    ap.add_argument("--eps-frac", type=float, default=0.15)
    ap.add_argument("--eps-end", type=float, default=0.05)
    ap.add_argument("--train-every", type=int, default=1)
    ap.add_argument("--huber", type=float, default=1.0,
                    help="Huber delta; 0 = plain MSE")
    ap.add_argument("--preset", choices=["default", "sb3zoo"], default=None,
                    help="sb3zoo = the RL Baselines3 Zoo tuned CartPole config")
    args = ap.parse_args()

    hp = dict(hidden=tuple(args.hidden), lr=args.lr,
              target_update=args.target_update, buffer=args.buffer,
              eps_frac=args.eps_frac, eps_end=args.eps_end,
              train_every=args.train_every, huber=args.huber)
    if args.preset == "sb3zoo":
        # Published tuned CartPole-v1 hyperparameters from RL Baselines3 Zoo.
        # Using someone else's tuning rather than our own is the point: it
        # removes any suspicion that the baseline was tuned to fail.
        hp = dict(hidden=(256, 256), lr=2.3e-3, target_update=100,
                  buffer=100_000, eps_frac=0.16, eps_end=0.04,
                  train_every=1, huber=1.0)

    os.makedirs(args.outdir, exist_ok=True)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    print("=" * 70)
    print("DQN BENCHMARK VALIDATION -- CartPole-v1")
    print("=" * 70)
    print("Same DQNAgent class as the maze experiments, unmodified.")
    print("Only hyperparameters differ (standard CartPole settings).")
    print(f"hyperparameters: {hp}")
    print(f"\n{args.seeds} seeds x {args.budget:,} env steps")
    print("Success criterion: mean return >= 475 over 100 consecutive episodes\n")

    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            rows = pool.starmap(run_one,
                                [(s, args.budget, False, hp) for s in seeds])
    else:
        rows = [run_one(s, args.budget, args.verbose, hp) for s in seeds]

    print(f"{'seed':>5} {'best 100-ep':>12} {'final 100-ep':>13} "
          f"{'solved':>7} {'at step':>10} {'time':>8}")
    for r in rows:
        at = f"{r['solved_at_step']:,}" if r["solved_at_step"] else "--"
        print(f"{r['seed']:>5} {r['best_100ep_mean']:>12.1f} "
              f"{r['final_100ep_mean']:>13.1f} "
              f"{'YES' if r['solved'] else 'no':>7} {at:>10} "
              f"{r['wallclock_s']:>7.0f}s")

    n_solved = sum(r["solved"] for r in rows)
    best = [r["best_100ep_mean"] for r in rows]
    steps = [r["solved_at_step"] for r in rows if r["solved_at_step"]]

    print("\n" + "=" * 70)
    print(f"solved {n_solved}/{len(rows)} seeds")
    print(f"best 100-episode mean: median {np.median(best):.1f}, "
          f"range [{min(best):.1f}, {max(best):.1f}]")
    if steps:
        print(f"steps to solve: median {np.median(steps):,.0f}, "
              f"range [{min(steps):,}, {max(steps):,}]")
    print("=" * 70)

    passed = n_solved >= max(1, int(0.6 * len(rows)))
    if passed:
        print("\nPASS -- the implementation reaches the published CartPole-v1")
        print("threshold under standard hyperparameters.")
        print("\nSentence for the paper:")
        print(f'  "To confirm the baseline is competitive rather than merely')
        print(f'   correct, the same DQN implementation solves CartPole-v1')
        print(f'   ({n_solved}/{len(rows)} seeds reach the canonical 475/500')
        print(f'   threshold, median {np.median(steps) if steps else float("nan"):,.0f}')
        print(f'   environment steps) under standard hyperparameters, with no')
        print(f'   change to the learner."')
    else:
        print("\nFAIL -- the implementation does not reach the CartPole-v1")
        print("threshold. Investigate before reporting any maze result: a")
        print("baseline that cannot solve CartPole cannot support a claim")
        print("about what DQN needs.")

    with open(os.path.join(args.outdir, "benchmark.json"), "w") as fh:
        json.dump({"task": "CartPole-v1", "criterion": "mean return >= 475 "
                                                       "over 100 episodes",
                   "budget_steps": args.budget, "hyperparameters":
                       {k: (list(v) if isinstance(v, tuple) else v)
                        for k, v in hp.items()},
                   "runs": rows,
                   "n_solved": n_solved, "n_seeds": len(rows),
                   "passed": passed}, fh, indent=2)
    print(f"\nwrote {args.outdir}/benchmark.json")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
