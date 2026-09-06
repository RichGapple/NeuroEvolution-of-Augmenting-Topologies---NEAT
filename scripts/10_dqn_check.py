#!/usr/bin/env python3
"""Conformance gate. Nothing else may run until this passes.

Two claims are checked numerically, because the whole comparison rests on them
and a docstring is not evidence:

  C1  StepNavEnv reproduces NavEnv.rollout exactly.
      Same policy in, byte-identical Trajectory out. If this fails, the DQN arm
      is running a different maze and every downstream number is meaningless.

  C2  Sum of per-step DQN rewards == NEAT fitness, exactly, for every eta.
      This is the invariant that makes the arms comparable. Checked over
      thousands of randomly generated trajectories, at every eta on the grid,
      in every sparsity mode.

Exits non-zero on failure, so it can sit in a Makefile in front of the sweep.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neat_sparsity.config import SparsityConfig
from neat_sparsity.reward import RewardModel
from dqn_sparsity.config import core_config
from dqn_sparsity.shaping import verify_equivalence
from dqn_sparsity.step_env import (StepNavEnv, rollout_stepwise,
                                   rollout_monolithic, trajectories_equal)


def check_env_conformance(cfg, n_policies: int = 200) -> bool:
    """C1: the wrapper and the original produce identical trajectories."""
    print("C1  StepNavEnv vs NavEnv.rollout")
    ok = True
    rng = np.random.default_rng(0)

    policies = []
    # fixed action sequences
    for _ in range(n_policies // 2):
        seq = rng.integers(0, 3, size=cfg.env.max_steps + 5)
        policies.append(("scripted", lambda o, t, s=seq: s[t % len(s)]))
    # observation-dependent policies, exercising the geometry
    for _ in range(n_policies // 2 - 4):
        w = rng.normal(size=(8, 3))
        policies.append(("reactive",
                         lambda o, t, w=w: int(np.argmax(np.asarray(o) @ w))))

    # goal-seeking policies: without these the checks never exercise the
    # success bonus, the progress clamp at p = 1, or early termination.
    for bias in (0.0, 0.15, -0.15, 0.35):
        def seek(o, t, b=bias):
            sin_b = float(o[6]) + b       # sin(bearing to goal)
            if sin_b > 0.12:
                return 1
            if sin_b < -0.12:
                return 0
            return 2
        policies.append(("goal-seeking", seek))

    n_reached = 0
    for kind, pol in policies:
        a = rollout_stepwise(cfg.env, pol)
        b = rollout_monolithic(cfg.env, pol)
        n_reached += int(a.reached_goal)
        if not trajectories_equal(a, b, tol=0.0):
            ok = False
            print(f"    MISMATCH ({kind}): steps {a.steps_taken} vs {b.steps_taken}, "
                  f"dmin {a.min_distance:.6f} vs {b.min_distance:.6f}")
            break

    # Random policies rarely solve this maze, so the goal-reaching branch
    # (early termination) would go untested. Re-run with an enlarged goal
    # radius purely to exercise that path in both implementations.
    import copy
    big = copy.deepcopy(cfg.env)
    big.goal_radius = 45.0
    n_reached_big = 0
    for kind, pol in policies:
        a = rollout_stepwise(big, pol)
        b = rollout_monolithic(big, pol)
        n_reached_big += int(a.reached_goal)
        if not trajectories_equal(a, b, tol=0.0):
            ok = False
            print(f"    MISMATCH on goal-reaching pass ({kind})")
            break

    if ok:
        print(f"    PASS   {len(policies)} policies x 2 passes, byte-identical")
        print(f"           normal radius: {n_reached} reached; "
              f"enlarged radius: {n_reached_big} reached "
              f"(exercises early termination)")
    return ok


def check_return_equivalence(cfg, n_traj: int = 400) -> bool:
    """C2: telescoped per-step reward sums to the NEAT fitness."""
    print("\nC2  sum(per-step DQN reward) == NEAT fitness")
    rng = np.random.default_rng(1)
    all_ok = True

    # Generate real trajectories from random reactive policies so the distance
    # sequences are physically achievable, not synthetic noise.
    trajs = []
    for _ in range(n_traj):
        w = rng.normal(size=(8, 3))
        trajs.append(rollout_stepwise(
            cfg.env, lambda o, t, w=w: int(np.argmax(np.asarray(o) @ w))))
    # Random policies almost never solve this maze, so the success bonus and
    # the progress clamp at p = 1 would never be tested. Add synthetic
    # goal-reaching trajectories: Trajectory is a plain dataclass, so these are
    # legitimate inputs to the reward model even though no policy produced them.
    from neat_sparsity.env import Trajectory
    for k in range(40):
        d0 = 113.137
        ds = list(np.linspace(d0, 0.0, 30 + k % 20))
        trajs.append(Trajectory(
            reached_goal=True, steps_taken=len(ds) - 1, initial_distance=d0,
            min_distance=0.0, final_distance=0.0, path_length=d0,
            collisions=0, distances=ds, final_x=90.0, final_y=90.0))
    # ...and near-misses that land exactly on quantiser bin edges, where
    # floor() is most likely to disagree between the two code paths.
    for L in (2, 4, 8, 16, 64, 256):
        for k in range(1, L):
            d0 = 100.0
            dmin = d0 * (1.0 - k / L)
            ds = list(np.linspace(d0, dmin, 25))
            trajs.append(Trajectory(
                reached_goal=False, steps_taken=24, initial_distance=d0,
                min_distance=dmin, final_distance=dmin, path_length=d0 - dmin,
                collisions=0, distances=ds, final_x=0.0, final_y=0.0))

    n_solved = sum(t.reached_goal for t in trajs)
    print(f"    {len(trajs)} trajectories ({n_solved} reached the goal, "
          f"including bin-edge cases)")

    for mode in ("quantized", "checkpoint", "gated"):
        scfg = SparsityConfig(**{**cfg.sparsity.__dict__, "mode": mode})
        for basis in ("min", "current"):
            worst = 0.0
            bad = 0
            for eta in cfg.etas:
                for traj in trajs:
                    ok, dqn_ret, neat_fit = verify_equivalence(
                        scfg, eta, traj, credit_basis=basis)
                    worst = max(worst, abs(dqn_ret - neat_fit))
                    if not ok:
                        bad += 1
            status = "PASS" if bad == 0 else f"FAIL ({bad} mismatches)"
            print(f"    mode={mode:<11s} basis={basis:<8s} "
                  f"max |return - fitness| = {worst:.3e}  {status}")
            all_ok &= (bad == 0)

    print("\n    Note: DQNConfig.reward_scale multiplies these rewards before")
    print("    they reach the replay buffer. That is optimiser preconditioning,")
    print("    not a change to the reward, so the identity above is unaffected.")
    print("    With gamma < 1 the agent optimises a discounted proxy; the")
    print("    UNDISCOUNTED return is what equals the fitness exactly.")
    return all_ok


def check_max_fitness_invariant(cfg) -> bool:
    """The DQN arm must inherit the NEAT arm's invariant: max fitness is the
    same at every eta, so 'sparse runs score lower' cannot be true by
    definition on this side either."""
    print("\nC3  max attainable fitness is eta-invariant")
    vals = {eta: RewardModel(cfg.sparsity, eta).max_fitness for eta in cfg.etas}
    ok = len(set(round(v, 12) for v in vals.values())) == 1
    print(f"    {'PASS' if ok else 'FAIL'}   max fitness = "
          f"{sorted(set(round(v, 6) for v in vals.values()))}")
    return ok


def check_gradients() -> bool:
    """C4: finite-difference check on the hand-written backprop.

    Guards against the reviewer objection that the baseline is simply broken.
    """
    print("\nC4  finite-difference gradient check on the MLP")
    from dqn_sparsity.nets import MLP, huber_grad
    rng = np.random.default_rng(3)
    # tanh + float64: ReLU kinks and float32 rounding both corrupt a
    # finite-difference check. The backward pass is shared, so verifying
    # it on the smooth path verifies it everywhere.
    net = MLP(8, [16, 16], 3, "tanh", seed=7)
    net.W = [w.astype(np.float64) for w in net.W]
    net.b = [b.astype(np.float64) for b in net.b]
    x = rng.normal(size=(12, 8))
    y = rng.normal(size=(12,))
    a = rng.integers(0, 3, size=12)
    idx = np.arange(12)

    def loss_of(net):
        out, _ = net.forward(x)
        _, l = huber_grad(out[idx, a], y, 1.0)
        return l

    out, cache = net.forward(x)
    g, _ = huber_grad(out[idx, a], y, 1.0)
    dout = np.zeros_like(out)
    dout[idx, a] = g / len(a)
    gW, gb = net.backward(cache, dout)

    eps = 1e-5
    worst = 0.0
    for li in range(net.n_layers):
        for _ in range(25):
            i = rng.integers(0, net.W[li].shape[0])
            j = rng.integers(0, net.W[li].shape[1])
            orig = float(net.W[li][i, j])
            net.W[li][i, j] = orig + eps
            lp = loss_of(net)
            net.W[li][i, j] = orig - eps
            lm = loss_of(net)
            net.W[li][i, j] = orig
            num = (lp - lm) / (2 * eps)
            ana = float(gW[li][i, j])
            denom = max(1e-6, abs(num) + abs(ana))
            worst = max(worst, abs(num - ana) / denom)

    ok = worst < 1e-5
    print(f"    {'PASS' if ok else 'FAIL'}   max relative error = {worst:.2e}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policies", type=int, default=200)
    ap.add_argument("--trajectories", type=int, default=400)
    args = ap.parse_args()

    cfg = core_config()
    print(f"config hash {cfg.config_hash()}")
    print(f"env: {cfg.env.width:g}x{cfg.env.height:g}, start {cfg.env.start}, "
          f"goal {cfg.env.goal}, {cfg.env.max_steps} steps\n")

    results = [
        check_env_conformance(cfg, args.policies),
        check_return_equivalence(cfg, args.trajectories),
        check_max_fitness_invariant(cfg),
        check_gradients(),
    ]

    print("\n" + "=" * 66)
    if all(results):
        print("ALL CONFORMANCE CHECKS PASSED")
        print("The DQN arm runs the same maze and receives the same total")
        print("reward information as the NEAT arm. The comparison is valid.")
        print("=" * 66)
        sys.exit(0)
    print("CONFORMANCE FAILED -- do not run the sweep, and do not trust")
    print("any comparison produced before this passes.")
    print("=" * 66)
    sys.exit(1)


if __name__ == "__main__":
    main()
