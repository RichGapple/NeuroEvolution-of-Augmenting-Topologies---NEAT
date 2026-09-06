#!/usr/bin/env python3
"""Test suite for the DQN arm. Run with: python tests/test_dqn.py

Covers the things that would silently invalidate the comparison if they broke:
environment conformance, exact return equivalence, gradient correctness,
determinism, and the guard rails in compare.py. Plain asserts, no pytest
dependency, matching the NEAT arm's tests/test_smoke.py.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neat_sparsity.config import SparsityConfig
from neat_sparsity.env import Trajectory
from neat_sparsity.reward import RewardModel
from dqn_sparsity.config import core_config, matched_to_neat
from dqn_sparsity.nets import MLP, Adam, huber_grad
from dqn_sparsity.agent import DQNAgent, ReplayBuffer
from dqn_sparsity.shaping import StepwiseReward, verify_equivalence
from dqn_sparsity.step_env import (StepNavEnv, rollout_stepwise,
                                   rollout_monolithic, trajectories_equal)
from dqn_sparsity.runner import run_single

PASS, FAIL = 0, 0


def check(name, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  PASS  {name}")
        PASS += 1
    except AssertionError as e:
        print(f"  FAIL  {name}: {e}")
        FAIL += 1
    except Exception as e:
        print(f"  ERROR {name}: {type(e).__name__}: {e}")
        FAIL += 1


# --------------------------------------------------------------------------- #
def test_env_conformance():
    cfg = core_config()
    rng = np.random.default_rng(0)
    for _ in range(40):
        w = rng.normal(size=(8, 3))
        pol = lambda o, t, w=w: int(np.argmax(np.asarray(o) @ w))
        a = rollout_stepwise(cfg.env, pol)
        b = rollout_monolithic(cfg.env, pol)
        assert trajectories_equal(a, b, 0.0), "StepNavEnv != NavEnv.rollout"


def test_env_deterministic():
    cfg = core_config()
    pol = lambda o, t: (t % 3)
    a = rollout_stepwise(cfg.env, pol)
    b = rollout_stepwise(cfg.env, pol)
    assert trajectories_equal(a, b, 0.0), "StepNavEnv is not deterministic"


def test_step_after_done_raises():
    env = StepNavEnv(core_config().env)
    env.reset()
    while not env.done:
        env.step(2)
    try:
        env.step(0)
        raise AssertionError("stepping a finished episode should raise")
    except RuntimeError:
        pass


# --------------------------------------------------------------------------- #
def test_return_equivalence_all_modes():
    cfg = core_config()
    rng = np.random.default_rng(1)
    trajs = []
    for _ in range(30):
        w = rng.normal(size=(8, 3))
        trajs.append(rollout_stepwise(
            cfg.env, lambda o, t, w=w: int(np.argmax(np.asarray(o) @ w))))
    for k in range(10):
        d0 = 113.137
        ds = list(np.linspace(d0, 0.0, 20 + k))
        trajs.append(Trajectory(True, len(ds) - 1, d0, 0.0, 0.0, d0, 0,
                                ds, 90.0, 90.0))
    for mode in ("quantized", "checkpoint", "gated"):
        s = SparsityConfig(**{**cfg.sparsity.__dict__, "mode": mode})
        for basis in ("min", "current"):
            for eta in cfg.etas:
                for t in trajs:
                    ok, ret, fit = verify_equivalence(s, eta, t,
                                                      credit_basis=basis)
                    assert ok, (f"mode={mode} basis={basis} eta={eta}: "
                                f"{ret} != {fit}")


def test_return_equivalence_bin_edges():
    """floor() at exact bin boundaries is where a mismatch would hide."""
    cfg = core_config()
    for L, eta in ((256, 0.0), (64, 0.25), (8, 0.625), (2, 0.875), (1, 1.0)):
        for k in range(L + 1):
            d0 = 100.0
            dmin = max(0.0, d0 * (1.0 - k / L))
            ds = list(np.linspace(d0, dmin, 12))
            t = Trajectory(False, 11, d0, dmin, dmin, d0 - dmin, 0, ds, 0.0, 0.0)
            ok, ret, fit = verify_equivalence(cfg.sparsity, eta, t)
            assert ok, f"L={L} eta={eta} k={k}: {ret} != {fit}"


def test_reward_scale_does_not_change_reported_units():
    """Scaling is learner-side only; nothing reported may move with it."""
    cfg = core_config()
    cfg.dqn.budget_steps = 2500
    cfg.dqn.warmup_steps = 10 ** 9        # pure random actions, no learning
    cfg.dqn.eval_points = 3
    t1, t2 = tempfile.mkdtemp(), tempfile.mkdtemp()
    try:
        cfg.dqn.reward_scale = 1.0
        a = run_single(cfg, 0.25, 5, t1)
        cfg.dqn.reward_scale = 50.0
        b = run_single(cfg, 0.25, 5, t2)
        assert abs(a["final_true_best"] - b["final_true_best"]) < 1e-9, \
            "reward_scale leaked into reported true objective"
        assert abs(a["mean_zero_reward_fraction"]
                   - b["mean_zero_reward_fraction"]) < 1e-9
    finally:
        shutil.rmtree(t1, ignore_errors=True)
        shutil.rmtree(t2, ignore_errors=True)


def test_gamma_is_contractive():
    """gamma = 1.0 removes the Bellman contraction; guard against regressing."""
    assert core_config().dqn.gamma < 1.0, \
        "gamma must be < 1: gamma=1.0 measured at random-policy level"


def test_max_fitness_invariant():
    cfg = core_config()
    vals = {RewardModel(cfg.sparsity, e).max_fitness for e in cfg.etas}
    assert len(vals) == 1, f"max fitness varies across eta: {vals}"
    assert abs(list(vals)[0] - 2.0) < 1e-12


def test_terminal_shaping_also_sums_right():
    """The ablation must still deliver the correct total, just all at once."""
    cfg = core_config()
    d0 = 100.0
    ds = list(np.linspace(d0, 30.0, 20))
    t = Trajectory(False, 19, d0, 30.0, 30.0, 70.0, 0, ds, 0.0, 0.0)
    for eta in cfg.etas:
        sh = StepwiseReward(cfg.sparsity, eta, mode="terminal")
        sh.reset(d0)
        total, dmin = 0.0, ds[0]
        for i in range(1, len(ds)):
            dmin = min(dmin, ds[i])
            last = i == len(ds) - 1
            total += sh.step_reward(
                {"distance": ds[i], "min_distance": dmin, "reached": False},
                done=last, traj_if_done=t if last else None)
        fit = RewardModel(cfg.sparsity, eta).fitness(t)
        assert abs(total - fit) < 1e-9, f"eta={eta}: {total} != {fit}"


# --------------------------------------------------------------------------- #
def test_gradient_finite_difference():
    rng = np.random.default_rng(3)
    net = MLP(8, [12, 12], 3, "tanh", seed=7)
    net.W = [w.astype(np.float64) for w in net.W]
    net.b = [b.astype(np.float64) for b in net.b]
    x = rng.normal(size=(10, 8))
    y = rng.normal(size=(10,))
    a = rng.integers(0, 3, size=10)
    idx = np.arange(10)

    def loss():
        out, _ = net.forward(x)
        return huber_grad(out[idx, a], y, 1.0)[1]

    out, cache = net.forward(x)
    g, _ = huber_grad(out[idx, a], y, 1.0)
    dout = np.zeros_like(out)
    dout[idx, a] = g / len(a)
    gW, gb = net.backward(cache, dout)

    eps, worst = 1e-5, 0.0
    for li in range(net.n_layers):
        for _ in range(15):
            i = rng.integers(0, net.W[li].shape[0])
            j = rng.integers(0, net.W[li].shape[1])
            o = float(net.W[li][i, j])
            net.W[li][i, j] = o + eps; lp = loss()
            net.W[li][i, j] = o - eps; lm = loss()
            net.W[li][i, j] = o
            num, ana = (lp - lm) / (2 * eps), float(gW[li][i, j])
            worst = max(worst, abs(num - ana) / max(1e-9, abs(num) + abs(ana)))
    assert worst < 1e-5, f"gradient check failed, rel err {worst:.2e}"


def test_adam_reduces_loss():
    rng = np.random.default_rng(5)
    net = MLP(6, [16], 2, "relu", seed=1)
    opt = Adam(net, 1e-2, clip=10.0)
    x = rng.normal(size=(64, 6)).astype(np.float32)
    y = rng.normal(size=(64, 2)).astype(np.float32)
    first = last = None
    for k in range(300):
        out, cache = net.forward(x)
        d, l = huber_grad(out, y, 1.0)
        opt.step(*net.backward(cache, d / len(x)))
        if k == 0:
            first = l
        last = l
    assert last < first * 0.9, f"Adam did not reduce loss: {first} -> {last}"


def test_target_network_lags():
    cfg = core_config()
    cfg.dqn.warmup_steps = 0
    cfg.dqn.target_update_every = 10 ** 9
    cfg.dqn.train_every = 1
    ag = DQNAgent(8, 3, cfg.dqn, seed=0)
    before = [w.copy() for w in ag.target.get_weights()]
    rng = np.random.default_rng(0)
    for _ in range(300):
        ag.observe(rng.normal(size=8).astype(np.float32), 0, 1.0,
                   rng.normal(size=8).astype(np.float32), False)
    assert ag.updates > 0, "no gradient updates ran"
    assert all(np.allclose(a, b) for a, b in
               zip(before, ag.target.get_weights())), "target net updated early"
    assert not all(np.allclose(a, b) for a, b in
                   zip(before, ag.q.get_weights())), "online net did not move"


def test_replay_diagnostics():
    buf = ReplayBuffer(100, 4, seed=0)
    for i in range(50):
        buf.add(np.zeros(4), 0, 0.0 if i % 2 else 1.0, np.zeros(4), False)
    assert abs(buf.zero_reward_fraction() - 0.5) < 1e-9
    assert buf.distinct_rewards() == 2
    assert 0.99 < buf.reward_entropy() <= 1.0
    for i in range(200):
        buf.add(np.zeros(4), 0, 0.0, np.zeros(4), False)
    assert buf.n == 100 and buf.distinct_rewards() == 1
    assert buf.reward_entropy() == 0.0


def test_epsilon_schedule():
    cfg = core_config().dqn
    assert abs(cfg.eps_at(0) - cfg.eps_start) < 1e-9
    assert abs(cfg.eps_at(cfg.budget_steps) - cfg.eps_end) < 1e-9
    mid = cfg.eps_at(int(0.5 * cfg.eps_decay_frac * cfg.budget_steps))
    assert cfg.eps_end < mid < cfg.eps_start


# --------------------------------------------------------------------------- #
def test_no_posix_only_imports():
    """Guard against reintroducing Unix-only modules at import scope.

    `import resource` at module scope broke the entire package on Windows with
    ModuleNotFoundError. Anything POSIX-only belongs behind a try/except inside
    sysinfo.py, never at the top of a module the package imports eagerly.
    """
    import ast, os
    banned = {"resource", "fcntl", "pwd", "grp", "termios", "posix", "syslog"}
    pkg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "dqn_sparsity")
    offenders = []
    for fn in sorted(os.listdir(pkg)):
        if not fn.endswith(".py") or fn == "sysinfo.py":
            continue
        tree = ast.parse(open(os.path.join(pkg, fn), encoding="utf-8").read())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            for n in names:
                if n in banned:
                    offenders.append(f"{fn}: import {n}")
    assert not offenders, "POSIX-only imports found: " + "; ".join(offenders)


def test_peak_rss_never_raises():
    """Cost reporting must degrade to NaN, never stop a run."""
    from dqn_sparsity.sysinfo import peak_rss_mb, describe_host
    v = peak_rss_mb()
    assert isinstance(v, float), "peak_rss_mb must return a float"
    assert v != v or v > 0, f"implausible peak RSS: {v}"
    h = describe_host()
    assert "platform" in h and h.get("gpu_used") is False


def test_maze_analysis():
    """Deceptiveness is computed, not assumed -- and A/B must differ on it."""
    from dqn_sparsity import mazes, maze_analysis
    cfg = core_config()
    a = mazes.apply_maze(cfg.env, "A")
    b = mazes.apply_maze(cfg.env, "B")
    assert mazes.identify(a) == "A" and mazes.identify(b) == "B"
    ga = maze_analysis.greedy_descent_optimum(a)
    gb = maze_analysis.greedy_descent_optimum(b)
    assert not ga["reaches_goal"], "maze A should be deceptive"
    assert gb["reaches_goal"], "maze B should NOT be deceptive"
    assert 0.70 < ga["progress"] < 0.74, f"maze A trap moved: {ga['progress']}"
    for m in (a, b):
        f = maze_analysis.step_feasibility(m)
        assert f["reachable"] and f["feasible"], "goal must be reachable in budget"


def test_config_hash_sensitivity():
    a = core_config()
    b = core_config()
    assert a.config_hash() == b.config_hash(), "hash is not stable"
    b.seeds = [999]
    assert a.config_hash() == b.config_hash(), "seeds must not change the hash"
    b.dqn.lr *= 2
    assert a.config_hash() != b.config_hash(), "lr must change the hash"
    c = core_config()
    c.env.max_steps += 1
    assert a.config_hash() != c.config_hash(), "env must change the hash"


def test_matched_to_neat():
    from neat_sparsity.config import core_config as neat_core
    n = neat_core()
    d = matched_to_neat(n)
    assert d.env is n.env and d.sparsity is n.sparsity
    assert d.etas == n.etas
    assert d.dqn.budget_steps == n.generations * n.neat.pop_size * n.env.max_steps


def test_run_single_end_to_end():
    cfg = core_config()
    cfg.dqn.budget_steps = 3000
    cfg.dqn.warmup_steps = 200
    cfg.dqn.eval_points = 5
    tmp = tempfile.mkdtemp()
    try:
        r = run_single(cfg, 0.5, 7, tmp)
        for k in ("final_true_best", "env_steps", "gradient_updates",
                  "mean_zero_reward_fraction", "peak_rss_mb", "config_hash"):
            assert k in r, f"summary missing {k}"
        assert r["env_steps"] == 3000
        assert 0.0 <= r["final_true_best"] <= 2.0
        assert os.path.exists(os.path.join(tmp, "eta=0.5", "seed=7", "blocks.csv"))
        assert os.path.exists(os.path.join(tmp, "eta=0.5", "seed=7", "run.json"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_run_reproducible():
    cfg = core_config()
    cfg.dqn.budget_steps = 2500
    cfg.dqn.warmup_steps = 200
    cfg.dqn.eval_points = 4
    t1, t2 = tempfile.mkdtemp(), tempfile.mkdtemp()
    try:
        a = run_single(cfg, 0.25, 11, t1)
        b = run_single(cfg, 0.25, 11, t2)
        for k in ("final_true_best", "gradient_updates", "final_td_loss",
                  "mean_zero_reward_fraction"):
            assert abs(float(a[k]) - float(b[k])) < 1e-9, \
                f"{k} not reproducible: {a[k]} vs {b[k]}"
    finally:
        shutil.rmtree(t1, ignore_errors=True)
        shutil.rmtree(t2, ignore_errors=True)


# --------------------------------------------------------------------------- #
def test_compare_guardrails():
    import pandas as pd
    from dqn_sparsity.compare import assert_comparable, crossover_point
    base = dict(config_hash="abc", sparsity_mode="quantized",
                final_true_best=1.0)
    n = pd.DataFrame([{**base, "eta": e, "seed": s, "arm": "neat"}
                      for e in (0.0, 1.0) for s in (1, 2, 3)])
    d = pd.DataFrame([{**base, "config_hash": "xyz", "eta": e, "seed": s,
                       "arm": "dqn"} for e in (0.0, 1.0) for s in (1, 2, 3)])
    assert_comparable(n, d)                      # differing hashes across arms is fine

    d2 = d.copy(); d2.loc[0, "sparsity_mode"] = "gated"
    try:
        assert_comparable(n, d2)
        raise AssertionError("should reject mismatched sparsity modes")
    except SystemExit:
        pass

    d3 = d.copy(); d3.loc[d3["eta"] == 1.0, "eta"] = 0.5
    try:
        assert_comparable(n, d3)
        raise AssertionError("should reject mismatched eta grids")
    except SystemExit:
        pass


def test_crossover_detection():
    from dqn_sparsity.compare import crossover_point
    etas = [0.0, 0.25, 0.5, 0.75, 1.0]
    dqn = {e: [2.0 - 2.0 * e] * 12 for e in etas}      # falls fast
    neat = {e: [1.0] * 12 for e in etas}               # flat
    c = crossover_point(neat, dqn, n_boot=200)
    assert c["eta"] is not None, "should find a crossover"
    assert abs(c["eta"] - 0.5) < 0.06, f"crossover at {c['eta']}, expected ~0.5"

    no = crossover_point({e: [0.1] * 12 for e in etas},
                         {e: [2.0] * 12 for e in etas}, n_boot=100)
    assert no["eta"] is None, "should report no crossover when one arm dominates"


# --------------------------------------------------------------------------- #
def main():
    print("=" * 66)
    print("DQN ARM TEST SUITE")
    print("=" * 66)

    print("\nenvironment conformance")
    check("StepNavEnv == NavEnv.rollout", test_env_conformance)
    check("StepNavEnv deterministic", test_env_deterministic)
    check("step() after done raises", test_step_after_done_raises)

    print("\nreward equivalence (the invariant the comparison rests on)")
    check("return == fitness, all modes", test_return_equivalence_all_modes)
    check("return == fitness at bin edges", test_return_equivalence_bin_edges)
    check("max fitness invariant across eta", test_max_fitness_invariant)
    check("reward_scale does not leak into reports",
          test_reward_scale_does_not_change_reported_units)
    check("gamma < 1 (Bellman contraction)", test_gamma_is_contractive)
    check("terminal shaping sums correctly", test_terminal_shaping_also_sums_right)

    print("\nlearner correctness")
    check("backprop vs finite differences", test_gradient_finite_difference)
    check("Adam reduces loss", test_adam_reduces_loss)
    check("target network lags online net", test_target_network_lags)
    check("replay buffer diagnostics", test_replay_diagnostics)
    check("epsilon schedule endpoints", test_epsilon_schedule)

    print("\nconfiguration and reproducibility")
    check("no POSIX-only imports (Windows safe)", test_no_posix_only_imports)
    check("peak RSS never raises", test_peak_rss_never_raises)
    check("maze A deceptive, B not", test_maze_analysis)
    check("config hash sensitivity", test_config_hash_sensitivity)
    check("matched_to_neat inherits env", test_matched_to_neat)
    check("run_single end to end", test_run_single_end_to_end)
    check("same seed -> same result", test_run_reproducible)

    print("\ncomparison guard rails")
    check("refuses mismatched configs", test_compare_guardrails)
    check("crossover detection", test_crossover_detection)

    print("\n" + "=" * 66)
    print(f"{PASS} passed, {FAIL} failed")
    print("=" * 66)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
