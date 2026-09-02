"""Fast correctness checks.  Run with:  python -m pytest tests -q   (or plain python)."""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from neat_sparsity.config import ExperimentConfig, NEATConfig
from neat_sparsity.env import NavEnv, Trajectory
from neat_sparsity.genome import Genome, InnovationTracker
from neat_sparsity.network import build_network
from neat_sparsity.population import Population, run_evolution
from neat_sparsity.reward import RewardModel, true_objective
from neat_sparsity.metrics import genome_topology, network_depth, n_distinct
from neat_sparsity.ablation import ablation_report, minimal_functional_network
from neat_sparsity.stats import (kruskal, dunn, cliffs_delta, threshold_analysis,
                                 bootstrap_ci)


def _cfg():
    c = ExperimentConfig()
    c.generations = 5
    c.neat.pop_size = 20
    c.env.max_steps = 60
    return c


def test_genome_roundtrip():
    from neat_sparsity.runner import genome_to_dict, genome_from_dict
    cfg = _cfg()
    rng = random.Random(0)
    tr = InnovationTracker(cfg.neat.num_inputs + cfg.neat.num_outputs)
    g = Genome.new_minimal(0, cfg.neat, tr, rng)
    for _ in range(20):
        g.mutate(cfg.neat, tr, rng)
    h = genome_from_dict(genome_to_dict(g))
    assert len(h.nodes) == len(g.nodes) and len(h.conns) == len(g.conns)
    net_g = build_network(g, cfg.neat)
    net_h = build_network(h, cfg.neat)
    x = [0.1] * cfg.neat.num_inputs
    assert np.allclose(net_g.activate(x), net_h.activate(x))
    print("genome roundtrip: ok")


def test_feedforward_acyclic():
    cfg = _cfg()
    rng = random.Random(1)
    tr = InnovationTracker(cfg.neat.num_inputs + cfg.neat.num_outputs)
    g = Genome.new_minimal(0, cfg.neat, tr, rng)
    for _ in range(200):
        g.mutate(cfg.neat, tr, rng)
    assert network_depth(g) >= 0, "cycle detected in a feedforward genome"
    print(f"acyclicity after 200 mutations: ok "
          f"({len(g.nodes)} nodes, {len(g.conns)} conns, depth {network_depth(g)})")


def test_determinism():
    cfg = _cfg()

    def run(seed):
        env = NavEnv(cfg.env)
        model = RewardModel(cfg.sparsity, 0.0, random.Random(seed))

        def ev(genomes):
            f, t, r = [], [], 0
            for g in genomes:
                traj = env.rollout(build_network(g, cfg.neat))
                f.append(model.fitness(traj))
                t.append(true_objective(traj, cfg.sparsity))
                r += int(traj.reached_goal)
            return f, t, r

        recs, champ, _ = run_evolution(cfg.neat, seed, ev, cfg.generations)
        return recs

    def same(r1, r2):
        # NaN-aware: TIR is NaN in generation 0 by construction
        if len(r1) != len(r2):
            return False
        for x, y in zip(r1, r2):
            for k in x:
                a_, b_ = x[k], y[k]
                if a_ != a_ and b_ != b_:
                    continue
                if a_ != b_:
                    return False
        return True

    a, b = run(7), run(7)
    assert same(a, b), "identical seeds produced different runs"
    c = run(8)
    assert not same(a, c), "different seeds produced identical runs"
    print("determinism: ok")


def test_sparsity_invariants():
    cfg = _cfg()
    t_ok = Trajectory(True, 10, 100.0, 0.0, 0.0, 100.0, 0, [100.0, 0.0])
    t_mid = Trajectory(False, 10, 100.0, 50.0, 50.0, 50.0, 0,
                       [100.0 - i for i in range(51)])
    prev = None
    for eta in np.linspace(0, 1, 9):
        m = RewardModel(cfg.sparsity, float(eta), random.Random(3))
        assert abs(m.fitness(t_ok) - 2.0) < 1e-9
        c = m.fitness(t_mid)
        if prev is not None:
            assert c <= prev + 1e-12, f"credit rose at eta={eta}"
        prev = c
    assert RewardModel(cfg.sparsity, 1.0).fitness(t_mid) == 0.0
    print("sparsity invariants: ok")


def test_ablation():
    cfg = _cfg()
    rng = random.Random(4)
    tr = InnovationTracker(cfg.neat.num_inputs + cfg.neat.num_outputs)
    g = Genome.new_minimal(0, cfg.neat, tr, rng)
    for _ in range(30):
        g.mutate(cfg.neat, tr, rng)
    rep = ablation_report(g, cfg)
    assert rep["n_enabled_connections"] > 0
    assert 0.0 <= rep["functional_conn_fraction"] <= 1.0
    seq = minimal_functional_network(g, cfg)
    assert seq["enabled_conns_after"] <= seq["enabled_conns_before"]
    assert seq["pruned_score"] >= seq["baseline_score"] - rep["tau"] - 1e-9
    print(f"ablation: ok (functional fraction "
          f"{rep['functional_conn_fraction']:.2f}, sequentially removable "
          f"{seq['removable_fraction']:.2f})")


def test_stats():
    rng = np.random.default_rng(0)
    g = {0.0: rng.normal(0, 1, 30), 0.5: rng.normal(0.5, 1, 30),
         1.0: rng.normal(2.5, 1, 30)}
    k = kruskal(g)
    assert k.pvalue < 0.05
    d = dunn(g)
    assert all("p_holm" in r for r in d)
    assert cliffs_delta(g[1.0], g[0.0]) > 0.5
    m, lo, hi = bootstrap_ci(g[0.0])
    assert lo <= m <= hi

    # a genuine break at 0.6
    gg = {}
    for e in np.linspace(0, 1, 9):
        base = 0.0 if e <= 0.6 else 5 * (e - 0.6)
        gg[float(e)] = base + rng.normal(0, 0.05, 25)
    thr = threshold_analysis(gg, n_boot=200)
    assert thr["piecewise_preferred"], thr
    assert abs(thr["breakpoint"] - 0.625) < 0.2, thr["breakpoint"]
    print(f"stats: ok (recovered breakpoint {thr['breakpoint']:.3f}, "
          f"dAIC {thr['delta_aic']:.0f})")


def test_end_to_end(tmpdir="/tmp/neat_smoke"):
    from neat_sparsity.runner import run_sweep
    cfg = _cfg()
    cfg.etas = [0.0, 1.0]
    cfg.seeds = [11, 12]
    res = run_sweep(cfg, tmpdir, workers=1, verbose=False)
    assert len(res) == 4
    assert os.path.exists(os.path.join(tmpdir, "summary.csv"))
    print("end-to-end sweep: ok")


if __name__ == "__main__":
    for fn in [test_genome_roundtrip, test_feedforward_acyclic, test_determinism,
               test_sparsity_invariants, test_ablation, test_stats, test_end_to_end]:
        fn()
    print("\nALL SMOKE TESTS PASSED")
