"""Phenotype: compile a genome into a callable network.

Feedforward genomes are compiled into a static evaluation order once, which is
what makes the ~10^6 forward passes per run affordable in pure Python.
Recurrent genomes (cfg.feedforward=False) fall back to a single synchronous
update per step with state carried between steps.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from .genome import Genome, INPUT, OUTPUT, HIDDEN


def _tanh(x: float) -> float:
    if x > 20.0:
        return 1.0
    if x < -20.0:
        return -1.0
    return math.tanh(x)


def _sigmoid(x: float) -> float:
    if x < -20.0:
        return 0.0
    if x > 20.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def _relu(x: float) -> float:
    return x if x > 0.0 else 0.0


ACTIVATIONS = {"tanh": _tanh, "sigmoid": _sigmoid, "relu": _relu,
               "identity": lambda x: x}


class FeedForwardNetwork:
    """Evaluation layers computed by Kahn topological sort over enabled edges."""

    __slots__ = ("input_keys", "output_keys", "node_evals", "values", "act", "out_act")

    def __init__(self, input_keys, output_keys, node_evals, act, out_act):
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.node_evals = node_evals   # list of (node_key, bias, [(src, w), ...], is_output)
        self.act = act
        self.out_act = out_act
        self.values: Dict[int, float] = {}

    def activate(self, inputs: Sequence[float]) -> List[float]:
        v = self.values
        v.clear()
        for k, x in zip(self.input_keys, inputs):
            v[k] = x
        act, out_act = self.act, self.out_act
        for key, bias, links, is_out in self.node_evals:
            s = bias
            for src, w in links:
                s += v.get(src, 0.0) * w
            v[key] = out_act(s) if is_out else act(s)
        return [v.get(k, 0.0) for k in self.output_keys]

    def reset(self) -> None:
        self.values.clear()


class RecurrentNetwork:
    __slots__ = ("input_keys", "output_keys", "node_evals", "state", "act", "out_act")

    def __init__(self, input_keys, output_keys, node_evals, act, out_act):
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.node_evals = node_evals
        self.act = act
        self.out_act = out_act
        self.state: Dict[int, float] = {}
        self.reset()

    def reset(self) -> None:
        self.state = {k: 0.0 for k, _, _, _ in self.node_evals}
        for k in self.input_keys:
            self.state[k] = 0.0

    def activate(self, inputs: Sequence[float]) -> List[float]:
        prev = dict(self.state)
        for k, x in zip(self.input_keys, inputs):
            prev[k] = x
            self.state[k] = x
        for key, bias, links, is_out in self.node_evals:
            s = bias
            for src, w in links:
                s += prev.get(src, 0.0) * w
            self.state[key] = self.out_act(s) if is_out else self.act(s)
        return [self.state.get(k, 0.0) for k in self.output_keys]


def _topological_order(node_keys: List[int], edges: List[Tuple[int, int]]) -> List[int]:
    indeg = {k: 0 for k in node_keys}
    adj: Dict[int, List[int]] = {k: [] for k in node_keys}
    for a, b in edges:
        if a in indeg and b in indeg:
            adj[a].append(b)
            indeg[b] += 1
    # deterministic: always pop the smallest available key
    ready = sorted(k for k in node_keys if indeg[k] == 0)
    order: List[int] = []
    while ready:
        k = ready.pop(0)
        order.append(k)
        for nxt in adj[k]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                # keep `ready` sorted
                lo, hi = 0, len(ready)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if ready[mid] < nxt:
                        lo = mid + 1
                    else:
                        hi = mid
                ready.insert(lo, nxt)
    return order


def build_network(genome: Genome, cfg):
    act = ACTIVATIONS[cfg.activation]
    out_act = ACTIVATIONS[cfg.output_activation]

    input_keys = sorted(k for k, n in genome.nodes.items() if n.type == INPUT)
    output_keys = sorted(k for k, n in genome.nodes.items() if n.type == OUTPUT)

    incoming: Dict[int, List[Tuple[int, float]]] = {k: [] for k in genome.nodes}
    edges: List[Tuple[int, int]] = []
    for ck in sorted(genome.conns.keys()):
        c = genome.conns[ck]
        if not c.enabled:
            continue
        src, dst = c.key
        if src in genome.nodes and dst in genome.nodes:
            incoming[dst].append((src, c.weight))
            edges.append((src, dst))

    all_keys = sorted(genome.nodes.keys())

    if cfg.feedforward:
        order = _topological_order(all_keys, edges)
        if len(order) != len(all_keys):        # cycle slipped in -> use recurrent
            evals = [(k, genome.nodes[k].bias, incoming[k], genome.nodes[k].type == OUTPUT)
                     for k in all_keys if genome.nodes[k].type != INPUT]
            return RecurrentNetwork(input_keys, output_keys, evals, act, out_act)
        evals = [(k, genome.nodes[k].bias, incoming[k], genome.nodes[k].type == OUTPUT)
                 for k in order if genome.nodes[k].type != INPUT]
        return FeedForwardNetwork(input_keys, output_keys, evals, act, out_act)

    evals = [(k, genome.nodes[k].bias, incoming[k], genome.nodes[k].type == OUTPUT)
             for k in all_keys if genome.nodes[k].type != INPUT]
    return RecurrentNetwork(input_keys, output_keys, evals, act, out_act)
