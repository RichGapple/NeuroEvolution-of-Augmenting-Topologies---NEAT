"""Genome representation, innovation bookkeeping, mutation and crossover.

Design notes relevant to the research question
----------------------------------------------
* Every structural mutation goes through `InnovationTracker`, which assigns a
  *global* innovation id and remembers, for each generation, which ids were
  created for the first time.  That bookkeeping is what makes the Topological
  Innovation Rate (Section 14 of the roadmap) measurable rather than estimated.
* Nothing here reads the reward.  Reward sparsity can therefore not leak into
  the mutation operators -- it can only act through selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

INPUT, OUTPUT, HIDDEN, BIAS = "input", "output", "hidden", "bias"


# --------------------------------------------------------------------------- #
@dataclass
class NodeGene:
    key: int
    type: str
    bias: float = 0.0

    def copy(self) -> "NodeGene":
        return NodeGene(self.key, self.type, self.bias)


@dataclass
class ConnGene:
    key: Tuple[int, int]          # (in_node, out_node)
    weight: float
    enabled: bool
    innovation: int

    def copy(self) -> "ConnGene":
        return ConnGene(self.key, self.weight, self.enabled, self.innovation)


# --------------------------------------------------------------------------- #
class InnovationTracker:
    """Global, per-run innovation numbering with per-generation memory.

    Matching structural mutations occurring in the same generation receive the
    same innovation number (the standard NEAT convention).
    """

    def __init__(self, start_node_key: int) -> None:
        self._conn: Dict[Tuple[int, int], int] = {}
        self._node_split: Dict[int, Tuple[int, int, int]] = {}  # conn innov -> (node, in_innov, out_innov)
        self._next_innovation = 0
        self._next_node_key = start_node_key
        # generation bookkeeping
        self.generation = 0
        self._first_seen: Dict[int, int] = {}   # innovation -> generation first created
        self.events_this_generation = 0         # structural mutation *events* attempted+applied

    # -- ids ----------------------------------------------------------------- #
    def _new_innovation(self) -> int:
        i = self._next_innovation
        self._next_innovation += 1
        self._first_seen[i] = self.generation
        return i

    def conn_innovation(self, key: Tuple[int, int]) -> int:
        if key not in self._conn:
            self._conn[key] = self._new_innovation()
        return self._conn[key]

    def node_split(self, conn_innovation: int) -> Tuple[int, int, int]:
        """Return (new_node_key, innov_in, innov_out) for splitting a connection."""
        if conn_innovation not in self._node_split:
            node_key = self._next_node_key
            self._next_node_key += 1
            self._node_split[conn_innovation] = (
                node_key,
                self._new_innovation(),
                self._new_innovation(),
            )
        return self._node_split[conn_innovation]

    # -- generation bookkeeping ---------------------------------------------- #
    def advance_generation(self) -> None:
        self.generation += 1
        self.events_this_generation = 0

    def first_seen(self, innovation: int) -> int:
        return self._first_seen.get(innovation, -1)

    @property
    def n_innovations(self) -> int:
        return self._next_innovation


# --------------------------------------------------------------------------- #
class Genome:
    __slots__ = ("key", "nodes", "conns", "fitness", "adjusted_fitness",
                 "species_id", "birth_generation", "novel_innovations")

    def __init__(self, key: int) -> None:
        self.key = key
        self.nodes: Dict[int, NodeGene] = {}
        self.conns: Dict[Tuple[int, int], ConnGene] = {}
        self.fitness: float = 0.0
        self.adjusted_fitness: float = 0.0
        self.species_id: int = -1
        self.birth_generation: int = 0
        self.novel_innovations: List[int] = []   # innovations created *by* this genome

    # -- construction --------------------------------------------------------- #
    @staticmethod
    def new_minimal(key: int, cfg, tracker: InnovationTracker, rng) -> "Genome":
        g = Genome(key)
        n_in, n_out = cfg.num_inputs, cfg.num_outputs
        # keys: 0..n_in-1 inputs, n_in..n_in+n_out-1 outputs, hidden from n_in+n_out
        for i in range(n_in):
            g.nodes[i] = NodeGene(i, INPUT, 0.0)
        for j in range(n_out):
            k = n_in + j
            g.nodes[k] = NodeGene(k, OUTPUT, rng.gauss(0.0, cfg.bias_replace_std))

        pairs = [(i, n_in + j) for i in range(n_in) for j in range(n_out)]
        if cfg.initial_connection == "sparse":
            n_keep = max(1, int(round(len(pairs) * cfg.initial_sparse_fraction)))
            rng.shuffle(pairs)
            pairs = sorted(pairs[:n_keep])
        for key_ in pairs:
            g.conns[key_] = ConnGene(
                key_, rng.gauss(0.0, cfg.weight_init_std), True,
                tracker.conn_innovation(key_),
            )
        return g

    def copy(self, new_key: Optional[int] = None) -> "Genome":
        g = Genome(self.key if new_key is None else new_key)
        g.nodes = {k: n.copy() for k, n in self.nodes.items()}
        g.conns = {k: c.copy() for k, c in self.conns.items()}
        g.species_id = self.species_id
        g.birth_generation = self.birth_generation
        return g

    # -- topology helpers ----------------------------------------------------- #
    def hidden_keys(self) -> List[int]:
        return sorted(k for k, n in self.nodes.items() if n.type == HIDDEN)

    def enabled_conns(self) -> List[ConnGene]:
        return [c for c in self.conns.values() if c.enabled]

    def creates_cycle(self, src: int, dst: int) -> bool:
        """Would adding src->dst create a cycle in the enabled graph?"""
        if src == dst:
            return True
        # walk forward from dst; if we reach src there is a cycle
        stack = [dst]
        seen = {dst}
        adj: Dict[int, List[int]] = {}
        for c in self.conns.values():
            if c.enabled:
                adj.setdefault(c.key[0], []).append(c.key[1])
        while stack:
            cur = stack.pop()
            for nxt in adj.get(cur, ()):
                if nxt == src:
                    return True
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return False

    # -- mutation -------------------------------------------------------------#
    def mutate(self, cfg, tracker: InnovationTracker, rng) -> int:
        """Apply mutations in place.  Returns the number of *structural* events."""
        structural = 0
        if rng.random() < cfg.add_node_prob:
            structural += int(self._mutate_add_node(cfg, tracker, rng))
        if rng.random() < cfg.add_conn_prob:
            structural += int(self._mutate_add_conn(cfg, tracker, rng))
        if cfg.delete_conn_prob and rng.random() < cfg.delete_conn_prob:
            structural += int(self._mutate_delete_conn(rng))
        if rng.random() < cfg.toggle_enable_prob:
            structural += int(self._mutate_toggle(rng))
        self._mutate_weights(cfg, rng)
        self._mutate_biases(cfg, rng)
        tracker.events_this_generation += structural
        return structural

    def _mutate_weights(self, cfg, rng) -> None:
        for c in self.conns.values():
            if rng.random() < cfg.weight_mutate_prob:
                if rng.random() < cfg.weight_perturb_prob:
                    c.weight += rng.gauss(0.0, cfg.weight_perturb_std)
                else:
                    c.weight = rng.gauss(0.0, cfg.weight_replace_std)
                c.weight = max(-cfg.weight_clamp, min(cfg.weight_clamp, c.weight))

    def _mutate_biases(self, cfg, rng) -> None:
        for n in self.nodes.values():
            if n.type == INPUT:
                continue
            if rng.random() < cfg.bias_mutate_prob:
                if rng.random() < cfg.bias_perturb_prob:
                    n.bias += rng.gauss(0.0, cfg.bias_perturb_std)
                else:
                    n.bias = rng.gauss(0.0, cfg.bias_replace_std)
                n.bias = max(-cfg.bias_clamp, min(cfg.bias_clamp, n.bias))

    def _mutate_add_node(self, cfg, tracker, rng) -> bool:
        candidates = [c for c in self.conns.values() if c.enabled]
        if not candidates:
            return False
        candidates.sort(key=lambda c: c.innovation)          # determinism
        conn = candidates[rng.randrange(len(candidates))]
        node_key, in_innov, out_innov = tracker.node_split(conn.innovation)
        if node_key in self.nodes:
            return False
        conn.enabled = False
        self.nodes[node_key] = NodeGene(node_key, HIDDEN, 0.0)
        src, dst = conn.key
        self.conns[(src, node_key)] = ConnGene((src, node_key), 1.0, True, in_innov)
        self.conns[(node_key, dst)] = ConnGene((node_key, dst), conn.weight, True, out_innov)
        self.novel_innovations.extend([in_innov, out_innov])
        return True

    def _mutate_add_conn(self, cfg, tracker, rng) -> bool:
        node_keys = sorted(self.nodes.keys())
        sources = [k for k in node_keys if self.nodes[k].type != OUTPUT]
        targets = [k for k in node_keys if self.nodes[k].type != INPUT]
        if not sources or not targets:
            return False
        for _ in range(cfg.add_conn_tries):
            src = sources[rng.randrange(len(sources))]
            dst = targets[rng.randrange(len(targets))]
            if (src, dst) in self.conns:
                continue
            if cfg.feedforward and self.creates_cycle(src, dst):
                continue
            innov = tracker.conn_innovation((src, dst))
            self.conns[(src, dst)] = ConnGene(
                (src, dst), rng.gauss(0.0, cfg.weight_init_std), True, innov)
            self.novel_innovations.append(innov)
            return True
        return False

    def _mutate_delete_conn(self, rng) -> bool:
        if len(self.conns) <= 1:
            return False
        keys = sorted(self.conns.keys())
        del self.conns[keys[rng.randrange(len(keys))]]
        return True

    def _mutate_toggle(self, rng) -> bool:
        if not self.conns:
            return False
        keys = sorted(self.conns.keys())
        c = self.conns[keys[rng.randrange(len(keys))]]
        if c.enabled and self.conns and sum(1 for x in self.conns.values() if x.enabled) <= 1:
            return False
        c.enabled = not c.enabled
        return True

    # -- crossover ------------------------------------------------------------ #
    @staticmethod
    def crossover(parent_a: "Genome", parent_b: "Genome", key: int, cfg, rng) -> "Genome":
        """`parent_a` is assumed to be the fitter parent (ties broken by size)."""
        if parent_b.fitness > parent_a.fitness:
            parent_a, parent_b = parent_b, parent_a
        elif parent_b.fitness == parent_a.fitness and len(parent_b.conns) < len(parent_a.conns):
            parent_a, parent_b = parent_b, parent_a

        child = Genome(key)
        b_by_innov = {c.innovation: c for c in parent_b.conns.values()}

        for ck in sorted(parent_a.conns.keys()):
            ca = parent_a.conns[ck]
            cb = b_by_innov.get(ca.innovation)
            if cb is None:                      # disjoint/excess -> from fitter parent
                child.conns[ck] = ca.copy()
            else:
                src = ca if rng.random() < 0.5 else cb
                gene = src.copy()
                gene.key = ca.key
                gene.innovation = ca.innovation
                if not (ca.enabled and cb.enabled):
                    gene.enabled = rng.random() >= cfg.inherit_disabled_prob
                child.conns[ck] = gene

        # nodes: union of nodes referenced by inherited connections + a's I/O
        for nk in sorted(parent_a.nodes.keys()):
            na = parent_a.nodes[nk]
            nb = parent_b.nodes.get(nk)
            node = na.copy()
            if nb is not None and rng.random() < 0.5:
                node.bias = nb.bias
            child.nodes[nk] = node
        for ck in child.conns:
            for nk in ck:
                if nk not in child.nodes:
                    src_node = parent_a.nodes.get(nk) or parent_b.nodes.get(nk)
                    child.nodes[nk] = src_node.copy() if src_node else NodeGene(nk, HIDDEN, 0.0)
        return child

    # -- compatibility --------------------------------------------------------- #
    @staticmethod
    def distance(a: "Genome", b: "Genome", cfg) -> float:
        """Standard NEAT compatibility distance (excess+disjoint pooled, as in
        most modern implementations, plus mean weight difference of matches)."""
        ai = {c.innovation: c for c in a.conns.values()}
        bi = {c.innovation: c for c in b.conns.values()}
        if not ai and not bi:
            return 0.0
        matching = ai.keys() & bi.keys()
        n_match = len(matching)
        n_disjoint = len(ai) + len(bi) - 2 * n_match
        if n_match:
            wdiff = sum(abs(ai[i].weight - bi[i].weight) for i in matching) / n_match
        else:
            wdiff = 0.0
        n = max(len(ai), len(bi), 1)
        # node genes (bias) contribute through the disjoint term only
        d = (cfg.c_disjoint * n_disjoint) / n + cfg.c_weight * wdiff
        return d
