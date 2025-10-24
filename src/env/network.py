# src/env/network.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple
import numpy as np
import networkx as nx

from env.scenarios import NetworkParams


@dataclass
class HomoParams:
    on: bool = False
    c: float = 25.0            # income scale: larger c = weaker income sorting
    lambda_triadic: float = 2.0  # weight for triadic closure bonus

@dataclass
class SocialNetwork:
    """
    Thin wrapper over a NetworkX Graph with helper methods used by the env.
    Handles fixed vs. endogenous (dynamic) networks.
    """
    S: int
    params: NetworkParams
    rng: np.random.Generator

    def __post_init__(self):
        # Dataclasses call __post_init__ with only `self`; fields are already set.
        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.S))
        self._init_graph()
        # hold attributes updated by env each period
        self._z = np.zeros(self.S)
        # homophily settings (tolerant if absent)
        hp = getattr(self.params.dynamics, "homophily", None)
        self.homo = HomoParams(**hp) if isinstance(hp, dict) else HomoParams(on=False)


    # ---------- Initialization ----------
    def _init_graph(self):
        t = self.params.type
        if t == "erdos_renyi":
            p = self.params.erdos_renyi.p_edge
            self.G = nx.erdos_renyi_graph(self.S, p, seed=int(self.rng.integers(0, 2**32 - 1)))
        elif t == "barabasi_albert":
            m = self.params.barabasi_albert.m_attach
            m = max(1, min(m, self.S - 1))
            self.G = nx.barabasi_albert_graph(self.S, m, seed=int(self.rng.integers(0, 2**32 - 1)))
        elif t == "watts_strogatz":
            k = self.params.watts_strogatz.k_nei
            beta = self.params.watts_strogatz.beta_rewire
            k = max(2, min(k, self.S - (self.S % 2 == 1)))  # ensure valid even k
            self.G = nx.watts_strogatz_graph(self.S, k, beta, seed=int(self.rng.integers(0, 2**32 - 1)))
        else:
            raise ValueError(f"Unknown network type: {t}")

        # Ensure simple undirected graph with no self-loops
        self.G.remove_edges_from(nx.selfloop_edges(self.G))

    # ---------- Queries ----------
    def neighbors(self, i: int) -> List[int]:
        return list(self.G.neighbors(i))

    def degree(self, i: int) -> int:
        return self.G.degree(i)

    def edges(self) -> Iterable[Tuple[int, int]]:
        return self.G.edges()

    # ---------- Neighborhood stats given status vector y ----------
    def avg_neighbor_y(self, i: int, y: np.ndarray) -> float:
        nbrs = self.neighbors(i)
        if not nbrs:
            return 0.0
        return float(np.mean(y[nbrs]))

    def max_neighbor_y(self, i: int, y: np.ndarray) -> float:
        nbrs = self.neighbors(i)
        if not nbrs:
            return 0.0
        return float(np.max(y[nbrs]))
    
    # NEW: called by env each step before maybe_update()
    def update_incomes(self, z_array: np.ndarray):
        self._z = np.asarray(z_array)

    def _candidate_add_partner(self, i: int) -> int:
        """Pick j != i not already linked, with income- and triadic-biased weights."""
        all_j = [j for j in range(self.S) if j != i and not self.G.has_edge(i, j)]
        if not all_j:
            return i
        if not self.homo.on:
            return int(self.rng.choice(all_j))

        zi = float(self._z[i])
        # income similarity weights
        inc_w = np.exp(-np.abs(self._z[all_j] - zi) / max(1e-6, self.homo.c))
        # triadic closure bonus = (#common neighbors)
        nbs_i = set(self.G.neighbors(i))
        tri_w = np.array([len(nbs_i.intersection(set(self.G.neighbors(j)))) for j in all_j], dtype=float)
        w = inc_w * (1.0 + self.homo.lambda_triadic * tri_w)
        if np.all(w <= 0):
            j = int(self.rng.choice(all_j))
        else:
            w = w / w.sum()
            j = int(self.rng.choice(all_j, p=w))
        return j

    # ---------- Endogenous link dynamics (simple, well-behaved) ----------
    def maybe_update(self):
        """
        Endogenous link dynamics (run only if params.dynamic=True).

        Schedule: execute once every `reevaluate_every` ticks.
        Drop step: each existing edge is removed independently with probability `drop_prob`.
        Add step: nodes are shuffled; each node i, with probability `add_prob` and if
                deg(i) < max_degree, proposes ONE link to j selected by
                `_candidate_add_partner(i)` which embeds homophily/triadic closure
                when enabled. Self-loops and duplicates are disallowed, and j must
                also satisfy deg(j) < max_degree.
        """
        if not self.params.dynamic:
            return

        rules = self.params.dynamics
        # honor reevaluation cadence
        tick = getattr(self, "_tick", 0)
        setattr(self, "_tick", tick + 1)
        if (tick % max(1, int(getattr(rules, "reevaluate_every", 1)))) != 0:
            return

        # -------- Drop existing edges --------
        to_drop = [(u, v) for (u, v) in list(self.G.edges())
                if self.rng.random() < float(rules.drop_prob)]
        if to_drop:
            self.G.remove_edges_from(to_drop)

        # -------- Add new edges (at most one proposal per node) --------
        max_deg = int(rules.max_degree)
        # iterate nodes in random order
        nodes = list(self.rng.permutation(self.S))
        for i in nodes:
            if self.G.degree(i) >= max_deg:
                continue
            if self.rng.random() >= float(rules.add_prob):
                continue

            j = self._candidate_add_partner(i)   # homophily/triadic-aware if enabled
            if j == i:
                continue
            if self.G.has_edge(i, j):
                continue
            if self.G.degree(j) >= max_deg:
                continue

            # add the undirected edge
            self.G.add_edge(i, j)

            