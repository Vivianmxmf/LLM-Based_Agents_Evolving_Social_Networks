# src/metrics/network_stats.py
from __future__ import annotations
import numpy as np
import networkx as nx
from typing import Tuple

def _edge_index(G: nx.Graph) -> np.ndarray:
    """Return E×2 array of undirected edge endpoints (each edge once)."""
    if G.number_of_edges() == 0:
        return np.empty((0, 2), dtype=int)
    return np.array(list(G.edges()), dtype=int)

def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson corr for paired arrays; returns np.nan on degenerate variance."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or b.size == 0:
        return float("nan")
    da = a - a.mean()
    db = b - b.mean()
    denom = np.sqrt(np.sum(da * da)) * np.sqrt(np.sum(db * db))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(da * db) / denom)

def income_homophily(G: nx.Graph, z: np.ndarray) -> float:
    """
    Pearson correlation of incomes across edges: corr(z_i, z_j) for (i,j) in E.
    """
    E = _edge_index(G)
    if E.size == 0:
        return float("nan")
    return _pearson(z[E[:, 0]], z[E[:, 1]])

def degree_assortativity(G: nx.Graph) -> float:
    """
    Pearson correlation of degrees across edges: corr(deg_i, deg_j) for (i,j) in E.
    """
    E = _edge_index(G)
    if E.size == 0:
        return float("nan")
    deg = dict(G.degree())
    di = np.array([deg[i] for i in E[:, 0]], dtype=float)
    dj = np.array([deg[j] for j in E[:, 1]], dtype=float)
    return _pearson(di, dj)

def clustering_coefficients(G: nx.Graph) -> Tuple[float, np.ndarray]:
    """
    Network-level clustering = average of node-level Cl_i.
    Returns (Cl, Cl_i array aligned with node ordering 0..S-1 if present).
    """
    if G.number_of_nodes() == 0:
        return float("nan"), np.array([])
    cl_map = nx.clustering(G)  # uses triangles / connected triplets formula
    # Align to node ids (assumed 0..S-1 as constructed in SocialNetwork)
    S = G.number_of_nodes()
    cl_vec = np.array([cl_map.get(i, 0.0) for i in range(S)], dtype=float)
    return float(np.mean(cl_vec)), cl_vec