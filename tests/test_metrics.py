# tests/test_metrics.py
import numpy as np
import networkx as nx
from metrics.inequality import gini
from metrics.network_stats import income_homophily, clustering_coefficients

def test_gini_known_values():
    assert gini(np.array([1.0, 1.0, 1.0])) == 0.0
    assert np.isclose(gini(np.array([0.0, 1.0])), 0.5)

def test_income_homophily_line_graph():
    G = nx.path_graph(3)  # edges: (0,1), (1,2)
    z = np.array([1.0, 2.0, 3.0])  # perfectly monotone across edges
    rho = income_homophily(G, z)
    assert np.isfinite(rho) and np.isclose(rho, 1.0)

def test_clustering_triangle():
    G = nx.complete_graph(3)  # triangle
    Cl, Cl_i = clustering_coefficients(G)
    assert np.isclose(Cl, 1.0)
    assert np.allclose(Cl_i, 1.0)