# tests/test_link_rules.py
import numpy as np
from env.scenarios import NetworkParams, ERParams, DynamicRules
from env.network import SocialNetwork
import networkx as nx

def test_endogenous_adds_edges_when_prob_one():
    S = 20
    rng = np.random.default_rng(123)
    params = NetworkParams(
        dynamic=True,
        type="erdos_renyi",
        erdos_renyi=ERParams(p_edge=0.0),              # start empty
        dynamics=DynamicRules(reevaluate_every=1, add_prob=1.0, drop_prob=0.0, max_degree=10),
    )
    net = SocialNetwork(S=S, params=params, rng=rng)
    assert net.G.number_of_edges() == 0  # start empty
    net.maybe_update()
    # with add_prob=1.0, each node proposes a link -> expect some edges
    assert net.G.number_of_edges() > 0
    # sanity: no self-loops
    assert len(list(nx.selfloop_edges(net.G))) == 0