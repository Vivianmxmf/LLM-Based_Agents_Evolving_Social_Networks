# tests/test_utility.py
import numpy as np
from env.utility import goods_utility, utility_total

def test_goods_utility_basic():
    x = np.array([1.0, 2.0, 0.0])
    y = np.array([1.0, 2.0, 5.0])
    xi = 0.6
    ug = goods_utility(x, y, xi)
    assert np.isclose(ug[0], 1.0)  # 1^0.6 * 1^0.4 = 1
    assert ug[2] == 0.0            # zero consumption -> zero utility

def test_utility_total_reduces_to_goods_when_no_network_terms():
    x = np.array([2.0, 3.0])
    y = np.array([1.0, 1.0])
    xi = 0.5
    gamma = np.zeros(2)
    phi = np.zeros(2)
    tau = np.zeros(2)
    y_max = np.zeros(2)
    U = utility_total(x, y, xi, gamma, phi, tau, y_max)
    ug = goods_utility(x, y, xi)
    assert np.allclose(U, ug)