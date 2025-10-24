# src/env/utility.py
from __future__ import annotations
import numpy as np

def goods_utility(x: np.ndarray, y: np.ndarray, xi: float) -> np.ndarray:
    """
    Cobb–Douglas u_goods = x^xi * y^(1 - xi)  (elementwise).
    Safe for x,y>=0. Returns zeros if either good is zero.
    """
    # avoid numerical warnings at 0^0: define 0^a = 0 for a>0
    ug = np.zeros_like(x, dtype=float)
    positive = (x > 0) & (y > 0)
    ug[positive] = np.power(x[positive], xi) * np.power(y[positive], 1.0 - xi)
    return ug

def relative_status(y: np.ndarray, ybar: np.ndarray) -> np.ndarray:
    """
    φ_i = y_i - avg_neighbor_y_i  (vectorized via provided ybar).
    """
    return y - ybar

def utility_total(
    x: np.ndarray,
    y: np.ndarray,
    xi: float,
    gamma: np.ndarray,
    phi: np.ndarray,
    tau: np.ndarray,
    y_max_nbr: np.ndarray,
) -> np.ndarray:
    """
    U_i = x_i^xi * y_i^(1−xi) + γ_i * φ_i + τ_i * max_{k in N(i)} y_k
    All inputs are 1D arrays of length S.
    """
    ug = goods_utility(x, y, xi)
    return ug + gamma * phi + tau * y_max_nbr