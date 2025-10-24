# src/metrics/inequality.py
from __future__ import annotations
import numpy as np

def gini(x: np.ndarray) -> float:
    """
    G(x) = sum_i sum_j |x_i - x_j| / (2 * S^2 * mean(x))
    Returns 0.0 for empty arrays or if mean==0 and all entries equal.
    """
    x = np.asarray(x, dtype=float).ravel()
    S = x.size
    if S == 0:
        return 0.0
    mu = float(np.mean(x))
    if mu == 0.0:
        # all zeros -> perfect equality
        return 0.0
    diffsum = np.abs(x[:, None] - x[None, :]).sum()
    return float(diffsum / (2.0 * (S**2) * mu))

def welfare(U: np.ndarray) -> float:
    """Aggregate welfare W = sum_i U_i."""
    return float(np.sum(U))

def status_share(p: float, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Share_i = (p * y_i) / z_i  (uses *income* z_i as in the paper).
    Safe-divide with zeros mapped to 0.0.
    """
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = (p * y) / z
    share = np.where(np.isfinite(share), share, 0.0)
    share = np.clip(share, 0.0, np.inf)
    return share