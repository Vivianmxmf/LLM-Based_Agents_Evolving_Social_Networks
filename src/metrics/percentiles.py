# src/metrics/percentiles.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple

def rank_percentiles(values: np.ndarray) -> np.ndarray:
    """
    Rank-based percentiles π_i ∈ [0,100] (average-rank for ties via stable argsort trick).
    Matches the spirit of the tax routine.
    """
    x = np.asarray(values, dtype=float).ravel()
    n = x.size
    if n <= 1:
        return np.zeros_like(x)
    order = x.argsort(kind="mergesort")        # stable
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    pi = 100.0 * (ranks - 1.0) / (n - 1.0)
    return pi

def assign_deciles(values: np.ndarray) -> np.ndarray:
    """
    Map values to income deciles d ∈ {1,…,10} using rank_percentiles.
    """
    pi = rank_percentiles(values)
    d = (pi // 10).astype(int) + 1
    d = np.minimum(d, 10)
    return d

def group_means_by_decile(
    z: np.ndarray, y: np.ndarray, phi: np.ndarray, U: np.ndarray, p: float, z_ref: np.ndarray | None = None
) -> pd.DataFrame:
    """
    Return DataFrame with one row per decile: columns [decile, count, z_mean, y_mean, phi_mean, U_mean, share_mean]
    where share_i = (p * y_i) / z_i (safe divide). If z_ref is provided, deciles
    are assigned based on z_ref (e.g., pre-tax income) instead of z.
    """
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    phi = np.asarray(phi, dtype=float)
    U = np.asarray(U, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        share = (p * y) / z
    share[~np.isfinite(share)] = 0.0
    share = np.clip(share, 0.0, np.inf)

    base_for_deciles = z_ref if z_ref is not None else z
    dec = assign_deciles(base_for_deciles)
    out = []
    for d in range(1, 11):
        mask = (dec == d)
        cnt = int(mask.sum())
        if cnt == 0:
            row = dict(decile=d, count=0, z_mean=np.nan, y_mean=np.nan, phi_mean=np.nan, U_mean=np.nan, share_mean=np.nan)
        else:
            row = dict(
                decile=d,
                count=cnt,
                z_mean=float(z[mask].mean()),
                y_mean=float(y[mask].mean()),
                phi_mean=float(phi[mask].mean()),
                U_mean=float(U[mask].mean()),
                share_mean=float(share[mask].mean()),
            )
        out.append(row)
    return pd.DataFrame(out, columns=["decile", "count", "z_mean", "y_mean", "phi_mean", "U_mean", "share_mean"])