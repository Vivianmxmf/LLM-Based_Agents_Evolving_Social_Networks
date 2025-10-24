# tests/test_percentiles.py
import numpy as np
from metrics.percentiles import rank_percentiles, assign_deciles, group_means_by_decile

def test_assign_deciles_bounds_and_counts():
    z = np.linspace(1, 100, 1000)
    dec = assign_deciles(z)
    assert dec.min() == 1 and dec.max() == 10
    # Each decile should have ~100 elems (allow small tolerance)
    counts = [int((dec == d).sum()) for d in range(1, 11)]
    assert sum(counts) == z.size
    assert all(abs(c - 100) <= 2 for c in counts)  # stable bucketing tolerance

def test_group_means_basic():
    z = np.array([1, 2, 3, 4, 100], dtype=float)
    y = np.array([1, 1, 1, 1, 10], dtype=float)
    phi = np.zeros_like(y)
    U = y.copy()
    p = 2.0
    df = group_means_by_decile(z, y, phi, U, p)
    assert set(df.columns) == {"decile", "count", "z_mean", "y_mean", "phi_mean", "U_mean", "share_mean"}
    assert df["count"].sum() == z.size
    # Top decile should include the outlier 100
    top_cnt = df.loc[df["decile"] == 10, "count"].iloc[0]
    assert int(top_cnt) >= 1