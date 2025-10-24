# [UPDATED] src/metrics/logging.py  — add percentile panel writer
from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

from .inequality import gini, welfare, status_share
from .network_stats import income_homophily, degree_assortativity, clustering_coefficients
from .percentiles import group_means_by_decile  # [NEW]

def compute_step_metrics(G: nx.Graph, z: np.ndarray, x: np.ndarray, y: np.ndarray, U: np.ndarray, p: float) -> dict:
    W = welfare(U)
    gz = gini(z)
    gU = gini(U)
    rho_z = income_homophily(G, z)
    assort = degree_assortativity(G)
    Cl, _ = clustering_coefficients(G)
    share = status_share(p, y, z)
    return {
        "W": W,
        "gini_z": gz,
        "gini_U": gU,
        "rho_z": rho_z,
        "assort": assort,
        "Cl": Cl,
        "share_mean": float(np.mean(share)),
        "y_mean": float(np.mean(y)),
        "x_mean": float(np.mean(x)),
        "U_mean": float(np.mean(U)),
    }

def write_panel(results_dir: Path, t: int, z: np.ndarray, x: np.ndarray, y: np.ndarray, U: np.ndarray, share: np.ndarray) -> None:
    panel_dir = results_dir / "logs"
    panel_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "t": t,
        "i": np.arange(z.size),
        "z": z,
        "x": x,
        "y": y,
        "U": U,
        "share": share,
    })
    path = panel_dir / "panel.csv"
    header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=header)

def write_step_summary(results_dir: Path, rows: list[dict]) -> None:
    out_dir = results_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "aggregates.csv", index=False)

def _deciles_from_income(z_pre: np.ndarray) -> np.ndarray:
    """Percentile ranks based on pre-tax income; stable across policy toggles."""
    order = z_pre.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(z_pre) + 1)
    pct = (ranks - 1) / (len(z_pre) - 1 + 1e-12)  # [0,1]
    return np.floor(10 * pct).clip(0, 9).astype(int) + 1  # 1..10

def write_percentile_panel(
    results_dir: Path,
    t: int,
    z_pre: np.ndarray,
    z_net: np.ndarray,
    y: np.ndarray,
    phi: np.ndarray,
    U: np.ndarray,
    p: float,
    z_ref: np.ndarray | None = None
) -> None:
    """Append one row per decile with BOTH pre-tax and net-income denominators.
       Output: results/<scenario>/logs/<SCENARIO>_panel_by_decile.csv
    """
    logs = results_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    out = logs / f"{results_dir.name}_panel_by_decile.csv"

    dec = _deciles_from_income(z_pre)
    df = pd.DataFrame({
        "t": t,
        "decile": dec,
        "z_mean": z_pre,
        "z_net_mean": z_net,
        "y_mean": y,
        "phi_mean": phi,
        "U_mean": U,
        "share_pretax": np.divide(p * y, z_pre, out=np.zeros_like(y), where=z_pre>0),
        "share_net":    np.divide(p * y, z_net, out=np.zeros_like(y), where=z_net>0),
    })

    # aggregate within decile (means)
    g = df.groupby(["t", "decile"], as_index=False).mean(numeric_only=True)

    # consistent column names used by your plotting scripts
    g = g.rename(columns={
        "share_pretax": "share_pretax_mean",
        "share_net":    "share_net_mean",
    })
    # append
    header = not out.exists()
    g.to_csv(out, mode="a", header=header, index=False)


# def write_percentile_panel(results_dir: Path, t: int, z: np.ndarray, y: np.ndarray, phi: np.ndarray, U: np.ndarray, p: float, z_ref: np.ndarray | None = None) -> None:
#     """
#     Append a decile-level panel at time t with means for z, y, φ, U, share.
#     """
#     dec_df = group_means_by_decile(z=z, y=y, phi=phi, U=U, p=p, z_ref=z_ref)
#     dec_df.insert(0, "t", t)
#     path = (results_dir / "logs")
#     path.mkdir(parents=True, exist_ok=True)
#     f = path / "panel_by_decile.csv"
#     header = not f.exists()
#     dec_df.to_csv(f, mode="a", index=False, header=header)