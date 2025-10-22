#!/usr/bin/env python3
"""Generate figures that highlight inequality effects described in the paper.

The script focuses on BR scenarios only and produces two bundles:

1. Fixed network comparison (EXP_LF_br vs EXP_HF_br by default)
   - Group-level outcomes for bottom/top halves of the income distribution
   - Scenario deltas (HF − LF) for the same metrics
   - Aggregate outcomes (e.g., welfare W)

2. Endogenous network comparison (EXP_LE_br vs EXP_HE_br by default)
   - Group-level outcomes and deltas
   - Network assortativity/clustering summaries

Use --help to customise scenario names, labels, ouput filenames, or metric
choices.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"
FIGS = PROJECT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data loaders


def _read_csv_smart(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, header=None)


def _load_deciles(scen: str) -> pd.DataFrame:
    path = RESULTS / scen / "logs" / "panel_by_decile.csv"
    df = _read_csv_smart(path)
    if "decile" not in df.columns:
        for col in df.columns:
            if str(col).lower().startswith("dec"):
                df = df.rename(columns={col: "decile"})
                break
        else:
            cols = list(df.columns)
            if len(cols) > 1:
                df = df.rename(columns={cols[1]: "decile"})
    return df


def _load_aggregates(scen: str) -> pd.DataFrame:
    path = RESULTS / scen / "logs" / "aggregates.csv"
    df = _read_csv_smart(path)
    expected = ["W", "gini_z", "gini_U", "rho_z", "assort", "Cl",
                "share_mean", "y_mean", "x_mean", "U_mean", "t", "rebate", "edges"]
    if df.shape[1] == len(expected):
        df.columns = expected
    if "t" in df.columns:
        df = df.sort_values("t").drop_duplicates("t", keep="last")
    return df


def _last_period_deciles(df: pd.DataFrame) -> pd.DataFrame:
    if "t" in df.columns:
        tmax = df["t"].max()
        df = df[df["t"] == tmax]
    return df.sort_values("decile")


# ---------------------------------------------------------------------------
# Helpers for group computations


MetricName = str
GroupName = str
ScenarioLabel = str


def _ensure_share(df: pd.DataFrame, price: float) -> pd.DataFrame:
    if "share_mean" in df.columns:
        return df
    if "y_mean" in df.columns and "z_net_mean" in df.columns:
        df = df.copy()
        df["share_mean"] = price * pd.to_numeric(df["y_mean"], errors="coerce") / \
            pd.to_numeric(df["z_net_mean"], errors="coerce").replace(0, np.nan)
        return df
    if "y_mean" in df.columns and "z_mean" in df.columns:
        df = df.copy()
        df["share_mean"] = price * pd.to_numeric(df["y_mean"], errors="coerce") / \
            pd.to_numeric(df["z_mean"], errors="coerce").replace(0, np.nan)
        return df
    raise ValueError("Unable to infer share_mean; need either explicit column or (y_mean with z_mean/z_net_mean).")


def _weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    mask = values.notna() & weights.notna()
    if not mask.any():
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def _compute_group_stats(df: pd.DataFrame, price: float, metrics: Mapping[MetricName, str]) -> Dict[MetricName, Dict[GroupName, float]]:
    df = _ensure_share(df, price)
    if "count" not in df.columns:
        df = df.copy()
        df["count"] = 1.0
    df["decile"] = pd.to_numeric(df["decile"], errors="coerce")
    groups = {
        "Bottom 50%": df["decile"] <= 5,
        "Top 50%": df["decile"] >= 6,
    }
    out: Dict[MetricName, Dict[GroupName, float]] = {m: {} for m in metrics}
    for metric in metrics:
        if metric not in df.columns:
            continue
        for gname, mask in groups.items():
            sub = df[mask]
            if sub.empty:
                val = float("nan")
            else:
                val = _weighted_mean(sub[metric], sub["count"])
            out.setdefault(metric, {})[gname] = val
    return out


def _final_aggregate_metrics(df: pd.DataFrame, metrics: Iterable[MetricName]) -> Dict[MetricName, float]:
    if df.empty:
        return {m: float("nan") for m in metrics}
    last = df.iloc[-1]
    result = {}
    for metric in metrics:
        result[metric] = float(pd.to_numeric(last.get(metric), errors="coerce"))
    return result


# ---------------------------------------------------------------------------
# Plotting helpers


def _setup_grid(num_metrics: int) -> Tuple[plt.Figure, np.ndarray]:
    ncols = 2
    nrows = math.ceil(num_metrics / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols / 2, 3.2 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax in axes[num_metrics:]:
        ax.remove()
    return fig, axes[:num_metrics]


def plot_group_levels(
    stats: Mapping[ScenarioLabel, Dict[MetricName, Dict[GroupName, float]]],
    metrics: Mapping[MetricName, str],
    groups: Sequence[GroupName],
    outpath: Path,
    title: str,
):
    labels = list(stats.keys())
    if not labels:
        raise ValueError("No scenario statistics provided for plotting.")
    num_metrics = len(metrics)
    fig, axes = _setup_grid(num_metrics)
    x = np.arange(len(groups), dtype=float)
    width = 0.35 if len(labels) == 2 else 0.8 / max(len(labels), 1)

    for ax, (metric, metric_label) in zip(axes, metrics.items()):
        for idx, label in enumerate(labels):
            values = [stats[label].get(metric, {}).get(g, np.nan) for g in groups]
            offset = (idx - (len(labels) - 1) / 2) * width
            ax.bar(x + offset, values, width=width, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    axes[0].figure.suptitle(title, fontsize=13)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=len(labels))
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_group_deltas(
    base_label: ScenarioLabel,
    compare_label: ScenarioLabel,
    stats: Mapping[ScenarioLabel, Dict[MetricName, Dict[GroupName, float]]],
    metrics: Mapping[MetricName, str],
    groups: Sequence[GroupName],
    outpath: Path,
    title: str,
):
    if base_label not in stats or compare_label not in stats:
        raise ValueError("Both baseline and comparison statistics are required for delta plot.")
    num_metrics = len(metrics)
    fig, axes = _setup_grid(num_metrics)
    x = np.arange(len(groups), dtype=float)

    for ax, (metric, metric_label) in zip(axes, metrics.items()):
        base_vals = stats[base_label].get(metric, {})
        comp_vals = stats[compare_label].get(metric, {})
        deltas = [comp_vals.get(g, np.nan) - base_vals.get(g, np.nan) for g in groups]
        ax.bar(x, deltas, width=0.5, color="#DD8452")
        ax.axhline(0, color="black", linewidth=1)
        for xpos, delta in zip(x, deltas):
            if not np.isnan(delta):
                ax.text(xpos, delta + np.sign(delta or 1) * 0.01, f"{delta:+.3f}",
                        ha="center", va="bottom" if delta >= 0 else "top", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_ylabel(f"Δ {metric_label}")
        ax.set_title(metric_label)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate_summary(
    stats: Mapping[ScenarioLabel, Dict[MetricName, float]],
    metrics: Mapping[MetricName, str],
    outpath: Path,
    title: str,
):
    labels = list(stats.keys())
    x = np.arange(len(metrics))
    width = 0.35 if len(labels) == 2 else 0.8 / max(len(labels), 1)

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for idx, scen_label in enumerate(labels):
        values = [stats[scen_label].get(metric, np.nan) for metric in metrics]
        offset = (idx - (len(labels) - 1) / 2) * width
        ax.bar(x + offset, values, width=width, label=scen_label)
    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics.values()), rotation=20, ha="right")
    ax.set_ylabel("Value (final period)")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI + orchestration


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Narrative inequality figures (BR scenarios only)")
    ap.add_argument("--lf", default="EXP_LF_br", help="Low-inequality fixed-network scenario")
    ap.add_argument("--hf", default="EXP_HF_br", help="High-inequality fixed-network scenario")
    ap.add_argument("--le", default="EXP_LE_br", help="Low-inequality endogenous-network scenario")
    ap.add_argument("--he", default="EXP_HE_br", help="High-inequality endogenous-network scenario")
    ap.add_argument("--lf_label", default="LF", help="Legend label for --lf")
    ap.add_argument("--hf_label", default="HF", help="Legend label for --hf")
    ap.add_argument("--le_label", default="LE", help="Legend label for --le")
    ap.add_argument("--he_label", default="HE", help="Legend label for --he")
    ap.add_argument("--price", type=float, default=2.0, help="Status good price p (used when reconstructing share)")
    ap.add_argument("--out_fixed_levels", default="inequal_fixed_group_levels.png",
                    help="Output PNG for fixed-network group metrics")
    ap.add_argument("--out_fixed_deltas", default="inequal_fixed_group_deltas.png",
                    help="Output PNG for fixed-network deltas")
    ap.add_argument("--out_fixed_aggregate", default="inequal_fixed_aggregate.png",
                    help="Output PNG for fixed-network aggregate summary")
    ap.add_argument("--out_endog_levels", default="inequal_endog_group_levels.png",
                    help="Output PNG for endogenous-network group metrics")
    ap.add_argument("--out_endog_deltas", default="inequal_endog_group_deltas.png",
                    help="Output PNG for endogenous-network deltas")
    ap.add_argument("--out_endog_network", default="inequal_endog_network_metrics.png",
                    help="Output PNG for endogenous-network mixing metrics")
    return ap.parse_args()


def _collect_group_stats(
    scenarios: Sequence[Tuple[str, ScenarioLabel]],
    price: float,
    metrics: Mapping[MetricName, str],
) -> Dict[ScenarioLabel, Dict[MetricName, Dict[GroupName, float]]]:
    stats: Dict[ScenarioLabel, Dict[MetricName, Dict[GroupName, float]]] = {}
    for scen, label in scenarios:
        try:
            df = _last_period_deciles(_load_deciles(scen))
        except FileNotFoundError as exc:
            print(f"[SKIP] {label}: {exc}")
            continue
        try:
            stats[label] = _compute_group_stats(df, price, metrics)
        except Exception as exc:
            print(f"[WARN] {label}: {exc}")
    return stats


def _collect_aggregate_stats(
    scenarios: Sequence[Tuple[str, ScenarioLabel]],
    metrics: Mapping[MetricName, str],
) -> Dict[ScenarioLabel, Dict[MetricName, float]]:
    out: Dict[ScenarioLabel, Dict[MetricName, float]] = {}
    for scen, label in scenarios:
        try:
            df = _load_aggregates(scen)
        except FileNotFoundError as exc:
            print(f"[SKIP] aggregates {label}: {exc}")
            continue
        out[label] = _final_aggregate_metrics(df, metrics)
    return out


def main():
    args = parse_args()

    group_metrics = {
        "share_mean": "Status share",
        "y_mean": "Status level (y)",
        "phi_mean": "Relative status (phi)",
        "U_mean": "Utility / welfare",
    }
    groups = ["Bottom 50%", "Top 50%"]

    # Fixed network LF vs HF -------------------------------------------------
    fixed_stats = _collect_group_stats([(args.lf, args.lf_label), (args.hf, args.hf_label)], args.price, group_metrics)
    if len(fixed_stats) >= 2:
        plot_group_levels(
            fixed_stats,
            group_metrics,
            groups,
            FIGS / args.out_fixed_levels,
            title=f"Fixed network: {args.lf_label} vs {args.hf_label}",
        )
        plot_group_deltas(
            base_label=args.lf_label,
            compare_label=args.hf_label,
            stats=fixed_stats,
            metrics=group_metrics,
            groups=groups,
            outpath=FIGS / args.out_fixed_deltas,
            title=f"Δ ({args.hf_label} − {args.lf_label}) by income half",
        )
    else:
        print("[SKIP] Fixed network group plots — insufficient data.")

    fixed_agg_metrics = {
        "W": "Aggregate welfare W",
        "share_mean": "Mean share",
        "U_mean": "Mean utility",
        "gini_U": "Gini(U)",
    }
    fixed_agg_stats = _collect_aggregate_stats([(args.lf, args.lf_label), (args.hf, args.hf_label)], fixed_agg_metrics)
    if len(fixed_agg_stats) >= 2:
        plot_aggregate_summary(
            fixed_agg_stats,
            fixed_agg_metrics,
            FIGS / args.out_fixed_aggregate,
            title=f"Fixed network aggregates: {args.lf_label} vs {args.hf_label}",
        )
    else:
        print("[SKIP] Fixed network aggregate plot — insufficient data.")

    # Endogenous network LE vs HE -------------------------------------------
    endog_stats = _collect_group_stats([(args.le, args.le_label), (args.he, args.he_label)], args.price, group_metrics)
    if len(endog_stats) >= 2:
        plot_group_levels(
            endog_stats,
            group_metrics,
            groups,
            FIGS / args.out_endog_levels,
            title=f"Endogenous network: {args.le_label} vs {args.he_label}",
        )
        plot_group_deltas(
            base_label=args.le_label,
            compare_label=args.he_label,
            stats=endog_stats,
            metrics=group_metrics,
            groups=groups,
            outpath=FIGS / args.out_endog_deltas,
            title=f"Δ ({args.he_label} − {args.le_label}) by income half",
        )
    else:
        print("[SKIP] Endogenous network group plots — insufficient data.")

    network_metrics = {
        "rho_z": "Income–income corr (rho_z)",
        "assort": "Degree assortativity",
        "Cl": "Clustering coefficient",
    }
    endog_network_stats = _collect_aggregate_stats([(args.le, args.le_label), (args.he, args.he_label)], network_metrics)
    if len(endog_network_stats) >= 2:
        plot_aggregate_summary(
            endog_network_stats,
            network_metrics,
            FIGS / args.out_endog_network,
            title=f"Network structure: {args.le_label} vs {args.he_label}",
        )
    else:
        print("[SKIP] Endogenous network metrics plot — insufficient data.")

    print(f"\nAll done. Figures written to {FIGS}")


if __name__ == "__main__":
    main()
