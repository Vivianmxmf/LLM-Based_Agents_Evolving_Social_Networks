#!/usr/bin/env python3
"""
Generate the standard figure bundle using BR scenarios only.
The script mirrors scripts/make_llm_figs.py but drops LLM dependencies.
It reads CSVs under results/<scenario>/logs and writes PNGs under figs/.

Usage (defaults assume standard scenario names):
    python scripts/make_br_figs.py
Override scenario sets via CLI flags; run with --help for details.
"""

from __future__ import annotations

import argparse
import glob
import json
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

# ---------- Utilities ------------------------------------------------------

def _read_csv_smart(path: Path) -> pd.DataFrame:
    """Read CSV with or without header and return a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, header=None)


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
            ren = {}
            if len(cols) >= 2:
                ren[cols[1]] = "decile"
            if len(cols) >= 3:
                ren[cols[2]] = "share_mean"
            if len(cols) >= 4:
                ren[cols[3]] = "y_mean"
            if len(cols) >= 6:
                ren[cols[5]] = "U_mean"
            if len(cols) >= 7:
                ren[cols[6]] = "z_mean"
            df = df.rename(columns=ren)
    return df


def _last_period_deciles(df: pd.DataFrame) -> pd.DataFrame:
    if "t" in df.columns:
        tmax = df["t"].max()
        return df[df["t"] == tmax].sort_values("decile")
    return df.sort_values("decile")


def _mean_ci(series: pd.Series) -> Tuple[float, float]:
    series = pd.to_numeric(series, errors="coerce").dropna()
    n = len(series)
    if n == 0:
        return float("nan"), float("nan")
    m = float(series.mean())
    if n == 1:
        return m, float("nan")
    s = float(series.std(ddof=1))
    ci = 1.96 * s / math.sqrt(n)
    return m, ci


def _savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _maybe(name: str, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except FileNotFoundError as e:
        print(f"[SKIP] {name} — missing file: {e}")
    except Exception as e:
        print(f"[WARN] {name} — {e}")


def _share_from_decile_df(df: pd.DataFrame, price: float) -> pd.Series:
    if "share_mean" in df.columns:
        return pd.to_numeric(df["share_mean"], errors="coerce")
    if {"y_mean", "z_net_mean"}.issubset(df.columns):
        y = pd.to_numeric(df["y_mean"], errors="coerce")
        z = pd.to_numeric(df["z_net_mean"], errors="coerce")
        return price * y / z.replace(0, np.nan)
    if {"y_mean", "z_mean"}.issubset(df.columns):
        y = pd.to_numeric(df["y_mean"], errors="coerce")
        z = pd.to_numeric(df["z_mean"], errors="coerce")
        return price * y / z.replace(0, np.nan)
    raise ValueError("Decile dataframe lacks share information; expected share_mean or (y_mean & z_[net_]mean).")


# ---------- Scenario helpers ----------------------------------------------

def _parse_scenario_pairs(csv_string: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for token in csv_string.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" in token:
            scen, label = token.split("=", 1)
        else:
            scen, label = token, token
        pairs.append((scen.strip(), label.strip()))
    return pairs


def _filter_by_file(pairs: Sequence[Tuple[str, str]], relative_file: str) -> List[Tuple[str, str]]:
    good: List[Tuple[str, str]] = []
    for scen, label in pairs:
        target = RESULTS / scen / relative_file
        if target.exists():
            good.append((scen, label))
    return good


# ---------- Figure helpers -------------------------------------------------

def plot_decile_share(scenarios: Sequence[Tuple[str, str]], price: float, title: str, outname: str):
    scenarios = list(scenarios)
    if not scenarios:
        raise FileNotFoundError("No scenarios provided for decile share plot.")

    series = []
    common = None
    for scen, label in scenarios:
        df = _last_period_deciles(_load_deciles(scen))
        shares = _share_from_decile_df(df, price)
        decs = np.array(sorted(pd.to_numeric(df["decile"], errors="coerce").dropna().unique()))
        if common is None:
            common = decs
        else:
            common = np.intersect1d(common, decs)
        series.append((scen, label, df, shares))

    if common is None or len(common) == 0:
        raise ValueError("No overlapping deciles across the requested scenarios.")

    x = np.array(sorted(common))
    plt.figure(figsize=(6.2, 4.2))
    for scen, label, df, shares in series:
        subset = df[df["decile"].isin(x)].sort_values("decile")
        y = pd.to_numeric(shares.loc[subset.index], errors="coerce").to_numpy()
        if len(y) != len(x):
            raise ValueError(f"{scen}: could not align deciles for plotting.")
        plt.plot(x, y, marker="o", label=label)
    plt.xticks(list(x))
    plt.xlabel("Income decile")
    plt.ylabel("Status share mean (p·y / z or z_net)")
    plt.title(title)
    plt.legend()
    _savefig(FIGS / outname)


def plot_grid_ci(prefix: str, k_vals: Iterable[int], outname: str, title: str):
    metrics = []
    for K in k_vals:
        rows = []
        for path in sorted(RESULTS.glob(f"{prefix}{K}_s*/logs/aggregates.csv")):
            scen = path.parts[-3]
            df = _load_aggregates(scen)
            last = df.iloc[-1]
            rows.append({
                "W": float(last.get("W", np.nan)),
                "gini_U": float(last.get("gini_U", np.nan)),
                "share_mean": float(last.get("share_mean", np.nan)),
            })
        solo = RESULTS / f"{prefix}{K}" / "logs" / "aggregates.csv"
        if not rows and solo.exists():
            df = _load_aggregates(f"{prefix}{K}")
            last = df.iloc[-1]
            rows.append({
                "W": float(last.get("W", np.nan)),
                "gini_U": float(last.get("gini_U", np.nan)),
                "share_mean": float(last.get("share_mean", np.nan)),
            })
        df = pd.DataFrame(rows)
        if df.empty:
            metrics.append((K, None))
        else:
            mW, cW = _mean_ci(df["W"])
            mG, cG = _mean_ci(df["gini_U"])
            mS, cS = _mean_ci(df["share_mean"])
            metrics.append((K, {"W": (mW, cW), "gini_U": (mG, cG), "share_mean": (mS, cS), "n": len(df)}))

    have = [m for _, m in metrics if m is not None]
    if not have:
        raise FileNotFoundError(f"No aggregates found for prefix {prefix}")

    Ks = [K for K, m in metrics if m is not None]
    valsW = [m["W"][0] for _, m in metrics if m is not None]
    errW = [m["W"][1] for _, m in metrics if m is not None]
    valsG = [m["gini_U"][0] for _, m in metrics if m is not None]
    errG = [m["gini_U"][1] for _, m in metrics if m is not None]
    valsS = [m["share_mean"][0] for _, m in metrics if m is not None]
    errS = [m["share_mean"][1] for _, m in metrics if m is not None]
    ns = [m["n"] for _, m in metrics if m is not None]

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6))
    for ax, vals, errs, name in zip(axes,
                                     [valsW, valsG, valsS],
                                     [errW, errG, errS],
                                     ["W", "Gini(U)", "Mean share"]):
        ax.bar([str(k) for k in Ks], vals, yerr=errs, capsize=5)
        ax.set_title(name)
    fig.suptitle(f"{title} (n={ns})", fontsize=12)
    _savefig(FIGS / outname)


def plot_share_delta(base: Tuple[str, str], variant: Tuple[str, str], price: float, outname: str, title: str):
    base_df = _last_period_deciles(_load_deciles(base[0]))
    var_df = _last_period_deciles(_load_deciles(variant[0]))

    common = np.intersect1d(base_df["decile"].unique(), var_df["decile"].unique())
    if len(common) == 0:
        raise ValueError("No overlapping deciles for delta plot")

    base_df = base_df[base_df["decile"].isin(common)].sort_values("decile")
    var_df = var_df[var_df["decile"].isin(common)].sort_values("decile")

    base_share = _share_from_decile_df(base_df, price).to_numpy()
    var_share = _share_from_decile_df(var_df, price).to_numpy()
    delta = var_share - base_share

    plt.figure(figsize=(6.2, 4.2))
    plt.axhline(0, color="gray", lw=1)
    plt.plot(common, delta, marker="o")
    plt.xticks(list(common))
    plt.xlabel("Income decile")
    plt.ylabel("Δ share (variant − base)")
    plt.title(f"{title}\n{variant[1]} minus {base[1]}")
    _savefig(FIGS / outname)


def plot_telemetry(scenarios: Sequence[str], outname: str, title: str):
    data = []
    for s in scenarios:
        js = RESULTS / s / "artifacts" / "llm_stats.json"
        if js.exists():
            try:
                obj = json.loads(js.read_text())
                data.append({
                    "scenario": s,
                    "success": float(obj.get("success_rate", np.nan)),
                    "fallback": float(obj.get("fallback_rate", np.nan)),
                    "latency": float(obj.get("latency_mean_s", np.nan)),
                })
            except Exception:
                pass
    if not data:
        raise FileNotFoundError("No telemetry JSONs located for the requested scenarios.")

    df = pd.DataFrame(data)
    idx = np.arange(len(df))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8.5, 4.2))
    ax1.bar(idx - width / 2, df["success"].values * 100, width=width, label="Success %")
    ax1.bar(idx + width / 2, df["fallback"].values * 100, width=width, label="Fallback %")
    ax1.set_xticks(idx)
    ax1.set_xticklabels(df["scenario"], rotation=30, ha="right")
    ax1.set_ylabel("%")
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(idx, df["latency"].values, marker="o", label="Mean latency (s)")
    ax2.set_ylabel("Seconds")
    fig.suptitle(title, fontsize=12)
    _savefig(FIGS / outname)


def plot_time_series(scenarios: Sequence[Tuple[str, str]], price: float, outname: str, title: str):
    scenarios = list(scenarios)
    if not scenarios:
        raise FileNotFoundError("No scenarios provided for time-series plot.")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for scen, label in scenarios:
        df = _load_aggregates(scen)
        if "t" not in df.columns:
            raise ValueError(f"{scen}: aggregates missing 't' column")
        df = df.sort_values("t")
        if "share_mean" in df.columns:
            share = pd.to_numeric(df["share_mean"], errors="coerce")
        else:
            if "y_mean" not in df.columns:
                raise ValueError(f"{scen}: aggregates missing 'y_mean' for share computation")
            y = pd.to_numeric(df["y_mean"], errors="coerce")
            if "z_net_mean" in df.columns:
                z = pd.to_numeric(df["z_net_mean"], errors="coerce").replace(0, np.nan)
            elif "z_mean" in df.columns:
                z = pd.to_numeric(df["z_mean"], errors="coerce").replace(0, np.nan)
            else:
                raise ValueError(f"{scen}: aggregates missing z_mean/z_net_mean for share computation")
            share = price * y / z
        axes[0].plot(df["t"], share, label=label)
        if "U_mean" not in df.columns:
            raise ValueError(f"{scen}: aggregates missing 'U_mean'")
        u_vals = pd.to_numeric(df["U_mean"], errors="coerce")
        axes[1].plot(df["t"], u_vals, label=label)

    axes[0].set_title("Mean share over time")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("mean share")
    axes[0].legend()

    axes[1].set_title("Mean utility over time")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("U_mean")
    axes[1].legend()

    fig.suptitle(title, fontsize=12)
    _savefig(FIGS / outname)




def plot_endog_decile_metrics(
    scenarios: Sequence[Tuple[str, str]],
    price: float,
    metrics: Mapping[str, str],
    outname: str,
    title: str,
):
    scenarios = list(scenarios)
    if len(scenarios) < 2:
        raise FileNotFoundError("Need at least two scenarios for endogenous comparison plot.")
    series_by_metric: Dict[str, Dict[str, pd.Series]] = {m: {} for m in metrics}
    common_deciles = None
    for scen, label in scenarios:
        df = _last_period_deciles(_load_deciles(scen)).copy()
        df["decile"] = pd.to_numeric(df["decile"], errors="coerce")
        df = df.dropna(subset=["decile"])
        df = df.sort_values("decile").reset_index(drop=True)
        df["share_mean"] = _share_from_decile_df(df, price).reindex(df.index)
        common = df["decile"].to_numpy()
        if common_deciles is None:
            common_deciles = common
        else:
            common_deciles = np.intersect1d(common_deciles, common)
        indexed = df.set_index("decile")
        for metric in metrics:
            if metric not in indexed.columns:
                continue
            vals = pd.to_numeric(indexed[metric], errors="coerce")
            series_by_metric[metric][label] = vals
    if common_deciles is None or len(common_deciles) == 0:
        raise ValueError("Could not find overlapping deciles for endogenous comparison plot.")
    common_deciles = np.array(sorted(common_deciles))
    n_metrics = len(metrics)
    ncols = 2
    nrows = math.ceil(n_metrics / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8, 5 * nrows))
    axes = np.atleast_1d(axes).flatten()
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for ax, (metric, label) in zip(axes, metrics.items()):
        for i, (scen_label, series) in enumerate(series_by_metric[metric].items()):
            aligned = series.reindex(common_deciles)
            ax.plot(common_deciles, aligned, marker="o", label=scen_label, color=palette[i % len(palette)])
        ax.set_xticks(common_deciles)
        ax.set_xlabel("Income decile")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(linestyle="--", alpha=0.3)
    if axes.size > n_metrics:
        for ax in axes[n_metrics:]:
            ax.remove()
    handles = []
    labels = []
    for metric_series in series_by_metric.values():
        for scen_label in metric_series.keys():
            if scen_label not in labels:
                labels.append(scen_label)
    if labels:
        handles = [plt.Line2D([0], [0], color=palette[i % len(palette)], marker='o', linestyle='-') for i, _ in enumerate(labels)]
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 0.98))
    #fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    _savefig(FIGS / outname)

def plot_rq4_net_vs_pretax(he_scen: str, her_scen: str, price: float, outname: str, title: str):
    he = _last_period_deciles(_load_deciles(he_scen))
    her = _last_period_deciles(_load_deciles(her_scen))

    for df in (he, her):
        if "z_net_mean" not in df.columns:
            df["z_net_mean"] = df.get("z_mean", pd.Series([np.nan] * len(df)))

    x_he = he["decile"].to_numpy()
    x_her = her["decile"].to_numpy()

    y_he_net = (price * he["y_mean"]) / he["z_net_mean"]
    y_her_net = (price * her["y_mean"]) / her["z_net_mean"]

    if "z_mean" in he.columns:
        y_he_pre = (price * he["y_mean"]) / he["z_mean"]
    else:
        y_he_pre = y_he_net
    if "z_mean" in her.columns:
        y_her_pre = (price * her["y_mean"]) / her["z_mean"]
    else:
        y_her_pre = y_her_net

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].plot(x_he, y_he_net, marker="o", label=f"{he_scen} (net)")
    axes[0].plot(x_her, y_her_net, marker="o", label=f"{her_scen} (net)")
    axes[0].set_title("Net-income share p·y / z_net")
    axes[0].set_xlabel("Income decile")
    axes[0].set_ylabel("share")
    axes[0].legend()

    axes[1].plot(x_he, y_he_pre, marker="o", label=f"{he_scen} (pretax)")
    axes[1].plot(x_her, y_her_pre, marker="o", label=f"{her_scen} (pretax)")
    axes[1].set_title("Pre-tax share p·y / z")
    axes[1].set_xlabel("Income decile")
    axes[1].set_ylabel("share")
    axes[1].legend()

    fig.suptitle(title, fontsize=12)
    _savefig(FIGS / outname)

    if np.allclose(y_her_net, y_her_pre, equal_nan=True):
        print("[NOTE] HER net vs pretax are identical; z_net == z.")
    if np.allclose(y_he_net, y_he_pre, equal_nan=True):
        print("[NOTE] HE net vs pretax are identical; z_net == z.")


def plot_distributions_if_available(scen: str, price: float):
    logs = RESULTS / scen / "logs"
    candidates = list(glob.glob(str(logs / "agents_final.csv"))) + \
                 list(glob.glob(str(logs / "snapshot_final.csv"))) + \
                 list(glob.glob(str(logs / "agents_t*.csv"))) + \
                 list(glob.glob(str(logs / "snapshot_t*.csv")))
    if not candidates:
        print(f"[SKIP] distributions: no agent snapshot in {logs}")
        return
    path = Path(sorted(candidates)[-1])
    df = _read_csv_smart(path)
    cols = df.columns
    if "z" not in cols and "z_mean" in cols:
        df = df.rename(columns={"z_mean": "z"})
    if "y" not in cols and "y_mean" in cols:
        df = df.rename(columns={"y_mean": "y"})
    if "z" not in df.columns or "y" not in df.columns:
        print(f"[SKIP] distributions: expected z/y columns in {path.name}")
        return
    z = pd.to_numeric(df.get("z_net", df["z"]), errors="coerce")
    y = pd.to_numeric(df["y"], errors="coerce")
    share = price * y / pd.to_numeric(df.get("z_net", df["z"]), errors="coerce").replace(0, np.nan)

    vals = z.dropna()
    if len(vals) > 0:
        plt.figure(figsize=(5.2, 3.8))
        plt.hist(vals, bins=40)
        plt.xlabel("income z")
        plt.ylabel("count")
        plt.title(f"{scen}: income histogram")
        _savefig(FIGS / f"{scen}_dist_income_hist.png")

        xs = np.sort(vals.values)
        ccdf = 1.0 - np.arange(1, len(xs) + 1) / len(xs)
        plt.figure(figsize=(5.2, 3.8))
        plt.loglog(xs, ccdf)
        plt.xlabel("z (log)")
        plt.ylabel("P(Z > z) (log)")
        plt.title(f"{scen}: income CCDF")
        _savefig(FIGS / f"{scen}_dist_income_ccdf_loglog.png")

    valsy = y.dropna()
    if len(valsy) > 0:
        plt.figure(figsize=(5.2, 3.8))
        plt.hist(valsy, bins=40)
        plt.xlabel("status y")
        plt.ylabel("count")
        plt.title(f"{scen}: status histogram")
        _savefig(FIGS / f"{scen}_dist_status_hist.png")

    vshare = share.dropna()
    if len(vshare) > 0:
        plt.figure(figsize=(5.2, 3.8))
        plt.hist(vshare, bins=40)
        plt.xlabel("share p·y / z{}".format("_net" if "z_net" in df.columns else ""))
        plt.ylabel("count")
        plt.title(f"{scen}: share histogram")
        _savefig(FIGS / f"{scen}_dist_share_hist.png")


# ---------- CLI ------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Make BR-only figures into figs/")
    ap.add_argument("--p", type=float, default=2.0, help="Status good price p (for share calculations)")

    ap.add_argument("--lf_br", default="EXP_LF_br", help="Comma-separated LF BR scenarios (use scen=Label to rename in legend)")
    ap.add_argument("--hf_br", default="EXP_HF_br", help="Comma-separated HF BR scenarios")
    ap.add_argument("--le_br", default="EXP_LE_br", help="Comma-separated LE BR scenarios")
    ap.add_argument("--he_br", default="EXP_HE_br", help="Comma-separated HE BR scenarios")

    ap.add_argument("--le_grid_prefix", default="EXP_LE_br_grid", help="Prefix for LE BR grid experiments (expects <prefix><K>[_s*])")
    ap.add_argument("--hf_grid_prefix", default="EXP_HF_br_grid", help="Prefix for HF BR grid experiments")
    ap.add_argument("--he_grid_prefix", default="EXP_HE_br_grid", help="Prefix for HE BR grid experiments")
    ap.add_argument("--grid_k_vals", default="11,21", help="Comma-separated K values to consider for grid CI plots")

    ap.add_argument("--he_delta_base", default="EXP_HE_br", help="Base scenario for HE share delta plot")
    ap.add_argument("--he_delta_variant", default="EXP_HE_br_nudge", help="Variant scenario for HE share delta plot")

    ap.add_argument("--telemetry_scenarios", default="EXP_LF_br,EXP_HF_br,EXP_LE_br,EXP_HE_br,EXP_HFR_br,EXP_HER_br",
                    help="Comma-separated scenarios for telemetry bars")

    ap.add_argument("--le_ts", default="EXP_LE_br, EXP_HE_br", help="Comma-separated scenarios for LE time-series plot")
    ap.add_argument("--he_ts", default="EXP_HE_br,EXP_LE_br", help="Comma-separated scenarios for HE time-series plot")

    ap.add_argument("--distribution_scenarios", default="EXP_HF_br",
                    help="Comma-separated scenarios for optional distribution snapshots")
    ap.add_argument("--skip_distributions", action="store_true", help="Skip optional distribution plots")
    ap.add_argument("--combined_deciles", default="EXP_LF_br=LF,EXP_HF_br=HF,EXP_HFR_br=HFR,EXP_LFR_br=LFR,EXP_HR_br=HR,EXP_LR_br=LR,EXP_HRF_br=HRF,EXP_LRF_br=LRF",
                    help="Comma-separated BR scenarios for the combined decile plot (use scen=Label to rename); leave blank to skip")
    ap.add_argument("--combined_out", default="EXP_BR_all_share.png",
                    help="Filename (under figs/) for the combined decile plot")
    ap.add_argument("--combined_title", default="Share by decile (BR scenarios)",
                    help="Title for the combined decile plot")
    ap.add_argument("--endog_compare", default="EXP_LE_br=LE,EXP_HE_br=HE",
                    help="Two scenarios (scen=Label) to compare for endogenous network metrics")
    ap.add_argument("--endog_compare_out", default="EXP_endog_BR_compare_metrics.png",
                    help="Filename (under figs/) for the endogenous comparison plot")
    ap.add_argument("--endog_compare_metrics",
                    default="share_mean:Status share,y_mean:Status level,phi_mean:Relative status,U_mean:Utility",
                    help="Comma-separated metric:label entries for the endogenous comparison plot")

    return ap.parse_args()




def _parse_metric_labels(csv: str) -> Mapping[str, str]:
    pairs: Dict[str, str] = {}
    for token in csv.split(','):
        token = token.strip()
        if not token:
            continue
        if ':' not in token:
            raise ValueError(f"Expected metric:label entry, got '{token}'")
        metric, label = token.split(':', 1)
        metric = metric.strip()
        label = label.strip()
        if not metric or not label:
            raise ValueError(f"Invalid metric:label entry '{token}'")
        pairs[metric] = label
    return pairs

def _split_ints(csv_string: str) -> List[int]:
    vals = []
    for token in csv_string.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            vals.append(int(token))
        except ValueError:
            raise ValueError(f"Could not parse integer from '{token}' in --grid_k_vals")
    return vals


# ---------- Main -----------------------------------------------------------

def main():
    args = parse_args()

    lf_pairs = _filter_by_file(_parse_scenario_pairs(args.lf_br), "logs/panel_by_decile.csv")
    hf_pairs = _filter_by_file(_parse_scenario_pairs(args.hf_br), "logs/panel_by_decile.csv")
    le_pairs = _filter_by_file(_parse_scenario_pairs(args.le_br), "logs/panel_by_decile.csv")
    he_pairs = _filter_by_file(_parse_scenario_pairs(args.he_br), "logs/panel_by_decile.csv")

    if lf_pairs:
        _maybe("LF share-by-decile BR -> figs/EXP_LF_BR_share.png",
               lambda: plot_decile_share(lf_pairs, args.p,
                                         "Fixed network (LF): share by decile (BR)",
                                         "EXP_LF_BR_share.png"))
    if hf_pairs:
        _maybe("HF share-by-decile BR -> figs/EXP_HF_BR_share.png",
               lambda: plot_decile_share(hf_pairs, args.p,
                                         "Fixed network (HF): share by decile (BR)",
                                         "EXP_HF_BR_share.png"))
    if le_pairs:
        _maybe("LE share-by-decile BR -> figs/EXP_LE_BR_share.png",
               lambda: plot_decile_share(le_pairs, args.p,
                                         "Endogenous (LE): share by decile (BR)",
                                         "EXP_LE_BR_share.png"))
    if he_pairs:
        _maybe("HE share-by-decile BR -> figs/EXP_HE_BR_share.png",
               lambda: plot_decile_share(he_pairs, args.p,
                                         "Endogenous (HE): share by decile (BR)",
                                         "EXP_HE_BR_share.png"))

    endog_pairs = _filter_by_file(_parse_scenario_pairs(args.endog_compare), "logs/panel_by_decile.csv")
    if len(endog_pairs) >= 2:
        metrics_map = _parse_metric_labels(args.endog_compare_metrics)
        _maybe(f"Endogenous comparison -> figs/{args.endog_compare_out}",
               lambda: plot_endog_decile_metrics(endog_pairs, args.p,
                                                metrics_map,
                                                args.endog_compare_out,
                                                f"Endogenous network comparison ({endog_pairs[0][1]} → {endog_pairs[1][1]})"))
    else:
        print("[SKIP] Endogenous comparison plot — need at least two scenarios with decile logs.")

    combined_pairs = []
    if args.combined_deciles.strip():
        combined_pairs = _filter_by_file(_parse_scenario_pairs(args.combined_deciles), "logs/panel_by_decile.csv")
        if combined_pairs:
            _maybe(f"Combined share-by-decile BR -> figs/{args.combined_out}",
                   lambda: plot_decile_share(combined_pairs, args.p, args.combined_title, args.combined_out))
        else:
            print("[SKIP] Combined share plot — no listed scenarios had decile data.")

    k_vals = _split_ints(args.grid_k_vals)
    if k_vals:
        _maybe("LE grid CI (BR) -> figs/EXP_LE_BR_grid_CI.png",
               lambda: plot_grid_ci(args.le_grid_prefix, k_vals,
                                    "EXP_LE_BR_grid_CI.png", "LE BR grid (K comparisons)"))
        _maybe("HF grid CI (BR) -> figs/EXP_HF_BR_grid_CI.png",
               lambda: plot_grid_ci(args.hf_grid_prefix, k_vals,
                                    "EXP_HF_BR_grid_CI.png", "HF BR grid (K comparisons)"))
        _maybe("HE grid CI (BR) -> figs/EXP_HE_BR_grid_CI.png",
               lambda: plot_grid_ci(args.he_grid_prefix, k_vals,
                                    "EXP_HE_BR_grid_CI.png", "HE BR grid (K comparisons)"))

    base_variant = _filter_by_file(_parse_scenario_pairs(f"{args.he_delta_base}"), "logs/panel_by_decile.csv")
    variant = _filter_by_file(_parse_scenario_pairs(f"{args.he_delta_variant}"), "logs/panel_by_decile.csv")
    if base_variant and variant:
        _maybe("HE delta share (BR variant) -> figs/EXP_HE_BR_delta_share.png",
               lambda: plot_share_delta(base_variant[0], variant[0], args.p,
                                        "EXP_HE_BR_delta_share.png",
                                        "HE BR norm change: Δshare by decile"))

    telemetry = [s.strip() for s in args.telemetry_scenarios.split(",") if s.strip()]
    if telemetry:
        _maybe("Telemetry (BR scenarios) -> figs/br_telemetry_bars.png",
               lambda: plot_telemetry(telemetry,
                                      "br_telemetry_bars.png",
                                      "BR telemetry (success/fallback/latency)"))

    le_ts_pairs = _filter_by_file(_parse_scenario_pairs(args.le_ts), "logs/aggregates.csv")
    he_ts_pairs = _filter_by_file(_parse_scenario_pairs(args.he_ts), "logs/aggregates.csv")

    if le_ts_pairs:
        _maybe("LE time series (BR) -> figs/EXP_LE_BR_ts_share_U.png",
               lambda: plot_time_series(le_ts_pairs, args.p,
                                        "EXP_LE_BR_ts_share_U.png",
                                        "Endogenous (LE): time series (share & U_mean, BR)"))
    if he_ts_pairs:
        _maybe("HE time series (BR) -> figs/EXP_HE_BR_ts_share_U.png",
               lambda: plot_time_series(he_ts_pairs, args.p,
                                        "EXP_HE_BR_ts_share_U.png",
                                        "Endogenous (HE): time series (share & U_mean, BR)"))

    _maybe("RQ4 net vs pretax (BR) -> figs/rq4_br_share_by_decile_net_vs_pretax.png",
           lambda: plot_rq4_net_vs_pretax("EXP_HE_br", "EXP_HER_br", args.p,
                                          "rq4_br_share_by_decile_net_vs_pretax.png",
                                          "RQ4 (BR): net vs pre-tax shares (HE vs HER)"))

    if not args.skip_distributions:
        dist_scenarios = [s.strip() for s in args.distribution_scenarios.split(",") if s.strip()]
        for scen in dist_scenarios:
            _maybe(f"Distributions ({scen})",
                   lambda scen=scen: plot_distributions_if_available(scen, args.p))

    print(f"\nAll done. Figures are under: {FIGS}")


if __name__ == "__main__":
    main()
