#!/usr/bin/env python3
# scripts/make_llm_figs.py
#
# One-stop plotter for LLM vs BR figures.
# It reads CSVs under results/<scenario>/logs and writes PNGs under figs/.
#
# Figures produced (if inputs present):
#  1) Fixed networks: share-by-decile BR vs LLM  (LF, HF)
#  2) Endogenous:     share-by-decile BR vs LLM  (LE, HE)
#  3) Grid CI bars:   LE / HF / HE  (K=11 vs 21, mean±95% CI for W, Gini(U), share)
#  4) Norm-nudge delta (HE LLM nudge vs base): Δshare by decile
#  5) Telemetry bars: success/fallback/latency (if llm_stats.json exist)
#  6) Time series:    mean share & mean U, BR vs LLM (LE, HE)
#  7) RQ4 net vs pre-tax: HE vs HER, shares by decile using z_net vs z
#  8) (Optional) Distribution snapshots if a per-agent snapshot exists
#
# Usage (defaults should Just Work if your scenarios use standard names):
#   python scripts/make_llm_figs.py
# Or override specific pairs/paths via flags; see --help.

from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"
FIGS = PROJECT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

# ---------- Utilities

def _read_csv_smart(path: Path) -> pd.DataFrame:
    """Read CSV that may or may not have a header. Return DataFrame."""
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        df = pd.read_csv(path)  # try with header
        # If columns are unnamed 0..N-1 but first row looks numeric, it's fine.
        return df
    except Exception:
        df = pd.read_csv(path, header=None)
        return df

def _load_aggregates(scen: str) -> pd.DataFrame:
    path = RESULTS / scen / "logs" / "aggregates.csv"
    df = _read_csv_smart(path)
    # Try to enforce expected columns
    expected = ["W","gini_z","gini_U","rho_z","assort","Cl",
                "share_mean","y_mean","x_mean","U_mean","t","rebate","edges"]
    if df.shape[1] == len(expected):
        df.columns = expected
    # If 't' exists, use the last unique t; else just last row
    if "t" in df.columns:
        # drop duplicates keeping last
        df = df.sort_values("t").drop_duplicates("t", keep="last")
    return df

def _load_deciles(scen: str) -> pd.DataFrame:
    path = RESULTS / scen / "logs" / "panel_by_decile.csv"
    df = _read_csv_smart(path)
    # best-guess rename if not already named
    # Expected: columns at least ['t','decile','share_mean','y_mean','U_mean','z_mean','z_net_mean?']
    if "decile" not in df.columns:
        # Try to infer basic schema (very robust fallback)
        # Heuristic: find likely decile column
        for col in df.columns:
            if str(col).lower().startswith("dec"):
                df = df.rename(columns={col:"decile"})
                break
        if "decile" not in df.columns:
            # assume columns: t, decile, share_mean, y_mean, x_mean, U_mean, z_mean, ...
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
    # else assume already last rows per decile
    return df.sort_values("decile")

def _ensure_series_equal_len(x: np.ndarray, y: np.ndarray, name: str):
    if len(x) != len(y):
        raise ValueError(f"{name}: length mismatch: len(x)={len(x)} vs len(y)={len(y)}")

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


def _final_share_mean(df: pd.DataFrame, price: float) -> float:
    work = df.sort_values("t") if "t" in df.columns else df
    last = work.iloc[-1]
    if "share_mean" in work.columns:
        val = last.get("share_mean")
    else:
        if "y_mean" not in work.columns:
            raise ValueError("Aggregates missing y_mean needed for share computation")
        y = float(last.get("y_mean"))
        if "z_net_mean" in work.columns:
            z = float(last.get("z_net_mean"))
        elif "z_mean" in work.columns:
            z = float(last.get("z_mean"))
        else:
            raise ValueError("Aggregates missing z_mean/z_net_mean needed for share computation")
        if z == 0:
            raise ValueError("Encountered zero income when computing share")
        val = price * y / z
    return float(val)

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

# ---------- Figure helpers

def plot_decile_share_br_vs_llm(br_scen: str, llm_scen: str, title: str, outname: str):
    br = _last_period_deciles(_load_deciles(br_scen))
    llm = _last_period_deciles(_load_deciles(llm_scen))
    dec = np.array(sorted(br["decile"].unique()))
    # Try align on common deciles
    common = np.intersect1d(br["decile"].unique(), llm["decile"].unique())
    br = br[br["decile"].isin(common)].sort_values("decile")
    llm = llm[llm["decile"].isin(common)].sort_values("decile")
    x = br["decile"].to_numpy()
    y_br = br["share_mean"].to_numpy()
    y_llm = llm["share_mean"].to_numpy()
    _ensure_series_equal_len(x, y_llm, "BR vs LLM decile align")

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(x, y_br, marker="o", label=f"{br_scen}")
    plt.plot(x, y_llm, marker="o", label=f"{llm_scen}")
    plt.xticks(list(x))
    plt.xlabel("Income decile")
    plt.ylabel("Status share mean (p y / z or z_net)")
    plt.title(title)
    plt.legend()

    _savefig(FIGS / outname)


def plot_share_gap_bars(pairs: List[Tuple[str, str, str]], price: float, outname: str, title: str):
    records = []
    for br_scen, llm_scen, label in pairs:
        dbr = _load_aggregates(br_scen)
        dll = _load_aggregates(llm_scen)
        try:
            br_share = _final_share_mean(dbr, price)
            llm_share = _final_share_mean(dll, price)
        except Exception as exc:
            raise ValueError(f"{label}: {exc}") from exc
        records.append((label, br_share, llm_share, br_scen, llm_scen))

    if not records:
        raise FileNotFoundError("No valid scenario pairs for share gap plot.")

    labels = [r[0] for r in records]
    br_vals = [r[1] for r in records]
    llm_vals = [r[2] for r in records]
    idx = np.arange(len(records))
    width = 0.35

    fig_width = max(6.5, 1.8 * len(records))
    fig, ax = plt.subplots(figsize=(fig_width, 4.2))
    ax.bar(idx - width / 2, br_vals, width=width, label="BR", color="#4C72B0")
    ax.bar(idx + width / 2, llm_vals, width=width, label="LLM", color="#DD8452")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Final mean status share")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    diffs = np.array(llm_vals) - np.array(br_vals)
    offset = max(max(br_vals + llm_vals), 1e-3) * 0.04
    for i, delta in enumerate(diffs):
        ymax = max(br_vals[i], llm_vals[i])
        ax.text(idx[i], ymax + offset, f"Δ={delta:.3f}", ha="center", va="bottom", fontsize=8)

    ax.legend()
    _savefig(FIGS / outname)


def plot_grid_ci(prefix: str, K_vals: List[int], outname: str, title: str):
    """prefix examples: 'EXP_LE_llm_grid', 'EXP_HF_llm_grid', 'EXP_HE_llm_grid'."""
    metrics = []
    for K in K_vals:
        rows = []
        # collect last-row aggregates for all seeds
        for path in sorted((RESULTS).glob(f"{prefix}{K}_s*/logs/aggregates.csv")):
            df = _load_aggregates(path.parts[-3])  # scenario folder name
            last = df.iloc[-1]
            rows.append({
                "W": float(last.get("W", np.nan)),
                "gini_U": float(last.get("gini_U", np.nan)),
                "share_mean": float(last.get("share_mean", np.nan)),
            })
        if not rows:
            # also allow a singleton (no _s*) at results/<prefixK>/logs/aggregates.csv
            solo = RESULTS / f"{prefix}{K}" / "logs" / "aggregates.csv"
            if solo.exists():
                df = _load_aggregates(f"{prefix}{K}")
                last = df.iloc[-1]
                rows.append({
                    "W": float(last.get("W", np.nan)),
                    "gini_U": float(last.get("gini_U", np.nan)),
                    "share_mean": float(last.get("share_mean", np.nan)),
                })
        df = pd.DataFrame(rows)
        n = len(df)
        if n == 0:
            metrics.append((K, None))
        else:
            mW, cW   = _mean_ci(df["W"])
            mG, cG   = _mean_ci(df["gini_U"])
            mS, cS   = _mean_ci(df["share_mean"])
            metrics.append((K, {"W":(mW,cW), "gini_U":(mG,cG), "share_mean":(mS,cS), "n":n}))

    # Plot three bars with errorbars for each K present
    have = [m for K,m in metrics if m is not None]
    if not have:
        raise FileNotFoundError("No aggregates found for any K in " + prefix)

    Ks = [K for K,m in metrics if m is not None]
    valsW = [m["W"][0] for K,m in metrics if m is not None]
    errW  = [m["W"][1] for K,m in metrics if m is not None]
    valsG = [m["gini_U"][0] for K,m in metrics if m is not None]
    errG  = [m["gini_U"][1] for K,m in metrics if m is not None]
    valsS = [m["share_mean"][0] for K,m in metrics if m is not None]
    errS  = [m["share_mean"][1] for K,m in metrics if m is not None]
    ns    = [m["n"] for K,m in metrics if m is not None]

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6))
    for ax, vals, errs, name in zip(axes, [valsW, valsG, valsS], [errW, errG, errS], ["W", "Gini(U)", "mean share"]):
        ax.bar([str(k) for k in Ks], vals, yerr=errs, capsize=5)
        ax.set_title(name)
    fig.suptitle(f"{title} (n={ns})", fontsize=12)
    _savefig(FIGS / outname)

def plot_nudge_delta(base_scen: str, nudge_scen: str, outname: str, title: str):
    base = _last_period_deciles(_load_deciles(base_scen))
    nudg = _last_period_deciles(_load_deciles(nudge_scen))
    common = np.intersect1d(base["decile"].unique(), nudg["decile"].unique())
    base = base[base["decile"].isin(common)].sort_values("decile")
    nudg = nudg[nudg["decile"].isin(common)].sort_values("decile")
    x = base["decile"].to_numpy()
    delta = nudg["share_mean"].to_numpy() - base["share_mean"].to_numpy()
    plt.figure(figsize=(6.2, 4.2))
    plt.axhline(0, color="gray", lw=1)
    plt.plot(x, delta, marker="o")
    plt.xticks(list(x))
    plt.xlabel("Income decile")
    plt.ylabel("Δ share (nudge − base)")
    plt.title(title)
    _savefig(FIGS / outname)

def plot_telemetry(scenarios: List[str], outname: str, title: str):
    # Expect results/<scen>/artifacts/llm_stats.json
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
        raise FileNotFoundError("No telemetry jsons found in provided scenarios.")
    df = pd.DataFrame(data)
    idx = np.arange(len(df))
    w = 0.35

    fig, ax1 = plt.subplots(figsize=(8.5, 4.2))
    ax1.bar(idx - w/2, df["success"].values*100, width=w, label="Success %")
    ax1.bar(idx + w/2, df["fallback"].values*100, width=w, label="Fallback %")
    ax1.set_xticks(idx)
    ax1.set_xticklabels(df["scenario"], rotation=30, ha="right")
    ax1.set_ylabel("%")
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(idx, df["latency"].values, marker="o", label="Mean latency (s)")
    ax2.set_ylabel("Seconds")
    fig.suptitle(title, fontsize=12)
    _savefig(FIGS / outname)

def plot_time_series(br_scen: str, llm_scen: str, outname: str, title: str):
    dbr = _load_aggregates(br_scen)
    dll = _load_aggregates(llm_scen)
    need = ["share_mean","U_mean"]
    for col in need:
        if col not in dbr.columns or col not in dll.columns:
            raise ValueError(f"Missing {col} in aggregates for {br_scen} or {llm_scen}")
    # Align on t if present
    if "t" in dbr.columns and "t" in dll.columns:
        # simple plot each on its own x
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
        axes[0].plot(dbr["t"], dbr["share_mean"], label=f"{br_scen}")
        axes[0].plot(dll["t"], dll["share_mean"], label=f"{llm_scen}")
        axes[0].set_title("Mean share over time")
        axes[0].set_xlabel("t"); axes[0].set_ylabel("mean share")
        axes[0].legend()
        axes[1].plot(dbr["t"], dbr["U_mean"], label=f"{br_scen}")
        axes[1].plot(dll["t"], dll["U_mean"], label=f"{llm_scen}")
        axes[1].set_title("Mean utility over time")
        axes[1].set_xlabel("t"); axes[1].set_ylabel("U_mean")
        axes[1].legend()
        fig.suptitle(title, fontsize=12)
        _savefig(FIGS / outname)
    else:
        raise ValueError("No 't' column in aggregates for time series.")

def plot_rq4_net_vs_pretax(he_scen: str, her_scen: str, p: float, outname: str, title: str):
    he  = _last_period_deciles(_load_deciles(he_scen))
    her = _last_period_deciles(_load_deciles(her_scen))
    # Use z_net_mean if present; else fall back to z_mean
    for df in (he, her):
        if "z_net_mean" not in df.columns:
            df["z_net_mean"] = df.get("z_mean", pd.Series([np.nan]*len(df)))
    x_he  = he["decile"].to_numpy()
    x_her = her["decile"].to_numpy()
    y_he_net  = (p*he["y_mean"] / he["z_net_mean"]).to_numpy()
    y_her_net = (p*her["y_mean"] / her["z_net_mean"]).to_numpy()
    y_he_pre  = (p*he["y_mean"] / he["z_mean"]).to_numpy() if "z_mean" in he.columns else y_he_net
    y_her_pre = (p*her["y_mean"] / her["z_mean"]).to_numpy() if "z_mean" in her.columns else y_her_net

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].plot(x_he, y_he_net, marker="o", label=f"{he_scen} (net)")
    axes[0].plot(x_her, y_her_net, marker="o", label=f"{her_scen} (net)")
    axes[0].set_title("Net-income share p y / z_net")
    axes[0].set_xlabel("Income decile"); axes[0].set_ylabel("share"); axes[0].legend()

    axes[1].plot(x_he, y_he_pre, marker="o", label=f"{he_scen} (pretax)")
    axes[1].plot(x_her, y_her_pre, marker="o", label=f"{her_scen} (pretax)")
    axes[1].set_title("Pre-tax share p y / z")
    axes[1].set_xlabel("Income decile"); axes[1].set_ylabel("share"); axes[1].legend()

    fig.suptitle(title, fontsize=12)
    _savefig(FIGS / outname)

    # Console check: identical net vs pretax?
    if np.allclose(y_her_net, y_her_pre, equal_nan=True):
        print("[NOTE] HER net vs pretax are identical. This suggests z_net == z in HER logging.")
    if np.allclose(y_he_net, y_he_pre, equal_nan=True):
        print("[NOTE] HE  net vs pretax are identical. This suggests z_net == z in HE logging.")

def plot_distributions_if_available(scen: str):
    """Optional: 4 distribution plots if an agent snapshot exists."""
    logs = RESULTS / scen / "logs"
    # Look for snapshots
    candidates = list(glob.glob(str(logs / "agents_final.csv"))) + \
                 list(glob.glob(str(logs / "snapshot_final.csv"))) + \
                 list(glob.glob(str(logs / "agents_t*.csv"))) + \
                 list(glob.glob(str(logs / "snapshot_t*.csv")))
    if not candidates:
        print(f"[SKIP] distributions: no agents snapshot in {logs}")
        return
    # Load the last alphabetically
    path = Path(sorted(candidates)[-1])
    df = _read_csv_smart(path)
    # Try to find columns
    # Expect something like: z, y, share (=p y / z or stored), degree, etc.
    cols = df.columns
    if "z" not in cols or "y" not in cols:
        print(f"[SKIP] distributions: expected z and y columns in {path.name}")
        return
    # share: compute if missing
    if "share" in cols:
        share = pd.to_numeric(df["share"], errors="coerce")
    else:
        # require p and maybe z_net; we don't know here, so use z
        p = 2.0
        share = p * pd.to_numeric(df["y"], errors="coerce") / pd.to_numeric(df["z"], errors="coerce")

    # income hist (log bins)
    vals = pd.to_numeric(df["z"], errors="coerce").dropna()
    if len(vals) > 0:
        plt.figure(figsize=(5.2,3.8))
        plt.hist(vals, bins=40)
        plt.xlabel("income z"); plt.ylabel("count"); plt.title(f"{scen}: income histogram")
        _savefig(FIGS / f"{scen}_dist_income_hist.png")

        # CCDF log-log
        xs = np.sort(vals.values)
        ccdf = 1.0 - np.arange(1, len(xs)+1)/len(xs)
        plt.figure(figsize=(5.2,3.8))
        plt.loglog(xs, ccdf)
        plt.xlabel("z (log)"); plt.ylabel("P(Z > z) (log)"); plt.title(f"{scen}: income CCDF")
        _savefig(FIGS / f"{scen}_dist_income_ccdf_loglog.png")

    # status y hist
    valsy = pd.to_numeric(df["y"], errors="coerce").dropna()
    if len(valsy) > 0:
        plt.figure(figsize=(5.2,3.8))
        plt.hist(valsy, bins=40)
        plt.xlabel("status y"); plt.ylabel("count"); plt.title(f"{scen}: status histogram")
        _savefig(FIGS / f"{scen}_dist_status_hist.png")

    # share hist
    vshare = share.dropna()
    if len(vshare) > 0:
        plt.figure(figsize=(5.2,3.8))
        plt.hist(vshare, bins=40)
        plt.xlabel("share p y / z"); plt.ylabel("count"); plt.title(f"{scen}: share histogram")
        _savefig(FIGS / f"{scen}_dist_share_hist.png")

# ---------- CLI & main

def _parse_br_llm_pairs(csv_string: str) -> List[Tuple[str, str, str]]:
    pairs: List[Tuple[str, str, str]] = []
    for token in csv_string.split(','):
        token = token.strip()
        if not token:
            continue
        label = None
        if '=' in token:
            token, label = token.split('=', 1)
            label = label.strip() or None
        if '/' not in token:
            raise ValueError(f"Could not parse pair '{token}'. Expected format BR/LLM or BR/LLM=Label")
        br_scen, llm_scen = [part.strip() for part in token.split('/', 1)]
        if not br_scen or not llm_scen:
            raise ValueError(f"Incomplete pair '{token}'.")
        if label is None:
            label = br_scen
        pairs.append((br_scen, llm_scen, label))
    return pairs


def _filter_pairs_with_aggregates(pairs: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    good: List[Tuple[str, str, str]] = []
    for br_scen, llm_scen, label in pairs:
        br_path = RESULTS / br_scen / 'logs' / 'aggregates.csv'
        llm_path = RESULTS / llm_scen / 'logs' / 'aggregates.csv'
        if br_path.exists() and llm_path.exists():
            good.append((br_scen, llm_scen, label))
    return good



def parse_args():
    ap = argparse.ArgumentParser(description="Make LLM vs BR figures into figs/")
    ap.add_argument("--p", type=float, default=2.0, help="Status good price p (for share calculations)")
    # Optional overrides (comma-separated scenario names)
    ap.add_argument("--lf_llm", default="EXP_LF_llm_grid21,EXP_LF_llm_grid11,EXP_LF_llm_small",
                    help="Try these LF LLM scenarios in order (first that exists is used)")
    ap.add_argument("--hf_llm", default="EXP_HF_llm_grid21,EXP_HF_llm_grid11,EXP_HF_llm_small",
                    help="Try these HF LLM scenarios in order")
    ap.add_argument("--le_llm", default="EXP_LE_llm_grid21,EXP_LE_llm_grid11,EXP_LE_llm_small",
                    help="Try these LE LLM scenarios in order")
    ap.add_argument("--he_llm", default="EXP_HE_llm_grid21,EXP_HE_llm_grid11,EXP_HE_llm_small",
                    help="Try these HE LLM scenarios in order")
    ap.add_argument("--he_nudge_base", default="EXP_HE_llm_grid11",
                    help="Base HE LLM scenario for nudge delta")
    ap.add_argument("--he_nudge", default="EXP_HE_llm_nudge",
                    help="Nudge scenario (LLM) for delta")
    ap.add_argument("--telemetry_scenarios",
                    default="EXP_LF_llm_small,EXP_HF_llm_small,EXP_LE_llm_small,EXP_HE_llm_small,EXP_HFR_llm_small,EXP_HER_llm_small",
                    help="Comma-separated LLM scenarios to read telemetry from")
    ap.add_argument("--fixed_share_pairs",
                    default="EXP_LF_br/EXP_LF_llm_small=LF,EXP_HF_br/EXP_HF_llm_small=HF,EXP_HFR_br/EXP_HFR_llm_small=HFR",
                    help="Scenario pairs (BR/LLM=Label) for fixed-network share summary; blank to skip")
    ap.add_argument("--fixed_share_out", default="EXP_fixed_BRvLLM_share_summary.png",
                    help="Output PNG name for fixed-network share summary")
    ap.add_argument("--fixed_share_title", default="Final share: BR vs LLM (Fixed networks)",
                    help="Title for fixed-network share summary")
    ap.add_argument("--endog_share_pairs",
                    default="EXP_LE_br/EXP_LE_llm_small=LE,EXP_HE_br/EXP_HE_llm_small=HE,EXP_HER_br/EXP_HER_llm_small=HER",
                    help="Scenario pairs (BR/LLM=Label) for endogenous-network share summary; blank to skip")
    ap.add_argument("--endog_share_out", default="EXP_endog_BRvLLM_share_summary.png",
                    help="Output PNG name for endogenous share summary")
    ap.add_argument("--endog_share_title", default="Final share: BR vs LLM (Endogenous networks)",
                    help="Title for endogenous share summary")
    ap.add_argument("--skip_distributions", action="store_true", help="Skip optional distribution plots")
    return ap.parse_args()

def _pick_existing(candidates_csv: str) -> Optional[str]:
    for s in [x.strip() for x in candidates_csv.split(",")]:
        if (RESULTS / s / "logs" / "panel_by_decile.csv").exists():
            return s
    return None


def main():
    args = parse_args()

    # Resolve LLM scenarios (first existing in candidate lists)
    lf_llm = _pick_existing(args.lf_llm)
    hf_llm = _pick_existing(args.hf_llm)
    le_llm = _pick_existing(args.le_llm)
    he_llm = _pick_existing(args.he_llm)

    # 1) Fixed: LF/HF share-by-decile BR vs LLM
    if lf_llm:
        _maybe("LF share-by-decile BR vs LLM -> figs/EXP_LF_BRvLLM_share.png",
               lambda: plot_decile_share_br_vs_llm("EXP_LF_br", lf_llm,
                     "Fixed network (LF): share by decile", "EXP_LF_BRvLLM_share.png"))
    if hf_llm:
        _maybe("HF share-by-decile BR vs LLM -> figs/EXP_HF_BRvLLM_share.png",
               lambda: plot_decile_share_br_vs_llm("EXP_HF_br", hf_llm,
                     "Fixed network (HF): share by decile", "EXP_HF_BRvLLM_share.png"))

    # 2) Endogenous: LE/HE share-by-decile BR vs LLM
    if le_llm:
        _maybe("LE share-by-decile BR vs LLM -> figs/EXP_LE_BRvLLM_share.png",
               lambda: plot_decile_share_br_vs_llm("EXP_LE_br", le_llm,
                     "Endogenous (LE): share by decile", "EXP_LE_BRvLLM_share.png"))
    if he_llm:
        _maybe("HE share-by-decile BR vs LLM -> figs/EXP_HE_BRvLLM_share.png",
               lambda: plot_decile_share_br_vs_llm("EXP_HE_br", he_llm,
                     "Endogenous (HE): share by decile", "EXP_HE_BRvLLM_share.png"))

    if args.fixed_share_pairs.strip():
        fixed_pairs = _filter_pairs_with_aggregates(_parse_br_llm_pairs(args.fixed_share_pairs))
        if fixed_pairs:
            _maybe(f"Fixed share summary -> figs/{args.fixed_share_out}",
                   lambda: plot_share_gap_bars(fixed_pairs, args.p, args.fixed_share_out, args.fixed_share_title))
        else:
            print("[SKIP] Fixed share summary — no matching BR/LLM aggregates found.")

    if args.endog_share_pairs.strip():
        endog_pairs = _filter_pairs_with_aggregates(_parse_br_llm_pairs(args.endog_share_pairs))
        if endog_pairs:
            _maybe(f"Endogenous share summary -> figs/{args.endog_share_out}",
                   lambda: plot_share_gap_bars(endog_pairs, args.p, args.endog_share_out, args.endog_share_title))
        else:
            print("[SKIP] Endogenous share summary — no matching BR/LLM aggregates found.")

    # 3) Grid CI bars: LE/HF/HE (K=11 vs 21)
    _maybe("LE grid CI -> figs/EXP_LE_llm_grid_CI.png",
           lambda: plot_grid_ci("EXP_LE_llm_grid", [11,21],
                    "EXP_LE_llm_grid_CI.png", "LE LLM grid (K=11 vs 21)"))
    _maybe("HF grid CI -> figs/EXP_HF_llm_grid_CI.png",
           lambda: plot_grid_ci("EXP_HF_llm_grid", [11,21],
                    "EXP_HF_llm_grid_CI.png", "HF LLM grid (K=11 vs 21)"))
    _maybe("HE grid CI -> figs/EXP_HE_llm_grid_CI.png",
           lambda: plot_grid_ci("EXP_HE_llm_grid", [11,21],
                    "EXP_HE_llm_grid_CI.png", "HE LLM grid (K=11 vs 21)"))

    # 4) Norm-nudge delta (HE)
    _maybe("HE nudge Δshare -> figs/EXP_HE_llm_nudge_delta_share.png",
           lambda: plot_nudge_delta(args.he_nudge_base, args.he_nudge,
                    "EXP_HE_llm_nudge_delta_share.png", "HE LLM norm nudge: Δshare by decile"))

    # 5) Telemetry bars (if present)
    tel = [s.strip() for s in args.telemetry_scenarios.split(",") if s.strip()]
    _maybe("Telemetry -> figs/llm_telemetry_bars.png",
           lambda: plot_telemetry(tel, "llm_telemetry_bars.png",
                                  "LLM telemetry (success/fallback/latency)"))

    # 6) Time series: share & U_mean (LE, HE)
    if le_llm:
        _maybe("LE time series -> figs/EXP_LE_BRvLLM_ts_share_U.png",
               lambda: plot_time_series("EXP_LE_br", le_llm,
                                        "EXP_LE_BRvLLM_ts_share_U.png",
                                        "Endogenous (LE): time series (share & U_mean)"))
    if he_llm:
        _maybe("HE time series -> figs/EXP_HE_BRvLLM_ts_share_U.png",
               lambda: plot_time_series("EXP_HE_br", he_llm,
                                        "EXP_HE_BRvLLM_ts_share_U.png",
                                        "Endogenous (HE): time series (share & U_mean)"))

    # 7) RQ4 net vs pretax: HE vs HER
    _maybe("RQ4 net vs pretax -> figs/rq4_share_by_decile_net_vs_pretax.png",
           lambda: plot_rq4_net_vs_pretax("EXP_HE_br", "EXP_HER_br", args.p,
                                          "rq4_share_by_decile_net_vs_pretax.png",
                                          "RQ4: net vs pre-tax shares (HE vs HER)"))

    # 8) Optional distributions if snapshots exist
    if not args.skip_distributions:
        for scen in [s for s in ["EXP_HF_llm_grid21", "EXP_HF_llm_grid11", "EXP_HF_llm_small"] if s]:
            _maybe(f"Distributions ({scen})", lambda s=scen: plot_distributions_if_available(s))

    print(f"\nAll done. Figures are under: {FIGS}")

if __name__ == "__main__":
    main()