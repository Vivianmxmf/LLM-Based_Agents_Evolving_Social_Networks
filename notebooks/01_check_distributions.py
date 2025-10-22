# notebooks/01_check_distributions.py
"""
Check distributions for a single scenario:
  - Income z (hist + log-log CCDF)
  - Final-period status y and utility U (hists)
  - Status share (p*y/z) histogram
  - y vs z scatter at final period

Usage:
  python notebooks/01_check_distributions.py --scenario HF
  # or omit --scenario to auto-pick the most recent results/*/logs
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

def find_latest_scenario() -> str | None:
    candidates = []
    for scen_dir in RESULTS_DIR.iterdir():
        if not scen_dir.is_dir():
            continue
        f = scen_dir / "logs" / "aggregates.csv"
        if f.exists():
            candidates.append((f.stat().st_mtime, scen_dir.name))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]

def loglog_ccdf(x: np.ndarray, ax: plt.Axes, title: str):
    """Plot CCDF on log-log axes to inspect tail behavior."""
    x = np.sort(x[x > 0.0])
    n = x.size
    if n == 0:
        return
    y = 1.0 - (np.arange(1, n + 1) / (n + 1.0))  # empirical CCDF
    ax.loglog(x, y, marker=".", linestyle="none")
    ax.set_xlabel("x")
    ax.set_ylabel("P(X ≥ x)")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", "-s", type=str, default=None, help="Scenario (e.g., HF, LF, HFR, HE)")
    args = ap.parse_args()

    scen = args.scenario or find_latest_scenario()
    if scen is None:
        raise SystemExit("No results found. Run a scenario first.")
    logs = RESULTS_DIR / scen / "logs"
    figs = RESULTS_DIR / scen / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    panel_f = logs / "panel.csv"
    agg_f = logs / "aggregates.csv"
    if not panel_f.exists():
        raise SystemExit(f"Missing panel at {panel_f}")
    if not agg_f.exists():
        raise SystemExit(f"Missing aggregates at {agg_f}")

    panel = pd.read_csv(panel_f)
    agg = pd.read_csv(agg_f)

    # incomes are constant across t; take the first appearance per agent
    # safer: groupby 'i' and take first z
    z0 = panel.sort_values(["i", "t"]).groupby("i", as_index=False)["z"].first()["z"].to_numpy()

    # final period snapshot
    t_last = int(panel["t"].max())
    snap = panel[panel["t"] == t_last].copy()
    y_last = snap["y"].to_numpy()
    U_last = snap["U"].to_numpy()
    share_last = snap["share"].to_numpy()

    # ----------------- Plots -----------------
    # 1) Income histogram (linear)
    fig1 = plt.figure()
    plt.hist(z0, bins=40)
    plt.xlabel("Income z")
    plt.ylabel("Count")
    plt.title(f"Income distribution (t=0) — {scen}")
    fig1.savefig(figs / f"dist_income_hist_{scen}.png", dpi=160, bbox_inches="tight")

    # 2) Income CCDF (log-log)
    fig2, ax2 = plt.subplots()
    loglog_ccdf(z0, ax2, f"Income CCDF (log-log) — {scen}")
    fig2.savefig(figs / f"dist_income_ccdf_loglog_{scen}.png", dpi=160, bbox_inches="tight")

    # 3) Final-period status histogram
    fig3 = plt.figure()
    plt.hist(y_last, bins=40)
    plt.xlabel("Status consumption y (final)")
    plt.ylabel("Count")
    plt.title(f"Status distribution at t={t_last} — {scen}")
    fig3.savefig(figs / f"dist_status_hist_t{t_last}_{scen}.png", dpi=160, bbox_inches="tight")

    # 4) Final-period utility histogram
    fig4 = plt.figure()
    plt.hist(U_last, bins=40)
    plt.xlabel("Utility U (final)")
    plt.ylabel("Count")
    plt.title(f"Utility distribution at t={t_last} — {scen}")
    fig4.savefig(figs / f"dist_utility_hist_t{t_last}_{scen}.png", dpi=160, bbox_inches="tight")

    # 5) Final-period status share histogram
    fig5 = plt.figure()
    plt.hist(share_last, bins=40)
    plt.xlabel("Status share p*y/z (final)")
    plt.ylabel("Count")
    plt.title(f"Status share distribution at t={t_last} — {scen}")
    fig5.savefig(figs / f"dist_share_hist_t{t_last}_{scen}.png", dpi=160, bbox_inches="tight")

    # 6) y vs z scatter (final)
    fig6 = plt.figure()
    plt.plot(snap["z"], snap["y"], linestyle="none", marker=".", alpha=0.6)
    plt.xlabel("Income z (final; identical to initial in current model)")
    plt.ylabel("Status y (final)")
    plt.title(f"Status vs income at t={t_last} — {scen}")
    fig6.savefig(figs / f"scatter_status_vs_income_t{t_last}_{scen}.png", dpi=160, bbox_inches="tight")

    # ----------------- Stats summary -----------------
    stats = {
        "scenario": scen,
        "t_last": t_last,
        "z_mean": float(np.mean(z0)),
        "z_p90": float(np.quantile(z0, 0.90)),
        "z_gini_proxy_from_panel": float(agg["gini_z"].iloc[-1]) if "gini_z" in agg.columns else np.nan,
        "y_mean_last": float(np.mean(y_last)),
        "U_mean_last": float(np.mean(U_last)),
        "share_mean_last": float(np.mean(share_last)),
    }
    pd.DataFrame([stats]).to_csv(figs / f"summary_stats_{scen}.csv", index=False)

    print(f"[ok] Saved figures & summary to {figs}")

if __name__ == "__main__":
    main()