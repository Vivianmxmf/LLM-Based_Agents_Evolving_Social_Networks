# notebooks/02_network_structure_starter.py
"""
Starter analysis for network structure & income-percentile panels.

Usage:
  # after you've run at least one scenario:
  python notebooks/02_network_structure_starter.py --scenario HF
  # or omit --scenario to auto-pick the most recent results/*/logs
"""

from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

def find_latest_scenario() -> str | None:
    """Return the scenario name with the newest aggregates.csv, or None if none found."""
    candidates = []
    if not RESULTS.exists():
        return None
    for scen_dir in RESULTS.iterdir():
        if not scen_dir.is_dir():
            continue
        f = scen_dir / "logs" / "aggregates.csv"
        if f.exists():
            candidates.append((f.stat().st_mtime, scen_dir.name))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", "-s", type=str, default=None,
                        help="Scenario folder name under results/ (e.g., HF, HE, LF, HFR)")
    args = parser.parse_args()

    scen = args.scenario or find_latest_scenario()
    if scen is None:
        raise SystemExit("No results found. Run a scenario first, e.g.: python -m src.run --config configs/HF.yaml")

    RES = RESULTS / scen / "logs" 
    agg_f = RES / str(str(args.scenario)[4:] + "_aggregates.csv")
    dec_f = RES / str(str(args.scenario)[4:] +"_panel_by_decile.csv")

    if not agg_f.exists():
        raise SystemExit(f"Missing {agg_f}. Run the scenario first.")
    if not dec_f.exists():
        raise SystemExit(f"Missing {dec_f}. Run the scenario first.")

    agg = pd.read_csv(agg_f)
    dec = pd.read_csv(dec_f)

    # --- Time series of network metrics ---
    fig1 = plt.figure()
    plt.plot(agg["t"], agg["rho_z"], label="Income homophily (ρ_z)")
    plt.plot(agg["t"], agg["assort"], label="Degree assortativity")
    plt.plot(agg["t"], agg["Cl"], label="Clustering coefficient")
    plt.xlabel("t")
    plt.ylabel("value")
    plt.title(f"Network structure over time — {scen}")
    plt.legend()
    fig1.savefig(RES / f"fig_network_structure_timeseries_{scen}.png", dpi=160, bbox_inches="tight")

    # --- Status share by decile (last period) ---
    t_last = int(dec["t"].max())
    dec_last = dec[dec["t"] == t_last].sort_values("decile")
    fig2 = plt.figure()
    plt.plot(dec_last["decile"], dec_last["share_mean"], marker="o")
    plt.xticks(range(1, 11))
    plt.xlabel("Income decile")
    plt.ylabel("Mean status share (p y / z)")
    plt.title(f"Status share by income decile at t={t_last} — {scen}")
    fig2.savefig(RES / f"fig_share_by_decile_last_{scen}.png", dpi=160, bbox_inches="tight")

    # --- Utility & status by decile (last period) ---
    fig3 = plt.figure()
    plt.plot(dec_last["decile"], dec_last["U_mean"], marker="o", label="U mean")
    plt.plot(dec_last["decile"], dec_last["y_mean"], marker="x", label="y mean")
    plt.xticks(range(1, 11))
    plt.xlabel("Income decile")
    plt.ylabel("level")
    plt.title(f"Utility & status by decile at t={t_last} — {scen}")
    plt.legend()
    fig3.savefig(RES / f"fig_U_y_by_decile_last_{scen}.png", dpi=160, bbox_inches="tight")

    print(f"[info] Using scenario: {scen}")
    print("Saved figures to", RES)

if __name__ == "__main__":
    main()