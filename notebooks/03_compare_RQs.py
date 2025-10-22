# notebooks/03_compare_RQs.py
"""
Compare scenarios on paper-style research questions (RQs).

Usage:
  python notebooks/03_compare_RQs.py --scenarios LF HF HFR HE
  # omit --scenarios to use the default set and auto-skip missing ones
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "figures"
OUT.mkdir(parents=True, exist_ok=True)

DEFAULT_SCENS = ["LF", "HF", "HFR", "HE"]

def load_agg(scen: str) -> pd.DataFrame | None:
    f = RES / scen / "logs" / "aggregates.csv"
    return pd.read_csv(f) if f.exists() else None

def load_decile_last(scen: str) -> pd.DataFrame | None:
    f = RES / scen / "logs" / "panel_by_decile.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    t_last = df["t"].max()
    return df[df["t"] == t_last].sort_values("decile").reset_index(drop=True)

def scenarios_present(names):
    avail = []
    for s in names:
        if (RES / s / "logs" / "aggregates.csv").exists():
            avail.append(s)
        else:
            print(f"[skip] No results for {s}")
    return avail

def plot_series(df_a: pd.DataFrame, df_b: pd.DataFrame, col: str, label_a: str, label_b: str, title: str, out: Path):
    fig = plt.figure()
    plt.plot(df_a["t"], df_a[col], label=label_a)
    plt.plot(df_b["t"], df_b[col], label=label_b)
    plt.xlabel("t"); plt.ylabel(col); plt.title(title); plt.legend()
    fig.savefig(out, dpi=160, bbox_inches="tight")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", "-s", nargs="*", default=None, help="Subset to compare, e.g., HF HFR")
    args = ap.parse_args()

    scens = scenarios_present(args.scenarios or DEFAULT_SCENS)
    if not scens:
        raise SystemExit("No scenarios available. Run some first.")

    # Load everything we may need
    agg = {s: load_agg(s) for s in scens}
    dec_last = {s: load_decile_last(s) for s in scens}

    # ---------------- RQ1: Effect of inequality (HF vs LF) ----------------
    if "HF" in agg and "LF" in agg and dec_last.get("HF") is not None and dec_last.get("LF") is not None:
        # Share by decile at final t
        fig = plt.figure()
        hf, lf = dec_last["HF"], dec_last["LF"]
        plt.plot(lf["decile"], lf["share_mean"], marker="o", label="LF")
        plt.plot(hf["decile"], hf["share_mean"], marker="x", label="HF")
        plt.xticks(range(1, 11))
        plt.xlabel("Income decile"); plt.ylabel("Mean status share (p y / z)")
        plt.title("RQ1: Inequality (LF vs HF) — status share by decile (final)")
        plt.legend()
        fig.savefig(OUT / "rq1_share_by_decile_LF_vs_HF.png", dpi=160, bbox_inches="tight")

        # Tiny last-period summary table
        last_LF = agg["LF"].iloc[-1][["W","gini_z","gini_U","share_mean"]]
        last_HF = agg["HF"].iloc[-1][["W","gini_z","gini_U","share_mean"]]
        tbl = pd.DataFrame([last_LF, last_HF], index=["LF","HF"]).round(3)
        tbl.to_csv(OUT / "rq1_summary_lastperiod_LF_vs_HF.csv")

    # ---------------- RQ2: Taxes (HF vs HFR) ----------------
    if "HF" in agg and "HFR" in agg and agg["HF"] is not None and agg["HFR"] is not None:
        plot_series(agg["HF"], agg["HFR"], "W", "HF", "HFR",
                    "RQ2: Welfare over time (tax off vs on)", OUT / "rq2_W_time_HF_vs_HFR.png")
        plot_series(agg["HF"], agg["HFR"], "gini_U", "HF", "HFR",
                    "RQ2: Utility Gini over time (tax off vs on)", OUT / "rq2_giniU_time_HF_vs_HFR.png")
        # save last-period deltas
        last = pd.DataFrame({
            "HF": agg["HF"].iloc[-1][["W","gini_U","gini_z","share_mean"]],
            "HFR": agg["HFR"].iloc[-1][["W","gini_U","gini_z","share_mean"]],
        }).T.round(3)
        last.to_csv(OUT / "rq2_summary_lastperiod_HF_vs_HFR.csv")

        # Distributional: share by decile under taxes
        if dec_last.get("HFR") is not None and dec_last.get("HF") is not None:
            fig = plt.figure()
            hf, hfr = dec_last["HF"], dec_last["HFR"]
            plt.plot(hf["decile"], hf["share_mean"], marker="x", label="HF")
            plt.plot(hfr["decile"], hfr["share_mean"], marker="o", label="HFR")
            plt.xticks(range(1, 11))
            plt.xlabel("Income decile"); plt.ylabel("Mean status share (p y / z)")
            plt.title("RQ2: Taxes — status share by decile (final)")
            plt.legend()
            fig.savefig(OUT / "rq2_share_by_decile_HF_vs_HFR.png", dpi=160, bbox_inches="tight")

    # ---------------- RQ3: Endogenous network (HF vs HE) ----------------
    if "HF" in agg and "HE" in agg and agg["HE"] is not None and agg["HF"] is not None:
        for col, fname in [("rho_z","rq3_rhoz_time_HF_vs_HE.png"),
                           ("assort","rq3_assort_time_HF_vs_HE.png"),
                           ("Cl","rq3_clustering_time_HF_vs_HE.png")]:
            plot_series(agg["HF"], agg["HE"], col, "HF (fixed)", "HE (endogenous)",
                        f"RQ3: {col} over time — fixed vs endogenous", OUT / fname)

    # ---------------- RQ4: Distributional pattern across scenarios ----------------
    # compute net-income share from panel for each scenario
    def load_decile_last_netshare(scen: str) -> pd.DataFrame | None:
        panel_f = RES / scen / "logs" / "panel.csv"
        dec_f = RES / scen / "logs" / "panel_by_decile.csv"
        if not panel_f.exists() or not dec_f.exists():
            return None
        p = 2.0  # match goods.p used in your configs
        panel = pd.read_csv(panel_f)
        # last period
        t_last = int(panel["t"].max())
        snap = panel[panel["t"] == t_last].copy()
        z_net = snap["x"] + p * snap["y"]
        share_net = (p * snap["y"]) / z_net.replace(0, np.nan)
        share_net = share_net.fillna(0.0).clip(lower=0.0)
        snap["share_net"] = share_net
        # deciles by pre-tax income like before
        dec = pd.read_csv(dec_f)
        dec_last_df = dec[dec["t"] == dec["t"].max()].sort_values("decile").reset_index(drop=True)
        # merge share_net means by decile (recompute deciles here)
        # quick deciles: ranks by z in last snapshot
        z_last = snap["z"].to_numpy()
        ranks = z_last.argsort().argsort()
        deciles = (100.0 * ranks / (len(ranks) - 1)) // 10 + 1
        snap["decile"] = deciles.astype(int).clip(1, 10)
        share_net_by_dec = snap.groupby("decile", as_index=False)["share_net"].mean()
        share_net_by_dec = share_net_by_dec.rename(columns={"share_net": "share_net_mean"})
        dec_last_df = dec_last_df.merge(share_net_by_dec[["decile","share_net_mean"]], on="decile", how="left")
        return dec_last_df

    # Replace RQ4 plotting loop: two panels side-by-side (net share and pre-tax share)
    present_dec = {s: d for s, d in dec_last.items() if d is not None}
    present_dec_net = {s: load_decile_last_netshare(s) for s in present_dec.keys()}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax1, ax2 = axes
    for s, d in present_dec_net.items():
        if d is not None:
            ax1.plot(d["decile"], d["share_net_mean"], marker="o", label=s)
    ax1.set_xticks(range(1, 11))
    ax1.set_xlabel("Income decile"); ax1.set_ylabel("Mean status share (p y / z_net)")
    ax1.set_title("Net-income share (final)"); ax1.legend()

    for s, d in present_dec.items():
        if d is not None:
            ax2.plot(d["decile"], d["share_mean"], marker="o", label=s)
    ax2.set_xticks(range(1, 11))
    ax2.set_xlabel("Income decile"); ax2.set_ylabel("Mean status share (p y / z)")
    ax2.set_title("Pre-tax share (final)"); ax2.legend()

    fig.suptitle("RQ4: Status share by decile (final) — selected scenarios")
    out_path = OUT / "rq4_share_by_decile_all_scenarios.png"
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"[ok] Saved RQ4 figure to {out_path.resolve()}")

    # ---------------- Text summary (quick) ----------------
    lines = []
    def add_line(s): lines.append(s)
    for key in ["LF","HF","HFR","HE"]:
        if key in agg and agg[key] is not None:
            last = agg[key].iloc[-1]
            add_line(
                f"{key}: W={last.W:.3f}, gini_z={last.gini_z:.3f}, gini_U={last.gini_U:.3f}, "
                f"share_mean={last.share_mean:.3f}, rho_z={last.rho_z:.3f}, assort={last.assort:.3f}, Cl={last.Cl:.3f}"
            )
    (OUT / "summary_RQs.txt").write_text("\n".join(lines) + "\n")
    print(f"[ok] Wrote figures & summary to {OUT}")

if __name__ == "__main__":
    main()