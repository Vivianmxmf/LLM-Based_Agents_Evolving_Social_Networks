#!/usr/bin/env python3
# scripts/make_four_distributions.py
#
# Usage:
#   python scripts/make_four_distributions.py \
#     --logs results/EXP_HF_br/logs \
#     --exp EXP_HF_br \
#     --p 2.0 \
#     --net true \
#     --outdir figs
#     [--agents results/EXP_HF_br/logs/agents_final.csv]
# If --agents is provided, the script uses that file directly.
#
# It expects an agent-level snapshot at the final period containing (at least):
#   z_mean or z, y, and (optionally) z_net (for taxed runs).
# Common filenames it will try (in this order):
#   agents_final.csv, agents_t*.csv, snapshot_final.csv
#
# Outputs (in --outdir):
#   <exp>_dist_income_hist.png
#   <exp>_dist_income_ccdf_loglog.png
#   <exp>_dist_status_hist_tfinal.png
#   <exp>_dist_share_hist_tfinal.png
import argparse, glob, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def find_agents_csv(logs_dir: str, explicit: str | None = None) -> str | None:
    """Return path to an agent-level snapshot CSV if found; else None.
    If `explicit` is given and exists, it is returned as-is.
    """
    if explicit and os.path.isfile(explicit):
        return explicit
    # search a few common names inside logs_dir
    cands = [
        os.path.join(logs_dir, "agents_final.csv"),
        os.path.join(logs_dir, "snapshot_final.csv"),
        os.path.join(logs_dir, "final_agents.csv"),
        os.path.join(logs_dir, "agents.csv"),
    ]
    cands += sorted(glob.glob(os.path.join(logs_dir, "agents_t*.csv")))
    cands += sorted(glob.glob(os.path.join(logs_dir, "snapshot_t*.csv")))
    for p in cands:
        if os.path.isfile(p):
            return p
    return None

def ccdf(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([]), np.array([])
    x_sorted = np.sort(x)
    n = x_sorted.size
    # CCDF: P(X >= x)
    y = 1.0 - np.arange(1, n + 1) / n
    return x_sorted, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs",  required=True, help="results/<SCENARIO>/logs directory")
    ap.add_argument("--exp",   required=True, help="experiment name, e.g. EXP_HF_br")
    ap.add_argument("--p",     type=float, default=2.0, help="status good price p")
    ap.add_argument("--net",   type=lambda s: s.lower() in {"1","true","yes"}, default=True,
                    help="use net income (z_net) if available")
    ap.add_argument("--outdir", default="figs", help="where to save pngs")
    ap.add_argument("--bins",   type=int, default=40)
    ap.add_argument("--agents", default=None,
                    help="optional direct path to an agent-level snapshot CSV (overrides --logs search)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    agents_csv = find_agents_csv(args.logs, args.agents)
    if agents_csv is None:
        msg = (
            f"Could not find an agent-level snapshot in {args.logs}.\n"
            "Expected one of: agents_final.csv, snapshot_final.csv, final_agents.csv, agents.csv,\n"
            "or time-stamped variants agents_t*.csv / snapshot_t*.csv.\n\n"
            "Fix: either (A) pass --agents /path/to/agents_final.csv, or (B) enable agent snapshots in your run.\n"
            "For option (B), add to your YAML:\n"
            "  logging:\n    save_agents_final: true\n"
            "and in your env after the last step write a CSV with columns z, y (and z_net if taxed).\n"
        )
        raise FileNotFoundError(msg)
    df = pd.read_csv(agents_csv)

    # Column normalization: allow either z or z_mean, and z_net if present
    # Prefer z_net for taxed runs when --net is true
    z_col = None
    if args.net and "z_net" in df.columns:
        z_col = "z_net"
    elif "z" in df.columns:
        z_col = "z"
    elif "z_mean" in df.columns:
        z_col = "z_mean"
    else:
        raise KeyError("No income column found (expected one of z_net, z, z_mean).")

    if "y" not in df.columns:
        # some logs call it y_mean in agent snapshot—try to be tolerant
        if "y_mean" in df.columns:
            df["y"] = df["y_mean"]
        else:
            raise KeyError("No status column found (expected y or y_mean).")

    z = pd.to_numeric(df[z_col], errors="coerce")
    y = pd.to_numeric(df["y"],   errors="coerce")
    z = z.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.replace([np.inf, -np.inf], np.nan).dropna()

    # Compute share; guard against division by zero
    share = (args.p * y) / z.replace(0, np.nan)

    # 1) Income histogram
    plt.figure()
    plt.hist(z, bins=args.bins, density=True)
    plt.xlabel("Income")
    plt.ylabel("Density")
    plt.title(f"Income distribution — {args.exp}")
    out1 = os.path.join(args.outdir, f"{args.exp}_dist_income_hist.png")
    plt.tight_layout(); plt.savefig(out1, dpi=200); plt.close()

    # 2) Income CCDF (log–log)
    xs, ys = ccdf(z.values)
    plt.figure()
    if xs.size:
        plt.loglog(xs, ys, marker=".", linestyle="none")
    plt.xlabel("Income (log)")
    plt.ylabel("CCDF  P(Z ≥ x) (log)")
    plt.title(f"Income CCDF (log–log) — {args.exp}")
    out2 = os.path.join(args.outdir, f"{args.exp}_dist_income_ccdf_loglog.png")
    plt.tight_layout(); plt.savefig(out2, dpi=200); plt.close()

    # 3) Status y histogram
    plt.figure()
    plt.hist(y, bins=args.bins, density=True)
    plt.xlabel("Status spending y")
    plt.ylabel("Density")
    plt.title(f"Status y — final snapshot — {args.exp}")
    out3 = os.path.join(args.outdir, f"{args.exp}_dist_status_hist_tfinal.png")
    plt.tight_layout(); plt.savefig(out3, dpi=200); plt.close()

    # 4) Status share histogram
    plt.figure()
    plt.hist(share.dropna(), bins=args.bins, density=True)
    plt.xlabel("Status share  s = p·y / z{}".format("_net" if (args.net and "z_net" in df.columns) else ""))
    plt.ylabel("Density")
    plt.title(f"Status share s — final snapshot — {args.exp}")
    out4 = os.path.join(args.outdir, f"{args.exp}_dist_share_hist_tfinal.png")
    plt.tight_layout(); plt.savefig(out4, dpi=200); plt.close()

    print("Saved:")
    for p in (out1,out2,out3,out4):
        print("  ", p)

if __name__ == "__main__":
    main()