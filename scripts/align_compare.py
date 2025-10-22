# scripts/align_compare.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def load_decile_last(scen: str) -> pd.DataFrame:
    p = Path("results")/scen/"logs"/"panel_by_decile.csv"
    df = pd.read_csv(p)
    t_last = df["t"].max()
    return df[df["t"]==t_last].sort_values("decile").reset_index(drop=True)

def sign_match(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.sign(a) == np.sign(b)))

def pack_metrics(name, brA, brB, llmA, llmB, metr, out):
    # Load
    A_br = load_decile_last(brA)
    B_br = load_decile_last(brB)
    A_llm = load_decile_last(llmA)
    B_llm = load_decile_last(llmB)

    dec = A_br["decile"].to_numpy()
    rows = []
    out_json = {"name": name, "pairs": {"br":[brA,brB], "llm":[llmA,llmB]}, "metrics":{}}

    for col in metr:
        d_br = (B_br[f"{col}"] - A_br[f"{col}"]).to_numpy()
        d_llm = (B_llm[f"{col}"] - A_llm[f"{col}"]).to_numpy()
        rmse = float(np.sqrt(np.mean((d_llm - d_br)**2)))
        sm   = sign_match(d_br, d_llm)
        rho_A = float(spearmanr(dec, A_br[f"{col}"]).correlation)
        rho_B = float(spearmanr(dec, B_br[f"{col}"]).correlation)
        rho_A_llm = float(spearmanr(dec, A_llm[f"{col}"]).correlation)
        rho_B_llm = float(spearmanr(dec, B_llm[f"{col}"]).correlation)

        rows.append([col, rmse, sm, rho_A, rho_B, rho_A_llm, rho_B_llm])
        out_json["metrics"][col] = {
            "rmse_delta": rmse,
            "sign_match_delta": sm,
            "spearman": {
                "A_br": rho_A, "B_br": rho_B,
                "A_llm": rho_A_llm, "B_llm": rho_B_llm
            }
        }

    out_df = pd.DataFrame(rows, columns=[
        "var","RMSE(Δ_llm−Δ_br)","SignMatch(Δ)","ρ(dec, A_br)","ρ(dec, B_br)","ρ(dec, A_llm)","ρ(dec, B_llm)"
    ])
    print(f"\n=== {name} ===")
    print(out_df.round(3).to_string(index=False))

    Path("results").mkdir(exist_ok=True)
    with open(Path("results")/f"{name}_alignment.json","w") as f:
        json.dump(out_json, f, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LLM vs BR alignment for one inequality/policy contrast")
    ap.add_argument("--br", nargs=2, required=True, help="Two BR scenarios: e.g. EXP_LF_br EXP_HF_br")
    ap.add_argument("--llm", nargs=2, required=True, help="Two LLM scenarios: e.g. EXP_LF_llm_small EXP_HF_llm_small")
    ap.add_argument("--name", required=True, help="Name tag for outputs, e.g. fixed_ineq")
    ap.add_argument("--vars", default="share_mean,y_mean,U_mean", help="Comma list from panel_by_decile")
    args = ap.parse_args()

    vars_ = [v.strip() for v in args.vars.split(",") if v.strip()]
    pack_metrics(args.name, args.br[0], args.br[1], args.llm[0], args.llm[1], vars_, Path("results"))