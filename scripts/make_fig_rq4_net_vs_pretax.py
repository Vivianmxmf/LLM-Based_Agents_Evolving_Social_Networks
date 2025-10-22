#!/usr/bin/env python3
# scripts/make_fig_rq4_net_vs_pretax.py
#
# Flexible comparison of net vs pre-tax status shares across scenarios.
# Provide one or more decile CSVs and the script plots all series (BR or LLM).

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
AXIS_LABEL_FONTSIZE = 20
AXIS_TICK_FONTSIZE = 20
TITLE_FONTSIZE = 20
LEGEND_FONTSIZE = 20
LINE_WIDTH = 4.0


def _load_decile(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "t" in df.columns:
        df = df[df["t"] == df["t"].max()].copy()

    if "decile" not in df.columns:
        if "income_decile" in df.columns:
            df = df.rename(columns={"income_decile": "decile"})
        else:
            raise ValueError(f"{path} is missing a 'decile' (or 'income_decile') column.")

    num_cols = df.select_dtypes(include="number").columns
    df = df.groupby("decile", as_index=False)[num_cols].mean()
    return df.sort_values("decile")


def _compute_share_net(df: pd.DataFrame, p: float) -> pd.Series:
    for c in ("share_net", "share_net_mean", "share_net_final"):
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    if {"y_mean", "z_net_mean"}.issubset(df.columns):
        return (p * pd.to_numeric(df["y_mean"], errors="coerce")) / pd.to_numeric(df["z_net_mean"], errors="coerce").replace(0, pd.NA)
    if "share_mean" in df.columns:
        return pd.to_numeric(df["share_mean"], errors="coerce")
    if {"y_mean", "z_mean"}.issubset(df.columns):
        return (p * pd.to_numeric(df["y_mean"], errors="coerce")) / pd.to_numeric(df["z_mean"], errors="coerce").replace(0, pd.NA)
    raise ValueError("Could not infer net-income share; supply share_net*, share_mean, or (y_mean & z_net_mean/z_mean).")


def _compute_share_pretax(df: pd.DataFrame, p: float) -> pd.Series:
    if {"y_mean", "z_mean"}.issubset(df.columns):
        return (p * pd.to_numeric(df["y_mean"], errors="coerce")) / pd.to_numeric(df["z_mean"], errors="coerce").replace(0, pd.NA)
    if "share_mean" in df.columns:
        return pd.to_numeric(df["share_mean"], errors="coerce")
    for c in ("share_net", "share_net_mean", "share_net_final"):
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    raise ValueError("Could not infer pre-tax share; provide share_mean or (y_mean & z_mean).")


def _parse_scenarios(entries: Sequence[str]) -> List[Tuple[Path, str]]:
    pairs: List[Tuple[Path, str]] = []
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        if "=" in entry:
            path_str, label = entry.split("=", 1)
        else:
            path_str, label = entry, Path(entry).stem
        pairs.append((Path(path_str), label.strip() or Path(path_str).stem))
    if len(pairs) < 2:
        raise ValueError("Need at least two scenario CSVs for comparison.")
    return pairs


def _collect_series(paths: List[Tuple[Path, str]], p: float) -> Tuple[List[int], Dict[str, pd.Series], Dict[str, pd.Series]]:
    net_series: Dict[str, pd.Series] = {}
    pretax_series: Dict[str, pd.Series] = {}
    common_deciles = None
    for path, label in paths:
        df = _load_decile(path)
        deciles = df["decile"].to_numpy()
        if common_deciles is None:
            common_deciles = deciles
        else:
            common_deciles = sorted(set(common_deciles).intersection(deciles))
        net_series[label] = _compute_share_net(df, p).reset_index(drop=True)
        pretax_series[label] = _compute_share_pretax(df, p).reset_index(drop=True)
        net_series[label].index = df["decile"].values
        pretax_series[label].index = df["decile"].values
    if not common_deciles:
        raise ValueError("Scenarios share no deciles in common.")
    common_deciles = sorted(common_deciles)
    for lbl in net_series:
        net_series[lbl] = net_series[lbl].reindex(common_deciles)
        pretax_series[lbl] = pretax_series[lbl].reindex(common_deciles)
    return common_deciles, net_series, pretax_series


def _plot_panel(deciles: Iterable[int], series_map: Dict[str, pd.Series], title: str, ylabel: str, color_cycle: Iterable[str], path: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for idx, (label, series) in enumerate(series_map.items()):
        ax.plot(deciles, series.values, marker="o", linewidth=LINE_WIDTH, label=label, color=color_cycle[idx % len(DEFAULT_COLORS)])
    ax.set_xticks(list(deciles))
    ax.set_xlabel("Income decile", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=AXIS_TICK_FONTSIZE)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare final net vs pre-tax status shares across scenarios.")
    parser.add_argument("inputs", nargs="+", help="Scenario decile CSVs with optional label, e.g. results/EXP_HE_br/logs/panel_by_decile.csv=HE_BR")
    parser.add_argument("--out", required=True, type=Path, help="Output combined PNG path")
    parser.add_argument("--p", type=float, default=2.0, help="Status good price p")
    args = parser.parse_args()

    scenarios = _parse_scenarios(args.inputs)
    deciles, net_series, pretax_series = _collect_series(scenarios, args.p)

    left_png = args.out.parent / f"{args.out.stem}_net.png"
    right_png = args.out.parent / f"{args.out.stem}_pretax.png"

    _plot_panel(deciles, net_series, "Net-income share (final)", "Mean status share (p·y / z_net)", DEFAULT_COLORS, left_png)
    _plot_panel(deciles, pretax_series, "Pre-tax share (final)", "Mean status share (p·y / z)", DEFAULT_COLORS, right_png)

    try:
        from PIL import Image
        left_img, right_img = Image.open(left_png), Image.open(right_png)
        height = max(left_img.height, right_img.height)
        width = left_img.width + right_img.width
        canvas = Image.new("RGB", (width, height), (255, 255, 255))
        canvas.paste(left_img, (0, 0))
        canvas.paste(right_img, (left_img.width, 0))
        canvas.save(args.out)
        print(f"Wrote combined figure -> {args.out}")
    except Exception:
        print("Pillow missing or failed; kept separate panels:", left_png, right_png)


if __name__ == "__main__":
    main()
