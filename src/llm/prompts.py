# src/llm/prompts.py
from __future__ import annotations
from textwrap import dedent

def build_decision_prompt(*, i: int, z_net: float, p: float, xi: float, tau: float, ybar: float,
                          grid_shares: list[float]) -> str:
    grid_str = ", ".join(f"{s:.2f}" for s in grid_shares)
    return dedent(f"""
    You choose a status share for agent {i}. Budget: x + p*y = z_net with p={p:.6g}, z_net={z_net:.6g}.
    Utility (per period, myopic): U = x^{xi:.3f} * y^{1-xi:.3f} + gamma*phi + tau*max_neighbor_y.
    Use tau={tau:.3f}. Relative status term uses neighbor average ybar={ybar:.6g}.
    Choose ONE share from this grid: [{grid_str}].
    Output strictly one line: INDEX=<k> where k is the 0-based index into the grid.
    """).strip()