from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
import json

from env.scenarios import SimConfig
from env.network import SocialNetwork
from env.utility import relative_status, utility_total
from agents.base import AgentView, AgentPolicy
from agents.best_response import BestResponsePolicy
from agents.random_agent import RandomPolicy


# [NEW] metrics imports
from metrics.logging import compute_step_metrics, write_panel, write_step_summary, write_percentile_panel  # [UPDATED]

from metrics.inequality import status_share

# ---------- Helpers: PLN sampler ----------

def sample_pln(
    S: int, m: float, alpha: float, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Pareto–Lognormal generator:
        w = Lognormal(ln m, σ) * Pareto(xm=1, α)
    """
    mu = np.log(m)
    ln_body = rng.lognormal(mean=mu, sigma=sigma, size=S)
    pareto_tail = (1.0 / (rng.random(S) ** (1.0 / alpha)))
    return ln_body * pareto_tail

# ---------- Core Environment ----------

@dataclass
class SimState:
    z: np.ndarray
    z_net: np.ndarray
    x: np.ndarray
    y: np.ndarray
    U: np.ndarray
    phi: np.ndarray
    ybar: np.ndarray
    y_max_nbr: np.ndarray

class StatusSimEnv:
    """
    Implements model dynamics, taxes, and myopic best responses.
    Now also emits per-step metrics faithful to the paper spec.
    """

    def __init__(self, cfg: SimConfig, results_dir: Optional[Path] = None):
        self.cfg = cfg
        self.S = cfg.run.population
        self.T = cfg.run.steps
        self.results_dir = (Path(cfg.run.results_dir) / cfg.run.scenario) if results_dir is None else results_dir

        # RNG
        self.rng = np.random.default_rng(cfg.run.seed)

        # Parameters
        self.theta = cfg.income.theta
        self.beta_flag = int(cfg.income.beta)
        self.eps_std = cfg.income.eps_std
        self.p = cfg.goods.p
        self.xi = cfg.goods.xi

        # Preferences
        g_mu = cfg.preferences.gamma_mean
        g_sd = cfg.preferences.gamma_std
        self.gamma = np.maximum(0.0, self.rng.normal(g_mu, g_sd, size=self.S))
        self.tau = self.rng.uniform(cfg.preferences.tau_min, cfg.preferences.tau_max, size=self.S)

        # Network
        self.net = SocialNetwork(self.S, cfg.network, self.rng)

        # Policy (best response by default)
        self.policy: AgentPolicy = BestResponsePolicy()
        # self.policy = RandomPolicy(self.rng)  # optional sanity baseline

        # Initialize state
        self.state = self._init_state()

        # Book-keeping
        self.t = 0
        self.summary_rows: list[dict] = []   # [UPDATED] renamed to reflect content

    # ---------- Initialization ----------
    def _init_state(self) -> SimState:
        income = self._draw_income(self.S)
        z_net, rebate, tax_rate = self._apply_taxes(income)

        y0 = np.zeros(self.S, dtype=float)
        x0 = z_net - self.p * y0
        ybar0 = np.zeros(self.S, dtype=float)
        y_max0 = np.zeros(self.S, dtype=float)
        phi0 = y0 - ybar0
        U0 = utility_total(x0, y0, self.xi, self.gamma, phi0, self.tau, y_max0)

        return SimState(
            z=income, z_net=z_net, x=x0, y=y0, U=U0, phi=phi0, ybar=ybar0, y_max_nbr=y_max0
        )
    
    # inside class StatusSimEnv
    def _snapshot_agents(self, t: int) -> None:
        """
        Write a single agent-level snapshot at time t.
        File: results/<scenario>/logs/agents_final.csv (or agents_tXXX.csv)
        """
        # Respect the logging toggle
        if not getattr(self.cfg.logging, "snapshot_agents", False):
            return

        logs_dir = Path(self.out_dir) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Robustly pick incomes: prefer explicit pre-/net- names if present
        z_pre = getattr(self, "z_pre_tax", getattr(self, "z", None))
        z_net = getattr(self, "z_net", z_pre)

        # Guard: if arrays live in a dict, adapt as needed
        assert z_pre is not None, "Expected z_pre_tax or z on env"

        # degree vector if your network exposes it; fall back to zeros
        try:
            deg = self.net.degrees()            # e.g. np.ndarray shape (S,)
        except Exception:
            try:
                deg = self.net.deg()
            except Exception:
                deg = np.zeros(self.S, dtype=int)

        # deciles by *pre-tax* income (consistent with your panels)
        # If pandas.qcut is unavailable in your env, replace with a simple rank bin.
        decile = pd.qcut(pd.Series(z_pre), 10, labels=False) + 1

        df = pd.DataFrame({
            "id": np.arange(self.S, dtype=int),
            "z_pre": z_pre,
            "z_net": z_net,
            "y": self.y,                                # current status consumption
            "x": z_net - self.goods.p * self.y,         # implied private good
            "degree": deg,
            "decile": decile.astype(int),
            "t": t,
        })

        name = "agents_final.csv" if t >= (self.T - 1) else f"agents_t{t:03d}.csv"
        df.to_csv(logs_dir / name, index=False)

    def _draw_income(self, S: int) -> np.ndarray:
        pln = sample_pln(
            S=S,
            m=self.cfg.income.pln.m,
            alpha=self.cfg.income.pln.alpha,
            sigma=self.cfg.income.pln.sigma,
            rng=self.rng,
        )
        eps = self.rng.normal(0.0, self.eps_std, size=S)
        z = self.theta + pln + (self.beta_flag * eps)
        return np.maximum(z, 1e-8)

    # ---------- Taxes & rebates ----------
    def _percentiles(self, values: np.ndarray) -> np.ndarray:
        order = values.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(values) + 1)
        if len(values) == 1:
            return np.zeros_like(values, dtype=float)
        return 100.0 * (ranks - 1) / (len(values) - 1)

    def _apply_taxes(self, z: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        if not self.cfg.tax.on:
            return z.copy(), 0.0, np.zeros_like(z)
        a, b = self.cfg.tax.a, self.cfg.tax.b
        pi = self._percentiles(z)
        t_i = a + b * pi
        T_total = float(np.sum(t_i * z))
        s = T_total / self.S
        z_net = (1.0 - t_i) * z + s
        return z_net, s, t_i

    # ---------- One simulation step ----------
    def step(self):
        s = self.state

        # Taxes (z is fixed unless you add shocks)
        z_net, rebate, t_i = self._apply_taxes(s.z)

        # Best responses
        new_y = np.zeros(self.S, dtype=float)
        new_ybar = np.zeros(self.S, dtype=float)
        new_ymax = np.zeros(self.S, dtype=float)

        for i in range(self.S):
            ybar_i = self.net.avg_neighbor_y(i, s.y)
            deg_i = self.net.degree(i)
            obs = AgentView(
                i=i, z_net=float(z_net[i]), xi=self.xi, p=self.p,
                tau_i=float(self.tau[i]), ybar_i=float(ybar_i), deg_i=int(deg_i)
            )
            new_y[i] = self.policy.act(obs)
            new_ybar[i] = ybar_i
            new_ymax[i] = self.net.max_neighbor_y(i, s.y)

        # Budgets & utilities
        new_x = z_net - self.p * new_y
        new_phi = new_y - new_ybar
        new_U = utility_total(new_x, new_y, self.xi, self.gamma, new_phi, self.tau, new_ymax)

        # Commit
        self.state = SimState(
            z=s.z, z_net=z_net, x=new_x, y=new_y, U=new_U,
            phi=new_phi, ybar=new_ybar, y_max_nbr=new_ymax
        )

        # [UPDATED] Optionally update links
        if self.cfg.network.dynamic and (self.t + 1) % self.cfg.network.dynamics.reevaluate_every == 0:
            # Use the income that actually enters choices (net when taxes are ON)
            incomes_for_network = z_net if self.cfg.tax.on else s.z
            self.net.update_incomes(incomes_for_network)
            self.net.maybe_update()

        # [NEW] Compute paper metrics + write per-agent panel (use net income when taxes are ON)
        z_for_metrics = z_net if self.cfg.tax.on else s.z
        z_for_deciles = s.z  # deciles by pre-tax income to show incidence
        # Use PRE-TAX income in denominator for "share" to reveal incidence
        share_vec = status_share(self.p, new_y, s.z)
        metrics = compute_step_metrics(self.net.G, z_for_metrics, new_x, new_y, new_U, self.p)
        metrics.update({
            "t": self.t,
            "rebate": float(rebate),
            "edges": int(self.net.G.number_of_edges()),
        })
        self.summary_rows.append(metrics)

        # Persist per-agent panel each step (compact; notebooks will aggregate)
        write_panel(self.results_dir, self.t, s.z, new_x, new_y, new_U, share_vec)

        # [NEW] Persist decile panel; file now logs both PRE-TAX and NET shares (share_pretax_mean, share_net_mean).
        write_percentile_panel(self.results_dir, self.t, s.z, z_net, new_y, new_phi, new_U, self.p, z_ref=z_for_deciles)
        self.t += 1

    # ---------- Run loop ----------
    def run(self) -> pd.DataFrame:
        start = time.perf_counter()
        # pretty progress bar with ETA; one tick per step
        for _ in tqdm(range(self.T), total=self.T, ncols=100,
                      desc=f"[{self.cfg.run.scenario}] sim", leave=True):
            self.step()
        elapsed = time.perf_counter() - start
        # [UPDATED] central place to flush aggregated metrics
        write_step_summary(self.results_dir, self.summary_rows)
        print(f"[{self.cfg.run.scenario}] done in {elapsed:0.2f}s "
              f"({self.T} steps, ~{(elapsed/self.T):0.3f}s/step)")

        from pathlib import Path
        art = self.results_dir / "artifacts"
        art.mkdir(parents=True, exist_ok=True)

        stats = {}
        if hasattr(self, "policy") and hasattr(self.policy, "get_stats"):
            try:
                s = self.policy.get_stats()
                if isinstance(s, dict) and s:
                    stats = {
                        "scenario": self.cfg.run.scenario,
                        "agent": self.cfg.run.agent,
                        "S": self.S,
                        "T": self.T,
                        "llm_stats": s
                    }
            except Exception:
                stats = {"scenario": self.cfg.run.scenario, "agent": self.cfg.run.agent, "error": "stats_unavailable"}

        with open(art / "llm_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

           # Final per-agent snapshot for figure scripts
        self._snapshot_agents(self.t if hasattr(self, "t") else self.T - 1)

        return pd.DataFrame(self.summary_rows)