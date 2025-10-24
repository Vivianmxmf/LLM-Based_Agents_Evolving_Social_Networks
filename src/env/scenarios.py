# src/env/scenarios.py
from __future__ import annotations
from pathlib import Path
from typing import Literal, Optional
import os
import yaml
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal

class LLMParams(BaseModel):
    grid_K: int = Field(11, ge=2, description="Number of discrete shares in [0,1].")
    max_tokens: int = Field(6, ge=1, le=64, description="Response cap for INDEX=…")


# ---------- Typed configuration (validated) ----------

class PLNParams(BaseModel):
    m: float = Field(..., gt=0, description="Lognormal body scale")
    alpha: float = Field(..., gt=1.0, description="Pareto tail exponent (>1)")
    sigma: float = Field(..., gt=0.0, description="Lognormal sigma (>0)")

class IncomeParams(BaseModel):
    theta: float = 5.0
    beta: int = Field(0, ge=0, le=1, description="0=low inequality; 1=high inequality")
    eps_std: float = 1.0
    pln: PLNParams

class GoodsParams(BaseModel):
    p: float = Field(2.0, gt=0.0)
    xi: float = Field(0.6, gt=0.0, lt=1.0)

class PrefParams(BaseModel):
    gamma_mean: float = 1.0
    gamma_std: float = Field(0.1, ge=0.0)
    tau_min: float = 0.0
    tau_max: float = 1.0

    @field_validator("tau_max")
    @classmethod
    def _tau_bounds(cls, v, info):
        tau_min = info.data.get("tau_min", 0.0)
        if v <= tau_min:
            raise ValueError("tau_max must be > tau_min")
        return v

class TaxParams(BaseModel):
    on: bool = False
    a: float = 0.0997
    b: float = 0.003

class ERParams(BaseModel):
    p_edge: float = Field(0.02, ge=0.0, le=1.0)

class BAParams(BaseModel):
    m_attach: int = Field(2, ge=1)

class WSParams(BaseModel):
    k_nei: int = Field(6, ge=2)
    beta_rewire: float = Field(0.05, ge=0.0, le=1.0)

class DynamicRules(BaseModel):
    reevaluate_every: int = Field(1, ge=1)
    add_prob: float = Field(0.03, ge=0.0, le=1.0)
    drop_prob: float = Field(0.02, ge=0.0, le=1.0)
    max_degree: int = Field(80, ge=2)

class NetworkParams(BaseModel):
    dynamic: bool = False
    type: Literal["erdos_renyi", "barabasi_albert", "watts_strogatz"] = "erdos_renyi"
    erdos_renyi: ERParams = ERParams()
    barabasi_albert: BAParams = BAParams()
    watts_strogatz: WSParams = WSParams()
    dynamics: DynamicRules = DynamicRules()

class LoggingParams(BaseModel):
    write_csv: bool = True
    write_parquet: bool = False
    snapshot_config: bool = True

class RunParams(BaseModel):
    scenario: str = "BASE"
    seed: Optional[int] = 42
    steps: int = Field(200, ge=1)
    population: int = Field(500, ge=2)
    results_dir: str = "results"
    log_every: int = Field(10, ge=1)
    agent: Literal["br", "llm", "random", "hybrid"] = "br" 

class SimConfig(BaseModel):
    run: RunParams
    income: IncomeParams
    goods: GoodsParams
    preferences: PrefParams
    tax: TaxParams
    network: NetworkParams
    logging: LoggingParams
    llm: Optional[LLMParams] = None

# ---------- Loading & merging ----------

def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def _normalize_tax_keys(d: dict) -> dict:
    """Fix YAML 1.1 quirk where key 'on' may parse as boolean True.
    Ensures tax dict has a string key 'on'.
    """
    if not isinstance(d, dict):
        return d
    tax = d.get("tax")
    if isinstance(tax, dict) and True in tax and "on" not in tax:
        # move boolean True key to 'on'
        val = bool(tax.pop(True))
        tax["on"] = val
    return d

def load_config(config_path: os.PathLike | str) -> SimConfig:
    """
    Load base.yaml and merge with the given scenario YAML.
    Returns a validated SimConfig.
    """
    cfg_dir = Path(config_path).resolve().parent
    base_path = cfg_dir / "base.yaml"
    with base_path.open("r") as f:
        base = _normalize_tax_keys(yaml.safe_load(f) or {})
    with Path(config_path).open("r") as f:
        over = _normalize_tax_keys(yaml.safe_load(f) or {})
    merged = _deep_merge(base, over)
    return SimConfig.model_validate(merged)

# ---------- Convenience helpers ----------

def scenario_summary(cfg: SimConfig) -> str:
    n = cfg.run.population
    steps = cfg.run.steps
    net_mode = "Endogenous" if cfg.network.dynamic else "Fixed"
    ineq = "High" if cfg.income.beta == 1 else "Low"
    tax = "On" if cfg.tax.on else "Off"
    return (
        f"[Scenario {cfg.run.scenario}] S={n}, T={steps} | "
        f"Inequality={ineq}, Taxes={tax}, Network={net_mode}({cfg.network.type}) | "
        f"p={cfg.goods.p}, xi={cfg.goods.xi} | theta={cfg.income.theta}, "
        f"PLN(m={cfg.income.pln.m}, alpha={cfg.income.pln.alpha}, sigma={cfg.income.pln.sigma})"
    )