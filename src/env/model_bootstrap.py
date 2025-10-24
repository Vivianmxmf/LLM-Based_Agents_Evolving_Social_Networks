# src/env/model_bootstrap.py  (replace its contents with:)
from __future__ import annotations
from env.scenarios import SimConfig
from env.model import StatusSimEnv
from agents.best_response import BestResponsePolicy
from agents.random_agent import RandomPolicy
from agents.llm_agent import LLMPolicy
from agents.hybrid_llm_br import HybridLLMBR

def build_env(cfg: SimConfig) -> StatusSimEnv:
    env = StatusSimEnv(cfg)
    if cfg.run.agent == "br":
        env.policy = BestResponsePolicy()
    elif cfg.run.agent == "random":
        env.policy = RandomPolicy(env.rng)
    elif cfg.run.agent == "llm":
        grid_K = (cfg.llm.grid_K if getattr(cfg, "llm", None) and cfg.llm.grid_K else 11)
        max_tok = (cfg.llm.max_tokens if getattr(cfg, "llm", None) and cfg.llm.max_tokens else 6)
        env.policy = LLMPolicy(grid_K=grid_K, max_tokens=max_tok)
    elif cfg.run.agent == "hybrid":
        grid_K = (cfg.llm.grid_K if getattr(cfg, "llm", None) and cfg.llm.grid_K else 11)
        max_tok = (cfg.llm.max_tokens if getattr(cfg, "llm", None) and cfg.llm.max_tokens else 6)
        env.policy = HybridLLMBR(frac_llm=0.2, rng=env.rng, llm_grid_K=grid_K, llm_max_tokens=max_tok)
    else:
        raise ValueError(f"Unknown agent: {cfg.run.agent}")
    return env