# (update) src/run.py  — minimal wiring to actually run the loop now
from __future__ import annotations
import argparse
from pathlib import Path
from utils.seed import set_all
from env.scenarios import load_config, scenario_summary
from env.model_bootstrap import build_env
import argparse

from env.model import StatusSimEnv
from env.scenarios import load_config  # if you already have a loader; keep as-is
from env import scenarios as _scen     # only if your loader lives there
from env.model_bootstrap import build_env

def parse_args():
    ap = argparse.ArgumentParser(description="Signature-Work Simulation")
    ap.add_argument("--config", type=str, required=True, help="Path to scenario YAML (e.g., configs/HE.yaml)")
    return ap.parse_args()

def _load_cfg(config_path: str) -> SimConfig:
    # Use your project’s existing loader if present
    try:
        return load_config(config_path)  # type: ignore[attr-defined]
    except Exception:
        # robust fallback if your run.py didn’t export load_config
        import yaml
        from env.scenarios import SimConfig
        data = yaml.safe_load(Path(config_path).read_text())
        return SimConfig.model_validate(data)

def main():
    ap = argparse.ArgumentParser(description="Status-sim runner")
    ap.add_argument("--config", "-c", required=True, help="Path to scenario YAML (under configs/)")
    ap.add_argument("--agent", choices=["br", "llm", "random"],
                    help="Override agent policy (br|llm|random). If set, scenario name is suffixed to avoid clobbering results.")
    args = ap.parse_args()

    # Resolve config path so running from within src/ works (expects configs/ at project root)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        candidate = (project_root / args.config).resolve()
        if candidate.exists():
            config_path = candidate
        else:
            config_path = (Path.cwd() / args.config).resolve()

    cfg = _load_cfg(str(config_path))

    # Ensure results write under project root "results/" (not src/results when running from src)
    project_root = Path(__file__).resolve().parent.parent
    abs_results_dir = (project_root / cfg.run.results_dir).resolve()
    cfg.run.results_dir = str(abs_results_dir)

    # Build env via factory (respects cfg.run.agent)
    env = build_env(cfg)

    # Optional CLI override for agent
    if args.agent:
        cfg.run.agent = args.agent  # type: ignore[assignment]
        # Suffix scenario to keep results tidy (e.g., EXP_HF_br -> EXP_HF_br_llm)
        suffix = f"_{args.agent}"
        if not cfg.run.scenario.endswith(suffix):
            cfg.run.scenario = f"{cfg.run.scenario}{suffix}"

    # Build env via factory (respects cfg.run.agent)
    env = build_env(cfg)

    print(f"[Scenario {cfg.run.scenario}] S={cfg.run.population}, T={cfg.run.steps} | "
          f"Inequality={'High' if cfg.income.beta else 'Low'}, Taxes={'On' if cfg.tax.on else 'Off'}, "
          f"Network={'Endogenous' if cfg.network.dynamic else 'Fixed'} | p={cfg.goods.p}, xi={cfg.goods.xi}")
    if getattr(cfg.run, "seed", None) is not None:
        print(f"Seed -> {cfg.run.seed}")

    env.run()
    
 

    out = Path(cfg.run.results_dir) / cfg.run.scenario / "logs" / "aggregates.csv"
    print(f"Saved per-step aggregates to: {out.resolve()}")

if __name__ == "__main__":
    main()