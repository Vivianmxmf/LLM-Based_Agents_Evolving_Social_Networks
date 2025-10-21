## LLM-Based Agents for Status Dynamics in Evolving Social Networks

A lightweight, reproducible codebase to replicate and extend the status–competition model (Antinyan et al., 2019) using:


* Scripted Best-Response (BR) agents (paper-faithful baseline), and
* Language-Model (LLM) agents (discrete policy over status–spending shares via Ollama).

It supports fixed vs. endogenous networks, low vs. high inequality, and progressive tax + equal rebate. The repo includes LLM telemetry, grid-size ablations, seed sweeps, and a “norm nudge” prompt experiment.


### Scenarios

We follow six canonical scenarios (plus LLM variants):

| Scenario | Inequality | Network | Redistribution |
| :--- | :--- | :--- | :--- |
| LF | Low | Fixed | Off |
| HF | High | Fixed | Off |
| LE | Low | Endogenous | Off |
| HE | High | Endogenous | Off |
| HFR | High | Fixed | On (tax+rebate) |
| HER | High | Endogenous | On (tax+rebate) |

### Agents

* **BR:** myopic best response $y_i^\*$ derived from the model’s utility (continuous choice).
* **LLM:** selects a discrete share $s \in \{0,\frac{1}{K-1},\dots,1\}$ via a one-line response (INDEX=k), mapped to $y = s \cdot z^{net}/p$. Strict parsing + BR fallback guarantee progress (and are fully logged).
* **Hybrid (optional):** call LLM with probability $\varphi$; otherwise use BR (compute efficient).

---

## 2) Repository layout

```markdown
signature-work/
├─ configs/                      # Scenario YAMLs (BR & LLM)
│  ├─ EXP_LF_br.yaml
│  ├─ EXP_HF_br.yaml
│  ├─ EXP_LE_br.yaml
│  ├─ EXP_HE_br.yaml
│  ├─ EXP_HFR_br.yaml
│  ├─ EXP_HER_br.yaml
│  ├─ EXP_*_llm_*.yaml          # LLM configs (grid K=11,21, nudges, etc.)
│  └─ base.yaml                  # Common defaults included by scenarios
├─ src/
│  ├─ run.py                     # Entry point (see §4)
│  ├─ env/
│  │  ├─ model.py                # StatusSimEnv (loop, net-income pipeline)
│  │  ├─ scenarios.py            # YAML loader & normalization helpers
│  │  ├─ model_bootstrap.py      # build_env(cfg)
│  │  └─ network.py              # SocialNetwork & link updates
│  ├─ agents/
│  │  ├─ br_agent.py             # Scripted best response
│  │  └─ llm_agent.py            # Discrete-share LLM policy
│  ├─ llm/
│  │  └─ client_ollama.py        # HTTP client to Ollama (timeout, retries)
│  ├─ utils/
│  │  ├─ seed.py                 # set_all(seed)
│  │  └─ logging.py              # CSV/JSON log helpers (aggregates, panels, telemetry)
│  └─ …
├─ scripts/
│  ├─ make_fig_rq4_net_vs_pretax.py     # HE vs HER: net- vs pre-tax share panels
│  ├─ make_four_distributions.py        # Distribution plots (income, share, status, utility)
│  ├─ summarize_grid_ci.py              # Compute mean±95%CI across seed runs
│  ├─ align_compare.py                  # Decile Δ alignment (LLM vs BR)
│  └─ …
├─ results/                    # Created at runtime (per scenario)
│  └─ EXP_*/ 
│     ├─ logs/
│     │  ├─ aggregates.csv
│     │  ├─ panel_by_decile.csv
│     │  ├─ agents_final.csv (if enabled)
│     │  └─ snapshots/ (if enabled)
│     └─ artifacts/
│        └─ llm_stats.json     # telemetry (LLM runs)
├─ figs/                       # Saved figures (recommended path convention)
└─ README.md                   # (this file)
```


---

## 3) Installation

### Python
```bash
# Recommended: Python 3.10–3.12
conda create -n sw312 python=3.12 -y
conda activate sw312
pip install -r requirements.txt
```


### LLM (Ollama)

For local, small-footprint models (Mac, 8 GB RAM):

```bash
# Install Ollama: https://ollama.com
ollama serve &                   # start background server on :11434
ollama pull llama3               # or mistral / codellama (consistent with your YAMLs)
export LLM_MODEL=llama3
export LLM_BASE_URL=http://localhost:11434
```


## 4) Running experiments

Basic (BR)
```bash
# From project root
python -m src.run --config configs/EXP_HF_br.yaml
```

LLM (discrete shares)
```bash
python -m src.run --config configs/EXP_HE_llm_grid11.yaml
```

The engine prints a scenario banner and seed, and writes logs to:
results/<SCENARIO>/logs/aggregates.csv
results/<SCENARIO>/logs/panel_by_decile.csv


LLM runs also write telemetry:
results/<SCENARIO>/artifacts/llm_stats.json


### Agent override (CLI)

Any scenario can be forced to a different agent:
```bash
python -m src.run --config configs/EXP_HF_br.yaml --agent llm
python -m src.run --config configs/EXP_LE_llm_grid11.yaml --agent br
The scenario name is auto-suffixed (e.g., _llm) to avoid clobbering outputs.
```



## 5) Key configuration notes (YAML)

Net income pipeline (policy on)
When tax.on: true, the simulator computes disposable income once at init:
$z^{net}_i = (1 - t_i)\, z_i + s$
and uses $z^{net}$ for budgets, choices, and logging in HER/HFR.
YAML 1.1 “on” quirk
Some parsers read on: as boolean. env/scenarios.py includes a normalizer:

```python
def _normalize_tax_keys(d: dict) -> dict:
    if not isinstance(d, dict): return d
    tax = d.get("tax")
    if isinstance(tax, dict) and True in tax and "on" not in tax:
        val = bool(tax.pop(True))
        tax["on"] = val
    return d
```

This ensures tax.on: true is honored before best responses are computed.
LLM grid size
Set the number of discrete shares with policy.grid.K (e.g., 11 or 21).
Smaller K ⇒ faster, more fallback; Larger K ⇒ better alignment, slightly slower.
Seeds & batch runs
Each YAML may include run.seed. For multi-seed sweeps, copy/patch configs or use shell loops (see §8).

## 6) Outputs & columns

logs/aggregates.csv
content... (per step; last row = endpoint)
W, gini_z, gini_U, rho_z, assort, Cl,
share_mean, y_mean, x_mean, U_mean, t, rebate, edges


logs/panel_by_decile.csv
content... (final period by income decile)
decile, z_mean (or z_net_mean under tax), y_mean, U_mean, share_mean, ...


artifacts/llm_stats.json
```json
{
  "scenario": "EXP_HE_llm_grid11",
  "llm_model": "llama3",
  "grid_K": 11,
  "calls": 1585,
  "success": 403,
  "fallback": { "parse": 1182, "timeout": 0 },
  "latency": { "mean_s": 0.331, "p95_s": 0.72, "max_s": 1.88 }
}
```


## 7) Reproducing key figures

Path convention: save figures under figs/<EXPERIMENT>_<meaning>.png.

RQ4: Net vs. pre-tax share (HE vs. HER)

Requires panel_by_decile.csv for both HE and HER; HER must include z_net_mean.

```bash
python scripts/make_fig_rq4_net_vs_pretax.py \
  --he  results/EXP_HE_br/logs/panel_by_decile.csv \
  --her results/EXP_HER_br/logs/panel_by_decile.csv \
  --out figs/rq4_share_by_decile_net_vs_pretax.png \
  --p 2.0
```
If the two curves look identical, you likely logged z_mean only. Re-run HER with net-income logging enabled so panel_by_decile.csv contains z_net_mean.


### Distributions (income, share, status, utility)
Needs an agent-level snapshot; if you don't see one, enable final snapshots in the config.
```bash
python scripts/make_four_distributions.py \
  --logs results/EXP_HF_br/logs \
  --out  figs/EXP_HF_br_four_dists.png
```

### Time-series and BR vs. LLM panels

Helpers in scripts/ read aggregates.csv and panel_by_decile.csv to:


plot LLM vs BR share-by-decile (EXP_*_BRvLLM_share.png);
plot time-series for share/utility (*_ts_share_U.png);
render grid CI summaries (*_llm_grid_CI.png) from seed sweeps via summarize_grid_ci.py.



## 8) Multi-seed sweeps & CI tables

Example: LE, K∈{11,21}, 5 seeds each
```bash
# LE, K=11
for s in 301 302 303 304 305; do
  cp configs/EXP_LE_llm_grid11.yaml /tmp/EXP_LE_llm_grid11_s${s}.yaml
  # (optional) yq/sed to set run.seed: s
  python -m src.run --config /tmp/EXP_LE_llm_grid11_s${s}.yaml
done
```
```bash
# LE, K=21
for s in 401 402 403 404 405; do
  cp configs/EXP_LE_llm_grid21.yaml /tmp/EXP_LE_llm_grid21_s${s}.yaml
  python -m src.run --config /tmp/EXP_LE_llm_grid21_s${s}.yaml
done
```


Summarize with 95% CIs
```bash
python scripts/summarize_grid_ci.py \
  --glob "results/EXP_LE_llm_grid11_s*/logs/aggregates.csv" \
  --label "LE LLM K=11" \
  --out   figs/EXP_LE_llm_grid11_CI.txt

python scripts/summarize_grid_ci.py \
  --glob "results/EXP_LE_llm_grid21_s*/logs/aggregates.csv" \
  --label "LE LLM K=21" \
  --out   figs/EXP_LE_llm_grid21_CI.txt
```


## 9) LLM “norm-nudge” experiment (language-only policy)

Example config: configs/EXP_HE_llm_nudge.yaml
Same numeric primitives as HE, but prepend a status-skeptical policy paragraph in the prompt.
```bash
python -m src.run --config configs/EXP_HE_llm_nudge.yaml
```

Compare decile shares vs. the HE LLM baseline:
```bash
python scripts/align_compare.py \
  --base results/EXP_HE_llm_grid11/logs/panel_by_decile.csv \
  --tgt  results/EXP_HE_llm_nudge/logs/panel_by_decile.csv \
  --metric share_mean \
  --out figs/EXP_HE_llm_nudge_delta_share.png
```


## 10) Troubleshooting

### A) HER identical to HE

Ensure tax.on: true is read correctly. The loader normalizes YAML booleans (on:) to a string key "on".
Confirm net-income pipeline is used at init and logged: z_net_mean must appear in HER panel_by_decile.csv.
Aggregates should reflect policy (e.g., lower gini_z, lower share_mean).

### B) LLM calls hang / timeout

Start Ollama: ollama serve &
Confirm env vars: LLM_BASE_URL=http://localhost:11434, LLM_MODEL=llama3 (or mistral).
Use small settings on low-RAM (e.g., S≤500, T≤200, K=11).
Increase timeout in llm/client_ollama.py if needed.

### C) Missing agent-level snapshots

make_four_distributions.py expects agents_final.csv or snapshot_t*.csv.
Enable snapshots in your YAML (e.g., a boolean like logging.snapshots: true or logging.agents_final: true, depending on your config schema).

### D) Paths in scripts

Use normalized CSV paths:

HE: results/EXP_HE_br/logs/panel_by_decile.csv
HER: results/EXP_HER_br/logs/panel_by_decile.csv
If a plotting script errors about shape mismatch, check that both inputs have exactly 10 deciles and the same column names (share_mean, z_net_mean).


## 11) Reproducibility checklist

Fix seeds (run.seed) and record them in result folder names (e.g., _s311).
Persist configs alongside outputs (the loader does this).
Use deterministic LLM decoding (temperature = 0) for baseline runs.
Keep all llm_stats.json artifacts; they explain fallback/latency behaviors.


## 12) Citing the reference model

Antinyan, A., et al. (2019). Social status competition and the impact of income inequality in evolving social networks. (Used here for ABM structure and comparisons.)


## 13) License & acknowledgments

Research/teaching use only.
Thanks to the original authors for the baseline design, and to the LLM-MAS community for open tools.


## 14) Quick start recipes

Run a full BR set (quick):
```bash
python -m src.run --config configs/EXP_LF_br.yaml
python -m src.run --config configs/EXP_HF_br.yaml
python -m src.run --config configs/EXP_LE_br.yaml
python -m src.run --config configs/EXP_HE_br.yaml
python -m src.run --config configs/EXP_HFR_br.yaml
python -m src.run --config configs/EXP_HER_br.yaml
```
Run tiny LLM baselines (K=11):
```bash
python -m src.run --config configs/EXP_LF_llm_grid11.yaml
python -m src.run --config configs/EXP_HF_llm_grid11.yaml
python -m src.run --config configs/EXP_LE_llm_grid11.yaml
python -m src.run --config configs/EXP_HE_llm_grid11.yaml
```
Grid ablation (HE, K=21, five seeds):
```bash
for s in 411 412 413 414 415; do
  cp configs/EXP_HE_llm_grid21.yaml /tmp/EXP_HE_llm_grid21_s${s}.yaml
  python -m src.run --config /tmp/EXP_HE_llm_grid21_s${s}.yaml
done
```
Make the RQ4 net vs. pre-tax figure:
```bash
python scripts/make_fig_rq4_net_vs_pretax.py \
  --he  results/EXP_HE_br/logs/panel_by_decile.csv \
  --her results/EXP_HER_br/logs/panel_by_decile.csv \
  --out figs/rq4_share_by_decile_net_vs_pretax.png \
  --p 2.0
```
Final notes

Figure paths standardized to figs/<EXPERIMENT>_<meaning>.png.
tax.on normalized at load time to avoid YAML 1.1 pitfalls.
HER/HFR net-income pipeline is enforced in env/model.py; HE ≠ HER once logging includes z_net.
On 8 GB RAM, stick with small populations/horizons and K=11. For higher fidelity, run K=21 and seed sweeps on a server (you can skip LLM telemetry on headless runs if Ollama isn’t available).
