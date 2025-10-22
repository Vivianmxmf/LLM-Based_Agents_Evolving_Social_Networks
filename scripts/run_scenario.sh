# scripts/run_scenario.sh
#!/usr/bin/env bash
set -euo pipefail
SCENARIO=${1:-"configs/HE.yaml"}
python -m src.run --config "$SCENARIO"