#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ "$(uname -s)" != Linux ]]; then
  echo 'Run this script inside WSL2 Ubuntu / Linux.' >&2
  exit 2
fi
python3.12 -c 'import sys; assert sys.version_info[:2] == (3, 12)'
nvidia-smi
task_env="$HOME/.venvs/worldcheck-42c"
if [[ ! -d "$task_env" ]]; then
  python3.12 -m venv "$task_env"
fi
"$task_env/bin/python" -m pip install --upgrade pip
"$task_env/bin/python" -m pip install -r agent_pipeline_v4_2/requirements-vllm.txt
"$task_env/bin/python" -m pip check
echo "PYTHON=$task_env/bin/python"
