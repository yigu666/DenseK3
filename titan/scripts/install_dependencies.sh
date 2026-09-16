#!/usr/bin/env bash
set -euo pipefail

readonly PROJECT_ROOT="${DENSEK3_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
readonly ENV_ROOT="${PROJECT_ROOT}/envs/titan"

cd "${PROJECT_ROOT}"
export PIP_CONFIG_FILE="${PROJECT_ROOT}/titan/env/pip.conf"
export PIP_CACHE_DIR="${PROJECT_ROOT}/cache/pip"
export HF_HOME="${PROJECT_ROOT}/cache/huggingface"
export XDG_CACHE_HOME="${PROJECT_ROOT}/cache/xdg"
export PIP_PROGRESS_BAR=off

"${ENV_ROOT}/bin/python" -m pip install \
  --progress-bar off \
  -r titan/env/requirements-titan.txt
"${ENV_ROOT}/bin/python" titan/scripts/probe_runtime.py
"${ENV_ROOT}/bin/python" -m pip check \
  > titan/manifests/pip-titan-check.txt
"${ENV_ROOT}/bin/python" -m pip freeze \
  > titan/manifests/pip-titan-freeze.txt
sha256sum \
  titan/env/requirements-titan.txt \
  titan/scripts/probe_runtime.py \
  titan/manifests/runtime-capability.json \
  > titan/manifests/runtime-files.sha256
