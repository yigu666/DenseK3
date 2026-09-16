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
export TORCH_CUDA_ARCH_LIST="7.5"

# Reconcile the locked Titan dependency set first.  Project and FLA metadata
# retain their canonical torch>=2.7 declarations, so both editable installs
# deliberately use --no-deps.  Actual compatibility is established by import,
# unit, and GPU smoke tests rather than falsifying package metadata.
"${ENV_ROOT}/bin/python" -m pip install \
  --progress-bar off \
  -r titan/env/requirements-titan.txt
# Expose the isolated FLA runtime only after dependency reconciliation; if it
# were visible to pip, Torch's exact triton==3.1 dependency would uninstall
# files from the overlay target.
export PYTHONPATH="${PROJECT_ROOT}/titan/runtime/triton-3.3${PYTHONPATH:+:${PYTHONPATH}}"
"${PROJECT_ROOT}/titan/scripts/prepare_fla_vendor.sh"
"${ENV_ROOT}/bin/python" -m pip install \
  --no-deps --no-build-isolation -e titan/vendor/flash-linear-attention
"${ENV_ROOT}/bin/python" -m pip install \
  --no-deps --no-build-isolation -e .

"${ENV_ROOT}/bin/python" - <<'PY'
import accelerate
import bitsandbytes
import datasets
import densek3_core
import einops
import fla
import pyarrow
import safetensors
import torch
import transformers

print("DENSEK3_IMPORT=PASS")
print(f"TORCH={torch.__version__}")
print(f"FLA={fla.__version__}")
print(f"TRANSFORMERS={transformers.__version__}")
print(f"DATASETS={datasets.__version__}")
print(f"PYARROW={pyarrow.__version__}")
print(f"BITSANDBYTES={bitsandbytes.__version__}")
PY

"${ENV_ROOT}/bin/python" -m pip freeze \
  > titan/manifests/pip-titan-project-freeze.txt
# A torch>=2.7 metadata warning is expected and preserved.  Do not hide it.
"${ENV_ROOT}/bin/python" -m pip check \
  > titan/manifests/pip-titan-project-check.txt || true
