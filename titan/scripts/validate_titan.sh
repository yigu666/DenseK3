#!/usr/bin/env bash
set -euo pipefail

readonly PROJECT_ROOT="${DENSEK3_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_ROOT}"
source titan/scripts/activate_titan.sh

mkdir -p titan/manifests/validation
ruff check pyproject.toml src tests scripts \
  | tee titan/manifests/validation/ruff.log
python -m pytest tests -q \
  | tee titan/manifests/validation/pytest.log
python titan/scripts/probe_runtime.py \
  | tee titan/manifests/validation/runtime.log

echo "TITAN_VALIDATION=PASS"
