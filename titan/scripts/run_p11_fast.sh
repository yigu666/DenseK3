#!/usr/bin/env bash

set -euo pipefail

readonly PROJECT_ROOT="${DENSEK3_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
readonly REQUIRED_TRITON_ROOT="$PROJECT_ROOT/titan/runtime/triton-3.3"
export REQUIRED_TRITON_ROOT

cd "$PROJECT_ROOT"
source titan/scripts/activate_titan.sh

python - <<'PY'
import os
from pathlib import Path

import triton

required = Path(os.environ["REQUIRED_TRITON_ROOT"]).resolve()
loaded = Path(triton.__file__).resolve()
if required not in loaded.parents:
    raise SystemExit(
        "P11.6 FAST requires the project-local Triton 3.3 overlay; "
        f"loaded {triton.__version__} from {loaded}"
    )
print(f"P11_6_FAST_TRITON_RUNTIME=PASS VERSION={triton.__version__}", flush=True)
PY

exec python titan/scripts/run_p11_fast.py "$@"
