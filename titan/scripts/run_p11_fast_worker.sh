#!/usr/bin/env bash

# Resumable background worker for P11.6 FAST.  It holds a project-local lock,
# forwards SIGTERM to the active runner, and retries only memory-envelope
# deferrals or a one-shot no-retry API pause.  Scientific failures stop it.

set -uo pipefail

readonly PROJECT_ROOT="${DENSEK3_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
readonly REPORT_DIR="$PROJECT_ROOT/titan/manifests/reproduction/p11-t/p11-6-fast"
readonly LOCK_PATH="$REPORT_DIR/automation/p11-6-fast.lock"
readonly PID_PATH="$REPORT_DIR/automation/p11-6-fast.pid"
readonly STATE_PATH="$REPORT_DIR/automation/p11-6-fast-worker.json"
readonly LAST_ERROR="$REPORT_DIR/p11-6-fast-last-error.json"
readonly RETRY_SECONDS=300

mkdir -p "$REPORT_DIR/automation"
cd "$PROJECT_ROOT"
exec 9>"$LOCK_PATH"
if ! flock -n 9; then
  printf '%s\n' "P11_6_FAST_WORKER=ALREADY_RUNNING"
  exit 0
fi
printf '%s\n' "$$" >"$PID_PATH"

child_pid=""

write_state() {
  local status="$1"
  local detail="$2"
  python - "$STATE_PATH" "$status" "$detail" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

path = Path(sys.argv[1])
record = {
    "stage": "P11.6-FAST-BACKGROUND-WORKER",
    "status": sys.argv[2],
    "detail": sys.argv[3],
    "completed_at": datetime.now().astimezone().isoformat(),
    "heldout_accessed": False,
    "p11_6_2m_automatically_allowed": False,
    "p11_7_allowed": False,
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY
}

cleanup() {
  rm -f "$PID_PATH"
}

stop_child() {
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill -TERM "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
}

handle_signal() {
  write_state "STOPPED_BY_SIGNAL" "Worker stopped; all trajectory, API, optimizer, scaler, and RNG progress is resumable."
  stop_child
  exit 143
}

trap cleanup EXIT
trap handle_signal INT TERM

source titan/scripts/activate_p11_kimi_api.sh
set +e
set -uo pipefail

write_state "RUNNING" "P11.6 FAST idempotent runner started."
while true; do
  error_before="MISSING"
  [[ -f "$LAST_ERROR" ]] && error_before="$(sha256sum "$LAST_ERROR" | awk '{print $1}')"
  titan/scripts/run_p11_fast.sh --phase all &
  child_pid=$!
  wait "$child_pid"
  rc=$?
  child_pid=""
  if [[ "$rc" -eq 0 ]]; then
    write_state "COMPLETE" "P11.6 FAST formal closeout completed."
    exit 0
  fi
  error_after="MISSING"
  [[ -f "$LAST_ERROR" ]] && error_after="$(sha256sum "$LAST_ERROR" | awk '{print $1}')"
  if [[ "$error_after" != "$error_before" ]]; then
    write_state "STOPPED_ERROR" "Runner persisted a new scientific or technical error; no automatic retry was made."
    exit "$rc"
  fi
  status="$(python - <<'PY'
import json
from pathlib import Path

root = Path("titan/manifests/reproduction/p11-t/p11-6-fast")
values = []
for path in root.rglob("*.json"):
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    status = record.get("status")
    if status in {"DEFERRED_GPU_MEMORY_UNSAFE", "PAUSED"}:
        values.append((path.stat().st_mtime_ns, status, record.get("stop_reason")))
    elif record.get("allowed") is False and "memory_envelope" in record:
        values.append((path.stat().st_mtime_ns, "DEFERRED_GPU_MEMORY_UNSAFE", None))
print(max(values)[1] if values else "UNKNOWN")
PY
)"
  if [[ "$status" == "DEFERRED_GPU_MEMORY_UNSAFE" || "$status" == "PAUSED" ]]; then
    write_state "WAITING_RETRYABLE_GATE" "Waiting for memory envelope or next unattempted API slot; failed/uncertain requests will never be retried."
    sleep "$RETRY_SECONDS"
    continue
  fi
  write_state "STOPPED_NONRETRYABLE" "Runner exited without a retryable persisted Gate."
  exit "$rc"
done
