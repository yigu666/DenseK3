#!/usr/bin/env bash

# Resumable Titan-only supervisor for the end of the bounded P11.5a probe.
#
# It waits for the already-running Wave 3 process, then invokes the existing
# idempotent P11.5a runner until Segment 3 and the formal closeout have either
# completed or produced a non-retryable failure. It never starts P11.6 training.

set -uo pipefail

readonly PROJECT_ROOT="${DENSEK3_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
readonly REPORT_DIR="${PROJECT_ROOT}/titan/manifests/reproduction/p11-t/p11-5a"
readonly WAVE_DIR="${REPORT_DIR}/wave-3"
readonly WAVE_PID_PATH="${WAVE_DIR}/p11-5a-wave-3.pid"
readonly WAVE_REPORT="${WAVE_DIR}/kimi-wave.json"
readonly SEGMENT_REPORT="${WAVE_DIR}/training-segment.json"
readonly SEGMENT_GATE_REPORT="${WAVE_DIR}/training-segment-gpu-gate.json"
readonly FINAL_REPORT="${REPORT_DIR}/p11-5a-final-check.json"
readonly LAST_ERROR_REPORT="${REPORT_DIR}/p11-5a-last-error.json"
readonly STATE_DIR="${REPORT_DIR}/automation"
readonly STATE_REPORT="${STATE_DIR}/p11-5a-auto-closeout.json"
readonly PID_PATH="${STATE_DIR}/p11-5a-auto-closeout.pid"
readonly LOCK_PATH="${STATE_DIR}/p11-5a-auto-closeout.lock"
readonly WAVE_POLL_SECONDS=60
readonly DEFERRED_RETRY_SECONDS=300

mkdir -p "${STATE_DIR}"
cd "${PROJECT_ROOT}"

exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
  printf '%s\n' "P11_5A_AUTO_CLOSEOUT=ALREADY_RUNNING"
  exit 0
fi

printf '%s\n' "$$" >"${PID_PATH}"

child_pid=""

timestamp() {
  date --iso-8601=seconds
}

log() {
  printf '%s %s\n' "$(timestamp)" "$*"
}

json_value() {
  local path="$1"
  local key="$2"
  python - "${path}" "${key}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
try:
    value = json.loads(path.read_text(encoding="utf-8"))
    for part in key.split("."):
        value = value[part]
except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError):
    print("MISSING")
else:
    if isinstance(value, bool):
        print(str(value).lower())
    elif value is None:
        print("null")
    else:
        print(value)
PY
}

write_state() {
  local status="$1"
  local stage="$2"
  local message="$3"
  local p11_6_allowed="${4:-null}"
  python - "${STATE_REPORT}" "${status}" "${stage}" "${message}" "${p11_6_allowed}" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

path = Path(sys.argv[1])
allowed_text = sys.argv[5]
allowed = None if allowed_text == "null" else allowed_text == "true"
record = {
    "stage": "P11.5a-AUTO-CLOSEOUT",
    "status": sys.argv[2],
    "current_stage": sys.argv[3],
    "message": sys.argv[4],
    "completed_at": datetime.now().astimezone().isoformat(),
    "p11_6_phase1_allowed": allowed,
    "p11_6_training_started": False,
    "p12_titan_migration_allowed": False,
    "heldout_accessed": False,
    "result_marker": f"P11_5A_AUTO_CLOSEOUT={sys.argv[2]}",
}
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY
}

stop_child() {
  if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
    kill -TERM "${child_pid}" 2>/dev/null || true
    wait "${child_pid}" 2>/dev/null || true
  fi
  child_pid=""
}

cleanup() {
  stop_child
  rm -f "${PID_PATH}"
}

handle_signal() {
  log "P11_5A_AUTO_CLOSEOUT=STOPPED_BY_SIGNAL"
  write_state "STOPPED_BY_SIGNAL" "CONTROL" "Supervisor stopped by SIGINT/SIGTERM; resumable artifacts were preserved."
  exit 143
}

trap cleanup EXIT
trap handle_signal INT TERM

file_hash() {
  if [[ -f "$1" ]]; then
    sha256sum "$1" | awk '{print $1}'
  else
    printf '%s\n' MISSING
  fi
}

run_probe_all() {
  local error_before
  local error_after
  local rc
  error_before="$(file_hash "${LAST_ERROR_REPORT}")"
  titan/scripts/run_p11_probe.sh --phase all &
  child_pid=$!
  wait "${child_pid}"
  rc=$?
  child_pid=""
  error_after="$(file_hash "${LAST_ERROR_REPORT}")"
  if [[ "${error_after}" != "${error_before}" ]]; then
    return 70
  fi
  return "${rc}"
}

log "P11_5A_AUTO_CLOSEOUT=STARTED PID=$$"
write_state "WAITING" "WAVE_3" "Waiting for the existing Wave 3 process to finish."

while true; do
  if [[ -f "${WAVE_PID_PATH}" ]]; then
    wave_pid="$(tr -d '[:space:]' <"${WAVE_PID_PATH}")"
    if [[ -n "${wave_pid}" ]] && kill -0 "${wave_pid}" 2>/dev/null; then
      sleep "${WAVE_POLL_SECONDS}"
      continue
    fi
  fi
  wave_status="$(json_value "${WAVE_REPORT}" status)"
  if [[ "${wave_status}" == "PASS" ]]; then
    break
  fi
  # A paused Kimi wave may be resumed safely: completed/failed attempt files
  # are immutable and the runner only consumes previously unattempted slots.
  if [[ "${wave_status}" == "PAUSED" ]] && \
    [[ "$(json_value "${WAVE_REPORT}" stop_reason)" == "PAUSED_AFTER_SINGLE_API_FAILURE" ]]; then
    break
  fi
  log "P11_5A_AUTO_CLOSEOUT=STOPPED_WAVE3_NOT_PASS STATUS=${wave_status}"
  write_state "STOPPED_WAVE3_NOT_PASS" "WAVE_3" "Wave 3 exited without a resumable PASS/PAUSED result."
  exit 2
done

# Load the secret only after the original Wave 3 process has exited. It is used
# solely if unattempted Kimi slots remain after a no-retry API failure.
if ! source titan/scripts/activate_p11_kimi_api.sh; then
  log "P11_5A_AUTO_CLOSEOUT=STOPPED_API_ACTIVATION_FAILED"
  write_state "STOPPED_API_ACTIVATION_FAILED" "WAVE_3" "Kimi API environment activation failed."
  exit 2
fi
# The API activation helper intentionally enables `set -e` for interactive
# safety. This supervisor must instead inspect the runner's non-zero deferred
# status and retry only the approved GPU-gate case.
set +e
set -uo pipefail

while true; do
  final_status="$(json_value "${FINAL_REPORT}" status)"
  if [[ "${final_status}" == "PASS" || "${final_status}" == "FAIL" ]]; then
    allowed="$(json_value "${FINAL_REPORT}" p11_6_phase1_allowed)"
    log "P11_5A_AUTO_CLOSEOUT=COMPLETE P11_6_PHASE1_ALLOWED=${allowed}"
    write_state "COMPLETE" "P11_5A_DECISION" "Formal P11.5a closeout completed; P11.6 training was not started." "${allowed}"
    exit 0
  fi

  log "P11_5A_AUTO_CLOSEOUT=RUNNING_IDEMPOTENT_ALL"
  write_state "RUNNING" "SEGMENT_3_OR_CLOSEOUT" "Running the first incomplete idempotent P11.5a phase."
  run_probe_all
  rc=$?

  final_status="$(json_value "${FINAL_REPORT}" status)"
  if [[ "${final_status}" == "PASS" || "${final_status}" == "FAIL" ]]; then
    continue
  fi
  if (( rc == 70 )); then
    log "P11_5A_AUTO_CLOSEOUT=STOPPED_RUNNER_ERROR"
    write_state "STOPPED_RUNNER_ERROR" "SEGMENT_3_OR_CLOSEOUT" "The P11.5a runner wrote a new error report; no automatic retry was attempted."
    exit 2
  fi

  wave_status="$(json_value "${WAVE_REPORT}" status)"
  wave_stop_reason="$(json_value "${WAVE_REPORT}" stop_reason)"
  if [[ "${wave_status}" == "PAUSED" ]] && \
    [[ "${wave_stop_reason}" == "PAUSED_AFTER_SINGLE_API_FAILURE" ]]; then
    log "P11_5A_AUTO_CLOSEOUT=CONTINUE_UNATTEMPTED_KIMI_SLOTS"
    write_state "WAITING" "WAVE_3_KIMI" "A failed API slot will not be retried; continuing only unattempted slots after the fixed interval."
    sleep "${WAVE_POLL_SECONDS}"
    continue
  fi
  if [[ "${wave_status}" != "PASS" ]]; then
    log "P11_5A_AUTO_CLOSEOUT=STOPPED_WAVE3_NONPASS STATUS=${wave_status}"
    write_state "STOPPED_WAVE3_NONPASS" "WAVE_3" "Wave 3 produced a non-resumable non-pass result."
    exit 2
  fi

  segment_status="$(json_value "${SEGMENT_REPORT}" status)"
  segment_gate_allowed="$(json_value "${SEGMENT_GATE_REPORT}" allowed)"
  if [[ "${segment_status}" != "PASS" && "${segment_gate_allowed}" == "false" ]]; then
    log "P11_5A_AUTO_CLOSEOUT=DEFERRED_GPU_MEMORY_OR_TEMPERATURE"
    write_state "WAITING" "GPU_GATE" "GPU memory/temperature gate is not currently safe; retrying after the fixed delay."
    sleep "${DEFERRED_RETRY_SECONDS}"
    continue
  fi
  if [[ "${segment_status}" == "PASS" && "${final_status}" == "DEFERRED_GPU_MEMORY_UNSAFE" ]]; then
    log "P11_5A_AUTO_CLOSEOUT=DEFERRED_GPU_MEMORY_OR_TEMPERATURE"
    write_state "WAITING" "GPU_GATE" "GPU memory/temperature gate is not currently safe; retrying after the fixed delay."
    sleep "${DEFERRED_RETRY_SECONDS}"
    continue
  fi

  log "P11_5A_AUTO_CLOSEOUT=STOPPED_UNEXPECTED_NONPASS RC=${rc}"
  write_state "STOPPED_UNEXPECTED_NONPASS" "SEGMENT_3_OR_CLOSEOUT" "Runner returned a non-pass state that is neither safely resumable nor GPU-deferred."
  exit 2
done
