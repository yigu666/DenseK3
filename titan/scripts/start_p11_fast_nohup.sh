#!/usr/bin/env bash

set -euo pipefail

readonly PROJECT_ROOT="${DENSEK3_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
readonly REPORT_DIR="$PROJECT_ROOT/titan/manifests/reproduction/p11-t/p11-6-fast"
readonly PID_PATH="$REPORT_DIR/automation/p11-6-fast.pid"
readonly LOG_PATH="$REPORT_DIR/p11-6-fast-nohup.log"

cd "$PROJECT_ROOT"
mkdir -p "$REPORT_DIR/automation"
if [[ -f "$PID_PATH" ]]; then
  pid="$(tr -d '[:space:]' <"$PID_PATH")"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    printf '%s\n' "P11_6_FAST_NOHUP=ALREADY_RUNNING PID=$pid"
    exit 0
  fi
fi

nohup titan/scripts/run_p11_fast_worker.sh </dev/null >>"$LOG_PATH" 2>&1 &
pid=$!
sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
  printf '%s\n' "P11_6_FAST_NOHUP=START_FAILED"
  tail -n 40 "$LOG_PATH"
  exit 2
fi
printf '%s\n' "P11_6_FAST_NOHUP=STARTED PID=$pid LOG=$LOG_PATH"
