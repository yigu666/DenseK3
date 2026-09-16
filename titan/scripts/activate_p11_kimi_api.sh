#!/usr/bin/env bash

set -euo pipefail

P11_ACTIVATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P11_ROOT="$(cd "${P11_ACTIVATE_DIR}/../.." && pwd)"
P11_SECRET="${P11_ROOT}/titan/secrets/moonshot-api-key"

if [[ ! -f "${P11_SECRET}" ]]; then
  echo "P11_KIMI_API_KEY_FILE=ABSENT" >&2
  return 2 2>/dev/null || exit 2
fi

if [[ "$(stat -c '%a' "${P11_SECRET}")" != "600" ]]; then
  echo "P11_KIMI_API_KEY_FILE_MODE=INVALID_REQUIRED_600" >&2
  return 2 2>/dev/null || exit 2
fi

MOONSHOT_API_KEY="$(<"${P11_SECRET}")"
if [[ -z "${MOONSHOT_API_KEY}" ]]; then
  echo "P11_KIMI_API_KEY_FILE=EMPTY" >&2
  unset MOONSHOT_API_KEY
  return 2 2>/dev/null || exit 2
fi
export MOONSHOT_API_KEY

unset P11_ACTIVATE_DIR P11_ROOT P11_SECRET
echo "P11_KIMI_API_KEY=LOADED_FROM_MODE_600_FILE"
