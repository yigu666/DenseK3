"""Run Titan-only P11-T preflight gates."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TITAN = ROOT / "titan"
sys.path.insert(0, str(TITAN))

from p11 import (  # noqa: E402
    KimiAPIConfig,
    KimiAPICostLedger,
    KimiAPITeacher,
    build_p11_asset_preflight,
    parse_kimi_trajectory,
)

REPORT_DIR = TITAN / "manifests/reproduction/p11-t"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_preflight() -> int:
    report = build_p11_asset_preflight(ROOT)
    report["completed_at"] = datetime.now().astimezone().isoformat()
    output = REPORT_DIR / "p11-t-asset-preflight.json"
    _write_json(output, report)
    print(report["result_marker"], flush=True)
    print(f"P11_T_TRAINING_ALLOWED={str(report['training_allowed']).lower()}", flush=True)
    print(f"P11_T_BLOCKERS={','.join(report['blockers']) or 'NONE'}", flush=True)
    print("P12_TITAN_MIGRATION_ALLOWED=false", flush=True)
    return 0 if report["status"] == "PASS" else 2


def run_api_probe() -> int:
    started_at = datetime.now().astimezone().isoformat()
    output = REPORT_DIR / "p11-t-kimi-api-probe.json"
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("status") == "PASS":
            print("P11_T_KIMI_API_PROBE=RESUMED_WITHOUT_API_CALL", flush=True)
            print("P11_T_TRAINING_ALLOWED=false", flush=True)
            print("P12_TITAN_MIGRATION_ALLOWED=false", flush=True)
            return 0
    try:
        config = KimiAPIConfig.from_json(TITAN / "configs/p11-kimi-api.json")
        teacher = KimiAPITeacher(config, ledger=KimiAPICostLedger(REPORT_DIR, config))
        result = teacher.complete(
            [
                {
                    "role": "system",
                    "content": "This is an API contract probe. Reply with exactly KEEP.",
                },
                {"role": "user", "content": "Return the required contract marker."},
            ],
            # Kimi K3 always reasons. 256 tokens keeps this one-shot probe cheap
            # while leaving enough room for low-effort reasoning plus `KEEP`.
            max_completion_tokens=256,
        )
        trajectory = parse_kimi_trajectory(result.text)
        checks = {
            "requested_model_returned": result.model == "kimi-k3",
            "keep_contract": trajectory.mode == "KEEP",
            "kimi_text_trajectory_present": bool(result.text.strip()),
            "runtime_output_logprobs_disabled": not config.logprobs
            and not result.output_logprobs,
            "usage_present": bool(result.usage),
            "completion_tokens_positive": result.usage.get("completion_tokens", 0) > 0,
            "secret_not_persisted": True,
        }
        report: dict[str, object] = {
            "stage": "P11.0-KIMI-K3-API-PROBE",
            "status": "PASS" if all(checks.values()) else "FAIL",
            "started_at": started_at,
            "completed_at": datetime.now().astimezone().isoformat(),
            "endpoint": config.base_url,
            "requested_model": config.model,
            "response_model": result.model,
            "finish_reason": result.finish_reason,
            "response_contract": trajectory.mode,
            "output_logprobs": list(result.output_logprobs),
            "teacher_signal": "TEXT_SPACE_TRAJECTORY_ONLY",
            "qwen_exact_token_logits_unchanged": True,
            "kimi_k3_runtime_limitation": "OUTPUT_LOGPROBS_UNSUPPORTED",
            "usage": result.usage,
            "reasoning_effort": config.reasoning_effort,
            "reasoning_content_persisted": False,
            "reasoning_content_present": result.reasoning_content_present,
            "reasoning_content_characters": result.reasoning_content_characters,
            "cost_cny": result.cost_cny,
            "checks": checks,
            "api_key_env": config.api_key_env,
            "api_key_persisted": False,
            "heldout_accessed": False,
            "training_performed": False,
            "p12_titan_migration_allowed": False,
        }
    except Exception as error:  # The exception envelope is itself a persistent Gate result.
        report = {
            "stage": "P11.0-KIMI-K3-API-PROBE",
            "status": "ERROR",
            "started_at": started_at,
            "completed_at": datetime.now().astimezone().isoformat(),
            "exception_type": type(error).__name__,
            "exception": str(error),
            "api_key_persisted": False,
            "heldout_accessed": False,
            "training_performed": False,
            "p12_titan_migration_allowed": False,
        }
    report["result_marker"] = (
        "P11_T_KIMI_API_PROBE=PASS" if report["status"] == "PASS" else "P11_T_KIMI_API_PROBE=FAIL"
    )
    _write_json(output, report)
    print(report["result_marker"], flush=True)
    print("P11_T_TRAINING_ALLOWED=false", flush=True)
    print("P12_TITAN_MIGRATION_ALLOWED=false", flush=True)
    return 0 if report["status"] == "PASS" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "api-probe"), default="preflight")
    args = parser.parse_args()
    return run_api_probe() if args.phase == "api-probe" else run_preflight()


if __name__ == "__main__":
    raise SystemExit(main())
