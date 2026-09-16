"""Rescore completed P6.2c metrics against the unchanged N0 formal Gate baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from densek3_core.recovery.p6_function_calibration import local_go_no_go

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "outputs/densek3-core/manifests/p6/p6-2c/p6-2c-qk-norm-bridge.json"
DEFAULT_FINAL = ROOT / "outputs/densek3-core/manifests/p6/p6-2c/p6-2c-final-check.json"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def save_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--final", type=Path, default=DEFAULT_FINAL)
    args = parser.parse_args()
    report_path = args.report.resolve()
    final_path = args.final.resolve()
    report = load_json(report_path)
    final = load_json(final_path)
    source_checks = {
        "stage": report["stage"] == "P6.2c" and final["stage"] == "P6.2c",
        "pre_fix_status": report["status"] == "NO_GO" and final["status"] == "NO_GO",
        "phase_a_positive": report["phase_a"]["decision"]["passed"] is True,
        "phase_b_complete": set(report["phase_b"]["layers"]) == {"3", "23"},
        "formal_rank": report["formal_kv_lora_rank"] == 512,
        "no_data_access": report["full_dev_accessed"] is False
        and report["heldout_accessed"] is False
        and report["canonical_accessed"] is False,
        "local_artifact_only": report["artifact"]["full_model_checkpoint"] is False
        and report["full_model_checkpoint_written"] is False,
        "artifact_hash": sha256_file(Path(report["artifact"]["path"]))
        == report["artifact"]["sha256"],
    }
    if not all(source_checks.values()):
        raise ValueError(f"P6.2c rescore source checks failed: {source_checks}")
    baseline = {
        layer_index: report["phase_a"]["layers"][str(layer_index)]["cases"]["N0"]
        for layer_index in (3, 23)
    }
    calibrated = {
        layer_index: report["phase_b"]["layers"][str(layer_index)]["calibration"][
            "best_validation"
        ]
        for layer_index in (3, 23)
    }
    corrected = local_go_no_go(
        baseline,
        calibrated,
        attention_core_improvement_min=0.20,
    )
    if not corrected["passed"]:
        raise ValueError(f"P6.2c corrected N0 formal Gate did not pass: {corrected}")
    previous = report["phase_b"]["local_decision"]
    report["phase_b"].update(
        {
            "status": "GO",
            "training_baseline": "N1_C3_DONOR_QK_NORM_PRESERVED",
            "formal_gate_baseline": "N0_C3_QK_NORM_REMOVED",
            "pre_fix_local_decision": previous,
            "local_decision": corrected,
        }
    )
    report.update(
        {
            "status": "GO",
            "p6_2c_scaleout_allowed": True,
            "result_marker": "P6_2C=GO",
            "rescore": {
                "status": "PASS",
                "reason": "RESTORE_UNCHANGED_P6_2B_N0_FORMAL_COMPARISON_BASELINE",
                "source_checks": source_checks,
                "training_performed": False,
                "optimizer_loaded": False,
                "dataset_loaded": False,
                "full_dev_accessed": False,
                "heldout_accessed": False,
                "canonical_accessed": False,
                "formal_gate_modified": False,
                "completed_at": now(),
            },
            "completed_at": now(),
        }
    )
    final.update(
        {
            "status": "GO",
            "phase_b_local_decision": corrected,
            "formal_gate_baseline": "N0_C3_QK_NORM_REMOVED",
            "p6_2c_scaleout_allowed": True,
            "p6_3_allowed": False,
            "p7_allowed": False,
            "result_marker": "P6_2C=GO",
            "rescore": report["rescore"],
            "completed_at": now(),
        }
    )
    save_json(report_path, report)
    save_json(final_path, final)
    print("P6_2C_FORMAL_RESCORE=PASS", flush=True)
    print("P6_2C_FORMAL_GATE_BASELINE=N0_C3_QK_NORM_REMOVED", flush=True)
    print("P6_2C=GO", flush=True)
    print("P6_2C_SCALEOUT_ALLOWED=true", flush=True)
    print("P6_3_ALLOWED=false", flush=True)
    print("P7_ALLOWED=false", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
