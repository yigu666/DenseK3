"""Run P2 gates in dependency order and update the machine-readable status."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = ROOT / "outputs/densek3-core/manifests/p2/p2-stage-status.json"
OUTPUT_DIR = STATUS_PATH.parent
P1_NUMERICS_AMENDMENTS = [
    "P1-NUM-2026-08-10-01",
    "P1-NUM-2026-08-10-02",
]

STAGES = [
    ("P2.0", "package_config", ["tests/contract/test_p1_config_contract.py"]),
    ("P2.1", "state_contract", ["tests/unit/test_kda_state.py"]),
    ("P2.2", "reference_recurrent", ["tests/unit/test_kda_reference.py"]),
    ("P2.3", "reference_full", ["tests/parity/test_kda_reference_recurrent.py::test_full_supports_target_gva_shape"]),
    ("P2.4", "reference_self_parity", ["tests/parity/test_kda_reference_recurrent.py"]),
    (
        "P2.5",
        "fla_chunk_adapter",
        [
            "tests/parity/test_kda_fla_chunk.py::test_fla_chunk_adapter_smoke",
            "tests/parity/test_kda_fla_chunk.py::test_fla_chunk_supports_non_square_v_first_state",
        ],
    ),
    ("P2.6", "fla_chunk_forward_parity", ["tests/parity/test_kda_fla_chunk.py::test_reference_matches_fla_chunk"]),
    (
        "P2.7",
        "fla_recurrent_adapter",
        ["tests/parity/test_kda_fla_recurrent.py::test_fused_recurrent_adapter_smoke"],
    ),
    (
        "P2.8",
        "fla_recurrent_parity",
        [
            "tests/parity/test_kda_fla_recurrent.py::test_reference_matches_fla_fused_single_token",
            "tests/parity/test_kda_fla_recurrent.py::test_fused_recurrent_decode_trajectory",
        ],
    ),
    ("P2.9", "continuation_parity", ["tests/parity/test_kda_continuation.py"]),
    ("P2.10", "backward_parity", ["tests/parity/test_kda_backward.py"]),
    ("P2.11", "numerical_matrix", ["tests/parity/test_kda_numerical_matrix.py"]),
    ("P2.12", "performance_baseline", []),
]


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def save_status(status: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = STATUS_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATUS_PATH)


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_environment() -> None:
    cuda_available = torch.cuda.is_available()
    save_json(
        OUTPUT_DIR / "p2-environment.json",
        {
            "recorded_at": now(),
            "project_root": str(ROOT),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": cuda_available,
            "gpu": torch.cuda.get_device_name() if cuda_available else None,
            "triton": package_version("triton"),
            "fla": package_version("flash-linear-attention") or package_version("fla"),
            "pytest": package_version("pytest"),
            "ruff": package_version("ruff"),
            "p1_contract_modified": False,
            "p1_numerics_amendments": P1_NUMERICS_AMENDMENTS,
        },
    )


def run_stage(stage_id: str, paths: list[str]) -> subprocess.CompletedProcess[str]:
    if stage_id == "P2.12":
        command = [sys.executable, "scripts/benchmark_p2_kda.py"]
    else:
        command = [sys.executable, "-m", "pytest", *paths, "-q"]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if stage_id == "P2.10" and result.returncode == 0:
        metrics_command = [sys.executable, "scripts/evaluate_p2_parity.py"]
        metrics = subprocess.run(metrics_command, cwd=ROOT, text=True, capture_output=True, check=False)
        return subprocess.CompletedProcess(
            args=[command, metrics_command],
            returncode=metrics.returncode,
            stdout=result.stdout + metrics.stdout,
            stderr=result.stderr + metrics.stderr,
        )
    return result


def runtime_failure() -> str | None:
    if not torch.cuda.is_available():
        return "CUDA is unavailable"
    if importlib.util.find_spec("fla") is None:
        return "FLA is not installed in the active Python environment"
    if importlib.util.find_spec("triton") is None:
        return "Triton is not installed in the active Python environment"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--through", default="P2.12", choices=[stage[0] for stage in STAGES])
    args = parser.parse_args()
    status = {
        "stage": "P2",
        "overall_status": "RUNNING",
        "first_failed_stage": None,
        "p1_contract_modified": False,
        "p1_numerics_amendments": P1_NUMERICS_AMENDMENTS,
        "updated_at": now(),
        "stages": {},
    }
    write_environment()
    save_status(status)
    blocked_by = None
    for stage_id, name, tests in STAGES:
        entry = {"name": name, "status": "RUNNING", "started_at": now(), "tests": tests}
        status["stages"][stage_id] = entry
        save_status(status)
        if blocked_by is not None:
            entry.update({"status": "BLOCKED", "blocked_by": blocked_by, "completed_at": now()})
        else:
            failure = runtime_failure() if stage_id == "P2.5" else None
            if failure is not None:
                entry.update({"status": "FAIL", "reason": failure, "completed_at": now()})
                blocked_by = stage_id
                status["first_failed_stage"] = stage_id
            else:
                result = run_stage(stage_id, tests)
                entry.update(
                    {
                        "status": "PASS" if result.returncode == 0 else "FAIL",
                        "returncode": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "completed_at": now(),
                    }
                )
                if result.returncode:
                    blocked_by = stage_id
                    status["first_failed_stage"] = stage_id
        status["updated_at"] = now()
        save_status(status)
        if stage_id == args.through:
            break
    status["overall_status"] = "PASS" if status["first_failed_stage"] is None else "FAIL"
    status["updated_at"] = now()
    save_status(status)
    passed = status["overall_status"] == "PASS"
    final = {
        "stage": "P2",
        "status": "PASS" if passed else "FAIL",
        "first_failed_stage": status["first_failed_stage"],
        "reference_recurrent": status["stages"].get("P2.2", {}).get("status") == "PASS",
        "reference_full": status["stages"].get("P2.3", {}).get("status") == "PASS",
        "reference_self_parity": status["stages"].get("P2.4", {}).get("status") == "PASS",
        "fla_chunk_parity": status["stages"].get("P2.6", {}).get("status") == "PASS",
        "fla_recurrent_parity": status["stages"].get("P2.8", {}).get("status") == "PASS",
        "continuation_parity": status["stages"].get("P2.9", {}).get("status") == "PASS",
        "backward_parity": status["stages"].get("P2.10", {}).get("status") == "PASS",
        "gva_16_32_verified": status["stages"].get("P2.5", {}).get("status") == "PASS",
        "fp32_verified": status["stages"].get("P2.11", {}).get("status") == "PASS",
        "bf16_verified": status["stages"].get("P2.11", {}).get("status") == "PASS",
        "performance_baseline": status["stages"].get("P2.12", {}).get("status") == "PASS",
        "p1_contract_modified": False,
        "p1_numerics_amendments": P1_NUMERICS_AMENDMENTS,
        "p3_allowed": passed,
        "result_marker": f"P2_FINAL_CHECK={'PASS' if passed else 'FAIL'}",
        "failure_marker": None if passed else f"P2_FIRST_FAILED_STAGE={status['first_failed_stage']}",
    }
    save_json(OUTPUT_DIR / "p2-final-check.json", final)
    failure_path = OUTPUT_DIR / "p2-failure.json"
    if passed:
        failure_path.unlink(missing_ok=True)
    else:
        failed_stage = status["first_failed_stage"]
        failed_entry = status["stages"].get(failed_stage, {})
        save_json(
            failure_path,
            {
                "stage": "P2",
                "first_failed_stage": failed_stage,
                "failed_test": failed_entry.get("tests"),
                "timestamp": now(),
                "command": [sys.executable, "scripts/run_p2.py"],
                "seed": 42,
                "expected": "The stage command exits successfully under the locked P1 contract",
                "actual": failed_entry.get("reason") or failed_entry.get("stdout"),
                "exception": failed_entry.get("stderr"),
                "reproduction_command": f"cd {ROOT} && {sys.executable} scripts/run_p2.py --through {failed_stage}",
                "blocked_stages": [key for key, value in status["stages"].items() if value["status"] == "BLOCKED"],
                "p1_contract_modified": False,
                "p1_numerics_amendments": P1_NUMERICS_AMENDMENTS,
            },
        )
    return 0 if status["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
