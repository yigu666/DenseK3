from __future__ import annotations

import json
import runpy
from pathlib import Path

import torch

from densek3_core.transplant.gdn_to_kda import convert_gdn_to_kda
from tests.p3.helpers import small_bridge_config, small_source_state


def runner_namespace() -> dict[str, object]:
    return runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/run_p3.py"))


def test_p3_functional_exception_is_persisted(tmp_path: Path) -> None:
    namespace = runner_namespace()
    output = tmp_path / "p3-functional-parity.json"
    report = namespace["save_functional_exception"](output, RuntimeError("diagnostic failure"), "trace")
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted == report
    assert persisted["stage"] == "P3.4"
    assert persisted["status"] == "EXCEPTION"
    assert persisted["diagnostics_completed"] is False
    assert persisted["first_failure"]["exception"] == "diagnostic failure"


def test_p3_functional_exception_preserves_partial_cases(tmp_path: Path) -> None:
    namespace = runner_namespace()
    output = tmp_path / "p3-functional-parity.json"
    partial = {
        "stage": "P3.4",
        "status": "RUNNING",
        "cases_total": 1,
        "cases_passed": 1,
        "cases_failed": 0,
        "cases": [{"T": 1, "passed": True}],
        "first_failure": None,
        "projection_diagnostics": {"status": "PASS"},
    }
    namespace["save_json"](output, partial)
    persisted = namespace["save_functional_exception"](output, RuntimeError("later failure"), "trace")
    assert persisted["cases_total"] == 1
    assert persisted["cases"] == partial["cases"]
    assert persisted["projection_diagnostics"] == {"status": "PASS"}
    assert persisted["exception_failure"]["exception"] == "later failure"


def test_p3_diagnostic_matrix_executes_on_small_bridge() -> None:
    namespace = runner_namespace()
    config = small_bridge_config()
    source = small_source_state()
    context = {
        "config": config,
        "source_state": source,
        "conversion": convert_gdn_to_kda(source, config),
        "device": torch.device("cpu"),
    }
    projection = namespace["_projection_diagnostics"](context)
    injection = namespace["_injection_isolation"](context)
    initial_state, distribution = namespace["_initial_state_isolation"](context)
    assert projection["status"] == "PASS"
    assert injection["status"] == "PASS"
    assert initial_state["status"] == "PASS"
    assert distribution["status"] == "PASS"
    assert set(initial_state["cases"]) == {"none", "zeros", "random"}


def _comparison_result(passed: bool) -> dict[str, object]:
    return {
        "passed": passed,
        "reference_nan_count": 0,
        "actual_nan_count": 0,
        "reference_inf_count": 0,
        "actual_inf_count": 0,
    }


def _approved_distribution() -> dict[str, object]:
    return {
        "num_elements": 330_240,
        "num_failed_allclose": 17,
        "relative_l2_error": 0.0062,
        "cosine_similarity": 0.99998,
        "mean_abs_error": 0.00075,
        "p99_9_abs_error": 0.00390625,
        "max_abs_error": 0.0625,
        "reference_nan_count": 0,
        "actual_nan_count": 0,
        "reference_inf_count": 0,
        "actual_inf_count": 0,
    }


def test_p3_bf16_fallback_requires_all_approved_evidence() -> None:
    namespace = runner_namespace()
    good = _comparison_result(True)
    failed_output = _comparison_result(False)
    cases = [
        {
            "dtype": "float32",
            "T": 129,
            "initial_state": "random",
            "standard_allclose_passed": True,
            "output": dict(good),
            "final_state": dict(good),
            "decay": dict(good),
            "beta": dict(good),
            "output_error_distribution": _approved_distribution(),
        },
        {
            "dtype": "bfloat16",
            "T": 129,
            "initial_state": "random",
            "standard_allclose_passed": False,
            "output": failed_output,
            "final_state": dict(good),
            "decay": dict(good),
            "beta": dict(good),
            "output_error_distribution": _approved_distribution(),
        },
    ]
    projection = {
        "status": "PASS",
        "cases": [
            {
                "dtype": "bfloat16",
                "forced_fp32_projection": {"all_exact": True},
            }
        ],
    }
    injection = {"status": "PASS", "both_injected_exact": True}
    initial_state = {"status": "PASS"}
    distribution = {"status": "PASS"}
    result = namespace["_apply_p3_functional_gate"](
        cases,
        projection,
        injection,
        initial_state,
        distribution,
    )
    assert result["passed"] is True
    assert result["standard_allclose"] is False
    assert result["pass_mode"] == "bf16_transplant_distribution_fallback"
    assert result["numerics_amendment"] == "P3-NUM-2026-08-10-01"
    assert cases[1]["fallback_accepted"] is True
    injection["both_injected_exact"] = False
    rejected = namespace["_apply_p3_functional_gate"](
        cases,
        projection,
        injection,
        initial_state,
        distribution,
    )
    assert rejected["passed"] is False


def test_p3_bf16_fallback_hard_max_abs_cap() -> None:
    namespace = runner_namespace()
    distribution = _approved_distribution()
    distribution["max_abs_error"] = 0.25
    result = namespace["_evaluate_bf16_distribution_fallback"](distribution)
    assert result["checks"]["max_abs_error"] is False
    assert result["passed"] is False
