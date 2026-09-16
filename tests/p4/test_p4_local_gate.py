from __future__ import annotations

import copy
import runpy
from pathlib import Path
from types import SimpleNamespace

import torch

from densek3_core.evaluation.p4_local_gate import (
    P4_REQUIRED_CAUSAL_CHECKS,
    P4_T129_HIDDEN_SHA256,
    P4_T129_INITIAL_STATE_SHA256,
    evaluate_output_distribution,
    evaluate_p4_amendment_evidence,
)


def _comparison(passed: bool) -> dict[str, object]:
    return {
        "passed": passed,
        "reference_nan_count": 0,
        "actual_nan_count": 0,
        "reference_inf_count": 0,
        "actual_inf_count": 0,
    }


def _distribution(**updates: float | int) -> dict[str, float | int]:
    value: dict[str, float | int] = {
        "relative_l2_error": 0.005701787128366677,
        "cosine_similarity": 0.9999837458330452,
        "mean_abs_error": 0.0005636099032884421,
        "p99_9_abs_error": 0.00390625,
        "failed_allclose_fraction": 1.2112403100775194e-05,
        "max_abs_error": 0.015625,
        "reference_nan_count": 0,
        "actual_nan_count": 0,
        "reference_inf_count": 0,
        "actual_inf_count": 0,
    }
    value.update(updates)
    return value


def _evidence_reports() -> tuple[dict[str, object], dict[str, object]]:
    causal = {
        "stage": "P4.4-L4-DIAG",
        # The historical runner intentionally used this status because the
        # then-unapproved T=129 ordinary allclose check failed.
        "status": "INCONCLUSIVE",
        "layer_index": 4,
        "diagnostics_completed": True,
        "p4_4_formal_status": "FAIL_UNCHANGED",
        "causal_checks": {name: True for name in P4_REQUIRED_CAUSAL_CHECKS},
        "artifact_consistency": {
            "recorded_layer_conversion": {
                "layer_index": 4,
                "source_hash": "source",
                "target_hash": "target",
            }
        },
    }
    t129 = {
        "stage": "P4.4-L4-T129-QKV-DISTRIBUTION",
        "status": "PASS",
        "diagnostic_only": True,
        "measurement_completed": True,
        "p4_4_formal_status": "FAIL_UNCHANGED",
        "numerics_amendment_approved": False,
        "layer_index": 4,
        "fixed_case": {
            "sequence_length": 129,
            "dtype": "bfloat16",
            "hidden_seed": 404129,
            "initial_state_mode": "random",
            "initial_state_seed": 404130,
            "initial_state_scale": 0.02,
            "hidden_sha256": P4_T129_HIDDEN_SHA256,
            "initial_state_sha256": P4_T129_INITIAL_STATE_SHA256,
        },
        "functional_comparisons": {
            "output": _comparison(False),
            "final_state": _comparison(True),
            "decay": _comparison(False),
            "beta": _comparison(True),
        },
        "final_output_distribution": _distribution(),
        "prior_evidence_checks": {
            "fp32_formal_strict_parity": True,
            "forced_fp32_strict_parity": True,
            "t7_both_injected_exact": True,
            "t129_both_injected_exact": True,
        },
        "artifact_consistency": {
            "passed": True,
            "layer_conversion": {
                "layer_index": 4,
                "source_hash": "source",
                "target_hash": "target",
            },
        },
        "parameter_embedding_audit": {"passed": True},
    }
    return causal, t129


def test_approved_evidence_accepts_historical_inconclusive_causal_envelope() -> None:
    causal, t129 = _evidence_reports()
    result = evaluate_p4_amendment_evidence(causal, t129)
    assert result["accepted"] is True
    assert result["causal_report_original_status"] == "INCONCLUSIVE"
    assert result["output_distribution_evaluation"]["passed"] is True
    assert all(result["acceptance_requires_all"].values())


def test_evidence_rejects_wrong_fixed_random_state_seed() -> None:
    causal, t129 = _evidence_reports()
    changed = copy.deepcopy(t129)
    changed["fixed_case"]["initial_state_seed"] = 404129
    result = evaluate_p4_amendment_evidence(causal, changed)
    assert result["accepted"] is False
    assert result["fixed_t129_case_checks"]["initial_state_seed"] is False


def test_output_distribution_requires_every_limit() -> None:
    assert evaluate_output_distribution(_distribution())["passed"] is True
    assert evaluate_output_distribution(_distribution(p99_9_abs_error=0.00401))["passed"] is False
    assert evaluate_output_distribution(_distribution(max_abs_error=0.12501))["passed"] is False


def _runner_namespace() -> dict[str, object]:
    return runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/run_p4.py"))


class _FixedMixer:
    def __init__(self, result: SimpleNamespace):
        self.result = result

    def forward(self, _hidden: torch.Tensor) -> SimpleNamespace:
        return self.result


def test_local_case_accepts_only_auxiliary_decay_with_approved_evidence() -> None:
    namespace = _runner_namespace()
    reference = SimpleNamespace(
        output=torch.ones(8, dtype=torch.bfloat16),
        final_state=torch.ones(8, dtype=torch.bfloat16),
        decay=torch.ones(8, dtype=torch.bfloat16),
        beta=torch.ones(8, dtype=torch.bfloat16),
    )
    actual = SimpleNamespace(
        output=reference.output.clone(),
        final_state=reference.final_state.clone(),
        decay=torch.full((8,), 1.125, dtype=torch.bfloat16),
        beta=reference.beta.clone(),
    )
    hidden = torch.zeros((1, 7, 1), dtype=torch.bfloat16)
    accepted = namespace["_local_case"](
        _FixedMixer(reference),
        _FixedMixer(actual),
        hidden,
        torch.bfloat16,
        p4_amendment_evidence_accepted=True,
    )
    rejected = namespace["_local_case"](
        _FixedMixer(reference),
        _FixedMixer(actual),
        hidden,
        torch.bfloat16,
        p4_amendment_evidence_accepted=False,
    )
    assert accepted["passed"] is True
    assert accepted["pass_mode"] == "auxiliary_intermediate_exception"
    assert accepted["p3_original_fallback_used"] is False
    assert rejected["passed"] is False


def test_local_output_uses_p4_distribution_rule_without_claiming_p3_fallback() -> None:
    namespace = _runner_namespace()
    reference_output = torch.ones(100_000, dtype=torch.bfloat16)
    actual_output = reference_output.clone()
    actual_output[0] += torch.tensor(0.015625, dtype=torch.bfloat16)
    reference = SimpleNamespace(
        output=reference_output,
        final_state=torch.ones(1, dtype=torch.bfloat16),
        decay=torch.ones(1, dtype=torch.bfloat16),
        beta=torch.ones(1, dtype=torch.bfloat16),
    )
    actual = SimpleNamespace(
        output=actual_output,
        final_state=reference.final_state.clone(),
        decay=reference.decay.clone(),
        beta=reference.beta.clone(),
    )
    hidden = torch.zeros((1, 129, 1), dtype=torch.bfloat16)
    result = namespace["_local_case"](
        _FixedMixer(reference),
        _FixedMixer(actual),
        hidden,
        torch.bfloat16,
        p4_amendment_evidence_accepted=True,
    )
    assert result["passed"] is True
    assert result["pass_mode"] == "bf16_transplant_distribution_fallback"
    assert result["p3_distribution_thresholds_used"] is True
    assert result["p3_original_fallback_preconditions_inherited"] is False
