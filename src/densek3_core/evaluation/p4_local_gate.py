"""Approved P4.4 local-transplant Gate hierarchy.

This module is intentionally limited to the single-mixer P4.4 replay.  It does
not define tolerances for full-model hidden states, logits, loss, generation, or
cache comparisons.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

P3_NUMERICS_AMENDMENT = "P3-NUM-2026-08-10-01"
P4_NUMERICS_AMENDMENT = "P4-NUM-2026-08-11-01"
P4_AMENDMENT_TYPE = "Gate hierarchy clarification"
P4_DIAGNOSTIC_LAYER = 4
P4_CAUSAL_EVIDENCE_FILENAME = "p4-layer-4-decay-diagnostics.json"
P4_T129_EVIDENCE_FILENAME = "p4-layer-4-t129-qkv-distribution.json"

P4_T129_HIDDEN_SEED = 404129
P4_T129_INITIAL_STATE_SEED = 404130
P4_T129_INITIAL_STATE_SCALE = 0.02
P4_T129_HIDDEN_SHA256 = "800c8dccaf5393e4579090de99b65e9e6100e5f64808638ea41648f3ba0300f7"
P4_T129_INITIAL_STATE_SHA256 = "c48f3a7c80ff6eb3580ca43f1d34813c514840733580f0f6af31e68e942806a5"

# P4 borrows these output-only threshold values from the approved P3
# BF16 GEMM-shape analysis.  It does not claim or modify the original P3
# amendment, whose decay-existing-Gate precondition remains unchanged.
P4_OUTPUT_DISTRIBUTION_LIMITS = {
    "relative_l2_error_max": 7.0e-3,
    "cosine_similarity_min": 0.99997,
    "mean_abs_error_max": 1.0e-3,
    "p99_9_abs_error_max": 4.0e-3,
    "failed_allclose_fraction_max": 1.0e-4,
    "max_abs_error_max": 1.25e-1,
    "nan_count_max": 0,
    "inf_count_max": 0,
}

P4_REQUIRED_CAUSAL_CHECKS = (
    "artifact_manifest_matches_recomputed_conversion",
    "parameter_embedding_exact",
    "fp32_formal_strict_parity",
    "original_bf16_failure_reproduced",
    "decay_path_mismatch_present_at_raw_a",
    "f_b_is_exact_replication",
    "recorded_logits_recompute_trace_decay_strictly",
    "forced_fp32_strict_parity",
    "forced_fp32_cast_decay_strict_parity",
    "t7_both_injected_output_and_state_exact",
    "t129_both_injected_output_and_state_exact",
    "seed_length_sweep_complete_and_finite",
)


def _comparison_is_finite(value: dict[str, Any]) -> bool:
    return all(
        value.get(name) == 0
        for name in (
            "reference_nan_count",
            "actual_nan_count",
            "reference_inf_count",
            "actual_inf_count",
        )
    )


def evaluate_output_distribution(distribution: dict[str, Any]) -> dict[str, Any]:
    """Evaluate only the approved output distribution thresholds."""
    limits = P4_OUTPUT_DISTRIBUTION_LIMITS
    nan_count = distribution.get("nan_count")
    if nan_count is None:
        nan_count = distribution.get("reference_nan_count", 0) + distribution.get("actual_nan_count", 0)
    inf_count = distribution.get("inf_count")
    if inf_count is None:
        inf_count = distribution.get("reference_inf_count", 0) + distribution.get("actual_inf_count", 0)
    checks = {
        "relative_l2_error": distribution["relative_l2_error"] <= limits["relative_l2_error_max"],
        "cosine_similarity": distribution["cosine_similarity"] >= limits["cosine_similarity_min"],
        "mean_abs_error": distribution["mean_abs_error"] <= limits["mean_abs_error_max"],
        "p99_9_abs_error": distribution["p99_9_abs_error"] <= limits["p99_9_abs_error_max"],
        "failed_allclose_fraction": (
            distribution["failed_allclose_fraction"] <= limits["failed_allclose_fraction_max"]
        ),
        "max_abs_error": distribution["max_abs_error"] <= limits["max_abs_error_max"],
        "nan_count": nan_count <= limits["nan_count_max"],
        "inf_count": inf_count <= limits["inf_count_max"],
    }
    return {
        "threshold_source": P3_NUMERICS_AMENDMENT,
        "thresholds_only": True,
        "limits": limits,
        "observed": {
            name: distribution[name]
            for name in (
                "relative_l2_error",
                "cosine_similarity",
                "mean_abs_error",
                "p99_9_abs_error",
                "failed_allclose_fraction",
                "max_abs_error",
            )
        },
        "observed_nan_count": nan_count,
        "observed_inf_count": inf_count,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _fixed_t129_case_checks(report: dict[str, Any]) -> dict[str, bool]:
    fixed = report.get("fixed_case", {})
    return {
        "layer_index": report.get("layer_index") == P4_DIAGNOSTIC_LAYER,
        "sequence_length": fixed.get("sequence_length") == 129,
        "dtype": fixed.get("dtype") == "bfloat16",
        "initial_state_mode": fixed.get("initial_state_mode") == "random",
        "hidden_seed": fixed.get("hidden_seed") == P4_T129_HIDDEN_SEED,
        "initial_state_seed": fixed.get("initial_state_seed") == P4_T129_INITIAL_STATE_SEED,
        "initial_state_scale": fixed.get("initial_state_scale") == P4_T129_INITIAL_STATE_SCALE,
        "hidden_sha256": fixed.get("hidden_sha256") == P4_T129_HIDDEN_SHA256,
        "initial_state_sha256": fixed.get("initial_state_sha256") == P4_T129_INITIAL_STATE_SHA256,
    }


def evaluate_p4_amendment_evidence(
    causal_report: dict[str, Any],
    t129_report: dict[str, Any],
) -> dict[str, Any]:
    """Revalidate the two frozen Layer-4 evidence reports.

    The older causal runner was deliberately marked ``INCONCLUSIVE`` because
    its T=129 random-state output missed ordinary allclose.  That status is not
    promoted or rewritten here.  Instead, this approved hierarchy checks the
    named causal controls and applies the separately measured output-only
    distribution rule.
    """
    causal = causal_report.get("causal_checks", {})
    causal_checks = {name: causal.get(name) is True for name in P4_REQUIRED_CAUSAL_CHECKS}
    causal_envelope = {
        "stage": causal_report.get("stage") == "P4.4-L4-DIAG",
        "layer_index": causal_report.get("layer_index") == P4_DIAGNOSTIC_LAYER,
        "diagnostics_completed": causal_report.get("diagnostics_completed") is True,
        "formal_gate_was_unchanged": causal_report.get("p4_4_formal_status") == "FAIL_UNCHANGED",
    }

    t129_envelope = {
        "stage": t129_report.get("stage") == "P4.4-L4-T129-QKV-DISTRIBUTION",
        "status": t129_report.get("status") == "PASS",
        "measurement_completed": t129_report.get("measurement_completed") is True,
        "diagnostic_only": t129_report.get("diagnostic_only") is True,
        "formal_gate_was_unchanged": t129_report.get("p4_4_formal_status") == "FAIL_UNCHANGED",
        "no_amendment_preapproved": t129_report.get("numerics_amendment_approved") is False,
    }
    fixed_case = _fixed_t129_case_checks(t129_report)
    functional = t129_report.get("functional_comparisons", {})
    output = functional.get("output", {})
    final_state = functional.get("final_state", {})
    beta = functional.get("beta", {})
    decay = functional.get("decay", {})
    output_distribution = t129_report.get("final_output_distribution", {})
    distribution_gate = evaluate_output_distribution(output_distribution)
    functional_checks = {
        "output_standard_gate_recorded_failed": output.get("passed") is False,
        "output_distribution_gate": distribution_gate["passed"],
        "final_state_existing_gate": final_state.get("passed") is True,
        "beta_existing_gate": beta.get("passed") is True,
        "decay_recorded_as_auxiliary_failure": decay.get("passed") is False,
        "all_functional_values_finite": bool(functional) and all(
            _comparison_is_finite(value) for value in functional.values()
        ),
    }
    prior = t129_report.get("prior_evidence_checks", {})
    causal_layer = causal_report.get("artifact_consistency", {}).get("recorded_layer_conversion", {})
    t129_layer = t129_report.get("artifact_consistency", {}).get("layer_conversion", {})
    cross_report_checks = {
        "t129_prior_fp32_strict": prior.get("fp32_formal_strict_parity") is True,
        "t129_prior_forced_fp32_strict": prior.get("forced_fp32_strict_parity") is True,
        "t129_prior_t7_injection_exact": prior.get("t7_both_injected_exact") is True,
        "t129_prior_t129_injection_exact": prior.get("t129_both_injected_exact") is True,
        "t129_artifact_consistent": t129_report.get("artifact_consistency", {}).get("passed") is True,
        "t129_parameter_embedding_exact": (
            t129_report.get("parameter_embedding_audit", {}).get("passed") is True
        ),
        "causal_and_t129_source_hash_match": (
            bool(causal_layer.get("source_hash"))
            and causal_layer.get("source_hash") == t129_layer.get("source_hash")
        ),
        "causal_and_t129_target_hash_match": (
            bool(causal_layer.get("target_hash"))
            and causal_layer.get("target_hash") == t129_layer.get("target_hash")
        ),
    }

    acceptance_requirements = {
        "fp32_strict_parity": causal_checks["fp32_formal_strict_parity"],
        "parameter_embedding_audit": (
            causal_checks["parameter_embedding_exact"]
            and cross_report_checks["t129_parameter_embedding_exact"]
        ),
        "forced_fp32_projection_parity": (
            causal_checks["forced_fp32_strict_parity"]
            and causal_checks["forced_fp32_cast_decay_strict_parity"]
        ),
        "causal_injection_isolation": (
            causal_checks["t7_both_injected_output_and_state_exact"]
            and causal_checks["t129_both_injected_output_and_state_exact"]
        ),
        "final_output_functional_gate": functional_checks["output_distribution_gate"],
        "final_state_existing_gate": functional_checks["final_state_existing_gate"],
        "beta_existing_gate": functional_checks["beta_existing_gate"],
        "finite_values": functional_checks["all_functional_values_finite"],
    }
    accepted = all(
        all(group.values())
        for group in (
            causal_envelope,
            causal_checks,
            t129_envelope,
            fixed_case,
            functional_checks,
            cross_report_checks,
            acceptance_requirements,
        )
    )
    return {
        "amendment": P4_NUMERICS_AMENDMENT,
        "amendment_type": P4_AMENDMENT_TYPE,
        "scope": "P4.4 single-layer qwen_compat transplant parity only",
        "architecture_modified": False,
        "converter_modified": False,
        "kda_recurrence_modified": False,
        "state_contract_modified": False,
        "p2_numerics_modified": False,
        "p3_numerics_modified": False,
        "causal_report_original_status": causal_report.get("status"),
        "causal_envelope_checks": causal_envelope,
        "causal_checks": causal_checks,
        "t129_envelope_checks": t129_envelope,
        "fixed_t129_case_checks": fixed_case,
        "t129_functional_checks": functional_checks,
        "cross_report_checks": cross_report_checks,
        "artifact_layer_identity": {
            "layer_index": t129_layer.get("layer_index"),
            "source_hash": t129_layer.get("source_hash"),
            "target_hash": t129_layer.get("target_hash"),
        },
        "output_distribution_evaluation": distribution_gate,
        "acceptance_requires_all": acceptance_requirements,
        "accepted": accepted,
    }


def load_and_evaluate_p4_amendment_evidence(output_dir: Path) -> dict[str, Any]:
    """Load co-located immutable diagnostic reports and re-evaluate them."""
    import hashlib
    import json

    causal_path = output_dir / P4_CAUSAL_EVIDENCE_FILENAME
    t129_path = output_dir / P4_T129_EVIDENCE_FILENAME
    missing = [str(path) for path in (causal_path, t129_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Approved P4.4 amendment requires diagnostic evidence: {missing}")
    causal_bytes = causal_path.read_bytes()
    t129_bytes = t129_path.read_bytes()
    causal_report = json.loads(causal_bytes)
    t129_report = json.loads(t129_bytes)
    result = evaluate_p4_amendment_evidence(causal_report, t129_report)
    result["evidence_files"] = {
        "causal": {
            "path": str(causal_path),
            "sha256": hashlib.sha256(causal_bytes).hexdigest(),
        },
        "t129_distribution": {
            "path": str(t129_path),
            "sha256": hashlib.sha256(t129_bytes).hexdigest(),
        },
    }
    return result
