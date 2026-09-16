"""Approved P4.6 composite cache-equivalence Gate hierarchy.

This module is deliberately scoped to the three frozen P4.6 full-vs-
incremental cases and the frozen batch-shape control.  It does not modify the
global BF16 allclose tolerances and it is not reusable by P4.5 or P4.7.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from densek3_core.transplant.full_model import verify_p4_artifact

P4_CACHE_NUMERICS_AMENDMENT = "P4-CACHE-NUM-2026-08-11-01"
P4_CACHE_AMENDMENT_TYPE = "Composite cache-equivalence Gate hierarchy"
P4_CACHE_EVIDENCE_FILENAME = "p4-cache-causal-diagnostics.json"
P4_CACHE_TOKEN_SEED = 20260811
P4_CACHE_CASES = ((1, 1), (7, 8), (64, 32))

# These limits are local to a deterministic case identified by prompt/decode,
# token seed, token hash, artifact hash, dtype and runtime evidence.  They are
# not a new model-wide or BF16-wide tolerance.
P4_CACHE_INTEGRATED_LOGIT_LIMITS: dict[tuple[int, int], dict[str, float | int]] = {
    (1, 1): {
        "relative_l2_error_max": 2.0e-2,
        "cosine_similarity_min": 0.99985,
        "mean_abs_error_max": 4.5e-2,
        "p99_9_abs_error_max": 1.75e-1,
        "failed_allclose_fraction_max": 7.5e-1,
        "max_abs_error_max": 2.5e-1,
        "nan_count_max": 0,
        "inf_count_max": 0,
    },
    (7, 8): {
        "relative_l2_error_max": 1.2e-2,
        "cosine_similarity_min": 0.99994,
        "mean_abs_error_max": 2.25e-2,
        "p99_9_abs_error_max": 1.0e-1,
        "failed_allclose_fraction_max": 5.0e-1,
        "max_abs_error_max": 2.5e-1,
        "nan_count_max": 0,
        "inf_count_max": 0,
    },
    (64, 32): {
        "relative_l2_error_max": 1.8e-2,
        "cosine_similarity_min": 0.99985,
        "mean_abs_error_max": 3.0e-2,
        "p99_9_abs_error_max": 1.25e-1,
        "failed_allclose_fraction_max": 6.5e-1,
        "max_abs_error_max": 5.0e-1,
        "nan_count_max": 0,
        "inf_count_max": 0,
    },
}

P4_CACHE_TENSOR_LIMITS: dict[str, float | int] = {
    "relative_l2_error_max": 2.0e-2,
    "cosine_similarity_min": 0.9998,
    "mean_abs_error_max": 2.0e-2,
    "max_abs_error_max": 3.75e-1,
    "nan_count_max": 0,
    "inf_count_max": 0,
}

P4_CACHE_BATCH_LOGIT_LIMITS: dict[str, float | int] = {
    "relative_l2_error_max": 8.0e-3,
    "cosine_similarity_min": 0.99996,
    "mean_abs_error_max": 1.25e-2,
    "p99_9_abs_error_max": 1.0e-1,
    "failed_allclose_fraction_max": 3.5e-1,
    "max_abs_error_max": 2.0e-1,
    "nan_count_max": 0,
    "inf_count_max": 0,
}

P4_CACHE_BATCH_STATE_LIMITS: dict[str, float | int] = {
    "relative_l2_error_max": 1.0e-2,
    "cosine_similarity_min": 0.99995,
    "mean_abs_error_max": 1.0e-4,
    "max_abs_error_max": 2.5e-2,
    "nan_count_max": 0,
    "inf_count_max": 0,
}

P4_CACHE_TOKEN_DECISION_LIMITS = {
    "top1_agreement_fraction_min": 1.0,
    "top10_overlap_count_min": 9,
    "nan_count_max": 0,
    "inf_count_max": 0,
}


def _finite_counts(value: dict[str, Any]) -> tuple[int, int]:
    nan_count = value.get("nan_count")
    if nan_count is None:
        nan_count = value.get("reference_nan_count", 0) + value.get("actual_nan_count", 0)
    inf_count = value.get("inf_count")
    if inf_count is None:
        inf_count = value.get("reference_inf_count", 0) + value.get("actual_inf_count", 0)
    return int(nan_count), int(inf_count)


def evaluate_numerical_envelope(
    observed: dict[str, Any],
    limits: dict[str, float | int],
) -> dict[str, Any]:
    """Evaluate a named min/max envelope without changing allclose."""
    nan_count, inf_count = _finite_counts(observed)
    checks: dict[str, bool] = {}
    observed_values: dict[str, Any] = {}
    for name, limit in limits.items():
        if name == "nan_count_max":
            checks["nan_count"] = nan_count <= limit
        elif name == "inf_count_max":
            checks["inf_count"] = inf_count <= limit
        elif name.endswith("_max"):
            candidate = name.removesuffix("_max")
            metric = candidate if candidate in observed else name
            checks[metric] = metric in observed and observed[metric] <= limit
            observed_values[metric] = observed.get(metric)
        elif name.endswith("_min"):
            candidate = name.removesuffix("_min")
            metric = candidate if candidate in observed else name
            checks[metric] = metric in observed and observed[metric] >= limit
            observed_values[metric] = observed.get(metric)
        else:
            raise ValueError(f"Unsupported numerical limit: {name}")
    return {
        "limits": dict(limits),
        "observed": observed_values,
        "observed_nan_count": nan_count,
        "observed_inf_count": inf_count,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _decision_scope_summary(
    top1_equal: torch.Tensor,
    overlap: torch.Tensor,
    reference_tie_count: torch.Tensor,
    actual_tie_count: torch.Tensor,
    mask: torch.Tensor,
    top_k: int,
) -> dict[str, Any]:
    selected_top1 = top1_equal[mask]
    selected_overlap = overlap[mask]
    selected_reference_ties = reference_tie_count[mask]
    selected_actual_ties = actual_tie_count[mask]
    token_count = selected_top1.numel()
    histogram = {
        str(count): int((selected_overlap == count).sum())
        for count in range(top_k + 1)
        if (selected_overlap == count).any()
    }
    return {
        "token_count": token_count,
        "top1_agreement_count": int(selected_top1.sum()),
        "top1_disagreement_count": int((~selected_top1).sum()),
        "top1_agreement_fraction": float(selected_top1.double().mean()) if token_count else 1.0,
        "top10_overlap_count_min": int(selected_overlap.min()) if token_count else top_k,
        "top10_overlap_fraction_mean": (
            float(selected_overlap.double().mean() / top_k) if token_count else 1.0
        ),
        "top10_exact_set_fraction": (
            float((selected_overlap == top_k).double().mean()) if token_count else 1.0
        ),
        "top10_overlap_histogram": histogram,
        "reference_positions_with_argmax_ties": int((selected_reference_ties > 1).sum()),
        "actual_positions_with_argmax_ties": int((selected_actual_ties > 1).sum()),
        "reference_max_argmax_tie_count": (
            int(selected_reference_ties.max()) if token_count else 0
        ),
        "actual_max_argmax_tie_count": int(selected_actual_ties.max()) if token_count else 0,
    }


def _decision_phase(position: int, prompt_length: int | None) -> str:
    if prompt_length is None:
        return "batch_shape"
    if position < prompt_length - 1:
        return "prefill_unused"
    if position == prompt_length - 1:
        return "prefill_last"
    return "decode"


def token_decision_metrics(
    reference: torch.Tensor,
    actual: torch.Tensor,
    *,
    top_k: int = 10,
    prompt_length: int | None = None,
) -> dict[str, Any]:
    """Measure all-token decisions plus generation-relevant diagnostics."""
    if reference.shape != actual.shape:
        raise ValueError(f"Decision shape mismatch: {reference.shape} != {actual.shape}")
    if reference.ndim != 3:
        raise ValueError("P4.6 token-decision logits must have shape [batch, sequence, vocabulary]")
    if reference.shape[-1] < top_k:
        raise ValueError(f"Vocabulary size {reference.shape[-1]} is smaller than top_k={top_k}")
    if prompt_length is not None and not 1 <= prompt_length <= reference.shape[1]:
        raise ValueError("prompt_length must be in [1, sequence_length]")

    # Greedy Top-1 and Top-10 candidate sets are intentionally independent.
    # Never derive the greedy decision from topk(...).indices[..., 0]: BF16
    # ties may be ordered differently by topk while argmax remains stable.
    reference_argmax_gpu = reference.argmax(dim=-1)
    actual_argmax_gpu = actual.argmax(dim=-1)
    reference_max_gpu = reference.max(dim=-1).values
    actual_max_gpu = actual.max(dim=-1).values
    reference_tie_count_gpu = (reference == reference_max_gpu.unsqueeze(-1)).sum(dim=-1)
    actual_tie_count_gpu = (actual == actual_max_gpu.unsqueeze(-1)).sum(dim=-1)

    reference_top_values_gpu, reference_top_gpu = reference.topk(top_k, dim=-1)
    actual_top_values_gpu, actual_top_gpu = actual.topk(top_k, dim=-1)
    reference_top = reference_top_gpu.detach().cpu()
    actual_top = actual_top_gpu.detach().cpu()
    reference_top_values = reference_top_values_gpu.detach().float().cpu()
    actual_top_values = actual_top_values_gpu.detach().float().cpu()
    reference_top1 = reference_argmax_gpu.detach().cpu()
    actual_top1 = actual_argmax_gpu.detach().cpu()
    reference_max = reference_max_gpu.detach().float().cpu()
    actual_max = actual_max_gpu.detach().float().cpu()
    reference_tie_count = reference_tie_count_gpu.detach().cpu()
    actual_tie_count = actual_tie_count_gpu.detach().cpu()
    top1_equal = reference_top1 == actual_top1
    topk_first_equal = reference_top[..., 0] == actual_top[..., 0]
    reference_topk_first_is_argmax = reference_top[..., 0] == reference_top1
    actual_topk_first_is_argmax = actual_top[..., 0] == actual_top1

    # A 10x10 broadcast is bounded even for the 96-token formal case and
    # avoids transferring the full vocabulary logits to CPU.
    overlap = (reference_top.unsqueeze(-1) == actual_top.unsqueeze(-2)).any(dim=-1).sum(dim=-1)
    decision_edge = (
        (~top1_equal)
        | (~topk_first_equal)
        | (~reference_topk_first_is_argmax)
        | (~actual_topk_first_is_argmax)
    )
    edge_coordinates = decision_edge.nonzero(as_tuple=False)
    decision_records = []
    for coordinate in edge_coordinates[:64].tolist():
        batch_index, position = coordinate
        index = (batch_index, position)
        reference_token = int(reference_top1[index])
        actual_token = int(actual_top1[index])
        reference_vector = reference[batch_index, position].detach().float()
        actual_vector = actual[batch_index, position].detach().float()
        reference_selected_logit = float(reference_vector[reference_token])
        reference_cross_logit = float(reference_vector[actual_token])
        actual_selected_logit = float(actual_vector[actual_token])
        actual_cross_logit = float(actual_vector[reference_token])
        reference_log_normalizer = torch.logsumexp(reference_vector, dim=-1)
        actual_log_normalizer = torch.logsumexp(actual_vector, dim=-1)
        decision_records.append(
            {
                "index": coordinate,
                "phase": _decision_phase(position, prompt_length),
                "generation_relevant": prompt_length is None or position >= prompt_length - 1,
                "hard_gate_position": prompt_length is None or position >= prompt_length - 1,
                "reference_argmax_token": reference_token,
                "actual_argmax_token": actual_token,
                "greedy_decision_mismatch": reference_token != actual_token,
                "reference_max_logit": float(reference_max[index]),
                "actual_max_logit": float(actual_max[index]),
                "reference_num_argmax_ties": int(reference_tie_count[index]),
                "actual_num_argmax_ties": int(actual_tie_count[index]),
                "reference_topk_first": int(reference_top[index][0]),
                "actual_topk_first": int(actual_top[index][0]),
                "topk_rank_order_mismatch": bool(not topk_first_equal[index]),
                "reference_topk_first_differs_from_argmax": bool(
                    not reference_topk_first_is_argmax[index]
                ),
                "actual_topk_first_differs_from_argmax": bool(
                    not actual_topk_first_is_argmax[index]
                ),
                "reference_top10": reference_top[index].tolist(),
                "actual_top10": actual_top[index].tolist(),
                "top10_overlap_count": int(overlap[index]),
                "reference_top1_minus_top2": float(
                    reference_top_values[index][0] - reference_top_values[index][1]
                ),
                "actual_top1_minus_top2": float(
                    actual_top_values[index][0] - actual_top_values[index][1]
                ),
                "reference_preferred_gap": reference_selected_logit - reference_cross_logit,
                "actual_preferred_gap": actual_selected_logit - actual_cross_logit,
                "actual_choice_rank_in_reference": 1 + int((reference_vector > reference_cross_logit).sum()),
                "reference_choice_rank_in_actual": 1 + int((actual_vector > actual_cross_logit).sum()),
                "reference_distribution": {
                    "reference_choice_logit": reference_selected_logit,
                    "actual_choice_logit": reference_cross_logit,
                    "reference_choice_probability": float(
                        torch.exp(reference_vector[reference_token] - reference_log_normalizer)
                    ),
                    "actual_choice_probability": float(
                        torch.exp(reference_vector[actual_token] - reference_log_normalizer)
                    ),
                },
                "actual_distribution": {
                    "actual_choice_logit": actual_selected_logit,
                    "reference_choice_logit": actual_cross_logit,
                    "actual_choice_probability": float(
                        torch.exp(actual_vector[actual_token] - actual_log_normalizer)
                    ),
                    "reference_choice_probability": float(
                        torch.exp(actual_vector[reference_token] - actual_log_normalizer)
                    ),
                },
            }
        )
    all_mask = torch.ones_like(top1_equal, dtype=torch.bool)
    all_summary = _decision_scope_summary(
        top1_equal,
        overlap,
        reference_tie_count,
        actual_tie_count,
        all_mask,
        top_k,
    )
    per_token = [
        {
            "index": [batch_index, position],
            "phase": _decision_phase(position, prompt_length),
            "generation_relevant": prompt_length is None or position >= prompt_length - 1,
            "hard_gate_position": prompt_length is None or position >= prompt_length - 1,
            "greedy_decision_match": bool(top1_equal[batch_index, position]),
            "reference_argmax_token": int(reference_top1[batch_index, position]),
            "actual_argmax_token": int(actual_top1[batch_index, position]),
            "reference_max_logit": float(reference_max[batch_index, position]),
            "actual_max_logit": float(actual_max[batch_index, position]),
            "reference_num_argmax_ties": int(reference_tie_count[batch_index, position]),
            "actual_num_argmax_ties": int(actual_tie_count[batch_index, position]),
            "reference_topk_first": int(reference_top[batch_index, position, 0]),
            "actual_topk_first": int(actual_top[batch_index, position, 0]),
            "topk_rank_order_mismatch": bool(not topk_first_equal[batch_index, position]),
            "top10_overlap_count": int(overlap[batch_index, position]),
        }
        for batch_index in range(reference.shape[0])
        for position in range(reference.shape[1])
    ]
    scope_summaries = {"all_tokens": all_summary}
    if prompt_length is not None:
        positions = torch.arange(reference.shape[1]).unsqueeze(0).expand(reference.shape[0], -1)
        scope_summaries.update(
            {
                "prefill_unused": _decision_scope_summary(
                    top1_equal,
                    overlap,
                    reference_tie_count,
                    actual_tie_count,
                    positions < prompt_length - 1,
                    top_k,
                ),
                "prefill_last": _decision_scope_summary(
                    top1_equal,
                    overlap,
                    reference_tie_count,
                    actual_tie_count,
                    positions == prompt_length - 1,
                    top_k,
                ),
                "decode": _decision_scope_summary(
                    top1_equal,
                    overlap,
                    reference_tie_count,
                    actual_tie_count,
                    positions >= prompt_length,
                    top_k,
                ),
                "generation_relevant": _decision_scope_summary(
                    top1_equal,
                    overlap,
                    reference_tie_count,
                    actual_tie_count,
                    positions >= prompt_length - 1,
                    top_k,
                ),
            }
        )
    nan_count = int(torch.isnan(reference).sum() + torch.isnan(actual).sum())
    inf_count = int(torch.isinf(reference).sum() + torch.isinf(actual).sum())
    metrics = {
        "token_count": all_summary["token_count"],
        "top_k": top_k,
        "top1_agreement_count": all_summary["top1_agreement_count"],
        "top1_disagreement_count": all_summary["top1_disagreement_count"],
        "top1_agreement_fraction": all_summary["top1_agreement_fraction"],
        "top10_overlap_count_min": all_summary["top10_overlap_count_min"],
        "top10_overlap_fraction_mean": all_summary["top10_overlap_fraction_mean"],
        "top10_exact_set_fraction": all_summary["top10_exact_set_fraction"],
        "top10_overlap_histogram": all_summary["top10_overlap_histogram"],
        "top1_disagreements": [
            item for item in decision_records if item["greedy_decision_mismatch"]
        ],
        "topk_rank_order_records": [
            item for item in decision_records if item["topk_rank_order_mismatch"]
        ],
        "decision_edge_records": decision_records,
        "decision_edge_records_truncated": edge_coordinates.shape[0] > len(decision_records),
        "scope_summaries": scope_summaries,
        "per_token": per_token,
        "generation_scope_diagnostic_only": False,
        "prefill_unused_diagnostic_only": prompt_length is not None,
        "formal_gate_scope": "generation_relevant" if prompt_length is not None else "batch_all_tokens",
        "nan_count": nan_count,
        "inf_count": inf_count,
    }
    formal_scope = (
        scope_summaries["generation_relevant"]
        if prompt_length is not None
        else scope_summaries["all_tokens"]
    )
    evaluation = evaluate_numerical_envelope(
        {**formal_scope, "nan_count": nan_count, "inf_count": inf_count},
        P4_CACHE_TOKEN_DECISION_LIMITS,
    )
    metrics["evaluation"] = evaluation
    metrics["passed"] = evaluation["passed"]
    return metrics


def evaluate_integrated_case(
    *,
    prompt_length: int,
    decode_length: int,
    standard_allclose_passed: bool,
    logits_distribution: dict[str, Any],
    cache_structural_passed: bool,
    cache_numerical_envelope_passed: bool,
    token_decisions: dict[str, Any],
) -> dict[str, Any]:
    """Apply the approved hierarchy to one deterministic integrated case."""
    split = (prompt_length, decode_length)
    if split not in P4_CACHE_INTEGRATED_LOGIT_LIMITS:
        raise ValueError(f"Unapproved P4.6 cache split: {split}")
    logits_envelope = evaluate_numerical_envelope(
        logits_distribution,
        P4_CACHE_INTEGRATED_LOGIT_LIMITS[split],
    )
    checks = {
        "cache_structure_and_lengths": cache_structural_passed,
        "cache_tensor_numerical_envelope": cache_numerical_envelope_passed,
        "integrated_logits_standard_or_envelope": standard_allclose_passed or logits_envelope["passed"],
        "token_decision_equivalence": token_decisions.get("passed") is True,
    }
    return {
        "split": {"prompt_length": prompt_length, "decode_length": decode_length},
        "standard_allclose_passed": standard_allclose_passed,
        "logits_envelope": logits_envelope,
        "token_decisions": token_decisions,
        "checks": checks,
        "pass_mode": (
            "standard_bf16_allclose"
            if standard_allclose_passed and all(checks.values())
            else "composite_bf16_cache_gate"
            if all(checks.values())
            else "fail"
        ),
        "passed": all(checks.values()),
    }


def evaluate_batch_control(
    *,
    standard_allclose_passed: bool,
    logits_distribution: dict[str, Any],
    state_evaluations: list[dict[str, Any]],
    token_decisions: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate B=2 against two B=1 executions as a shape control."""
    logits_envelope = evaluate_numerical_envelope(logits_distribution, P4_CACHE_BATCH_LOGIT_LIMITS)
    checks = {
        "batch_logits_standard_or_envelope": standard_allclose_passed or logits_envelope["passed"],
        "all_kda_states_within_envelope": bool(state_evaluations)
        and all(item.get("passed") is True for item in state_evaluations),
        "batch_token_decision_equivalence": token_decisions.get("passed") is True,
    }
    return {
        "standard_allclose_passed": standard_allclose_passed,
        "logits_envelope": logits_envelope,
        "state_evaluations": state_evaluations,
        "token_decisions": token_decisions,
        "checks": checks,
        "pass_mode": (
            "standard_bf16_allclose"
            if standard_allclose_passed and all(checks.values())
            else "batch_shape_numerical_envelope"
            if all(checks.values())
            else "fail"
        ),
        "passed": all(checks.values()),
    }


def evaluate_direct_cache_evidence(
    report: dict[str, Any],
    artifact_verification: dict[str, Any],
) -> dict[str, Any]:
    """Revalidate the frozen v2 causal report as a formal hard prerequisite."""
    direct_cases = report.get("direct_cases", [])
    expected_splits = [(1, 2), (7, 15), (64, 96)]
    split_checks = [
        (case.get("prompt_length"), case.get("sequence_length")) == expected
        for case, expected in zip(direct_cases, expected_splits, strict=False)
    ]
    semantic_checks = []
    for case in direct_cases:
        named = case.get("cache_semantic_checks", {})
        semantic_checks.append(
            case.get("direct_cache_contracts_passed") is True
            and bool(named)
            and all(value is True for value in named.values())
        )

    integrated_cases = report.get("integrated_cases", [])
    integrated_identity = [
        {
            "prompt_length": case.get("prompt_length"),
            "decode_length": case.get("decode_length"),
            "sequence_length": case.get("sequence_length"),
            "token_sha256": case.get("token_sha256"),
        }
        for case in integrated_cases
    ]
    integrated_identity_checks = [
        identity["prompt_length"] == prompt
        and identity["decode_length"] == decode
        and identity["sequence_length"] == prompt + decode
        and isinstance(identity["token_sha256"], str)
        and len(identity["token_sha256"]) == 64
        for identity, (prompt, decode) in zip(integrated_identity, P4_CACHE_CASES, strict=False)
    ]
    batch_token_sha256 = report.get("batch_isolation", {}).get("token_sha256")

    causal = report.get("causal_summary", {})
    evidence_artifact = report.get("artifact_verification", {})
    forced_tails = causal.get("forced_fp32_projection_tails", [])
    tail_checks = (
        len(forced_tails) == 1
        and forced_tails[0].get("sequence_length") == 96
        and len(forced_tails[0].get("tails", [])) == 1
        and forced_tails[0]["tails"][0].get("tensor") == "output_gate"
        and forced_tails[0]["tails"][0].get("num_failed_allclose") == 1
    )
    checks = {
        "stage": report.get("stage") == "P4.6-CACHE-DIAG",
        "diagnostic_status": report.get("status") == "PASS",
        "diagnostic_only": report.get("diagnostic_only") is True,
        "diagnostics_completed": report.get("diagnostics_completed") is True,
        "formal_gate_was_unchanged": report.get("formal_gate_modified") is False,
        "formal_status_was_fail_unchanged": report.get("p4_6_formal_status") == "FAIL_UNCHANGED",
        "no_prior_cache_amendment": report.get("numerics_amendment_approved") is False,
        "token_seed": report.get("token_seed") == P4_CACHE_TOKEN_SEED,
        "direct_case_count": len(direct_cases) == len(expected_splits),
        "direct_splits": len(split_checks) == len(expected_splits) and all(split_checks),
        "direct_cache_semantics": len(semantic_checks) == len(expected_splits) and all(semantic_checks),
        "integrated_case_count": len(integrated_identity) == len(P4_CACHE_CASES),
        "integrated_case_identity": len(integrated_identity_checks) == len(P4_CACHE_CASES)
        and all(integrated_identity_checks),
        "batch_case_identity": isinstance(batch_token_sha256, str) and len(batch_token_sha256) == 64,
        "causal_classification": causal.get("classification")
        == "SUPPORTED_NUMERICAL_PROPAGATION_NOT_CACHE_LAYOUT",
        "causal_direct_contracts": causal.get("direct_cache_contracts_passed") is True,
        "causal_seen_tokens": causal.get("seen_tokens_correct") is True,
        "causal_integrated_failure_reproduced": causal.get("integrated_failure_reproduced") is True,
        "batch_isolation_persisted": causal.get("batch_isolation_persisted") is True,
        "forced_fp32_one_element_tail_retained": tail_checks,
        "artifact_status": artifact_verification.get("status") == "PASS",
        "artifact_parameter_hash_match": bool(artifact_verification.get("parameter_hash"))
        and artifact_verification.get("parameter_hash") == evidence_artifact.get("parameter_hash"),
        "artifact_manifest_hash_match": bool(artifact_verification.get("manifest_hash"))
        and artifact_verification.get("manifest_hash") == evidence_artifact.get("manifest_hash"),
    }
    return {
        "evidence_original_status": report.get("status"),
        "checks": checks,
        "forced_fp32_tail": forced_tails,
        "artifact_identity": {
            "parameter_hash": artifact_verification.get("parameter_hash"),
            "manifest_hash": artifact_verification.get("manifest_hash"),
        },
        "integrated_case_identity": integrated_identity,
        "batch_token_sha256": batch_token_sha256,
        "passed": all(checks.values()),
    }


def load_and_evaluate_direct_cache_evidence(
    output_dir: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    """Load immutable v2 evidence and bind it to the replayed artifact."""
    evidence_path = output_dir / P4_CACHE_EVIDENCE_FILENAME
    if not evidence_path.is_file():
        raise FileNotFoundError(f"Approved P4.6 composite Gate requires {evidence_path}")
    evidence_bytes = evidence_path.read_bytes()
    report = json.loads(evidence_bytes)
    artifact = verify_p4_artifact(artifact_dir, verify_tensor_hashes=False)
    result = evaluate_direct_cache_evidence(report, artifact)
    result["amendment"] = P4_CACHE_NUMERICS_AMENDMENT
    result["scope"] = "P4.6 deterministic cache-equivalence replay only"
    result["evidence_file"] = {
        "path": str(evidence_path),
        "sha256": hashlib.sha256(evidence_bytes).hexdigest(),
    }
    return result


__all__ = [
    "P4_CACHE_AMENDMENT_TYPE",
    "P4_CACHE_BATCH_LOGIT_LIMITS",
    "P4_CACHE_BATCH_STATE_LIMITS",
    "P4_CACHE_CASES",
    "P4_CACHE_EVIDENCE_FILENAME",
    "P4_CACHE_INTEGRATED_LOGIT_LIMITS",
    "P4_CACHE_NUMERICS_AMENDMENT",
    "P4_CACHE_TENSOR_LIMITS",
    "P4_CACHE_TOKEN_DECISION_LIMITS",
    "P4_CACHE_TOKEN_SEED",
    "evaluate_batch_control",
    "evaluate_direct_cache_evidence",
    "evaluate_integrated_case",
    "evaluate_numerical_envelope",
    "load_and_evaluate_direct_cache_evidence",
    "token_decision_metrics",
]
