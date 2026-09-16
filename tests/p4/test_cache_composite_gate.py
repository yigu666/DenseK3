from __future__ import annotations

import json
import runpy
from collections import namedtuple
from pathlib import Path

import pytest
import torch

from densek3_core.evaluation.p4_cache_gate import (
    P4_CACHE_BATCH_STATE_LIMITS,
    P4_CACHE_NUMERICS_AMENDMENT,
    evaluate_batch_control,
    evaluate_direct_cache_evidence,
    evaluate_integrated_case,
    evaluate_numerical_envelope,
    token_decision_metrics,
)


def passing_decisions() -> dict[str, object]:
    return {
        "top1_agreement_fraction": 1.0,
        "top10_overlap_fraction_mean": 1.0,
        "top10_overlap_count_min": 10,
        "nan_count": 0,
        "inf_count": 0,
        "passed": True,
    }


def test_token_decision_gate_requires_every_top1_and_retains_top10_overlap() -> None:
    reference = torch.arange(24, dtype=torch.float32).view(1, 2, 12)
    same = token_decision_metrics(reference, reference.clone())
    assert same["top1_agreement_fraction"] == 1.0
    assert same["top10_overlap_count_min"] == 10
    assert same["passed"] is True

    changed = reference.clone()
    changed[0, 0, 10], changed[0, 0, 11] = changed[0, 0, 11].clone(), changed[0, 0, 10].clone()
    mismatch = token_decision_metrics(reference, changed)
    assert mismatch["top1_disagreement_count"] == 1
    assert mismatch["top10_overlap_count_min"] == 10
    assert mismatch["passed"] is False


def test_greedy_top1_uses_argmax_even_when_topk_tie_order_differs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits = torch.full((1, 1, 221), -1.0)
    logits[..., 11] = 5.0
    logits[..., 83] = 5.0
    logits[..., 220] = 5.0
    original_topk = torch.Tensor.topk
    topk_result = namedtuple("TopKResult", ("values", "indices"))

    def force_token_220_first(
        value: torch.Tensor,
        k: int,
        dim: int | None = None,
        largest: bool = True,
        sorted: bool = True,
    ) -> object:
        result = original_topk(value, k, dim=dim, largest=largest, sorted=sorted)
        values = result.values.clone()
        indices = result.indices.clone()
        token_220_position = int((indices[0, 0] == 220).nonzero(as_tuple=False)[0])
        values[0, 0, 0], values[0, 0, token_220_position] = (
            values[0, 0, token_220_position].clone(),
            values[0, 0, 0].clone(),
        )
        indices[0, 0, 0], indices[0, 0, token_220_position] = (
            indices[0, 0, token_220_position].clone(),
            indices[0, 0, 0].clone(),
        )
        return topk_result(values, indices)

    monkeypatch.setattr(torch.Tensor, "topk", force_token_220_first)
    metrics = token_decision_metrics(logits, logits.clone(), prompt_length=1)
    token = metrics["per_token"][0]
    assert token["reference_argmax_token"] == 11
    assert token["actual_argmax_token"] == 11
    assert token["reference_topk_first"] == 220
    assert token["actual_topk_first"] == 220
    assert token["reference_num_argmax_ties"] == 3
    assert token["actual_num_argmax_ties"] == 3
    assert metrics["passed"] is True


def test_top10_candidate_changes_do_not_change_greedy_metric() -> None:
    reference = torch.arange(12, 0, -1, dtype=torch.float32).view(1, 1, 12)
    actual = reference.clone()
    actual[..., 9] = -10.0
    actual[..., 11] = 2.5
    metrics = token_decision_metrics(reference, actual, prompt_length=1)
    scope = metrics["scope_summaries"]["generation_relevant"]
    assert scope["top1_agreement_fraction"] == 1.0
    assert scope["top10_overlap_count_min"] == 9
    assert metrics["per_token"][0]["reference_argmax_token"] == 0
    assert metrics["per_token"][0]["actual_argmax_token"] == 0
    assert metrics["passed"] is True

    below_minimum = reference.clone()
    below_minimum[..., 8] = -10.0
    below_minimum[..., 9] = -10.0
    below_minimum[..., 10] = 3.5
    below_minimum[..., 11] = 3.25
    rejected = token_decision_metrics(reference, below_minimum, prompt_length=1)
    assert rejected["scope_summaries"]["generation_relevant"]["top1_agreement_fraction"] == 1.0
    assert rejected["scope_summaries"]["generation_relevant"]["top10_overlap_count_min"] == 8
    assert rejected["passed"] is False


def test_token_decision_rescore_separates_unused_prompt_from_generation_path() -> None:
    reference = torch.arange(60, dtype=torch.float32).view(1, 5, 12)
    actual = reference.clone()
    actual[0, 1, 10], actual[0, 1, 11] = (
        actual[0, 1, 11].clone(),
        actual[0, 1, 10].clone(),
    )
    metrics = token_decision_metrics(reference, actual, prompt_length=4)
    scopes = metrics["scope_summaries"]
    assert metrics["formal_gate_scope"] == "generation_relevant"
    assert metrics["passed"] is True
    assert scopes["all_tokens"]["top1_agreement_fraction"] == 0.8
    assert scopes["prefill_unused"]["top1_disagreement_count"] == 1
    assert scopes["prefill_last"]["top1_agreement_fraction"] == 1.0
    assert scopes["decode"]["top1_agreement_fraction"] == 1.0
    assert scopes["generation_relevant"]["top1_agreement_fraction"] == 1.0

    mismatch = metrics["top1_disagreements"][0]
    assert mismatch["phase"] == "prefill_unused"
    assert mismatch["generation_relevant"] is False
    assert mismatch["hard_gate_position"] is False
    assert mismatch["greedy_decision_mismatch"] is True
    assert mismatch["top10_overlap_count"] == 10
    assert mismatch["actual_choice_rank_in_reference"] == 2
    assert mismatch["reference_choice_rank_in_actual"] == 2
    assert mismatch["reference_top1_minus_top2"] == 1.0
    assert mismatch["actual_top1_minus_top2"] == 1.0
    assert [item["phase"] for item in metrics["per_token"]] == [
        "prefill_unused",
        "prefill_unused",
        "prefill_unused",
        "prefill_last",
        "decode",
    ]


def test_integrated_gate_is_case_scoped_and_keeps_standard_allclose_separate() -> None:
    distribution = {
        "relative_l2_error": 0.015,
        "cosine_similarity": 0.9999,
        "mean_abs_error": 0.02,
        "p99_9_abs_error": 0.1,
        "failed_allclose_fraction": 0.6,
        "max_abs_error": 0.3,
        "nan_count": 0,
        "inf_count": 0,
    }
    result = evaluate_integrated_case(
        prompt_length=64,
        decode_length=32,
        standard_allclose_passed=False,
        logits_distribution=distribution,
        cache_structural_passed=True,
        cache_numerical_envelope_passed=True,
        token_decisions=passing_decisions(),
    )
    assert result["passed"] is True
    assert result["pass_mode"] == "composite_bf16_cache_gate"
    assert result["standard_allclose_passed"] is False

    distribution["max_abs_error"] = 0.75
    rejected = evaluate_integrated_case(
        prompt_length=64,
        decode_length=32,
        standard_allclose_passed=False,
        logits_distribution=distribution,
        cache_structural_passed=True,
        cache_numerical_envelope_passed=True,
        token_decisions=passing_decisions(),
    )
    assert rejected["passed"] is False


def test_batch_control_requires_each_state_envelope_and_token_decisions() -> None:
    logits = {
        "relative_l2_error": 0.0069,
        "cosine_similarity": 0.99998,
        "mean_abs_error": 0.01,
        "p99_9_abs_error": 0.08,
        "failed_allclose_fraction": 0.28,
        "max_abs_error": 0.15,
        "nan_count": 0,
        "inf_count": 0,
    }
    state = evaluate_numerical_envelope(
        {
            "relative_l2_error": 0.008,
            "cosine_similarity": 0.99997,
            "mean_abs_error": 5e-5,
            "max_abs_error": 0.015,
            "nan_count": 0,
            "inf_count": 0,
        },
        P4_CACHE_BATCH_STATE_LIMITS,
    )
    result = evaluate_batch_control(
        standard_allclose_passed=False,
        logits_distribution=logits,
        state_evaluations=[state],
        token_decisions=passing_decisions(),
    )
    assert result["passed"] is True
    assert result["pass_mode"] == "batch_shape_numerical_envelope"

    state["passed"] = False
    rejected = evaluate_batch_control(
        standard_allclose_passed=False,
        logits_distribution=logits,
        state_evaluations=[state],
        token_decisions=passing_decisions(),
    )
    assert rejected["passed"] is False


def diagnostic_evidence() -> dict[str, object]:
    named_checks = {
        "all_values_finite": True,
        "convolution_outputs_pass": True,
        "convolution_states_exact": True,
        "fla_recurrence_output_pass": True,
        "fla_recurrence_state_pass": True,
        "gqa_append_exact": True,
        "position_sequence_exact": True,
        "reference_recurrence_output_pass": True,
        "reference_recurrence_state_pass": True,
    }
    direct = [
        {
            "prompt_length": prompt,
            "sequence_length": sequence,
            "direct_cache_contracts_passed": True,
            "cache_semantic_checks": dict(named_checks),
        }
        for prompt, sequence in ((1, 2), (7, 15), (64, 96))
    ]
    integrated = [
        {
            "prompt_length": prompt,
            "decode_length": decode,
            "sequence_length": prompt + decode,
            "token_sha256": str(index) * 64,
        }
        for index, (prompt, decode) in enumerate(((1, 1), (7, 8), (64, 32)), start=1)
    ]
    return {
        "stage": "P4.6-CACHE-DIAG",
        "status": "PASS",
        "diagnostic_only": True,
        "diagnostics_completed": True,
        "formal_gate_modified": False,
        "p4_6_formal_status": "FAIL_UNCHANGED",
        "numerics_amendment_approved": False,
        "token_seed": 20260811,
        "direct_cases": direct,
        "integrated_cases": integrated,
        "batch_isolation": {"token_sha256": "b" * 64},
        "causal_summary": {
            "classification": "SUPPORTED_NUMERICAL_PROPAGATION_NOT_CACHE_LAYOUT",
            "direct_cache_contracts_passed": True,
            "seen_tokens_correct": True,
            "integrated_failure_reproduced": True,
            "batch_isolation_persisted": True,
            "forced_fp32_projection_tails": [
                {
                    "sequence_length": 96,
                    "tails": [
                        {
                            "tensor": "output_gate",
                            "num_failed_allclose": 1,
                        }
                    ],
                }
            ],
        },
        "artifact_verification": {
            "status": "PASS",
            "parameter_hash": "parameters",
            "manifest_hash": "manifest",
        },
    }


def test_direct_semantics_evidence_is_bound_to_the_artifact() -> None:
    report = diagnostic_evidence()
    artifact = {
        "status": "PASS",
        "parameter_hash": "parameters",
        "manifest_hash": "manifest",
    }
    accepted = evaluate_direct_cache_evidence(report, artifact)
    assert accepted["passed"] is True
    assert accepted["integrated_case_identity"][2]["sequence_length"] == 96

    artifact["parameter_hash"] = "different"
    rejected = evaluate_direct_cache_evidence(report, artifact)
    assert rejected["passed"] is False
    assert rejected["checks"]["artifact_parameter_hash_match"] is False


def test_replay_mode_requires_frozen_p4_0_through_p4_5(tmp_path: Path) -> None:
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/run_p4.py"))
    output = tmp_path / "manifests"
    output.mkdir()
    for stage, _, filename, marker in namespace["STAGES"][:6]:
        (output / filename).write_text(
            json.dumps(
                {
                    "stage": stage,
                    "status": "PASS",
                    "result_marker": marker + "=PASS",
                }
            ),
            encoding="utf-8",
        )
    replay = namespace["load_replay_prerequisites"]({"output_dir": output})
    assert replay["passed"] is True
    assert len(replay["reports"]) == 6


def test_cache_amendment_id_is_frozen() -> None:
    assert P4_CACHE_NUMERICS_AMENDMENT == "P4-CACHE-NUM-2026-08-11-01"
