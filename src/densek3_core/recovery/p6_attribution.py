"""Metric and decision helpers for the no-training P6.1b attribution."""

from __future__ import annotations

from typing import Any

import torch

from densek3_core.recovery.p5_transition import hidden_drift
from densek3_core.recovery.p6_mla_probe import P6NoPEFullAttention, P6NoPEMLA, nope_attention_trace

TRACE_FIELDS = (
    "q",
    "k",
    "v",
    "attention_scores_visible",
    "attention_core",
    "attention_output",
    "gated_attention_output",
    "mixer_output",
)


@torch.no_grad()
def trace_case(
    module: P6NoPEFullAttention | P6NoPEMLA,
    residual_hidden_states: torch.Tensor,
    normalized_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    trace = nope_attention_trace(module, normalized_hidden_states, attention_mask)
    trace["post_attention_residual_hidden"] = residual_hidden_states + trace["mixer_output"]
    return trace


def compare_traces(reference: dict[str, torch.Tensor], actual: dict[str, torch.Tensor]) -> dict[str, Any]:
    fields = (*TRACE_FIELDS, "post_attention_residual_hidden")
    return {name: hidden_drift(reference[name], actual[name]) for name in fields}


def trace_hashes(trace: dict[str, torch.Tensor]) -> dict[str, str]:
    """Hash attribution inputs/outputs so a repeated server run can prove identity."""
    from densek3_core.recovery.p6_mla_probe import tensor_sha256

    return {
        name: tensor_sha256(trace[name])
        for name in (*TRACE_FIELDS, "post_attention_residual_hidden")
    }


def _mixer(effect: dict[str, Any]) -> float:
    return float(effect["mixer_output"]["relative_l2_error"])


def classify_attribution(effects: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Classify the largest measured function shock; this is not a capability Gate."""
    required = {"B_to_C", "B_to_D", "B_to_E", "B_to_F", "F_to_G", "F_to_H", "F_to_C"}
    if set(effects) != required:
        raise ValueError(f"P6.1b effects must be exactly {sorted(required)}")
    magnitudes = {
        "LOW_RANK_CAPACITY": _mixer(effects["B_to_F"]),
        "LATENT_NORMALIZATION": _mixer(effects["F_to_G"]),
        "QK_NORMALIZATION": max(_mixer(effects["B_to_E"]), _mixer(effects["F_to_H"])),
        "GEMM_SPLIT_NUMERICS": _mixer(effects["B_to_D"]),
    }
    total = _mixer(effects["B_to_C"])
    remaining_joint = _mixer(effects["F_to_C"])
    remaining_isolated = max(
        magnitudes["LATENT_NORMALIZATION"],
        magnitudes["QK_NORMALIZATION"],
        magnitudes["GEMM_SPLIT_NUMERICS"],
    )
    interaction_ratio = remaining_joint / max(remaining_isolated, torch.finfo(torch.float64).eps)
    low_rank_explains_total = magnitudes["LOW_RANK_CAPACITY"] / max(
        total,
        torch.finfo(torch.float64).eps,
    )
    if low_rank_explains_total >= 0.75:
        dominant = "LOW_RANK_CAPACITY"
    elif remaining_joint >= 0.35 * total and interaction_ratio >= 1.25:
        dominant = "INTERACTION"
    else:
        dominant = max(magnitudes, key=magnitudes.__getitem__)
    ordered_atomic = sorted(magnitudes, key=magnitudes.__getitem__, reverse=True)
    secondary = next((name for name in ordered_atomic if name != dominant), ordered_atomic[0])
    recommendations = {
        "LOW_RANK_CAPACITY": "REDESIGN_RANK512_INITIALIZATION_WITHOUT_CHANGING_RANK_YET",
        "LATENT_NORMALIZATION": "REDESIGN_LATENT_RMSNORM_SCALE_OR_PLACEMENT",
        "QK_NORMALIZATION": "ABSORB_DONOR_QK_NORMALIZATION_INTO_MLA_REPARAMETERIZATION",
        "GEMM_SPLIT_NUMERICS": "PRESERVE_OR_FUSE_PACKED_Q_GATE_PROJECTION_SEMANTICS",
        "INTERACTION": "BUILD_A_STAGED_JOINT_SEMANTIC_BRIDGE_FOR_NORM_COMPRESSION_AND_PROJECTION",
    }
    return {
        "dominant_factor": dominant,
        "secondary_factor": secondary,
        "recommended_redesign": recommendations[dominant],
        "atomic_mixer_relative_l2": magnitudes,
        "total_B_to_C_mixer_relative_l2": total,
        "remaining_F_to_C_mixer_relative_l2": remaining_joint,
        "remaining_joint_to_largest_isolated_ratio": interaction_ratio,
        "low_rank_B_to_F_fraction_of_total_B_to_C": low_rank_explains_total,
        "interaction_assessment": (
            "DOMINANT_INTERACTION" if dominant == "INTERACTION" else "MEASURED_NOT_DOMINANT"
        ),
        "classification_is_attribution_not_capability_gate": True,
    }


__all__ = ["TRACE_FIELDS", "classify_attribution", "compare_traces", "trace_case", "trace_hashes"]
