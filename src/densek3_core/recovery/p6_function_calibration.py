"""Local attention-function calibration helpers for the P6.2b pilot."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from densek3_core.recovery.p5_corpus import PackedTokenDataset
from densek3_core.recovery.p6_attribution import TRACE_FIELDS, trace_case
from densek3_core.recovery.p6_initialization import DriftAccumulator
from densek3_core.recovery.p6_mla_probe import (
    P6NoPEFullAttention,
    P6NoPEMLA,
    tensor_sha256,
)

TRAINABLE_PARAMETER_NAMES = (
    "kv_a_proj.weight",
    "kv_a_layernorm.weight",
    "kv_b_proj.weight",
)


def select_document_disjoint_sequences(
    corpus_manifest: dict[str, Any],
    *,
    fit_sequences: int,
    validation_sequences: int,
) -> dict[str, Any]:
    """Select fit from the start and validation from the end with zero document overlap."""
    train = corpus_manifest["splits"]["train"]
    sequence_length = int(train["sequence_length"])
    sequence_count = int(train["sequence_count"])
    intervals = [
        (
            int(item["token_start"]),
            int(item["eos_position"]) + 1,
            str(item["document_id"]),
        )
        for item in train["memberships"]
    ]

    def documents_for_sequence(index: int) -> set[str]:
        start = index * sequence_length
        end = start + sequence_length
        documents = {
            document_id
            for document_start, document_end, document_id in intervals
            if document_start < end and document_end > start
        }
        if not documents:
            raise ValueError(f"P6.2b packed train sequence {index} has no document membership")
        return documents

    fit_indices = list(range(fit_sequences))
    if fit_sequences + validation_sequences > sequence_count:
        raise ValueError("P6.2b requested more sequences than the frozen train split contains")
    fit_documents = set().union(*(documents_for_sequence(index) for index in fit_indices))
    validation_indices = []
    validation_documents: set[str] = set()
    for index in range(sequence_count - 1, fit_sequences - 1, -1):
        documents = documents_for_sequence(index)
        if documents & fit_documents:
            continue
        validation_indices.append(index)
        validation_documents.update(documents)
        if len(validation_indices) == validation_sequences:
            break
    if len(validation_indices) != validation_sequences:
        raise ValueError("P6.2b could not construct a document-disjoint validation split")
    validation_indices.sort()
    overlap = fit_documents & validation_documents
    if overlap:
        raise AssertionError("P6.2b fit/validation document overlap is non-zero")
    return {
        "fit_sequence_indices": fit_indices,
        "validation_sequence_indices": validation_indices,
        "fit_sequence_count": len(fit_indices),
        "validation_sequence_count": len(validation_indices),
        "fit_documents": sorted(fit_documents),
        "validation_documents": sorted(validation_documents),
        "fit_document_count": len(fit_documents),
        "validation_document_count": len(validation_documents),
        "document_overlap": [],
        "document_overlap_count": 0,
        "selection_algorithm": "FIT_FROM_START_VALIDATION_FROM_END_WITH_DOCUMENT_EXCLUSION",
        "sequence_length": sequence_length,
    }


@torch.no_grad()
def capture_frozen_p5_inputs(
    model: torch.nn.Module,
    train: PackedTokenDataset,
    layer_indices: list[int],
    sequence_indices: list[int],
    *,
    token_prefix: int | None,
    marker: str,
) -> dict[str, Any]:
    """Capture target-layer residual and normalized inputs from the untouched P5 model."""
    if train.split != "train":
        raise ValueError("P6.2b calibration inputs must come from the frozen train split")
    device = next(model.parameters()).device
    residual = {index: [] for index in layer_indices}
    normalized = {index: [] for index in layer_indices}
    token_hashes = []
    for ordinal, sequence_index in enumerate(sequence_indices, start=1):
        token_array = train[sequence_index]
        if token_prefix is not None:
            token_array = token_array[:token_prefix]
        tokens = torch.from_numpy(token_array.copy()).unsqueeze(0).to(device)
        captured: dict[int, torch.Tensor] = {}
        handles = []
        for layer_index in layer_indices:
            layer = model.model.layers[layer_index]
            handles.append(
                layer.input_layernorm.register_forward_pre_hook(
                    lambda _module, inputs, index=layer_index, values=captured: values.__setitem__(
                        index,
                        inputs[0].detach(),
                    )
                )
            )
        try:
            model.model(input_ids=tokens, use_cache=False, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        if set(captured) != set(layer_indices):
            raise RuntimeError("P6.2b did not capture every frozen P5 target layer")
        for layer_index in layer_indices:
            value = captured[layer_index]
            residual[layer_index].append(value.cpu())
            normalized[layer_index].append(
                model.model.layers[layer_index].input_layernorm(value).cpu()
            )
        token_hashes.append(tensor_sha256(tokens))
        print(f"{marker}={ordinal}/{len(sequence_indices)}", flush=True)
    return {
        "residual": residual,
        "normalized": normalized,
        "manifest": {
            "split": "train",
            "sequence_indices": sequence_indices,
            "token_prefix": token_prefix,
            "effective_activation_tokens_per_layer": sum(
                value.shape[1] for value in residual[layer_indices[0]]
            ),
            "token_sha256": token_hashes,
            "per_layer_normalized_input_sha256": {
                str(index): [tensor_sha256(value) for value in normalized[index]]
                for index in layer_indices
            },
            "source_model": "UNTOUCHED_FROZEN_P5",
        },
    }


def freeze_local_calibration_scope(module: P6NoPEMLA) -> dict[str, Any]:
    """Freeze all parameters except the rank-512 latent representation itself."""
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    named = dict(module.named_parameters())
    if not set(TRAINABLE_PARAMETER_NAMES) <= set(named):
        raise ValueError("P6.2b target module does not expose the frozen trainable contract")
    for name in TRAINABLE_PARAMETER_NAMES:
        named[name].requires_grad_(True)
    trainable = [name for name, parameter in module.named_parameters() if parameter.requires_grad]
    if trainable != list(TRAINABLE_PARAMETER_NAMES):
        raise AssertionError(f"P6.2b trainable scope mismatch: {trainable}")
    return {
        "trainable_parameter_names": trainable,
        "trainable_parameter_count": sum(named[name].numel() for name in trainable),
        "frozen_parameter_names": [
            name for name, parameter in module.named_parameters() if not parameter.requires_grad
        ],
        "q_projection_frozen": not named["q_proj.weight"].requires_grad,
        "gate_projection_frozen": not named["g_proj.weight"].requires_grad,
        "output_projection_frozen": not named["o_proj.weight"].requires_grad,
        "kv_b_semantic_slices": "INTERLEAVED_PER_HEAD_K_UP_AND_V_UP",
    }


def differentiable_function_outputs(
    module: P6NoPEFullAttention | P6NoPEMLA,
    normalized_hidden_states: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return differentiable local attention-core and mixer outputs."""
    q, k, v, gate, _ = module.project_qkvg(normalized_hidden_states)
    core = F.scaled_dot_product_attention(q, k, v, is_causal=q.shape[2] > 1)
    batch, _, sequence, _ = core.shape
    attention_output = core.transpose(1, 2).contiguous().reshape(batch, sequence, -1)
    mixer_output = module.o_proj(attention_output * torch.sigmoid(gate))
    return {"attention_core": core, "mixer_output": mixer_output}


def normalized_mse(actual: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    epsilon = torch.finfo(actual.dtype).eps
    return (actual - reference).square().mean() / reference.square().mean().clamp_min(epsilon)


@torch.no_grad()
def local_validation_metrics(
    donor: torch.nn.Module,
    student: P6NoPEMLA,
    residual_batches: list[torch.Tensor],
    normalized_batches: list[torch.Tensor],
) -> dict[str, Any]:
    """Evaluate BF16 local function quality and attention-probability divergence."""
    if len(residual_batches) != len(normalized_batches) or not residual_batches:
        raise ValueError("P6.2b validation batches are empty or misaligned")
    teacher = P6NoPEFullAttention(donor, split_q_gate=False, apply_qk_norm=True).eval()
    device = next(student.parameters()).device
    fields = (*TRACE_FIELDS, "post_attention_residual_hidden")
    accumulators = {name: DriftAccumulator() for name in fields}
    probability = DriftAccumulator()
    teacher_student_kl_sum = 0.0
    student_teacher_kl_sum = 0.0
    probability_rows = 0
    for residual_cpu, normalized_cpu in zip(
        residual_batches,
        normalized_batches,
        strict=True,
    ):
        residual = residual_cpu.to(device)
        normalized = normalized_cpu.to(device)
        mask = torch.ones(normalized.shape[:2], device=device)
        teacher_trace = trace_case(teacher, residual, normalized, mask)
        student_trace = trace_case(student, residual, normalized, mask)
        for name in fields:
            accumulators[name].update(teacher_trace[name], student_trace[name])
        teacher_scores = torch.matmul(
            teacher_trace["q"].float(),
            teacher_trace["k"].float().transpose(-1, -2),
        ) / math.sqrt(teacher_trace["q"].shape[-1])
        student_scores = torch.matmul(
            student_trace["q"].float(),
            student_trace["k"].float().transpose(-1, -2),
        ) / math.sqrt(student_trace["q"].shape[-1])
        sequence = normalized.shape[1]
        visible = torch.ones((sequence, sequence), device=device, dtype=torch.bool).tril()
        teacher_scores = teacher_scores.masked_fill(~visible.view(1, 1, sequence, sequence), -torch.inf)
        student_scores = student_scores.masked_fill(~visible.view(1, 1, sequence, sequence), -torch.inf)
        teacher_probability = torch.softmax(teacher_scores, dim=-1)
        student_probability = torch.softmax(student_scores, dim=-1)
        probability.update(teacher_probability, student_probability)
        epsilon = torch.finfo(torch.float32).tiny
        teacher_log = teacher_probability.clamp_min(epsilon).log()
        student_log = student_probability.clamp_min(epsilon).log()
        teacher_student_kl_sum += float(
            (teacher_probability * (teacher_log - student_log)).sum(dim=-1).sum()
        )
        student_teacher_kl_sum += float(
            (student_probability * (student_log - teacher_log)).sum(dim=-1).sum()
        )
        probability_rows += teacher_probability.numel() // sequence
    metrics = {name: accumulator.finalize() for name, accumulator in accumulators.items()}
    metrics["attention_probability"] = {
        **probability.finalize(),
        "teacher_to_student_kl_mean": teacher_student_kl_sum / probability_rows,
        "student_to_teacher_kl_mean": student_teacher_kl_sum / probability_rows,
    }
    return metrics


def local_go_no_go(
    baseline: dict[int, dict[str, Any]],
    calibrated: dict[int, dict[str, Any]],
    *,
    attention_core_improvement_min: float,
) -> dict[str, Any]:
    if set(baseline) != {3, 23} or set(calibrated) != {3, 23}:
        raise ValueError("P6.2b pilot decisions require exactly layers 3 and 23")
    per_layer = {}
    for layer_index in (3, 23):
        base_mixer = float(baseline[layer_index]["mixer_output"]["relative_l2_error"])
        new_mixer = float(calibrated[layer_index]["mixer_output"]["relative_l2_error"])
        base_core = float(baseline[layer_index]["attention_core"]["relative_l2_error"])
        new_core = float(calibrated[layer_index]["attention_core"]["relative_l2_error"])
        mixer_improvement = (base_mixer - new_mixer) / max(
            base_mixer,
            torch.finfo(torch.float64).eps,
        )
        core_improvement = (base_core - new_core) / max(
            base_core,
            torch.finfo(torch.float64).eps,
        )
        checks = {
            "finite": all(
                math.isfinite(value)
                for value in (base_mixer, new_mixer, base_core, new_core)
            ),
            "attention_core_improvement": core_improvement >= attention_core_improvement_min,
        }
        if layer_index == 23:
            checks.update(
                {
                    "mixer_absolute": new_mixer <= 0.40,
                    "mixer_relative_improvement": mixer_improvement >= 0.45,
                }
            )
        else:
            checks.update(
                {
                    "mixer_absolute": new_mixer <= 0.32,
                    "mixer_not_worse_than_20_percent": new_mixer <= 1.20 * base_mixer,
                }
            )
        per_layer[str(layer_index)] = {
            "baseline_mixer_relative_l2": base_mixer,
            "calibrated_mixer_relative_l2": new_mixer,
            "mixer_improvement_fraction": mixer_improvement,
            "baseline_attention_core_relative_l2": base_core,
            "calibrated_attention_core_relative_l2": new_core,
            "attention_core_improvement_fraction": core_improvement,
            "checks": checks,
            "passed": all(checks.values()),
        }
    passed = all(item["passed"] for item in per_layer.values())
    return {
        "per_layer": per_layer,
        "passed": passed,
        "decision": "GO" if passed else "NO_GO",
        "p6_2b_scaleout_allowed": passed,
        "attention_core_improvement_min": attention_core_improvement_min,
    }


def extension_outcome(
    local_decision: dict[str, Any],
    layer23_mixer_by_step: dict[int, float],
    *,
    plateau_last64_absolute_improvement_max: float,
) -> dict[str, Any]:
    """Classify the one-time extension without changing the local GO Gate."""
    required_steps = {192, 224, 256}
    if set(layer23_mixer_by_step) != required_steps:
        raise ValueError("P6.2b-2 plateau assessment requires steps 192, 224, and 256")
    first_interval = layer23_mixer_by_step[192] - layer23_mixer_by_step[224]
    second_interval = layer23_mixer_by_step[224] - layer23_mixer_by_step[256]
    last64_improvement = layer23_mixer_by_step[192] - layer23_mixer_by_step[256]
    plateau = (
        not local_decision["passed"]
        and last64_improvement <= plateau_last64_absolute_improvement_max
    )
    if local_decision["passed"]:
        status = "GO"
    elif plateau:
        status = "NO_GO_PLATEAU"
    else:
        status = "NO_GO_WITHIN_64K"
    return {
        "status": status,
        "passed": local_decision["passed"],
        "layer23_mixer_by_step": {
            str(step): layer23_mixer_by_step[step] for step in sorted(required_steps)
        },
        "layer23_mixer_improvement_192_to_224": first_interval,
        "layer23_mixer_improvement_224_to_256": second_interval,
        "layer23_mixer_improvement_last64": last64_improvement,
        "plateau_last64_absolute_improvement_max": plateau_last64_absolute_improvement_max,
        "plateau": plateau,
        "plateau_rule_is_reason_label_not_go_gate": True,
        "further_budget_extension_allowed": False,
    }


def qk_norm_bridge_phase_a_decision(
    n0_metrics: dict[str, Any],
    n1_metrics: dict[str, Any],
    *,
    mixer_improvement_fraction_min: float,
    mixer_relative_l2_signal_max: float,
) -> dict[str, Any]:
    """Decide whether the frozen Q/K-norm bridge merits fresh calibration.

    N0 is the C3 initialization with donor Q/K RMSNorm removed. N1 is the
    otherwise identical C3 initialization with the donor norms preserved.
    This diagnostic signal does not alter the formal post-calibration Gate.
    """
    n0_mixer = float(n0_metrics["mixer_output"]["relative_l2_error"])
    n1_mixer = float(n1_metrics["mixer_output"]["relative_l2_error"])
    finite = math.isfinite(n0_mixer) and math.isfinite(n1_mixer)
    improvement = (n0_mixer - n1_mixer) / max(
        n0_mixer,
        torch.finfo(torch.float64).eps,
    )
    checks = {
        "finite": finite,
        "relative_improvement_signal": improvement >= mixer_improvement_fraction_min,
        "absolute_error_signal": n1_mixer <= mixer_relative_l2_signal_max,
    }
    passed = checks["finite"] and (
        checks["relative_improvement_signal"] or checks["absolute_error_signal"]
    )
    return {
        "status": "POSITIVE_SIGNAL" if passed else "NO_SIGNAL",
        "passed": passed,
        "n0_mixer_relative_l2": n0_mixer,
        "n1_mixer_relative_l2": n1_mixer,
        "mixer_improvement_fraction": improvement,
        "mixer_improvement_fraction_min": mixer_improvement_fraction_min,
        "mixer_relative_l2_signal_max": mixer_relative_l2_signal_max,
        "checks": checks,
        "formal_local_gate_modified": False,
    }


__all__ = [
    "TRAINABLE_PARAMETER_NAMES",
    "capture_frozen_p5_inputs",
    "differentiable_function_outputs",
    "extension_outcome",
    "freeze_local_calibration_scope",
    "local_go_no_go",
    "local_validation_metrics",
    "normalized_mse",
    "qk_norm_bridge_phase_a_decision",
    "select_document_disjoint_sequences",
]
