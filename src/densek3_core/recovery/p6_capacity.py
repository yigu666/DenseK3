"""Teacher-forced rank-capacity helpers for the P6.2a diagnostic."""

from __future__ import annotations

import statistics
from typing import Any

import torch

from densek3_core.recovery.p5_corpus import PackedTokenDataset
from densek3_core.recovery.p6_mla_probe import P6NoPEMLA, tensor_sha256

CAPACITY_FIELDS = (
    "k",
    "v",
    "attention_core",
    "attention_output",
    "mixer_output",
    "post_attention_residual_hidden",
)


def _distribution(values: torch.Tensor) -> dict[str, float | int]:
    values = values.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
    if not values.numel():
        raise ValueError("P6.2a latent RMS distribution cannot be empty")
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("P6.2a latent RMS distribution is non-finite")
    quantiles = torch.quantile(values, torch.tensor([0.01, 0.5, 0.99], dtype=torch.float64))
    return {
        "count": values.numel(),
        "minimum": float(values.min()),
        "p01": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p99": float(quantiles[2]),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "rms": float(values.square().mean().sqrt()),
    }


@torch.no_grad()
def latent_rms_statistics(
    module: P6NoPEMLA,
    normalized_batches: list[torch.Tensor],
    *,
    prefix: int,
) -> dict[str, Any]:
    """Measure raw and scale-matched latent per-token RMS on frozen inputs."""
    if not normalized_batches:
        raise ValueError("P6.2a normalized calibration batches cannot be empty")
    device = next(module.parameters()).device
    raw_values = []
    normalized_values = []
    for batch_cpu in normalized_batches:
        batch = batch_cpu[:, :prefix].to(device)
        latent = module.kv_a_proj(batch)
        normalized = module.kv_a_layernorm(latent)
        raw_values.append(latent.float().square().mean(dim=-1).sqrt().cpu())
        normalized_values.append(normalized.float().square().mean(dim=-1).sqrt().cpu())
    raw = torch.cat(raw_values)
    normalized = torch.cat(normalized_values)
    return {
        "rank": module.kv_lora_rank,
        "tokens": raw.numel(),
        "raw_per_token_rms": _distribution(raw),
        "scale_matched_per_token_rms": _distribution(normalized),
        "gamma": _distribution(module.kv_a_layernorm.weight.float()),
        "all_finite": bool(torch.isfinite(raw).all() and torch.isfinite(normalized).all()),
    }


@torch.no_grad()
def collect_scaleout_inputs(
    model: torch.nn.Module,
    train: PackedTokenDataset,
    layer_indices: list[int],
    sequence_indices: list[int],
    *,
    prefix: int,
) -> dict[str, Any]:
    """Capture inputs produced by the in-memory eight-MLA rank-512 assembly."""
    if train.split != "train":
        raise ValueError("P6.2a scale-out inputs must use the frozen train split")
    device = next(model.parameters()).device
    residual = {index: [] for index in layer_indices}
    normalized = {index: [] for index in layer_indices}
    token_hashes = []
    for ordinal, sequence_index in enumerate(sequence_indices, start=1):
        tokens = torch.from_numpy(train[sequence_index].copy()).unsqueeze(0).to(device)
        captured: dict[int, torch.Tensor] = {}
        handles = []
        for layer_index in layer_indices:
            layer = model.model.layers[layer_index]
            handles.append(
                layer.input_layernorm.register_forward_pre_hook(
                    lambda _module, inputs, index=layer_index, values=captured: values.__setitem__(
                        index,
                        inputs[0][:, :prefix].detach(),
                    )
                )
            )
        try:
            model.model(input_ids=tokens, use_cache=False, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        if set(captured) != set(layer_indices):
            raise RuntimeError("P6.2a did not capture every scale-out layer input")
        for layer_index in layer_indices:
            value = captured[layer_index]
            residual[layer_index].append(value.cpu())
            normalized[layer_index].append(
                model.model.layers[layer_index].input_layernorm(value).cpu()
            )
        token_hashes.append(tensor_sha256(tokens))
        print(
            f"P6_2A_SCALEOUT_INPUT_CAPTURE={ordinal}/{len(sequence_indices)}",
            flush=True,
        )
    return {
        "residual": residual,
        "normalized": normalized,
        "manifest": {
            "split": "train",
            "sequence_indices": sequence_indices,
            "prefix": prefix,
            "token_sha256": token_hashes,
            "assembled_model_rank": 512,
            "in_memory_only": True,
            "checkpoint_written": False,
            "per_layer_normalized_input_sha256": {
                str(index): [tensor_sha256(value) for value in normalized[index]]
                for index in layer_indices
            },
        },
    }


def capacity_diagnostics(
    teacher_forced_by_rank: dict[int, dict[str, Any]],
    scaleout_r512: dict[str, Any],
    *,
    rank_elasticity_high_min: float,
    scaleout_amplification_high_min: float,
) -> dict[str, Any]:
    """Compute continuous rank and scale-out attribution diagnostics."""
    if set(teacher_forced_by_rank) != {512, 768, 1024}:
        raise ValueError("P6.2a teacher-forced ranks must be exactly 512, 768, and 1024")
    rank_elasticity = {}
    scaleout_amplification = {}
    monotonic = {}
    epsilon = torch.finfo(torch.float64).eps
    for field in CAPACITY_FIELDS:
        errors = {
            rank: float(teacher_forced_by_rank[rank][field]["relative_l2_error"])
            for rank in (512, 768, 1024)
        }
        denominator = max(errors[512], epsilon)
        rank_elasticity[field] = (errors[512] - errors[1024]) / denominator
        scaleout_amplification[field] = (
            float(scaleout_r512[field]["relative_l2_error"]) / denominator
        )
        monotonic[field] = errors[512] >= errors[768] >= errors[1024]
    mixer_elasticity = rank_elasticity["mixer_output"]
    mixer_amplification = scaleout_amplification["mixer_output"]
    high_elasticity = mixer_elasticity >= rank_elasticity_high_min
    high_amplification = mixer_amplification >= scaleout_amplification_high_min
    if high_elasticity and high_amplification:
        classification = "MIXED_CAPACITY_AND_DISTRIBUTION_SHIFT"
    elif high_elasticity:
        classification = "INTRINSIC_RANK_CAPACITY_LIMITED"
    elif high_amplification:
        classification = "DISTRIBUTION_SHIFT_ACCUMULATION_LIMITED"
    else:
        classification = "LOW_RANK_ELASTICITY_OBJECTIVE_OR_INITIALIZATION_LIMITED"
    return {
        "rank_elasticity": rank_elasticity,
        "scaleout_amplification": scaleout_amplification,
        "teacher_forced_error_monotonic_by_rank": monotonic,
        "primary_mixer_rank_elasticity": mixer_elasticity,
        "primary_mixer_scaleout_amplification": mixer_amplification,
        "classification": classification,
        "classification_thresholds": {
            "rank_elasticity_high_min": rank_elasticity_high_min,
            "scaleout_amplification_high_min": scaleout_amplification_high_min,
        },
        "classification_is_diagnostic_not_gate": True,
    }


def aggregate_capacity_diagnostics(layers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not layers:
        raise ValueError("P6.2a requires at least one layer diagnostic")
    elasticities = [
        float(layer["diagnostics"]["primary_mixer_rank_elasticity"])
        for layer in layers.values()
    ]
    amplifications = [
        float(layer["diagnostics"]["primary_mixer_scaleout_amplification"])
        for layer in layers.values()
    ]
    classifications: dict[str, int] = {}
    for layer in layers.values():
        label = str(layer["diagnostics"]["classification"])
        classifications[label] = classifications.get(label, 0) + 1
    return {
        "median_mixer_rank_elasticity": statistics.median(elasticities),
        "minimum_mixer_rank_elasticity": min(elasticities),
        "maximum_mixer_rank_elasticity": max(elasticities),
        "median_mixer_scaleout_amplification": statistics.median(amplifications),
        "minimum_mixer_scaleout_amplification": min(amplifications),
        "maximum_mixer_scaleout_amplification": max(amplifications),
        "layer_classification_counts": dict(sorted(classifications.items())),
        "continuous_diagnostics_are_primary": True,
        "heuristic_labels_are_not_formal_gates": True,
    }


__all__ = [
    "CAPACITY_FIELDS",
    "aggregate_capacity_diagnostics",
    "capacity_diagnostics",
    "collect_scaleout_inputs",
    "latent_rms_statistics",
]
