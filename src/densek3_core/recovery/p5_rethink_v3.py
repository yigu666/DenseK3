"""Initialization-first contracts for the P5 RETHINK v3 no-training probe."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from densek3_core.modeling import DenseK3ForCausalLM
from densek3_core.modeling.modeling_densek3 import DenseK3KDA
from densek3_core.recovery.p5_transition import canonical_json_hash

P5_RETHINK_V3_LOWER_BOUND = -5.0
P5_RETHINK_V3_PROBABILITY_EPSILON = 2.0**-20
P5_RETHINK_V3_FORMULA = {
    "p4_log_decay": "-exp(A_log) * softplus(raw_decay + dt_bias)",
    "k3_log_decay": "lower_bound * sigmoid(exp(A_log) * (raw_decay + dt_bias))",
    "i1_pivot": "raw_decay = 0",
    "i1_match": "log-decay value and first derivative with respect to raw_decay",
    "unrepresentable": "P4 log_decay < lower_bound cannot be represented exactly by K3",
}
P5_RETHINK_V3_FORMULA_HASH = canonical_json_hash(P5_RETHINK_V3_FORMULA)


def named_tensor_sha256(values: dict[str, torch.Tensor]) -> str:
    """Hash tensor names, dtypes, shapes, and bytes in deterministic lexical order."""
    digest = hashlib.sha256()
    for name in sorted(values):
        value = values[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build_analytic_i1_manifest(
    changed: dict[str, torch.Tensor],
    layers: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the canonical provenance shared by in-memory application and replay."""
    return {
        "initialization": "I1_RETENTION_MATCHED_ANALYTIC_LOCAL_VALUE_AND_SLOPE",
        "formula": P5_RETHINK_V3_FORMULA,
        "formula_hash": P5_RETHINK_V3_FORMULA_HASH,
        "calibration_used": False,
        "random_initialization_used": False,
        "changed_tensor_names": sorted(changed),
        "changed_tensor_hash": named_tensor_sha256(changed),
        "layers": layers,
        "aggregate": {
            "representable_heads": sum(item["representable_heads"] for item in layers),
            "lower_clipped_heads": sum(item["lower_clipped_heads"] for item in layers),
            "upper_clipped_heads": sum(item["upper_clipped_heads"] for item in layers),
            "total_heads": sum(item["heads"] for item in layers),
            "clipped_heads": [
                {"layer_index": item["layer_index"], **head}
                for item in layers
                for head in item["clipped_heads"]
            ],
        },
    }


def replay_analytic_i1_from_checkpoint(
    checkpoint_dir: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Recompute the 48 I1 tensors from frozen P4 using the formal mapping device."""
    from safetensors import safe_open

    checkpoint_dir = Path(checkpoint_dir).resolve()
    index = json.loads(
        (checkpoint_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_map: dict[str, str] = index["weight_map"]
    selected = [
        name
        for name in weight_map
        if name.endswith((".self_attn.A_log", ".self_attn.dt_bias"))
    ]
    if len(selected) != 48:
        raise ValueError(f"I1 replay expected 48 frozen P4 source tensors, found {len(selected)}")
    source = {}
    source_storage_dtypes = {}
    by_shard: dict[str, list[str]] = {}
    for name in selected:
        by_shard.setdefault(weight_map[name], []).append(name)
    for shard, names in sorted(by_shard.items()):
        with safe_open(checkpoint_dir / shard, framework="pt", device="cpu") as handle:
            for name in sorted(names):
                value = handle.get_tensor(name)
                source[name] = value
                source_storage_dtypes[name] = str(value.dtype).removeprefix("torch.")
    changed = {}
    layers = []
    layer_indices = sorted(
        int(name.split(".")[2]) for name in source if name.endswith(".self_attn.A_log")
    )
    for layer_index in layer_indices:
        prefix = f"model.layers.{layer_index}.self_attn"
        a_name = f"{prefix}.A_log"
        bias_name = f"{prefix}.dt_bias"
        # The formal indexed runtime always materializes A_log and dt_bias as
        # FP32, independently of their safetensors storage dtype. Replay must
        # reproduce that cast before the analytic FP64 calculation; otherwise
        # BF16 storage values follow a different cast/rounding path than the
        # checkpoint that was actually materialized by apply_analytic_i1().
        a_log = source[a_name].to(dtype=torch.float32).to(device)
        dt_bias = source[bias_name].to(dtype=torch.float32).to(device)
        if dt_bias.numel() % a_log.numel():
            raise ValueError(f"I1 replay source shapes are incompatible at layer {layer_index}")
        new_a_log, new_dt_bias, layer_report = analytic_retention_match(
            a_log,
            dt_bias,
            key_dim=dt_bias.numel() // a_log.numel(),
        )
        changed[a_name] = new_a_log
        changed[bias_name] = new_dt_bias
        layers.append({"layer_index": layer_index, **layer_report})
    if len(layers) != 24:
        raise ValueError(f"I1 replay expected 24 KDA layers, found {len(layers)}")
    manifest = build_analytic_i1_manifest(changed, layers)
    manifest["replay_source"] = {
        "storage_dtypes": source_storage_dtypes,
        "formal_runtime_dtype": "float32",
        "runtime_dtype_cast_applied": any(
            dtype != "float32" for dtype in source_storage_dtypes.values()
        ),
    }
    return changed, manifest


def analytic_retention_match(
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    key_dim: int,
    lower_bound: float = P5_RETHINK_V3_LOWER_BOUND,
    probability_epsilon: float = P5_RETHINK_V3_PROBABILITY_EPSILON,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Locally invert K3 at raw=0, matching feasible P4 retention value and slope."""
    if a_log.ndim != 1 or dt_bias.numel() != a_log.numel() * key_dim:
        raise ValueError("I1 expects A_log=[HV] and dt_bias=[HV*K]")
    if lower_bound >= 0:
        raise ValueError("I1 requires a negative K3 lower bound")
    if not 0.0 < probability_epsilon < 0.5:
        raise ValueError("I1 probability epsilon must be in (0, 0.5)")
    heads = a_log.numel()
    bias = dt_bias.detach().double().reshape(heads, key_dim)
    per_head_bias = bias[:, :1]
    repeat_spread = (bias - per_head_bias).abs().amax(dim=1)
    if bool((repeat_spread != 0).any()):
        raise ValueError(
            "I1 analytic value+slope mapping requires P4 dt_bias to be repeated exactly within each head"
        )

    scale = a_log.detach().double().exp().reshape(heads, 1)
    p4_magnitude = scale * F.softplus(per_head_bias)
    bound_magnitude = -float(lower_bound)
    raw_probability = p4_magnitude / bound_magnitude
    probability = raw_probability.clamp(probability_epsilon, 1.0 - probability_epsilon)
    p4_slope_magnitude = scale * torch.sigmoid(per_head_bias)
    new_scale = p4_slope_magnitude / (
        bound_magnitude * probability * (1.0 - probability)
    )
    new_a_log = new_scale.log().reshape_as(a_log)
    logit_probability = probability.log() - torch.log1p(-probability)
    new_per_head_bias = logit_probability / new_scale
    new_dt_bias = new_per_head_bias.expand(heads, key_dim).reshape_as(dt_bias)

    target_magnitude = bound_magnitude * torch.sigmoid(new_scale * new_per_head_bias)
    target_slope_magnitude = (
        bound_magnitude
        * probability
        * (1.0 - probability)
        * new_scale
    )
    representable = (raw_probability >= probability_epsilon) & (
        raw_probability <= 1.0 - probability_epsilon
    )
    value_error = (target_magnitude - p4_magnitude).abs()
    slope_error = (target_slope_magnitude - p4_slope_magnitude).abs()
    p4_retention = torch.exp(-p4_magnitude)
    i1_retention = torch.exp(-target_magnitude)
    clipped_heads = []
    for head_index in torch.nonzero(~representable, as_tuple=False).flatten().tolist():
        clipped_heads.append(
            {
                "head_index": int(head_index),
                "clip_side": (
                    "lower"
                    if bool(raw_probability[head_index] < probability_epsilon)
                    else "upper"
                ),
                "p4_target_log_decay": float(-p4_magnitude[head_index]),
                "i1_realized_log_decay": float(-target_magnitude[head_index]),
                "log_decay_absolute_error": float(value_error[head_index]),
                "p4_target_retention": float(p4_retention[head_index]),
                "i1_realized_retention": float(i1_retention[head_index]),
                "retention_absolute_error": float(
                    (i1_retention[head_index] - p4_retention[head_index]).abs()
                ),
            }
        )
    report = {
        "heads": heads,
        "key_dim": key_dim,
        "lower_bound": lower_bound,
        "probability_epsilon": probability_epsilon,
        "dt_bias_repeat_exact": True,
        "dt_bias_max_within_head_spread": float(repeat_spread.max()),
        "representable_heads": int(representable.sum()),
        "lower_clipped_heads": int((raw_probability < probability_epsilon).sum()),
        "upper_clipped_heads": int((raw_probability > 1.0 - probability_epsilon).sum()),
        "clipped_heads": clipped_heads,
        "p4_pivot_log_decay_min": float((-p4_magnitude).min()),
        "p4_pivot_log_decay_max": float((-p4_magnitude).max()),
        "i1_pivot_log_decay_min": float((-target_magnitude).min()),
        "i1_pivot_log_decay_max": float((-target_magnitude).max()),
        "representable_value_max_abs_error": (
            float(value_error[representable].max()) if bool(representable.any()) else None
        ),
        "slope_max_abs_error": float(slope_error.max()),
        "new_a_log_min": float(new_a_log.min()),
        "new_a_log_max": float(new_a_log.max()),
        "new_dt_bias_min": float(new_dt_bias.min()),
        "new_dt_bias_max": float(new_dt_bias.max()),
        "all_finite": bool(torch.isfinite(new_a_log).all() and torch.isfinite(new_dt_bias).all()),
    }
    if not report["all_finite"]:
        raise FloatingPointError(f"I1 analytic mapping produced non-finite parameters: {report}")
    return new_a_log.to(a_log.dtype), new_dt_bias.to(dt_bias.dtype), report


@torch.no_grad()
def apply_analytic_i1(model: DenseK3ForCausalLM) -> dict[str, Any]:
    """Apply deterministic I1 to all KDA A_log/dt_bias tensors only."""
    if model.config.kda_gate_mode != "qwen_compat":
        raise ValueError("I1 must start from the frozen P4 qwen_compat parameterization")
    before_objects = {name: id(parameter) for name, parameter in model.named_parameters()}
    before_versions = {name: parameter._version for name, parameter in model.named_parameters()}
    changed: dict[str, torch.Tensor] = {}
    layers = []
    for layer in model.model.layers:
        module = layer.self_attn
        if not isinstance(module, DenseK3KDA):
            continue
        new_a_log, new_dt_bias, layer_report = analytic_retention_match(
            module.A_log,
            module.dt_bias,
            key_dim=module.key_dim,
        )
        module.A_log.copy_(new_a_log)
        module.dt_bias.copy_(new_dt_bias)
        a_name = f"model.layers.{module.layer_idx}.self_attn.A_log"
        bias_name = f"model.layers.{module.layer_idx}.self_attn.dt_bias"
        changed[a_name] = module.A_log
        changed[bias_name] = module.dt_bias
        layers.append({"layer_index": module.layer_idx, **layer_report})
    if len(layers) != 24:
        raise ValueError(f"I1 expected 24 KDA layers, found {len(layers)}")
    after_objects = {name: id(parameter) for name, parameter in model.named_parameters()}
    after_versions = {name: parameter._version for name, parameter in model.named_parameters()}
    observed_changed = {
        name for name in before_versions if after_versions[name] != before_versions[name]
    }
    expected_changed = set(changed)
    checks = {
        "parameter_objects_unchanged": before_objects == after_objects,
        "exactly_A_log_and_dt_bias_mutated": observed_changed == expected_changed,
        "changed_tensor_count": len(changed) == 48,
        "all_layers_finite": all(item["all_finite"] for item in layers),
        "dt_bias_repeat_contract": all(item["dt_bias_repeat_exact"] for item in layers),
    }
    if not all(checks.values()):
        raise ValueError(f"I1 application contract failed: {checks}")
    return {**build_analytic_i1_manifest(changed, layers), "checks": checks}


__all__ = [
    "P5_RETHINK_V3_FORMULA",
    "P5_RETHINK_V3_FORMULA_HASH",
    "P5_RETHINK_V3_LOWER_BOUND",
    "P5_RETHINK_V3_PROBABILITY_EPSILON",
    "analytic_retention_match",
    "apply_analytic_i1",
    "build_analytic_i1_manifest",
    "named_tensor_sha256",
    "replay_analytic_i1_from_checkpoint",
]
