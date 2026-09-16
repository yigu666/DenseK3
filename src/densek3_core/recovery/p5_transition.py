"""Deterministic, weight-preserving P4 to P5 KDA parameterization transition."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import AbstractContextManager
from typing import Any

import torch

from densek3_core.kda.metrics import error_metrics
from densek3_core.kda.reference import kimi_k3_decay, qwen_compat_decay
from densek3_core.modeling import DenseK3Config, DenseK3ForCausalLM, DenseK3HybridCache
from densek3_core.modeling.modeling_densek3 import DenseK3KDA

P5_GATE_LOWER_BOUND = -5.0
P5_CONFIG_CHANGES = {
    "kda_gate_mode": {"from": "qwen_compat", "to": "kimi_k3"},
    "kda_gate_lower_bound": {"from": None, "to": P5_GATE_LOWER_BOUND},
    "kda_output_gate_activation": {"from": "silu", "to": "sigmoid"},
}


def canonical_json_hash(value: Any) -> str:
    """Hash a JSON-compatible value with stable key and separator rules."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_p5_target_config(p4_config: DenseK3Config) -> DenseK3Config:
    """Clone a frozen P4 config and apply only the three P1-approved P5 changes."""
    p4_config.validate_p4()
    payload = p4_config.to_dict()
    payload.update(
        {
            "kda_gate_mode": "kimi_k3",
            "kda_gate_lower_bound": P5_GATE_LOWER_BOUND,
            "kda_output_gate_activation": "sigmoid",
        }
    )
    target = DenseK3Config.from_dict(payload)
    if target.kda_parameterization != "p5_k3_style":
        raise ValueError("P5 target config did not select the K3-style parameterization")
    return target


def _parameter_snapshot(model: torch.nn.Module) -> dict[str, tuple[Any, ...]]:
    return {
        name: (
            id(parameter),
            parameter.data_ptr(),
            parameter._version,
            tuple(parameter.shape),
            str(parameter.dtype),
            parameter.requires_grad,
        )
        for name, parameter in model.named_parameters()
    }


def parameter_schema(model: torch.nn.Module) -> list[dict[str, Any]]:
    """Return the deterministic model parameter schema without copying tensor bytes."""
    return [
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "numel": parameter.numel(),
            "requires_grad": parameter.requires_grad,
        }
        for name, parameter in model.named_parameters()
    ]


def apply_p5_parameterization(model: DenseK3ForCausalLM) -> dict[str, Any]:
    """Switch an in-memory P4 model to P5 without replacing or mutating a tensor."""
    model.config.validate_p4()
    before = _parameter_snapshot(model)
    target_config = build_p5_target_config(model.config)
    model.config.kda_gate_mode = target_config.kda_gate_mode
    model.config.kda_gate_lower_bound = target_config.kda_gate_lower_bound
    model.config.kda_output_gate_activation = target_config.kda_output_gate_activation
    model.config.kda_output_gate_alpha = None
    model.config.validate_architecture()

    kda_layers = 0
    for layer in model.model.layers:
        if isinstance(layer.self_attn, DenseK3KDA):
            if layer.self_attn.config is not model.config:
                raise ValueError("KDA layer does not share the model configuration object")
            layer.self_attn.o_norm.activation = "sigmoid"
            layer.self_attn.o_norm.alpha = None
            kda_layers += 1
    if kda_layers != 24:
        raise ValueError(f"P5 transition expected 24 KDA layers, found {kda_layers}")

    after = _parameter_snapshot(model)
    checks = {
        "parameter_names_exact": before.keys() == after.keys(),
        "parameter_objects_exact": before == after,
        "parameter_count_unchanged": len(before) == len(after),
        "topology_24_kda_8_gqa": len(model.config.kda_indices) == 24 and len(model.config.gqa_indices) == 8,
        "target_gate_mode": model.config.kda_gate_mode == "kimi_k3",
        "target_lower_bound": model.config.kda_gate_lower_bound == P5_GATE_LOWER_BOUND,
        "target_output_gate": model.config.kda_output_gate_activation == "sigmoid",
        "target_output_gate_alpha": model.config.kda_output_gate_alpha is None,
        "all_kda_output_gates_switched": all(
            layer.self_attn.o_norm.activation == "sigmoid"
            for layer in model.model.layers
            if isinstance(layer.self_attn, DenseK3KDA)
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"P5 parameterization transition invariant failed: {checks}")
    schema = parameter_schema(model)
    return {
        "status": "PASS",
        "transition_type": "configuration_only_no_tensor_conversion",
        "config_changes": P5_CONFIG_CHANGES,
        "checks": checks,
        "kda_layers_switched": kda_layers,
        "parameter_tensor_count": len(schema),
        "parameter_numel": sum(item["numel"] for item in schema),
        "parameter_schema_hash": canonical_json_hash(schema),
        "optimizer_state_reused": False,
    }


def tensor_statistics(value: torch.Tensor) -> dict[str, Any]:
    """Compute exact CPU quantiles and distribution summaries for one tensor."""
    flat = value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    finite = torch.isfinite(flat)
    finite_values = flat[finite]
    quantile_points = torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], dtype=torch.float64)
    quantiles = (
        torch.quantile(finite_values, quantile_points)
        if finite_values.numel()
        else torch.full_like(quantile_points, torch.nan)
    )
    quantile_values = [float(value) for value in quantiles] if finite_values.numel() else [None] * 5
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "num_elements": flat.numel(),
        "nan_count": int(torch.isnan(flat).sum()),
        "inf_count": int(torch.isinf(flat).sum()),
        "finite_fraction": float(finite.double().mean()) if flat.numel() else 1.0,
        "min": float(finite_values.min()) if finite_values.numel() else None,
        "max": float(finite_values.max()) if finite_values.numel() else None,
        "mean": float(finite_values.mean()) if finite_values.numel() else None,
        "std": float(finite_values.std(unbiased=False)) if finite_values.numel() else None,
        "rms": float(finite_values.square().mean().sqrt()) if finite_values.numel() else None,
        "p01": quantile_values[0],
        "p05": quantile_values[1],
        "p50": quantile_values[2],
        "p95": quantile_values[3],
        "p99": quantile_values[4],
    }


class KDAParameterizationObserver(AbstractContextManager["KDAParameterizationObserver"]):
    """Capture decay, beta, and output-gate distributions at every KDA input."""

    def __init__(self, model: DenseK3ForCausalLM):
        self.model = model
        self.layers: list[dict[str, Any]] = []
        self._handles: list[Any] = []

    def _hook(self, module: DenseK3KDA, args: tuple[Any, ...]) -> None:
        hidden_states: torch.Tensor = args[0]
        attention_mask: torch.Tensor | None = args[1]
        sequence = hidden_states.shape[1]
        if attention_mask is not None:
            current_mask = attention_mask[:, -sequence:].to(hidden_states.dtype)
            hidden_states = hidden_states * current_mask.unsqueeze(-1)
        with torch.no_grad():
            raw_decay = module.f_b_proj(module.f_a_proj(hidden_states)).view(
                hidden_states.shape[0],
                sequence,
                module.value_heads,
                module.key_dim,
            )
            beta = torch.sigmoid(module.b_proj(hidden_states))
            gate_logits = module.g_proj(hidden_states).view(
                hidden_states.shape[0],
                sequence,
                module.value_heads,
                module.value_dim,
            )
            if module.config.kda_gate_mode == "qwen_compat":
                decay = qwen_compat_decay(raw_decay, module.A_log, module.dt_bias)
            elif module.config.kda_gate_mode == "kimi_k3":
                decay = kimi_k3_decay(
                    raw_decay,
                    module.A_log,
                    module.dt_bias,
                    lower_bound=module.config.kda_gate_lower_bound,
                )
            else:
                raise ValueError(f"Unknown KDA gate mode: {module.config.kda_gate_mode}")
            gate_float = gate_logits.float()
            if module.o_norm.alpha is not None:
                gate = (1.0 - module.o_norm.alpha) * torch.nn.functional.silu(
                    gate_float
                ) + module.o_norm.alpha * torch.sigmoid(gate_float)
            else:
                gate = (
                    torch.nn.functional.silu(gate_float)
                    if module.o_norm.activation == "silu"
                    else torch.sigmoid(gate_float)
                )
        decay_stats = tensor_statistics(decay)
        decay_float = decay.detach().float()
        decay_stats.update(
            {
                "fraction_within_0_01_of_zero": float((decay_float >= -0.01).float().mean()),
                "fraction_within_0_01_of_lower_bound": (
                    None
                    if module.config.kda_gate_lower_bound is None
                    else float((decay_float <= module.config.kda_gate_lower_bound + 0.01).float().mean())
                ),
            }
        )
        beta_stats = tensor_statistics(beta)
        beta_float = beta.detach().float()
        beta_stats.update(
            {
                "fraction_le_0_01": float((beta_float <= 0.01).float().mean()),
                "fraction_ge_0_99": float((beta_float >= 0.99).float().mean()),
            }
        )
        gate_stats = tensor_statistics(gate)
        gate_flat = gate.detach().float()
        near_zero = gate_flat.abs() <= 0.01
        sigmoid_saturated = (gate_flat <= 0.01) | (gate_flat >= 0.99)
        gate_stats.update(
            {
                "fraction_le_0_01": float((gate_flat <= 0.01).float().mean()),
                "fraction_ge_0_99": float((gate_flat >= 0.99).float().mean()),
                "negative_fraction": float((gate_flat < 0).float().mean()),
                "near_zero_fraction": float(near_zero.float().mean()),
                "sigmoid_saturation_fraction": (
                    float(sigmoid_saturated.float().mean()) if module.o_norm.activation == "sigmoid" else None
                ),
            }
        )
        self.layers.append(
            {
                "layer_index": module.layer_idx,
                "mode": module.config.kda_gate_mode,
                "lower_bound": module.config.kda_gate_lower_bound,
                "output_gate_activation": module.o_norm.activation,
                "output_gate_alpha": module.o_norm.alpha,
                "raw_decay_logits": tensor_statistics(raw_decay),
                "log_decay": decay_stats,
                "retention": tensor_statistics(decay.exp()),
                "beta": beta_stats,
                "output_gate_logits": tensor_statistics(gate_logits),
                "output_gate": gate_stats,
            }
        )

    def __enter__(self) -> KDAParameterizationObserver:
        if self._handles:
            raise RuntimeError("KDA parameterization observer is already active")
        for layer in self.model.model.layers:
            if isinstance(layer.self_attn, DenseK3KDA):
                self._handles.append(layer.self_attn.register_forward_pre_hook(self._hook))
        return self

    def __exit__(self, *exc: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return None


def cache_state_statistics(cache: DenseK3HybridCache, config: DenseK3Config) -> list[dict[str, Any]]:
    """Summarize the fixed-size KDA recurrent state at all 24 layers."""
    layers = []
    for layer_index in config.kda_indices:
        state = cache.kda_layer(layer_index).recurrent_state
        if state is None:
            raise ValueError(f"KDA cache layer {layer_index} has no recurrent state")
        layers.append({"layer_index": layer_index, "recurrent_state": tensor_statistics(state)})
    return layers


def hidden_drift(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    """Measure architecture-shock drift without applying an acceptance tolerance."""
    metrics = error_metrics(reference, actual)
    reference_rms = float(reference.detach().float().square().mean().sqrt())
    actual_rms = float(actual.detach().float().square().mean().sqrt())
    metrics.update(
        {
            "shape": list(reference.shape),
            "reference_rms": reference_rms,
            "actual_rms": actual_rms,
            "rms_ratio": actual_rms / max(reference_rms, torch.finfo(torch.float32).eps),
        }
    )
    return metrics


def token_decision_drift(reference_logits: torch.Tensor, actual_logits: torch.Tensor) -> dict[str, Any]:
    """Compare actual greedy decisions and independently computed top-10 sets."""
    if reference_logits.shape != actual_logits.shape:
        raise ValueError("Token-decision logits must have identical shapes")
    reference_argmax = reference_logits.argmax(dim=-1)
    actual_argmax = actual_logits.argmax(dim=-1)
    reference_top10 = reference_logits.topk(10, dim=-1).indices
    actual_top10 = actual_logits.topk(10, dim=-1).indices
    overlaps = []
    for expected, observed in zip(
        reference_top10.reshape(-1, 10).cpu().tolist(),
        actual_top10.reshape(-1, 10).cpu().tolist(),
        strict=True,
    ):
        overlaps.append(len(set(expected) & set(observed)))
    matches = reference_argmax == actual_argmax
    return {
        "positions": matches.numel(),
        "argmax_agreement_fraction": float(matches.float().mean()) if matches.numel() else 1.0,
        "argmax_disagreements": int((~matches).sum()),
        "last_reference_argmax": reference_argmax[:, -1].cpu().tolist(),
        "last_actual_argmax": actual_argmax[:, -1].cpu().tolist(),
        "last_argmax_match": bool(torch.equal(reference_argmax[:, -1], actual_argmax[:, -1])),
        "top10_min_overlap": min(overlaps, default=10),
        "top10_mean_overlap": sum(overlaps) / len(overlaps) if overlaps else 1.0,
    }


def iter_finite_reports(value: Any) -> Iterator[bool]:
    """Yield finiteness checks from nested tensor-statistic reports."""
    if isinstance(value, dict):
        if "nan_count" in value and "inf_count" in value:
            yield value["nan_count"] == 0 and value["inf_count"] == 0
        for child in value.values():
            yield from iter_finite_reports(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_finite_reports(child)


__all__ = [
    "KDAParameterizationObserver",
    "P5_CONFIG_CHANGES",
    "P5_GATE_LOWER_BOUND",
    "apply_p5_parameterization",
    "build_p5_target_config",
    "cache_state_statistics",
    "canonical_json_hash",
    "hidden_drift",
    "iter_finite_reports",
    "parameter_schema",
    "tensor_statistics",
    "token_decision_drift",
]
