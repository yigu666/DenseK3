"""P5 decay/output-gate 2x2 attribution contracts and pure analysis helpers."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch

from densek3_core.modeling import DenseK3ForCausalLM
from densek3_core.modeling.modeling_densek3 import DenseK3KDA
from densek3_core.recovery.p5_transition import P5_GATE_LOWER_BOUND, canonical_json_hash


@dataclass(frozen=True)
class P5FactorialCase:
    """One cell in the P4/K3 decay by SiLU/Sigmoid output-gate design."""

    case_id: str
    decay: str
    output_gate: str
    formal_checkpoint_configuration: bool

    @property
    def gate_mode(self) -> str:
        return "qwen_compat" if self.decay == "p4" else "kimi_k3"

    @property
    def lower_bound(self) -> float | None:
        return None if self.decay == "p4" else P5_GATE_LOWER_BOUND

    @property
    def output_gate_activation(self) -> str:
        return self.output_gate.lower()

    def manifest(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "gate_mode": self.gate_mode,
            "lower_bound": self.lower_bound,
            "output_gate_activation": self.output_gate_activation,
            "diagnostic_only_mixed_configuration": not self.formal_checkpoint_configuration,
        }


P5_FACTORIAL_CASES = {
    "A": P5FactorialCase("A", "p4", "silu", True),
    "B": P5FactorialCase("B", "k3", "silu", False),
    "C": P5FactorialCase("C", "p4", "sigmoid", False),
    "D": P5FactorialCase("D", "k3", "sigmoid", True),
}
P5_FACTORIAL_DEFINITION_HASH = canonical_json_hash({case_id: case.manifest() for case_id, case in P5_FACTORIAL_CASES.items()})


def _parameter_identity(model: torch.nn.Module) -> dict[str, tuple[Any, ...]]:
    return {
        name: (
            id(parameter),
            parameter.data_ptr(),
            parameter._version,
            tuple(parameter.shape),
            str(parameter.dtype),
        )
        for name, parameter in model.named_parameters()
    }


def apply_factorial_case(model: DenseK3ForCausalLM, case_id: str) -> dict[str, Any]:
    """Select a diagnostic cell without changing a tensor or weakening config validation."""
    if case_id not in P5_FACTORIAL_CASES:
        raise ValueError(f"Unknown P5 factorial case: {case_id}")
    case = P5_FACTORIAL_CASES[case_id]
    before = _parameter_identity(model)
    model.config.kda_gate_mode = case.gate_mode
    model.config.kda_gate_lower_bound = case.lower_bound
    model.config.kda_output_gate_activation = case.output_gate_activation
    model.config.kda_output_gate_alpha = None
    kda_layers = 0
    for layer in model.model.layers:
        if isinstance(layer.self_attn, DenseK3KDA):
            layer.self_attn.o_norm.activation = case.output_gate_activation
            layer.self_attn.o_norm.alpha = None
            kda_layers += 1
    after = _parameter_identity(model)
    formal_validation = None
    if case.formal_checkpoint_configuration:
        model.config.validate_architecture()
        formal_validation = True
    checks = {
        "parameter_identity_unchanged": before == after,
        "all_kda_layers_selected": kda_layers == len(model.config.kda_indices),
        "gate_mode": model.config.kda_gate_mode == case.gate_mode,
        "lower_bound": model.config.kda_gate_lower_bound == case.lower_bound,
        "output_gate": model.config.kda_output_gate_activation == case.output_gate_activation,
        "module_output_gates": all(
            layer.self_attn.o_norm.activation == case.output_gate_activation
            and layer.self_attn.o_norm.alpha is None
            for layer in model.model.layers
            if isinstance(layer.self_attn, DenseK3KDA)
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"P5 factorial case application failed: {checks}")
    return {
        "case": case.manifest(),
        "case_definition_hash": P5_FACTORIAL_DEFINITION_HASH,
        "formal_config_validation": formal_validation,
        "formal_validation_deliberately_skipped": not case.formal_checkpoint_configuration,
        "parameter_tensor_count": len(after),
        "kda_layer_count": kda_layers,
        "checks": checks,
    }


def scalar_factorial_effect(values: dict[str, float]) -> dict[str, float | None]:
    """Calculate main and interaction effects for A/B/C/D scalar measurements."""
    if set(values) != set(P5_FACTORIAL_CASES):
        raise ValueError("Factorial scalar values must contain exactly A, B, C, and D")
    if not all(math.isfinite(float(value)) for value in values.values()):
        raise ValueError("Factorial scalar values must be finite")
    a, b, c, d = (float(values[case_id]) for case_id in "ABCD")
    decay_effect = b - a
    gate_effect = c - a
    joint_effect = d - a
    interaction = joint_effect - decay_effect - gate_effect
    return {
        "A": a,
        "B": b,
        "C": c,
        "D": d,
        "decay_effect_B_minus_A": decay_effect,
        "gate_effect_C_minus_A": gate_effect,
        "joint_effect_D_minus_A": joint_effect,
        "additive_prediction": a + decay_effect + gate_effect,
        "interaction_D_minus_B_minus_C_plus_A": interaction,
        "interaction_fraction_of_joint": (
            interaction / joint_effect if abs(joint_effect) > torch.finfo(torch.float64).eps else None
        ),
    }


def first_generation_divergence(reference: list[int], actual: list[int]) -> dict[str, Any]:
    """Return the first differing generated-token offset and aligned agreement."""
    aligned = min(len(reference), len(actual))
    matches = [reference[index] == actual[index] for index in range(aligned)]
    first = next((index for index, match in enumerate(matches) if not match), None)
    if first is None and len(reference) != len(actual):
        first = aligned
    return {
        "reference_length": len(reference),
        "actual_length": len(actual),
        "exact_match": reference == actual,
        "first_divergence_offset": first,
        "common_prefix_tokens": aligned if first is None else first,
        "aligned_token_agreement": sum(matches) / aligned if aligned else 1.0,
    }


__all__ = [
    "P5_FACTORIAL_CASES",
    "P5_FACTORIAL_DEFINITION_HASH",
    "P5FactorialCase",
    "apply_factorial_case",
    "first_generation_divergence",
    "scalar_factorial_effect",
]
