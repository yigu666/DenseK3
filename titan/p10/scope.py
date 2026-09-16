"""Semantic trainable-scope construction for Titan P10."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from densek3_core.modeling.modeling_densek3 import DenseK3KDA, DenseK3MLA

from .lora import LoRAHandle, MergeableLoRALinear


@dataclass
class P10TrainableScope:
    lora: list[LoRAHandle]
    direct_model: dict[str, nn.Parameter]
    direct_attnres: dict[str, nn.Parameter]
    manifest: dict[str, object]

    def trainable_parameters(self) -> list[nn.Parameter]:
        values = [
            parameter
            for handle in self.lora
            for parameter in (handle.module.lora_A, handle.module.lora_B)
        ]
        values.extend(self.direct_model.values())
        values.extend(self.direct_attnres.values())
        return values


def _parameter_name_by_identity(model: nn.Module) -> dict[int, str]:
    return {id(value): name for name, value in model.named_parameters()}


def build_p10_trainable_scope(
    model: nn.Module,
    attnres: nn.Module,
    *,
    rank: int,
    alpha: float,
    reduced_titan_scope: bool = False,
) -> P10TrainableScope:
    """Freeze the source, attach the exact approved LoRA set, then unfreeze direct params."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in attnres.parameters():
        parameter.requires_grad_(True)

    names_before = _parameter_name_by_identity(model)
    direct_values: dict[str, nn.Parameter] = {}

    def direct(parameter: nn.Parameter) -> None:
        name = names_before.get(id(parameter))
        if name is None:
            raise KeyError("Direct P10 parameter is not owned by the source model")
        if name in direct_values:
            return
        parameter.requires_grad_(True)
        direct_values[name] = parameter

    # Global/pre-norm RMS scales remain native low-dimensional parameters.
    for layer in model.model.layers:
        direct(layer.input_layernorm.weight)
        direct(layer.post_attention_layernorm.weight)
        mixer = layer.self_attn
        if isinstance(mixer, DenseK3KDA):
            kda_direct = (mixer.o_norm.weight,) if reduced_titan_scope else (
                mixer.A_log,
                mixer.dt_bias,
                mixer.f_a_proj.weight,
                mixer.f_b_proj.weight,
                mixer.b_proj.weight,
                mixer.o_norm.weight,
            )
            for parameter in kda_direct:
                direct(parameter)
        elif isinstance(mixer, DenseK3MLA):
            direct(mixer.kv_a_layernorm.weight)
            direct(mixer.q_norm.weight)
            direct(mixer.k_norm.weight)
        else:
            raise TypeError(f"P10 source contains unsupported mixer: {type(mixer).__name__}")
    direct(model.model.norm.weight)

    handles: list[LoRAHandle] = []

    def attach(parent: nn.Module, attribute: str, name: str, family: str) -> None:
        base = getattr(parent, attribute)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"P10 LoRA target {name} is not nn.Linear: {type(base).__name__}")
        module = MergeableLoRALinear(base, rank=rank, alpha=alpha)
        setattr(parent, attribute, module)
        handles.append(LoRAHandle(name, parent, attribute, module, family))

    for layer_index, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        for attribute in ("gate_proj", "up_proj", "down_proj"):
            attach(mlp, attribute, f"model.layers.{layer_index}.mlp.{attribute}", "MLP")
        mixer = layer.self_attn
        if isinstance(mixer, DenseK3KDA):
            kda_targets = ("o_proj",) if reduced_titan_scope else (
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            )
            for attribute in kda_targets:
                attach(
                    mixer,
                    attribute,
                    f"model.layers.{layer_index}.self_attn.{attribute}",
                    "KDA",
                )
        else:
            for attribute in ("q_proj", "g_proj", "kv_a_proj", "kv_b_proj", "o_proj"):
                attach(
                    mixer,
                    attribute,
                    f"model.layers.{layer_index}.self_attn.{attribute}",
                    "MLA",
                )

    attnres_values = dict(attnres.named_parameters())
    trainable_names = [name for name, value in model.named_parameters() if value.requires_grad]
    lora_tensor_count = len(handles) * 2
    direct_tensor_count = len(direct_values)
    expected = {
        "lora_modules": 160 if reduced_titan_scope else 232,
        "lora_tensors": 320 if reduced_titan_scope else 464,
        "direct_model_tensors": 113 if reduced_titan_scope else 233,
        "attnres_tensors": 130,
        "kda_layers": 24,
        "mla_layers": 8,
    }
    observed = {
        "lora_modules": len(handles),
        "lora_tensors": lora_tensor_count,
        "direct_model_tensors": direct_tensor_count,
        "attnres_tensors": len(attnres_values),
        "kda_layers": sum(isinstance(layer.self_attn, DenseK3KDA) for layer in model.model.layers),
        "mla_layers": sum(isinstance(layer.self_attn, DenseK3MLA) for layer in model.model.layers),
    }
    excluded_lora_tensors = []
    excluded_direct_parameters = []
    frozen_audit_parameters = []
    if reduced_titan_scope:
        for layer_index, layer in enumerate(model.model.layers):
            mixer = layer.self_attn
            if not isinstance(mixer, DenseK3KDA):
                continue
            for attribute in ("q_proj", "k_proj", "v_proj"):
                base_name = f"model.layers.{layer_index}.self_attn.{attribute}"
                excluded_lora_tensors.extend((f"lora.{base_name}.A", f"lora.{base_name}.B"))
                frozen_audit_parameters.append(f"{base_name}.weight")
            for parameter in (
                mixer.A_log,
                mixer.dt_bias,
                mixer.f_a_proj.weight,
                mixer.f_b_proj.weight,
                mixer.b_proj.weight,
            ):
                name = names_before[id(parameter)]
                excluded_direct_parameters.append(f"direct_master.model.{name}")
                frozen_audit_parameters.append(name)
    excluded_optimizer_tensors = [*excluded_lora_tensors, *excluded_direct_parameters]
    checks = {
        "expected_counts": observed == expected,
        "embedding_frozen": not model.model.embed_tokens.weight.requires_grad,
        "lm_head_frozen": not model.lm_head.weight.requires_grad,
        "lora_fp32": all(
            parameter.dtype == torch.float32
            for handle in handles
            for parameter in (handle.module.lora_A, handle.module.lora_B)
        ),
        "lora_zero_function": all(
            bool(torch.count_nonzero(handle.module.lora_B) == 0) for handle in handles
        ),
        "all_attnres_trainable": all(value.requires_grad for value in attnres_values.values()),
        "only_approved_model_params": len(trainable_names) == lora_tensor_count + direct_tensor_count,
        "reduced_excluded_count": (
            len(excluded_optimizer_tensors) == 264 if reduced_titan_scope else not excluded_optimizer_tensors
        ),
        "reduced_excluded_lora_absent": all(
            item.name.split(".self_attn.")[-1] not in {"q_proj", "k_proj", "v_proj"}
            for item in handles
            if item.family == "KDA"
        ) if reduced_titan_scope else True,
        "reduced_recurrent_direct_frozen": all(
            not dict(model.named_parameters())[name].requires_grad
            for name in frozen_audit_parameters
        ) if reduced_titan_scope else True,
    }
    manifest = {
        "expected": expected,
        "observed": observed,
        "checks": checks,
        "passed": all(checks.values()),
        "scope_policy": (
            "P10_TITAN_REDUCED_SCOPE_2026_08_21"
            if reduced_titan_scope
            else "P10_CANONICAL_FULL_SCOPE"
        ),
        "lora_targets": [
            {"name": item.name, "family": item.family, "rank": rank, "alpha": alpha}
            for item in handles
        ],
        "direct_model_parameters": sorted(direct_values),
        "direct_attnres_parameters": sorted(attnres_values),
        "excluded_optimizer_tensor_count": len(excluded_optimizer_tensors),
        "excluded_optimizer_tensors": sorted(excluded_optimizer_tensors),
        "frozen_audit_parameter_names": sorted(frozen_audit_parameters),
        "trainable_parameter_count": sum(value.numel() for value in [
            *[parameter for handle in handles for parameter in (handle.module.lora_A, handle.module.lora_B)],
            *direct_values.values(),
            *attnres_values.values(),
        ]),
    }
    if not manifest["passed"]:
        raise ValueError(f"P10 trainable scope mismatch: {manifest}")
    return P10TrainableScope(handles, direct_values, attnres_values, manifest)


def build_p10_titan_reduced_scope(
    model: nn.Module,
    attnres: nn.Module,
    *,
    rank: int,
    alpha: float,
) -> P10TrainableScope:
    """Build the authorized Titan-only 563-tensor reduced P10-T scope."""
    return build_p10_trainable_scope(
        model,
        attnres,
        rank=rank,
        alpha=alpha,
        reduced_titan_scope=True,
    )
