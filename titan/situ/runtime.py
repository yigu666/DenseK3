"""Instance-local SiTU-GLU runtime and deterministic down-projection folding."""

from __future__ import annotations

import hashlib
import types
from typing import Any

import torch
from torch import nn

from .activation import SITU_BETA, SITU_LINEAR_BETA, situ_glu


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def enable_situ_glu_runtime(
    model: nn.Module,
    *,
    beta: float = SITU_BETA,
    linear_beta: float = SITU_LINEAR_BETA,
) -> None:
    """Patch one loaded model instance; canonical DenseK3 sources stay untouched."""

    for layer_index, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        if hasattr(mlp, "_titan_p9_original_forward"):
            raise RuntimeError(f"Titan P9 SiTU runtime already enabled at layer {layer_index}")
        mlp._titan_p9_original_forward = mlp.forward  # type: ignore[attr-defined]

        def patched(this: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
            gate = this.gate_proj(hidden)
            up = this.up_proj(hidden)
            activation = situ_glu(gate, up, beta=beta, linear_beta=linear_beta)
            return this.down_proj(activation)

        mlp.forward = types.MethodType(patched, mlp)


def disable_situ_glu_runtime(model: nn.Module) -> None:
    for layer in model.model.layers:
        mlp = layer.mlp
        original = getattr(mlp, "_titan_p9_original_forward", None)
        if original is not None:
            mlp.forward = original
            delattr(mlp, "_titan_p9_original_forward")


@torch.no_grad()
def fold_down_projection_scales(
    model: nn.Module,
    scales: dict[int, torch.Tensor],
    selected_layers: set[int],
) -> dict[str, Any]:
    """Fold S1 channel scales into W_down columns with deterministic casting."""

    records = []
    for layer_index, layer in enumerate(model.model.layers):
        weight = layer.mlp.down_proj.weight
        before = tensor_sha256(weight)
        if layer_index in selected_layers:
            scale = scales[layer_index].to(device=weight.device, dtype=torch.float32)
            if scale.ndim != 1 or scale.numel() != weight.shape[1]:
                raise ValueError(f"Invalid P9 scale for layer {layer_index}: {scale.shape}")
            if not bool(torch.isfinite(scale).all()):
                raise FloatingPointError(f"Non-finite P9 scale at layer {layer_index}")
            folded = weight.float() * scale.unsqueeze(0)
            weight.copy_(folded.to(weight.dtype))
        after = tensor_sha256(weight)
        records.append(
            {
                "layer": layer_index,
                "selection": "S1_ACTIVATION_MATCHED" if layer_index in selected_layers else "S0_RAW",
                "before_sha256": before,
                "after_sha256": after,
                "changed": before != after,
            }
        )
    return {
        "records": records,
        "selected_s1_layers": sorted(selected_layers),
        "changed_layers": [item["layer"] for item in records if item["changed"]],
        "passed": all(item["changed"] == (item["layer"] in selected_layers) for item in records),
    }
