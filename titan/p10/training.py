"""Numerically stable FP16 optimization helpers for Titan P10."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


def audit_gradient_coverage(
    parameters: list[nn.Parameter],
    parameter_names: dict[int, str],
) -> dict[str, object]:
    """Require every optimizer tensor in the frozen P10 scope to receive a gradient."""
    missing = [
        parameter_names[id(parameter)]
        for parameter in parameters
        if parameter.grad is None
    ]
    return {
        "complete": not missing,
        "optimizer_tensors": len(parameters),
        "gradient_tensors": len(parameters) - len(missing),
        "missing_gradient_count": len(missing),
        "missing_gradient_names": missing,
    }


@dataclass
class MasterBinding:
    name: str
    source: nn.Parameter
    master: nn.Parameter


class FP32MasterParameters:
    """Keep AdamW state and updates in FP32 while the model remains FP16."""

    def __init__(self, parameters: dict[str, nn.Parameter]):
        self.bindings = [
            MasterBinding(
                name,
                parameter,
                nn.Parameter(parameter.detach().float().clone(), requires_grad=True),
            )
            for name, parameter in sorted(parameters.items())
        ]

    def optimizer_parameters(self) -> list[nn.Parameter]:
        return [item.master for item in self.bindings]

    def copy_scaled_grads_from_model(self) -> None:
        for item in self.bindings:
            if item.source.grad is None:
                item.master.grad = None
            else:
                item.master.grad = item.source.grad.detach().float().clone()

    @torch.no_grad()
    def copy_to_model(self) -> None:
        for item in self.bindings:
            item.source.copy_(item.master.to(item.source.dtype))

    @torch.no_grad()
    def refresh_from_model(self) -> None:
        for item in self.bindings:
            item.master.copy_(item.source.float())


def cosine_learning_rate(
    completed_steps: int,
    total_steps: int,
    base_learning_rate: float,
    *,
    warmup_ratio: float,
    minimum_ratio: float,
) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    warmup = max(1, int(total_steps * warmup_ratio))
    if completed_steps < warmup:
        return base_learning_rate * (completed_steps + 1) / warmup
    progress = min(1.0, (completed_steps - warmup) / max(1, total_steps - warmup))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_learning_rate * (minimum_ratio + (1.0 - minimum_ratio) * cosine)
