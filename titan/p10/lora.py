"""Mergeable, instance-local LoRA used only while optimizing Titan P10."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class MergeableLoRALinear(nn.Module):
    """A frozen linear plus FP32 rank adapter that can be merged in-place."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if base.bias is not None:
            raise ValueError("P10 only supports the bias-free DenseK3 linears")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.base.weight.requires_grad_(False)
        self.lora_A = nn.Parameter(
            torch.empty(
                self.rank,
                base.in_features,
                device=base.weight.device,
                dtype=torch.float32,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                base.out_features,
                self.rank,
                device=base.weight.device,
                dtype=torch.float32,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        base = self.base(value)
        low_rank = F.linear(F.linear(value.float(), self.lora_A), self.lora_B)
        return base + (low_rank * self.scaling).to(base.dtype)

    @torch.no_grad()
    def delta_norm(self, *, row_chunk: int = 1024) -> float:
        squared = torch.zeros((), device=self.lora_A.device, dtype=torch.float64)
        for start in range(0, self.out_features, row_chunk):
            delta = self.lora_B[start : start + row_chunk] @ self.lora_A
            squared += delta.double().square().sum() * (self.scaling**2)
        return float(squared.sqrt().cpu())

    @torch.no_grad()
    def merge(self, *, row_chunk: int = 1024) -> nn.Linear:
        weight = self.base.weight
        for start in range(0, self.out_features, row_chunk):
            end = min(start + row_chunk, self.out_features)
            delta = self.lora_B[start:end] @ self.lora_A
            merged = weight[start:end].float() + delta * self.scaling
            weight[start:end].copy_(merged.to(weight.dtype))
        return self.base


@dataclass(frozen=True)
class LoRAHandle:
    name: str
    parent: nn.Module
    attribute: str
    module: MergeableLoRALinear
    family: str


def merge_all_lora(handles: Iterable[LoRAHandle]) -> list[dict[str, object]]:
    report = []
    for handle in handles:
        delta_norm = handle.module.delta_norm()
        base = handle.module.merge()
        setattr(handle.parent, handle.attribute, base)
        report.append(
            {
                "name": handle.name,
                "family": handle.family,
                "rank": handle.module.rank,
                "alpha": handle.module.alpha,
                "delta_norm": delta_norm,
            }
        )
    return report
