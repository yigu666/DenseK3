"""Exact SiTU-GLU reference used by the Titan P9 overlay."""

from __future__ import annotations

import torch

SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0


def situ_glu(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    beta: float = SITU_BETA,
    linear_beta: float = SITU_LINEAR_BETA,
) -> torch.Tensor:
    """Apply K3 SiTU-GLU in FP32 and restore the projection dtype.

    The sigmoid intentionally receives the uncapped gate.  This matches the
    pinned Kimi reference and is not equivalent to sigmoid(beta*tanh(g/beta)).
    """

    if gate.shape != up.shape:
        raise ValueError(f"SiTU gate/up shape mismatch: {gate.shape} != {up.shape}")
    if beta <= 0.0 or linear_beta <= 0.0:
        raise ValueError("SiTU beta values must be positive")
    output_dtype = gate.dtype
    gate32 = gate.float()
    up32 = up.float()
    capped_gate = beta * torch.tanh(gate32 / beta)
    capped_up = linear_beta * torch.tanh(up32 / linear_beta)
    return (capped_gate * torch.sigmoid(gate32) * capped_up).to(output_dtype)


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """DenseK3/Qwen SwiGLU runtime semantics."""

    if gate.shape != up.shape:
        raise ValueError(f"SwiGLU gate/up shape mismatch: {gate.shape} != {up.shape}")
    return torch.nn.functional.silu(gate) * up
