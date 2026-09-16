"""Small numerical helpers for Titan P8 diagnostics."""

from __future__ import annotations

import math
from typing import Any

import torch


def tensor_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    if actual.shape != reference.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} vs {reference.shape}")
    a = actual.detach().double()
    r = reference.detach().double()
    difference = a - r
    return {
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(r).all()),
        "exact": bool(torch.equal(actual, reference)),
        "max_abs": float(difference.abs().max()),
        "mean_abs": float(difference.abs().mean()),
        "relative_l2": float(difference.norm() / r.norm().clamp_min(1e-30)),
        "cosine": float(
            torch.nn.functional.cosine_similarity(a.reshape(1, -1), r.reshape(1, -1))
        ),
        "rms_ratio": float(
            a.square().mean().sqrt() / r.square().mean().sqrt().clamp_min(1e-30)
        ),
    }


def routing_summary(alpha: torch.Tensor) -> dict[str, float | int]:
    if alpha.ndim != 3:
        raise ValueError("Depth-routing alpha must have [S,B,T] shape")
    alpha64 = alpha.detach().double()
    entropy = -(alpha64 * alpha64.clamp_min(1e-30).log()).sum(dim=0)
    source_count = int(alpha.shape[0])
    uniform_entropy = math.log(source_count)
    return {
        "source_count": source_count,
        "entropy_mean": float(entropy.mean()),
        "entropy_min": float(entropy.min()),
        "uniform_entropy": uniform_entropy,
        "entropy_gap_mean": uniform_entropy - float(entropy.mean()),
        "alpha_max_mean": float(alpha64.max(dim=0).values.mean()),
        "alpha_min_mean": float(alpha64.min(dim=0).values.mean()),
        "alpha_max": float(alpha64.max()),
        "alpha_min": float(alpha64.min()),
        "embedding_weight_mean": float(alpha64[0].mean()),
        "recent_weight_mean": float(alpha64[-1].mean()),
        "content_std": float(alpha64.std(dim=(1, 2), unbiased=False).max()),
    }
