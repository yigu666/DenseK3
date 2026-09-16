"""Numerical comparison metrics for P2 result manifests."""

from __future__ import annotations

import torch

REFERENCE_FP32_RTOL = 1e-5
REFERENCE_FP32_ATOL = 1e-6
OPTIMIZED_FP32_OUTPUT_RTOL = 1e-4
OPTIMIZED_FP32_OUTPUT_ATOL = 1e-4
OPTIMIZED_FP32_STATE_RTOL = 2e-3
OPTIMIZED_FP32_STATE_ATOL = 2e-3
OPTIMIZED_BF16_RTOL = 6e-3
OPTIMIZED_BF16_ATOL = 6e-3
INPUT_GRADIENT_RTOL = 8e-3
INPUT_GRADIENT_ATOL = 8e-3
GATE_GRADIENT_RTOL = 1e-2
GATE_GRADIENT_ATOL = 1e-2


def optimized_output_tolerance(dtype: torch.dtype) -> tuple[float, float]:
    """Return the approved P2 tolerance for optimized FLA token outputs."""
    if dtype == torch.float32:
        return OPTIMIZED_FP32_OUTPUT_RTOL, OPTIMIZED_FP32_OUTPUT_ATOL
    if dtype == torch.bfloat16:
        return OPTIMIZED_BF16_RTOL, OPTIMIZED_BF16_ATOL
    raise ValueError(f"No approved optimized output tolerance for {dtype}")


def optimized_state_tolerance(dtype: torch.dtype) -> tuple[float, float]:
    """Return the approved P2 tolerance for optimized FLA final states."""
    if dtype == torch.float32:
        return OPTIMIZED_FP32_STATE_RTOL, OPTIMIZED_FP32_STATE_ATOL
    if dtype == torch.bfloat16:
        return OPTIMIZED_BF16_RTOL, OPTIMIZED_BF16_ATOL
    raise ValueError(f"No approved optimized state tolerance for {dtype}")


def error_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float | int]:
    """Compute the P2 elementwise and vector metrics in FP64."""
    reference_flat = reference.detach().double().reshape(-1)
    actual_flat = actual.detach().double().reshape(-1)
    difference = actual_flat - reference_flat
    reference_norm = torch.linalg.vector_norm(reference_flat)
    difference_norm = torch.linalg.vector_norm(difference)
    denominator = max(float(reference_norm), torch.finfo(torch.float64).eps)
    if reference_flat.numel() == 0:
        cosine = 1.0
    elif float(reference_norm) == 0.0 or float(torch.linalg.vector_norm(actual_flat)) == 0.0:
        cosine = float(reference_norm == torch.linalg.vector_norm(actual_flat))
    else:
        cosine = float(torch.nn.functional.cosine_similarity(reference_flat, actual_flat, dim=0))
    return {
        "max_abs_error": float(difference.abs().max()) if difference.numel() else 0.0,
        "mean_abs_error": float(difference.abs().mean()) if difference.numel() else 0.0,
        "relative_l2_error": float(difference_norm) / denominator,
        "cosine_similarity": cosine,
        "reference_nan_count": int(torch.isnan(reference_flat).sum()),
        "actual_nan_count": int(torch.isnan(actual_flat).sum()),
        "reference_inf_count": int(torch.isinf(reference_flat).sum()),
        "actual_inf_count": int(torch.isinf(actual_flat).sum()),
    }
