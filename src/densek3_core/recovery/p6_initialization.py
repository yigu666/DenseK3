"""Deterministic P6.1c activation-aware rank-512 initialization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from densek3_core.modeling.modeling_densek3 import DenseK3RMSNorm
from densek3_core.recovery.p6_mla_probe import (
    P6KVFactorization,
    build_full_kv_weight,
    tensor_sha256,
)


def covariance_from_batches(
    batches: list[torch.Tensor],
    *,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Accumulate `X^T X / N` deterministically in IEEE FP32."""
    if not batches:
        raise ValueError("P6.1c activation batches cannot be empty")
    hidden_size = batches[0].shape[-1]
    compute_device = batches[0].device if device is None else device
    covariance = torch.zeros((hidden_size, hidden_size), dtype=torch.float32, device=compute_device)
    tokens = 0
    batch_hashes = []
    original_precision = torch.get_float32_matmul_precision()
    cuda_backend = getattr(torch.backends, "cuda", None)
    matmul_backend = None if cuda_backend is None else getattr(cuda_backend, "matmul", None)
    original_tf32 = None
    if compute_device.type == "cuda" and matmul_backend is not None:
        original_tf32 = matmul_backend.allow_tf32
        matmul_backend.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        for batch in batches:
            flat = batch.detach().reshape(-1, hidden_size).to(device=compute_device, dtype=torch.float32)
            covariance.addmm_(flat.transpose(0, 1), flat)
            tokens += flat.shape[0]
            batch_hashes.append(tensor_sha256(batch))
        covariance /= tokens
        covariance.copy_((covariance + covariance.transpose(0, 1)) * 0.5)
    finally:
        torch.set_float32_matmul_precision(original_precision)
        if original_tf32 is not None:
            matmul_backend.allow_tf32 = original_tf32
    return covariance, {
        "tokens": tokens,
        "sequences": len(batches),
        "hidden_size": hidden_size,
        "batch_sha256": batch_hashes,
        "compute_device": str(compute_device),
        "compute_dtype": "float32",
        "tf32_disabled": compute_device.type == "cuda",
        "covariance_sha256_fp32": tensor_sha256(covariance),
        "all_finite": bool(torch.isfinite(covariance).all()),
    }


def _canonicalize_svd_signs(u: torch.Tensor, vh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pivots = vh.abs().argmax(dim=1)
    signs = vh[torch.arange(vh.shape[0], device=vh.device), pivots].sign()
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return u * signs.unsqueeze(0), vh * signs.unsqueeze(1)


@dataclass
class DriftAccumulator:
    """Accumulate exact global drift metrics without retaining calibration tensors."""

    elements: int = 0
    reference_square_sum: float = 0.0
    actual_square_sum: float = 0.0
    difference_square_sum: float = 0.0
    dot_sum: float = 0.0
    absolute_difference_sum: float = 0.0
    maximum_absolute_error: float = 0.0
    reference_nan_count: int = 0
    actual_nan_count: int = 0
    reference_inf_count: int = 0
    actual_inf_count: int = 0

    @torch.no_grad()
    def update(self, reference: torch.Tensor, actual: torch.Tensor) -> None:
        if reference.shape != actual.shape:
            raise ValueError(
                f"P6.1c drift shape mismatch: {tuple(reference.shape)} != {tuple(actual.shape)}"
            )
        reference_flat = reference.detach().float().reshape(-1)
        actual_flat = actual.detach().float().reshape(-1)
        reference_nan = torch.isnan(reference_flat)
        actual_nan = torch.isnan(actual_flat)
        reference_inf = torch.isinf(reference_flat)
        actual_inf = torch.isinf(actual_flat)
        reference_values = torch.nan_to_num(reference_flat, nan=0.0, posinf=0.0, neginf=0.0)
        actual_values = torch.nan_to_num(actual_flat, nan=0.0, posinf=0.0, neginf=0.0)
        difference = actual_values - reference_values
        self.elements += reference_flat.numel()
        values = torch.stack(
            (
                reference_values.square().sum(),
                actual_values.square().sum(),
                difference.square().sum(),
                (reference_values * actual_values).sum(),
                difference.abs().sum(),
                difference.abs().max() if difference.numel() else difference.new_zeros(()),
            )
        ).cpu()
        counts = torch.stack(
            (
                reference_nan.sum(),
                actual_nan.sum(),
                reference_inf.sum(),
                actual_inf.sum(),
            )
        ).cpu()
        self.reference_square_sum += float(values[0])
        self.actual_square_sum += float(values[1])
        self.difference_square_sum += float(values[2])
        self.dot_sum += float(values[3])
        self.absolute_difference_sum += float(values[4])
        self.maximum_absolute_error = max(self.maximum_absolute_error, float(values[5]))
        self.reference_nan_count += int(counts[0])
        self.actual_nan_count += int(counts[1])
        self.reference_inf_count += int(counts[2])
        self.actual_inf_count += int(counts[3])

    def finalize(self) -> dict[str, float | int]:
        if not self.elements:
            raise ValueError("P6.1c drift accumulator cannot finalize without elements")
        epsilon = torch.finfo(torch.float64).eps
        reference_norm = self.reference_square_sum**0.5
        actual_norm = self.actual_square_sum**0.5
        cosine_denominator = reference_norm * actual_norm
        cosine = (
            self.dot_sum / cosine_denominator
            if cosine_denominator > epsilon
            else float(reference_norm == actual_norm)
        )
        cosine = min(1.0, max(-1.0, cosine))
        reference_rms = (self.reference_square_sum / self.elements) ** 0.5
        actual_rms = (self.actual_square_sum / self.elements) ** 0.5
        return {
            "elements": self.elements,
            "max_abs_error": self.maximum_absolute_error,
            "mean_abs_error": self.absolute_difference_sum / self.elements,
            "relative_l2_error": self.difference_square_sum**0.5 / max(reference_norm, epsilon),
            "cosine_similarity": cosine,
            "reference_rms": reference_rms,
            "actual_rms": actual_rms,
            "rms_ratio": actual_rms / max(reference_rms, epsilon),
            "reference_nan_count": self.reference_nan_count,
            "actual_nan_count": self.actual_nan_count,
            "reference_inf_count": self.reference_inf_count,
            "actual_inf_count": self.actual_inf_count,
        }


@torch.no_grad()
def activation_weighted_rank_factorization(
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    covariance: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rank: int,
    eigenvalue_relative_floor: float = 1e-8,
) -> P6KVFactorization:
    """Solve `min ||X(W-W_hat)^T||_F` with a deterministic covariance metric."""
    if covariance.dtype != torch.float32 or covariance.device != k_weight.device:
        raise ValueError("P6.1c covariance must be FP32 on the donor weight device")
    if num_heads % num_kv_heads:
        raise ValueError("P6.1c target heads must be divisible by donor KV heads")
    # As in W0, factor only the unique four-head [K;V] matrix. Repeating each
    # complete block G times changes singular values by sqrt(G), not the right
    # singular subspace, and avoids an unnecessarily large 8192x2560 SVD.
    base = torch.cat(
        (
            k_weight.detach().float().reshape(num_kv_heads, head_dim, -1),
            v_weight.detach().float().reshape(num_kv_heads, head_dim, -1),
        ),
        dim=1,
    ).reshape(num_kv_heads * 2 * head_dim, -1)
    full = build_full_kv_weight(
        k_weight,
        v_weight,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    ).detach().float()
    original_precision = torch.get_float32_matmul_precision()
    cuda_backend = getattr(torch.backends, "cuda", None)
    matmul_backend = None if cuda_backend is None else getattr(cuda_backend, "matmul", None)
    original_tf32 = None
    if covariance.device.type == "cuda" and matmul_backend is not None:
        original_tf32 = matmul_backend.allow_tf32
        matmul_backend.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    finally:
        torch.set_float32_matmul_precision(original_precision)
        if original_tf32 is not None:
            matmul_backend.allow_tf32 = original_tf32
    descending = torch.arange(
        eigenvalues.numel() - 1,
        -1,
        -1,
        device=eigenvalues.device,
    )
    eigenvalues = eigenvalues[descending].clamp_min(0)
    eigenvectors = eigenvectors[:, descending]
    threshold = max(float(eigenvalues[0]) * eigenvalue_relative_floor, torch.finfo(torch.float32).eps)
    active = eigenvalues > threshold
    if int(active.sum()) < rank:
        raise ValueError("P6.1c activation covariance has fewer active directions than rank")
    eigenvalues = eigenvalues[active]
    eigenvectors = eigenvectors[:, active]
    sqrt_metric = eigenvectors * eigenvalues.sqrt().unsqueeze(0)
    inverse_sqrt_metric = eigenvectors * eigenvalues.rsqrt().unsqueeze(0)
    original_precision = torch.get_float32_matmul_precision()
    if covariance.device.type == "cuda" and matmul_backend is not None:
        original_tf32 = matmul_backend.allow_tf32
        matmul_backend.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        whitened_weight = base @ sqrt_metric
        u, singular, vh = torch.linalg.svd(whitened_weight, full_matrices=False)
    finally:
        torch.set_float32_matmul_precision(original_precision)
        if original_tf32 is not None:
            matmul_backend.allow_tf32 = original_tf32
    u, vh = _canonicalize_svd_signs(u, vh)
    retained = min(rank, singular.numel())
    group_size = num_heads // num_kv_heads
    sqrt_group = float(group_size) ** 0.5
    retained_full = singular[:retained] * sqrt_group
    sqrt_singular = retained_full.sqrt()
    full_u = (
        u[:, :retained]
        .reshape(num_kv_heads, 2 * head_dim, retained)
        .repeat_interleave(group_size, dim=0)
        .reshape(num_heads * 2 * head_dim, retained)
        / sqrt_group
    )
    kv_b = full_u * sqrt_singular.unsqueeze(0)
    right_factor = vh[:retained] @ inverse_sqrt_metric.transpose(0, 1)
    kv_a = sqrt_singular.unsqueeze(1) * right_factor
    reconstruction = kv_b @ kv_a
    difference = full - reconstruction
    activation_total = singular.square().sum()
    activation_retained = singular[:retained].square().sum()
    activation_residual = (activation_total - activation_retained).clamp_min(0)
    report = {
        "algorithm": "EXACT_COVARIANCE_WEIGHTED_SVD",
        "compute_device": str(covariance.device),
        "covariance_compute_dtype": "float32",
        "factorization_compute_dtype": "float32",
        "tf32_disabled": covariance.device.type == "cuda",
        "rank": retained,
        "full_shape": list(full.shape),
        "unique_base_shape": list(base.shape),
        "covariance_active_rank": int(active.sum()),
        "covariance_eigenvalue_relative_floor": eigenvalue_relative_floor,
        "activation_weighted_relative_error": float((activation_residual / activation_total).sqrt()),
        "activation_weighted_explained_squared_mass": float(activation_retained / activation_total),
        "weight_frobenius_relative_error": float(
            torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(full)
        ),
        "weight_maximum_absolute_error": float(difference.abs().max()),
        "all_finite": bool(torch.isfinite(kv_a).all() and torch.isfinite(kv_b).all()),
        "kv_a_sha256_fp32": tensor_sha256(kv_a),
        "kv_b_sha256_fp32": tensor_sha256(kv_b),
    }
    if not report["all_finite"]:
        raise FloatingPointError("P6.1c activation-weighted factorization is non-finite")
    return P6KVFactorization(kv_a_weight=kv_a, kv_b_weight=kv_b, report=report)


@torch.no_grad()
def fit_scale_matched_rmsnorm(
    latent_batches: list[torch.Tensor],
    *,
    eps: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Fit per-channel RMSNorm gamma by deterministic closed-form least squares."""
    if not latent_batches:
        raise ValueError("P6.1c latent batches cannot be empty")
    width = latent_batches[0].shape[-1]
    numerator = torch.zeros(width, dtype=torch.float64)
    denominator = torch.zeros(width, dtype=torch.float64)
    raw_error = 0.0
    elements = 0
    for latent in latent_batches:
        values = latent.detach().cpu().reshape(-1, width).double()
        normalized = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + eps)
        numerator += (normalized * values).sum(dim=0)
        denominator += normalized.square().sum(dim=0)
        raw_error += float((normalized - values).square().sum())
        elements += values.numel()
    gamma = (numerator / denominator.clamp_min(torch.finfo(torch.float64).eps)).float()
    fitted_error = 0.0
    for latent in latent_batches:
        values = latent.detach().cpu().reshape(-1, width).double()
        normalized = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + eps)
        fitted_error += float((normalized * gamma.double() - values).square().sum())
    report = {
        "algorithm": "PER_CHANNEL_CLOSED_FORM_RMSNORM_SCALE_MATCH",
        "tokens": sum(batch.numel() // width for batch in latent_batches),
        "channels": width,
        "ones_rmse": (raw_error / elements) ** 0.5,
        "fitted_rmse": (fitted_error / elements) ** 0.5,
        "gamma_min": float(gamma.min()),
        "gamma_mean": float(gamma.mean()),
        "gamma_max": float(gamma.max()),
        "gamma_sha256_fp32": tensor_sha256(gamma),
        "all_finite": bool(torch.isfinite(gamma).all()),
    }
    if not report["all_finite"]:
        raise FloatingPointError("P6.1c scale-matched RMSNorm gamma is non-finite")
    return gamma, report


@torch.no_grad()
def apply_rmsnorm_scale(module: DenseK3RMSNorm, gamma: torch.Tensor) -> None:
    if module.weight.shape != gamma.shape:
        raise ValueError("P6.1c latent RMSNorm gamma shape mismatch")
    module.weight.copy_(gamma.to(device=module.weight.device, dtype=module.weight.dtype))


__all__ = [
    "DriftAccumulator",
    "activation_weighted_rank_factorization",
    "apply_rmsnorm_scale",
    "covariance_from_batches",
    "fit_scale_matched_rmsnorm",
]
