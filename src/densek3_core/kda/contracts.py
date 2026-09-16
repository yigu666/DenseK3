"""Tensor and architecture contracts for DenseK3 KDA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class KDAContract:
    """KDA dimensions read directly from the locked P1 YAML."""

    qk_num_heads: int
    value_num_heads: int
    qk_head_dim: int
    value_head_dim: int
    short_conv_kernel_size: int
    decay_projection_rank: int
    qk_l2norm: bool
    beta_activation: str
    allow_negative_eigenvalues: bool
    recurrent_state_dtype: str
    recurrent_state_layout: str
    training_backend: str
    short_inference_backend: str

    @property
    def value_heads_per_qk_head(self) -> int:
        return self.value_num_heads // self.qk_num_heads

    @property
    def torch_state_dtype(self) -> torch.dtype:
        if self.recurrent_state_dtype != "float32":
            raise ValueError(f"Unsupported recurrent state dtype: {self.recurrent_state_dtype}")
        return torch.float32

    def state_shape(self, batch_size: int) -> tuple[int, int, int, int]:
        """Return the physical V-first state shape `[B, HV, V, K]`."""
        return batch_size, self.value_num_heads, self.value_head_dim, self.qk_head_dim

    def validate(self) -> None:
        if self.qk_num_heads <= 0 or self.value_num_heads <= 0:
            raise ValueError("KDA head counts must be positive")
        if self.value_num_heads % self.qk_num_heads:
            raise ValueError("value_num_heads must be divisible by qk_num_heads")
        if self.value_heads_per_qk_head != 2:
            raise ValueError("DenseK3 P1 requires an explicit 2:1 GVA mapping")
        if self.qk_head_dim <= 0 or self.value_head_dim <= 0:
            raise ValueError("KDA head dimensions must be positive")
        if self.recurrent_state_dtype != "float32":
            raise ValueError("DenseK3 P1 requires an FP32 recurrent state")
        if self.recurrent_state_layout != "v_first":
            raise ValueError("DenseK3 P1 requires V-first recurrent state layout")
        if not self.qk_l2norm:
            raise ValueError("DenseK3 P1 requires Q/K L2 normalization")
        if self.beta_activation != "sigmoid":
            raise ValueError("DenseK3 P1 requires sigmoid beta activation")
        if self.allow_negative_eigenvalues:
            raise ValueError("DenseK3 P1 disables negative eigenvalues")
        if self.training_backend != "fla_chunk":
            raise ValueError("DenseK3 P1 training backend must be fla_chunk")
        if self.short_inference_backend != "fla_fused_recurrent":
            raise ValueError("DenseK3 P1 short inference backend must be fla_fused_recurrent")

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> KDAContract:
        backends = values.get("backends")
        if not isinstance(backends, dict):
            raise ValueError("P1 KDA backends must be a mapping")
        contract = cls(
            qk_num_heads=int(values["qk_num_heads"]),
            value_num_heads=int(values["value_num_heads"]),
            qk_head_dim=int(values["qk_head_dim"]),
            value_head_dim=int(values["value_head_dim"]),
            short_conv_kernel_size=int(values["short_conv_kernel_size"]),
            decay_projection_rank=int(values["decay_projection_rank"]),
            qk_l2norm=bool(values["qk_l2norm"]),
            beta_activation=str(values["beta_activation"]),
            allow_negative_eigenvalues=bool(values["allow_negative_eigenvalues"]),
            recurrent_state_dtype=str(values["recurrent_state_dtype"]),
            recurrent_state_layout=str(values["recurrent_state_layout"]),
            training_backend=str(backends["training"]),
            short_inference_backend=str(backends["short_inference"]),
        )
        contract.validate()
        return contract


def validate_operator_shapes(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[int, int, int, int, int, int]:
    """Validate full-sequence operator tensors and return logical dimensions."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or decay.ndim != 4 or beta.ndim != 3:
        raise ValueError("Expected q/k/v/decay rank 4 and beta rank 3")
    batch, sequence, qk_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2:]
    if k.shape != q.shape:
        raise ValueError(f"q and k shapes differ: {tuple(q.shape)} vs {tuple(k.shape)}")
    if v.shape[:2] != (batch, sequence):
        raise ValueError("v batch/sequence dimensions do not match q")
    if value_heads % qk_heads:
        raise ValueError("value heads must be divisible by Q/K heads")
    if decay.shape != (batch, sequence, value_heads, key_dim):
        raise ValueError("decay must have shape [B,T,HV,K]")
    if beta.shape != (batch, sequence, value_heads):
        raise ValueError("beta must have shape [B,T,HV]")
    devices = {tensor.device for tensor in (q, k, v, decay, beta)}
    if len(devices) != 1:
        raise ValueError("All KDA inputs must be on the same device")
    return batch, sequence, qk_heads, value_heads, key_dim, value_dim
