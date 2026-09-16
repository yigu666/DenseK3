"""Single-layer P6 GQA-to-strict-NoPE-MLA conversion probe."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from densek3_core.modeling.hybrid_cache import DenseK3HybridCache
from densek3_core.modeling.modeling_densek3 import DenseK3GQA, DenseK3RMSNorm

P6_PROBE_LAYER = 3
P6_NUM_HEADS = 16
P6_HEAD_DIM = 256
P6_KV_LORA_RANK = 512
P6_CACHE_MODE = "expanded_reference"


def tensor_sha256(value: torch.Tensor) -> str:
    data = value.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(data.numpy().tobytes()).hexdigest()


def split_gated_query_weight(weight: torch.Tensor, *, num_heads: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    expected = (num_heads * 2 * head_dim, weight.shape[1])
    if tuple(weight.shape) != expected:
        raise ValueError(f"Packed GQA query weight must have shape {expected}")
    packed = weight.reshape(num_heads, 2, head_dim, weight.shape[1])
    return packed[:, 0].reshape(num_heads * head_dim, -1), packed[:, 1].reshape(num_heads * head_dim, -1)


def expand_gqa_kv_weight(weight: torch.Tensor, *, num_heads: int, num_kv_heads: int, head_dim: int) -> torch.Tensor:
    expected = (num_kv_heads * head_dim, weight.shape[1])
    if tuple(weight.shape) != expected or num_heads % num_kv_heads:
        raise ValueError("Donor GQA K/V weight is incompatible with the target heads")
    return weight.reshape(num_kv_heads, head_dim, -1).repeat_interleave(
        num_heads // num_kv_heads,
        dim=0,
    ).reshape(num_heads * head_dim, -1)


def build_full_kv_weight(
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    k_full = expand_gqa_kv_weight(
        k_weight,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    ).reshape(num_heads, head_dim, -1)
    v_full = expand_gqa_kv_weight(
        v_weight,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    ).reshape(num_heads, head_dim, -1)
    return torch.cat((k_full, v_full), dim=1).reshape(num_heads * 2 * head_dim, -1)


def _repeat_kv_blocks(blocks: torch.Tensor, repeats: int) -> torch.Tensor:
    """Repeat complete `[K;V]` head blocks, never their individual rows."""
    return blocks.repeat_interleave(repeats, dim=0)


def _canonicalize_svd_signs(u: torch.Tensor, vh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pivots = vh.abs().argmax(dim=1)
    signs = vh[torch.arange(vh.shape[0], device=vh.device), pivots].sign()
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return u * signs.unsqueeze(0), vh * signs.unsqueeze(1)


@dataclass
class P6KVFactorization:
    kv_a_weight: torch.Tensor
    kv_b_weight: torch.Tensor
    report: dict[str, Any]


@torch.no_grad()
def deterministic_rank_factorization(
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rank: int,
) -> P6KVFactorization:
    """Factor the four unique GQA K/V head blocks, then repeat their left factor."""
    if rank <= 0:
        raise ValueError("MLA factorization rank must be positive")
    if k_weight.device != v_weight.device:
        raise ValueError("P6 donor K/V weights must share a device")
    compute_device = k_weight.device
    # The 16-head full matrix repeats each of the four donor K/V head blocks
    # four times. Factoring the 2048xD unique matrix is mathematically exact
    # with respect to factoring the repeated 8192xD matrix and is much cheaper.
    base = torch.cat(
        (
            k_weight.float().reshape(num_kv_heads, head_dim, -1),
            v_weight.float().reshape(num_kv_heads, head_dim, -1),
        ),
        dim=1,
    ).reshape(num_kv_heads * 2 * head_dim, -1)
    original_precision = torch.get_float32_matmul_precision()
    original_tf32 = None
    cuda_backend = getattr(torch.backends, "cuda", None)
    matmul_backend = None if cuda_backend is None else getattr(cuda_backend, "matmul", None)
    if compute_device.type == "cuda" and matmul_backend is not None:
        original_tf32 = matmul_backend.allow_tf32
        matmul_backend.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        u, singular, vh = torch.linalg.svd(base, full_matrices=False)
        u, vh = _canonicalize_svd_signs(u, vh)
    finally:
        torch.set_float32_matmul_precision(original_precision)
        if original_tf32 is not None:
            matmul_backend.allow_tf32 = original_tf32
    retained = min(rank, singular.numel())
    u_r = u[:, :retained]
    s_r = singular[:retained]
    vh_r = vh[:retained]
    group_size = num_heads // num_kv_heads
    # Singular values of a row-block repeated G times scale by sqrt(G).
    sqrt_group = float(group_size) ** 0.5
    full_u_factor = _repeat_kv_blocks(
        u_r.reshape(num_kv_heads, 2 * head_dim, retained),
        group_size,
    ).reshape(num_heads * 2 * head_dim, retained) / sqrt_group
    full_singular = singular * sqrt_group
    retained_full = s_r * sqrt_group
    sqrt_s = retained_full.sqrt()
    kv_b = full_u_factor * sqrt_s.unsqueeze(0)
    kv_a = sqrt_s.unsqueeze(1) * vh_r
    base_reconstruction = (u_r * s_r.unsqueeze(0)) @ vh_r
    residual = base - base_reconstruction
    base_norm = torch.linalg.vector_norm(base)
    full = build_full_kv_weight(
        k_weight,
        v_weight,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    ).float()
    full_reconstruction = kv_b @ kv_a
    full_heads = full.reshape(num_heads, 2 * head_dim, -1)
    reconstructed_heads = full_reconstruction.reshape(num_heads, 2 * head_dim, -1)
    k_full, v_full = full_heads[:, :head_dim], full_heads[:, head_dim:]
    k_reconstructed = reconstructed_heads[:, :head_dim]
    v_reconstructed = reconstructed_heads[:, head_dim:]

    def reconstruction_error(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
        difference = reference - actual
        return {
            "frobenius_relative_error": float(
                torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference)
            ),
            "maximum_absolute_error": float(difference.abs().max()),
        }

    report = {
        "algorithm": "DETERMINISTIC_TRUNCATED_SVD_UNIQUE_GQA_BLOCKS",
        "compute_device": str(compute_device),
        "compute_dtype": "float32",
        "tf32_disabled": compute_device.type == "cuda",
        "sign_canonicalization": "VH_MAX_ABS_PIVOT_POSITIVE",
        "rank": retained,
        "full_shape": list(full.shape),
        "unique_base_shape": list(base.shape),
        "frobenius_relative_error": float(torch.linalg.vector_norm(residual) / base_norm),
        "spectral_relative_error": (
            float(full_singular[retained] / full_singular[0])
            if retained < full_singular.numel()
            else 0.0
        ),
        "maximum_absolute_error": float((full - full_reconstruction).abs().max()),
        "k_reconstruction": reconstruction_error(k_full, k_reconstructed),
        "v_reconstruction": reconstruction_error(v_full, v_reconstructed),
        "explained_squared_singular_mass": float(
            retained_full.square().sum() / full_singular.square().sum()
        ),
        "singular_value_max": float(full_singular[0]),
        "singular_value_first_excluded": (
            float(full_singular[retained]) if retained < full_singular.numel() else 0.0
        ),
        "all_finite": bool(torch.isfinite(kv_a).all() and torch.isfinite(kv_b).all()),
        "kv_a_sha256_fp32": tensor_sha256(kv_a),
        "kv_b_sha256_fp32": tensor_sha256(kv_b),
    }
    if not report["all_finite"]:
        raise FloatingPointError("P6 SVD factorization produced non-finite weights")
    return P6KVFactorization(kv_a_weight=kv_a, kv_b_weight=kv_b, report=report)


def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    past_length: int,
    dropout: float,
    training: bool,
) -> torch.Tensor:
    batch, _, sequence, _ = q.shape
    total_length = k.shape[2]
    if attention_mask is not None:
        if attention_mask.shape[-1] == sequence and past_length:
            prefix = torch.ones((batch, past_length), device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat((prefix, attention_mask), dim=-1)
        if attention_mask.shape != (batch, total_length):
            raise ValueError("P6 expanded attention mask must cover the cached sequence")
    all_visible = attention_mask is None or bool(attention_mask.bool().all())
    native_causal = all_visible and past_length == 0 and sequence > 1
    no_mask_decode = all_visible and sequence == 1
    if native_causal or no_mask_decode:
        sdpa_mask = None
    else:
        query_positions = past_length + torch.arange(sequence, device=q.device)
        key_positions = torch.arange(total_length, device=q.device)
        sdpa_mask = key_positions.view(1, 1, 1, -1) <= query_positions.view(1, 1, -1, 1)
        if attention_mask is not None:
            sdpa_mask = sdpa_mask & attention_mask[:, None, None, :].bool()
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=sdpa_mask,
        dropout_p=dropout if training else 0.0,
        is_causal=native_causal,
    )


class _P6NoPEAttentionBase(nn.Module):
    position_encoding = "none"
    cache_mode = P6_CACHE_MODE

    def __init__(self, donor: DenseK3GQA):
        super().__init__()
        self.layer_idx = donor.layer_idx
        self.num_heads = donor.num_heads
        self.head_dim = donor.head_dim
        self.dropout = donor.dropout
        hidden_size = donor.q_proj.in_features
        device = donor.q_proj.weight.device
        dtype = donor.q_proj.weight.dtype
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False, device=device, dtype=dtype)
        with torch.no_grad():
            self.o_proj.weight.copy_(donor.o_proj.weight)

    def _finish(
        self,
        hidden_states: torch.Tensor,
        gate: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cache: DenseK3HybridCache | None,
        use_cache: bool,
    ) -> torch.Tensor:
        batch, sequence, _ = hidden_states.shape
        if cache is not None:
            k, v = cache.update_gqa(k, v, self.layer_idx)
        elif use_cache:
            raise ValueError("use_cache=True requires a DenseK3HybridCache")
        past_length = k.shape[2] - sequence
        output = _attention(
            q,
            k,
            v,
            attention_mask,
            past_length=past_length,
            dropout=self.dropout,
            training=self.training,
        )
        output = output.transpose(1, 2).contiguous().reshape(batch, sequence, -1)
        return self.o_proj(output * torch.sigmoid(gate))


class P6NoPEFullAttention(_P6NoPEAttentionBase):
    """Case B: donor projections and norms, activation-expanded K/V, no RoPE.

    Keeping the donor packed Q/gate and compact GQA K/V GEMMs makes A-to-B a
    clean position-encoding ablation.  K/V are expanded only after projection.
    """

    case = "B_FULL_RANK_NOPE"

    def __init__(self, donor: DenseK3GQA, *, split_q_gate: bool = False, apply_qk_norm: bool = True):
        super().__init__(donor)
        hidden_size = donor.k_proj.in_features
        device = donor.k_proj.weight.device
        dtype = donor.k_proj.weight.dtype
        self.num_key_value_heads = donor.num_key_value_heads
        self.group_size = self.num_heads // self.num_key_value_heads
        self.split_q_gate = split_q_gate
        self.apply_qk_norm = apply_qk_norm
        if split_q_gate:
            self.q_proj = nn.Linear(
                hidden_size,
                self.num_heads * self.head_dim,
                bias=False,
                device=device,
                dtype=dtype,
            )
            self.g_proj = nn.Linear(
                hidden_size,
                self.num_heads * self.head_dim,
                bias=False,
                device=device,
                dtype=dtype,
            )
        else:
            self.packed_q_proj = nn.Linear(
                hidden_size,
                self.num_heads * 2 * self.head_dim,
                bias=False,
                device=device,
                dtype=dtype,
            )
        self.k_proj = nn.Linear(
            hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.v_proj = nn.Linear(
            hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.q_norm = DenseK3RMSNorm(self.head_dim, donor.q_norm.eps).to(device=device, dtype=dtype)
        self.k_norm = DenseK3RMSNorm(self.head_dim, donor.k_norm.eps).to(device=device, dtype=dtype)
        q_weight, gate_weight = split_gated_query_weight(
            donor.q_proj.weight,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        with torch.no_grad():
            if split_q_gate:
                self.q_proj.weight.copy_(q_weight)
                self.g_proj.weight.copy_(gate_weight)
            else:
                self.packed_q_proj.weight.copy_(donor.q_proj.weight)
            self.k_proj.weight.copy_(donor.k_proj.weight)
            self.v_proj.weight.copy_(donor.v_proj.weight)
            self.q_norm.weight.copy_(donor.q_norm.weight)
            self.k_norm.weight.copy_(donor.k_norm.weight)

    def project_qkvg(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, sequence, _ = hidden_states.shape
        if self.split_q_gate:
            q_raw = self.q_proj(hidden_states).view(batch, sequence, self.num_heads, self.head_dim)
            gate = self.g_proj(hidden_states).view(batch, sequence, self.num_heads, self.head_dim)
        else:
            packed = self.packed_q_proj(hidden_states).view(
                batch,
                sequence,
                self.num_heads,
                2,
                self.head_dim,
            )
            q_raw, gate = packed.unbind(dim=3)
        k_raw = self.k_proj(hidden_states).view(
            batch,
            sequence,
            self.num_key_value_heads,
            self.head_dim,
        )
        v_compact = self.v_proj(hidden_states).view(
            batch,
            sequence,
            self.num_key_value_heads,
            self.head_dim,
        )
        q = (self.q_norm(q_raw) if self.apply_qk_norm else q_raw).transpose(1, 2)
        normalized_k = self.k_norm(k_raw) if self.apply_qk_norm else k_raw
        k = normalized_k.transpose(1, 2).repeat_interleave(self.group_size, dim=1)
        v = v_compact.transpose(1, 2).repeat_interleave(self.group_size, dim=1)
        return q, k, v, gate.reshape(batch, sequence, -1), k_raw.repeat_interleave(
            self.group_size,
            dim=2,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None,
        cache: DenseK3HybridCache | None,
        use_cache: bool,
    ) -> torch.Tensor:
        del position_ids
        q, k, v, gate, _ = self.project_qkvg(hidden_states)
        return self._finish(hidden_states, gate, q, k, v, attention_mask, cache, use_cache)


class P6NoPEMLA(_P6NoPEAttentionBase):
    """Case C: rank-512 strict-NoPE MLA with expanded reference cache."""

    case = "C_RANK512_MLA_NOPE"

    def __init__(
        self,
        donor: DenseK3GQA,
        factorization: P6KVFactorization,
        *,
        split_q_gate: bool = True,
        apply_qk_norm: bool = False,
        apply_latent_norm: bool = True,
    ):
        super().__init__(donor)
        hidden_size = donor.k_proj.in_features
        rank = factorization.kv_a_weight.shape[0]
        device = donor.k_proj.weight.device
        dtype = donor.k_proj.weight.dtype
        self.kv_lora_rank = rank
        self.q_lora_rank = None
        self.qk_nope_head_dim = self.head_dim
        self.qk_pos_head_dim = 0
        self.value_head_dim = self.head_dim
        self.split_q_gate = split_q_gate
        self.apply_qk_norm = apply_qk_norm
        self.apply_latent_norm = apply_latent_norm
        if split_q_gate:
            self.q_proj = nn.Linear(
                hidden_size,
                self.num_heads * self.head_dim,
                bias=False,
                device=device,
                dtype=dtype,
            )
            self.g_proj = nn.Linear(
                hidden_size,
                self.num_heads * self.head_dim,
                bias=False,
                device=device,
                dtype=dtype,
            )
        else:
            self.packed_q_proj = nn.Linear(
                hidden_size,
                self.num_heads * 2 * self.head_dim,
                bias=False,
                device=device,
                dtype=dtype,
            )
        self.kv_a_proj = nn.Linear(hidden_size, rank, bias=False, device=device, dtype=dtype)
        self.kv_a_layernorm = DenseK3RMSNorm(rank, donor.q_norm.eps).to(device=device, dtype=dtype)
        self.kv_b_proj = nn.Linear(rank, self.num_heads * 2 * self.head_dim, bias=False, device=device, dtype=dtype)
        if apply_qk_norm:
            self.q_norm = DenseK3RMSNorm(self.head_dim, donor.q_norm.eps).to(device=device, dtype=dtype)
            self.k_norm = DenseK3RMSNorm(self.head_dim, donor.k_norm.eps).to(device=device, dtype=dtype)
        q_weight, gate_weight = split_gated_query_weight(
            donor.q_proj.weight,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        with torch.no_grad():
            if split_q_gate:
                self.q_proj.weight.copy_(q_weight)
                self.g_proj.weight.copy_(gate_weight)
            else:
                self.packed_q_proj.weight.copy_(donor.q_proj.weight)
            if apply_qk_norm:
                self.q_norm.weight.copy_(donor.q_norm.weight)
                self.k_norm.weight.copy_(donor.k_norm.weight)
            self.kv_a_proj.weight.copy_(factorization.kv_a_weight)
            self.kv_b_proj.weight.copy_(factorization.kv_b_weight)
            self.kv_a_layernorm.weight.fill_(1)

    def project_kv(
        self,
        hidden_states: torch.Tensor,
        *,
        apply_latent_norm: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, sequence, _ = hidden_states.shape
        latent = self.kv_a_proj(hidden_states)
        projected_latent = self.kv_a_layernorm(latent) if apply_latent_norm else latent
        kv = self.kv_b_proj(projected_latent).view(
            batch,
            sequence,
            self.num_heads,
            2,
            self.head_dim,
        )
        k, v = kv.unbind(dim=3)
        return k, v, latent

    def project_qkvg(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, sequence, _ = hidden_states.shape
        if self.split_q_gate:
            q = self.q_proj(hidden_states).view(batch, sequence, self.num_heads, self.head_dim)
            gate = self.g_proj(hidden_states)
        else:
            packed = self.packed_q_proj(hidden_states).view(
                batch,
                sequence,
                self.num_heads,
                2,
                self.head_dim,
            )
            q, gate_per_head = packed.unbind(dim=3)
            gate = gate_per_head.reshape(batch, sequence, -1)
        k, v, latent = self.project_kv(hidden_states, apply_latent_norm=self.apply_latent_norm)
        if self.apply_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), gate, latent

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None,
        cache: DenseK3HybridCache | None,
        use_cache: bool,
    ) -> torch.Tensor:
        del position_ids
        q, k, v, gate, _ = self.project_qkvg(hidden_states)
        return self._finish(
            hidden_states,
            gate,
            q,
            k,
            v,
            attention_mask,
            cache,
            use_cache,
        )


@torch.no_grad()
def nope_attention_trace(
    module: P6NoPEFullAttention | P6NoPEMLA,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """Return common no-cache tensors for P6.1b attribution cases B through H."""
    q, k, v, gate, auxiliary = module.project_qkvg(hidden_states)
    batch, _, sequence, head_dim = q.shape
    scores = torch.matmul(q, k.transpose(-1, -2)) / float(head_dim) ** 0.5
    visible = torch.ones((sequence, sequence), device=scores.device, dtype=torch.bool).tril()
    visible_scores = scores.masked_select(visible.view(1, 1, sequence, sequence))
    core = _attention(
        q,
        k,
        v,
        attention_mask,
        past_length=0,
        dropout=0.0,
        training=False,
    )
    attention_output = core.transpose(1, 2).contiguous().reshape(batch, sequence, -1)
    gated_output = attention_output * torch.sigmoid(gate)
    mixer_output = module.o_proj(gated_output)
    return {
        "q": q,
        "k": k,
        "v": v,
        "attention_scores_visible": visible_scores,
        "attention_core": core,
        "attention_output": attention_output,
        "gated_attention_output": gated_output,
        "mixer_output": mixer_output,
        "auxiliary": auxiliary,
    }


def conversion_manifest(donor: DenseK3GQA, factorization: P6KVFactorization) -> dict[str, Any]:
    q_weight, gate_weight = split_gated_query_weight(
        donor.q_proj.weight,
        num_heads=donor.num_heads,
        head_dim=donor.head_dim,
    )
    return {
        "stage": "P6.1",
        "layer_index": donor.layer_idx,
        "source": "GQA_WITH_PARTIAL_ROPE",
        "target": "STRICT_NOPE_MLA_RANK512",
        "cache_mode": P6_CACHE_MODE,
        "true_latent_cache": False,
        "mapping": {
            "q_proj": {"operation": "UNPACK", "shape": list(q_weight.shape), "sha256": tensor_sha256(q_weight)},
            "g_proj": {"operation": "UNPACK", "shape": list(gate_weight.shape), "sha256": tensor_sha256(gate_weight)},
            "kv": {"operation": "DETERMINISTIC_TRUNCATED_SVD", **factorization.report},
            "o_proj": {"operation": "COPY", "shape": list(donor.o_proj.weight.shape)},
            "q_norm": {"operation": "EXCLUDE", "shape": list(donor.q_norm.weight.shape), "reason": "data-dependent"},
            "k_norm": {"operation": "EXCLUDE", "shape": list(donor.k_norm.weight.shape), "reason": "data-dependent"},
            "kv_a_layernorm": {"operation": "CONSTRUCT_ONES", "shape": [P6_KV_LORA_RANK]},
        },
        "target_contract": {
            "heads": P6_NUM_HEADS,
            "q_lora_rank": None,
            "kv_lora_rank": P6_KV_LORA_RANK,
            "qk_nope_head_dim": P6_HEAD_DIM,
            "qk_pos_head_dim": 0,
            "value_head_dim": P6_HEAD_DIM,
            "position_encoding": "none",
            "output_gate": "full_rank_sigmoid",
            "output_gate_bias": False,
        },
    }


__all__ = [
    "P6_CACHE_MODE",
    "P6_HEAD_DIM",
    "P6_KV_LORA_RANK",
    "P6_NUM_HEADS",
    "P6_PROBE_LAYER",
    "P6KVFactorization",
    "P6NoPEFullAttention",
    "P6NoPEMLA",
    "_attention",
    "build_full_kv_weight",
    "conversion_manifest",
    "deterministic_rank_factorization",
    "expand_gqa_kv_weight",
    "split_gated_query_weight",
    "tensor_sha256",
    "nope_attention_trace",
]
