"""Independent, auditable PyTorch KDA reference implementation."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from densek3_core.kda.contracts import validate_operator_shapes
from densek3_core.kda.state import validate_state_tensor


def l2_normalize_kda(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Match FLA's sum-of-squares L2 normalization in FP32."""
    x_float = x.float()
    return x_float * torch.rsqrt(x_float.square().sum(dim=-1, keepdim=True) + eps)


def qwen_compat_decay(raw_decay: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor | None = None) -> torch.Tensor:
    """Produce P4 log-space decay from raw channel logits."""
    if raw_decay.ndim < 2:
        raise ValueError("raw_decay must end in [HV,K]")
    value_heads, key_dim = raw_decay.shape[-2:]
    if tuple(a_log.shape) != (value_heads,):
        raise ValueError(f"a_log must have shape {(value_heads,)}")
    logits = raw_decay.float()
    if dt_bias is not None:
        if dt_bias.numel() != value_heads * key_dim:
            raise ValueError(f"dt_bias must contain {value_heads * key_dim} values")
        logits = logits + dt_bias.float().reshape(value_heads, key_dim)
    return -a_log.float().exp().reshape(value_heads, 1) * F.softplus(logits)


def kimi_k3_decay(
    raw_decay: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float = -5.0,
) -> torch.Tensor:
    """Produce P5 lower-bounded log-space decay from raw channel logits."""
    if raw_decay.ndim < 2:
        raise ValueError("raw_decay must end in [HV,K]")
    value_heads, key_dim = raw_decay.shape[-2:]
    if tuple(a_log.shape) != (value_heads,):
        raise ValueError(f"a_log must have shape {(value_heads,)}")
    if lower_bound >= 0:
        raise ValueError("Kimi K3 decay lower_bound must be negative")
    logits = raw_decay.float()
    if dt_bias is not None:
        if dt_bias.numel() != value_heads * key_dim:
            raise ValueError(f"dt_bias must contain {value_heads * key_dim} values")
        logits = logits + dt_bias.float().reshape(value_heads, key_dim)
    return lower_bound * torch.sigmoid(a_log.float().exp().reshape(value_heads, 1) * logits)


def kda_recurrent_step(
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    decay_t: torch.Tensor,
    beta_t: torch.Tensor,
    state: torch.Tensor,
    *,
    scale: float | None = None,
    normalize_qk: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one KDA token update using an FP32 V-first state."""
    if q_t.ndim != 3 or k_t.ndim != 3 or v_t.ndim != 3 or decay_t.ndim != 3 or beta_t.ndim != 2:
        raise ValueError("Expected step tensors q/k/v/decay rank 3 and beta rank 2")
    batch, qk_heads, key_dim = q_t.shape
    value_heads, value_dim = v_t.shape[1:]
    if k_t.shape != q_t.shape:
        raise ValueError("q_t and k_t must have matching shapes")
    if value_heads % qk_heads:
        raise ValueError("value heads must be divisible by Q/K heads")
    if decay_t.shape != (batch, value_heads, key_dim):
        raise ValueError("decay_t must have shape [B,HV,K]")
    if beta_t.shape != (batch, value_heads):
        raise ValueError("beta_t must have shape [B,HV]")
    validate_state_tensor(state, batch, value_heads, value_dim, key_dim, q_t.device)

    q = l2_normalize_kda(q_t) if normalize_qk else q_t.float()
    k = l2_normalize_kda(k_t) if normalize_qk else k_t.float()
    group_size = value_heads // qk_heads
    q = q.repeat_interleave(group_size, dim=1)
    k = k.repeat_interleave(group_size, dim=1)
    q = q * (key_dim**-0.5 if scale is None else scale)
    v = v_t.float()
    decay = decay_t.float()
    beta = beta_t.float()

    next_state = state.clone()
    next_state = next_state * decay.exp().unsqueeze(-2)
    prediction = torch.einsum("bhk,bhvk->bhv", k, next_state)
    delta = beta.unsqueeze(-1) * (v - prediction)
    next_state = next_state + delta.unsqueeze(-1) * k.unsqueeze(-2)
    output = torch.einsum("bhk,bhvk->bhv", q, next_state)
    return output.to(v_t.dtype), next_state


def kda_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    scale: float | None = None,
    normalize_qk: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run full-sequence KDA by repeatedly applying the single-step oracle."""
    batch, sequence, _, value_heads, key_dim, value_dim = validate_operator_shapes(q, k, v, decay, beta)
    if initial_state is None:
        state = torch.zeros((batch, value_heads, value_dim, key_dim), dtype=torch.float32, device=q.device)
    else:
        validate_state_tensor(initial_state, batch, value_heads, value_dim, key_dim, q.device)
        state = initial_state.clone()
    outputs = []
    for token_index in range(sequence):
        output_t, state = kda_recurrent_step(
            q[:, token_index],
            k[:, token_index],
            v[:, token_index],
            decay[:, token_index],
            beta[:, token_index],
            state,
            scale=scale,
            normalize_qk=normalize_qk,
        )
        outputs.append(output_t)
    output = v.new_empty((batch, 0, value_heads, value_dim)) if sequence == 0 else torch.stack(outputs, dim=1)
    return output, state if output_final_state else None


def kda_reference_from_raw(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_decay: torch.Tensor,
    beta_logits: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    *,
    gate_mode: str = "qwen_compat",
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the reference with the same raw gate inputs accepted by FLA adapters."""
    if gate_mode == "qwen_compat":
        decay = qwen_compat_decay(raw_decay, a_log, dt_bias)
    elif gate_mode == "kimi_k3":
        decay = kimi_k3_decay(raw_decay, a_log, dt_bias)
    else:
        raise ValueError(f"Unknown KDA gate mode: {gate_mode}")
    return kda_reference(
        q,
        k,
        v,
        decay,
        torch.sigmoid(beta_logits),
        initial_state=initial_state,
        output_final_state=output_final_state,
    )


def kda_reference_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run independent packed sequences without sharing recurrent state."""
    batch, total_tokens, _, value_heads, key_dim, value_dim = validate_operator_shapes(q, k, v, decay, beta)
    if batch != 1:
        raise ValueError("Packed KDA inputs must have batch size 1")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must have shape [N+1]")
    offsets = [int(value) for value in cu_seqlens.detach().cpu().tolist()]
    if offsets[0] != 0 or offsets[-1] != total_tokens or any(
        left >= right for left, right in zip(offsets, offsets[1:], strict=False)
    ):
        raise ValueError("cu_seqlens must define non-empty contiguous sequences")
    sequence_count = len(offsets) - 1
    if initial_state is not None:
        validate_state_tensor(initial_state, sequence_count, value_heads, value_dim, key_dim, q.device)
    output = torch.zeros_like(v)
    final_states = []
    for sequence_index, (start, end) in enumerate(zip(offsets, offsets[1:], strict=False)):
        state_i = None if initial_state is None else initial_state[sequence_index : sequence_index + 1]
        output_i, final_state_i = kda_reference(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            decay[:, start:end],
            beta[:, start:end],
            initial_state=state_i,
            output_final_state=True,
        )
        output[:, start:end] = output_i
        final_states.append(final_state_i)
    final_state = torch.cat(final_states, dim=0)
    return output, final_state if output_final_state else None


def kda_reference_padded(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run left-padded dense batches and zero padding outputs."""
    batch, sequence, _, value_heads, key_dim, value_dim = validate_operator_shapes(q, k, v, decay, beta)
    if attention_mask.shape != (batch, sequence):
        raise ValueError("attention_mask must have shape [B,T]")
    if initial_state is not None:
        validate_state_tensor(initial_state, batch, value_heads, value_dim, key_dim, q.device)
    output = torch.zeros_like(v)
    final_states = []
    for batch_index in range(batch):
        valid = attention_mask[batch_index].bool()
        valid_count = int(valid.sum())
        if valid_count == 0:
            state_i = (
                torch.zeros((1, value_heads, value_dim, key_dim), dtype=torch.float32, device=q.device)
                if initial_state is None
                else initial_state[batch_index : batch_index + 1].clone()
            )
            final_states.append(state_i)
            continue
        first_valid = sequence - valid_count
        if not torch.all(valid[first_valid:]) or torch.any(valid[:first_valid]):
            raise ValueError("Only left padding is supported")
        state_i = None if initial_state is None else initial_state[batch_index : batch_index + 1]
        output_i, final_state_i = kda_reference(
            q[batch_index : batch_index + 1, first_valid:],
            k[batch_index : batch_index + 1, first_valid:],
            v[batch_index : batch_index + 1, first_valid:],
            decay[batch_index : batch_index + 1, first_valid:],
            beta[batch_index : batch_index + 1, first_valid:],
            initial_state=state_i,
            output_final_state=True,
        )
        output[batch_index : batch_index + 1, first_valid:] = output_i
        final_states.append(final_state_i)
    final_state = torch.cat(final_states, dim=0)
    return output, final_state if output_final_state else None


def default_scale(key_dim: int) -> float:
    """Return the P1 query scale."""
    return 1.0 / math.sqrt(key_dim)
