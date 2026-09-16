"""Thin adapters from DenseK3 tensor contracts to FLA KDA operators."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from densek3_core.kda.contracts import validate_operator_shapes
from densek3_core.kda.state import validate_state_tensor


class FLANotAvailableError(RuntimeError):
    """Raised when the pinned FLA runtime cannot be imported."""


@dataclass(frozen=True)
class FLABackendOptions:
    """P1-locked fused gate options."""

    gate_mode: str = "qwen_compat"
    allow_negative_eigenvalues: bool = False
    state_v_first: bool = True

    @property
    def safe_gate(self) -> bool:
        return self.gate_mode == "kimi_k3"

    @property
    def lower_bound(self) -> float | None:
        return -5.0 if self.gate_mode == "kimi_k3" else None

    def validate(self) -> None:
        if self.gate_mode not in {"qwen_compat", "kimi_k3"}:
            raise ValueError(f"Unknown KDA gate mode: {self.gate_mode}")
        if self.allow_negative_eigenvalues:
            raise ValueError("DenseK3 P1 disables negative eigenvalues")
        if not self.state_v_first:
            raise ValueError("DenseK3 P1 requires V-first recurrent state")


def _load_fla_operators():
    try:
        from fla.ops.kda import chunk_kda, fused_recurrent_kda
    except (ImportError, ModuleNotFoundError) as exc:
        raise FLANotAvailableError("FLA KDA operators are unavailable in this Python environment") from exc
    return chunk_kda, fused_recurrent_kda


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_decay: torch.Tensor,
    beta_logits: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
) -> None:
    batch, _, _, value_heads, key_dim, value_dim = validate_operator_shapes(q, k, v, raw_decay, beta_logits)
    if value_heads // q.shape[2] != 2:
        raise ValueError("DenseK3 P1 requires an explicit 2:1 GVA mapping")
    if initial_state is not None:
        state_batch = batch if cu_seqlens is None else cu_seqlens.numel() - 1
        validate_state_tensor(initial_state, state_batch, value_heads, value_dim, key_dim, q.device)


def fla_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_decay: torch.Tensor,
    beta_logits: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    *,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    options: FLABackendOptions | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Call FLA chunk KDA using only P1-locked options."""
    options = options or FLABackendOptions()
    options.validate()
    _validate_inputs(q, k, v, raw_decay, beta_logits, initial_state, cu_seqlens)
    chunk_kda, _ = _load_fla_operators()
    return chunk_kda(
        q=q,
        k=k,
        v=v,
        g=raw_decay,
        beta=beta_logits,
        A_log=a_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=options.allow_negative_eigenvalues,
        safe_gate=options.safe_gate,
        lower_bound=options.lower_bound,
        state_v_first=options.state_v_first,
        cu_seqlens=cu_seqlens,
    )


def fla_fused_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_decay: torch.Tensor,
    beta_logits: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    *,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    options: FLABackendOptions | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Call FLA fused recurrent KDA using only P1-locked options."""
    options = options or FLABackendOptions()
    options.validate()
    _validate_inputs(q, k, v, raw_decay, beta_logits, initial_state, cu_seqlens)
    _, fused_recurrent_kda = _load_fla_operators()
    return fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=raw_decay,
        beta=beta_logits,
        A_log=a_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=options.allow_negative_eigenvalues,
        lower_bound=options.lower_bound,
        state_v_first=options.state_v_first,
        cu_seqlens=cu_seqlens,
    )
