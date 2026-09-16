"""P4 DenseK3 24-KDA/8-GQA text-only causal language model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from densek3_core.kda.fla_backend import FLABackendOptions, fla_chunk_kda, fla_fused_recurrent_kda
from densek3_core.kda.reference import kda_reference_from_raw
from densek3_core.modeling.configuration_densek3 import DenseK3Config
from densek3_core.modeling.hybrid_cache import DenseK3HybridCache, KDACacheLayer


@dataclass
class DenseK3ModelOutput(ModelOutput):
    """Backbone output with the explicit hybrid cache."""

    last_hidden_state: torch.Tensor | None = None
    past_key_values: DenseK3HybridCache | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None


@dataclass
class DenseK3CausalLMOutput(ModelOutput):
    """Causal-LM output used by the P4 evaluator and generation smoke tests."""

    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    past_key_values: DenseK3HybridCache | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None


class DenseK3RMSNorm(nn.Module):
    """Multiplicative RMSNorm; Qwen `(1 + weight)` tensors are converted once."""

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        variance = hidden_states.float().square().mean(dim=-1, keepdim=True)
        normalized = hidden_states.float() * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight.float()).to(dtype)


class DenseK3GatedRMSNorm(nn.Module):
    """Per-value-head gated RMSNorm shared by P4 and P5 KDA."""

    def __init__(
        self,
        head_dim: int,
        eps: float,
        activation: str = "silu",
        alpha: float | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.eps = eps
        if activation not in {"silu", "sigmoid"}:
            raise ValueError(f"Unknown gated RMSNorm activation: {activation}")
        if alpha is not None and not 0.0 <= alpha <= 1.0:
            raise ValueError("Output-gate homotopy alpha must be in [0, 1]")
        self.activation = activation
        self.alpha = alpha

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        variance = hidden_states.float().square().mean(dim=-1, keepdim=True)
        normalized = hidden_states.float() * torch.rsqrt(variance + self.eps)
        normalized = self.weight * normalized.to(dtype)
        gate_float = gate.float()
        if self.alpha is not None:
            activated_gate = (1.0 - self.alpha) * F.silu(gate_float) + self.alpha * torch.sigmoid(
                gate_float
            )
        else:
            activated_gate = F.silu(gate_float) if self.activation == "silu" else torch.sigmoid(gate_float)
        return (normalized * activated_gate).to(dtype)


class DenseK3MLP(nn.Module):
    """Unmodified Qwen SwiGLU MLP."""

    def __init__(self, config: DenseK3Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_partial_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids: torch.LongTensor,
    rotary_dim: int,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply text-mode partial RoPE to `[B,H,T,D]` query and key tensors."""
    if rotary_dim == 0:
        return q, k
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, device=q.device).float() / rotary_dim))
    frequencies = position_ids.float().unsqueeze(-1) * inv_freq
    embedding = torch.cat((frequencies, frequencies), dim=-1).unsqueeze(1)
    cos = embedding.cos().to(q.dtype)
    sin = embedding.sin().to(q.dtype)
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q = torch.cat((q_rot * cos + _rotate_half(q_rot) * sin, q_pass), dim=-1)
    k = torch.cat((k_rot * cos + _rotate_half(k_rot) * sin, k_pass), dim=-1)
    return q, k


class DenseK3GQA(nn.Module):
    """The original gated Qwen GQA parameterization at eight locked layers."""

    def __init__(self, config: DenseK3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.group_size = self.num_heads // self.num_key_value_heads
        self.rotary_dim = config.rotary_dim
        self.rope_theta = config.rope_theta
        self.dropout = config.attention_dropout
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = DenseK3RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = DenseK3RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None,
        cache: DenseK3HybridCache | None,
        use_cache: bool,
    ) -> torch.Tensor:
        batch, sequence, _ = hidden_states.shape
        packed_q = self.q_proj(hidden_states).view(batch, sequence, self.num_heads, 2, self.head_dim)
        q, gate = packed_q.unbind(dim=3)
        q = self.q_norm(q).transpose(1, 2)
        gate = gate.reshape(batch, sequence, self.num_heads * self.head_dim)
        k = self.k_norm(
            self.k_proj(hidden_states).view(batch, sequence, self.num_key_value_heads, self.head_dim)
        ).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, sequence, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        q, k = _apply_partial_rope(q, k, position_ids, self.rotary_dim, self.rope_theta)

        if cache is not None:
            k, v = cache.update_gqa(k, v, self.layer_idx)
        elif use_cache:
            raise ValueError("use_cache=True requires a DenseK3HybridCache")
        total_length = k.shape[2]
        past_length = total_length - sequence
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        if attention_mask is not None:
            if attention_mask.shape[-1] == sequence and past_length:
                prefix = torch.ones(
                    (batch, past_length), device=attention_mask.device, dtype=attention_mask.dtype
                )
                attention_mask = torch.cat((prefix, attention_mask), dim=-1)
            if attention_mask.shape != (batch, total_length):
                raise ValueError("GQA attention_mask must cover the full cached sequence")
        all_tokens_visible = attention_mask is None or bool(attention_mask.bool().all())
        use_native_causal = all_tokens_visible and past_length == 0 and sequence > 1
        no_mask_decode = all_tokens_visible and sequence == 1
        if use_native_causal or no_mask_decode:
            sdpa_mask = None
        else:
            query_positions = past_length + torch.arange(sequence, device=hidden_states.device)
            key_positions = torch.arange(total_length, device=hidden_states.device)
            allowed = key_positions.view(1, 1, 1, -1) <= query_positions.view(1, 1, -1, 1)
            if attention_mask is not None:
                allowed = allowed & attention_mask[:, None, None, :].bool()
            sdpa_mask = allowed
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=use_native_causal,
        )
        output = output.transpose(1, 2).contiguous().view(batch, sequence, -1)
        return self.o_proj(output * torch.sigmoid(gate))


def _expanded_nope_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    past_length: int,
    dropout: float,
    training: bool,
) -> torch.Tensor:
    """Reference SDPA used by P6 MLA before P7 introduces a latent cache."""
    batch, _, sequence, _ = q.shape
    total_length = k.shape[2]
    if attention_mask is not None:
        if attention_mask.shape[-1] == sequence and past_length:
            prefix = torch.ones(
                (batch, past_length),
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat((prefix, attention_mask), dim=-1)
        if attention_mask.shape != (batch, total_length):
            raise ValueError("MLA attention_mask must cover the full expanded cache")
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


class DenseK3MLA(nn.Module):
    """Native P6 strict-NoPE rank-512 MLA with an expanded reference cache."""

    position_encoding = "none"
    q_lora_rank = None
    qk_pos_head_dim = 0
    cache_mode = "expanded_reference"

    def __init__(self, config: DenseK3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.head_dim = config.mla_qk_nope_head_dim
        self.qk_nope_head_dim = config.mla_qk_nope_head_dim
        self.value_head_dim = config.mla_value_head_dim
        self.kv_lora_rank = config.mla_kv_lora_rank
        self.apply_qk_norm = config.mla_qk_norm
        self.dropout = config.attention_dropout
        width = self.num_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, width, bias=False)
        self.g_proj = nn.Linear(config.hidden_size, width, bias=False)
        self.kv_a_proj = nn.Linear(config.hidden_size, self.kv_lora_rank, bias=False)
        self.kv_a_layernorm = DenseK3RMSNorm(self.kv_lora_rank, config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.head_dim + self.value_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(self.num_heads * self.value_head_dim, config.hidden_size, bias=False)
        if self.apply_qk_norm:
            self.q_norm = DenseK3RMSNorm(self.head_dim, config.rms_norm_eps)
            self.k_norm = DenseK3RMSNorm(self.head_dim, config.rms_norm_eps)

    def project_kv(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, sequence, _ = hidden_states.shape
        latent = self.kv_a_proj(hidden_states)
        projected = self.kv_a_layernorm(latent)
        kv = self.kv_b_proj(projected).view(
            batch,
            sequence,
            self.num_heads,
            self.head_dim + self.value_head_dim,
        )
        k, v = torch.split(kv, (self.head_dim, self.value_head_dim), dim=-1)
        return k, v, latent

    def project_qkvg(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, sequence, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch, sequence, self.num_heads, self.head_dim)
        gate = self.g_proj(hidden_states)
        k, v, latent = self.project_kv(hidden_states)
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
        batch, sequence, _ = hidden_states.shape
        q, k, v, gate, _ = self.project_qkvg(hidden_states)
        if cache is not None:
            k, v = cache.update_gqa(k, v, self.layer_idx)
        elif use_cache:
            raise ValueError("use_cache=True requires a DenseK3HybridCache")
        past_length = k.shape[2] - sequence
        output = _expanded_nope_attention(
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


def _causal_depthwise_conv(
    projected: torch.Tensor,
    conv: nn.Conv1d,
    previous_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal depthwise convolution with an explicit `kernel_size - 1` cache."""
    kernel_size = conv.kernel_size[0]
    x = projected.transpose(1, 2)
    if previous_state is None:
        padded = F.pad(x, (kernel_size - 1, 0))
        output = F.conv1d(
            x,
            conv.weight,
            bias=None,
            padding=kernel_size - 1,
            groups=x.shape[1],
        )[..., : x.shape[-1]]
    else:
        expected = (x.shape[0], x.shape[1], kernel_size - 1)
        if tuple(previous_state.shape) != expected:
            raise ValueError(f"Invalid convolution state shape: expected {expected}, got {tuple(previous_state.shape)}")
        padded = torch.cat((previous_state.to(x.dtype), x), dim=-1)
        output = F.conv1d(padded, conv.weight, bias=None, groups=x.shape[1])
    output = output.transpose(1, 2)
    new_state = padded[..., -(kernel_size - 1) :].detach()
    return F.silu(output), new_state


class DenseK3KDA(nn.Module):
    """KDA mixer with the locked P4 compatibility and P5 target modes."""

    def __init__(self, config: DenseK3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.qk_heads = config.kda_qk_num_heads
        self.value_heads = config.kda_value_num_heads
        self.key_dim = config.kda_qk_head_dim
        self.value_dim = config.kda_value_head_dim
        qk_width = self.qk_heads * self.key_dim
        value_width = self.value_heads * self.value_dim
        kernel = config.kda_conv_kernel_size
        # P4 production values are always transplanted. Zeros keep small randomly
        # initialized test models finite without adding a hidden conversion path.
        self.A_log = nn.Parameter(torch.zeros(self.value_heads))
        self.dt_bias = nn.Parameter(torch.zeros(self.value_heads * self.key_dim))
        self.q_proj = nn.Linear(config.hidden_size, qk_width, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, qk_width, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, value_width, bias=False)
        self.q_conv1d = nn.Conv1d(qk_width, qk_width, kernel, groups=qk_width, bias=False)
        self.k_conv1d = nn.Conv1d(qk_width, qk_width, kernel, groups=qk_width, bias=False)
        self.v_conv1d = nn.Conv1d(value_width, value_width, kernel, groups=value_width, bias=False)
        self.f_a_proj = nn.Linear(config.hidden_size, config.kda_decay_projection_rank, bias=False)
        self.f_b_proj = nn.Linear(
            config.kda_decay_projection_rank, self.value_heads * self.key_dim, bias=False
        )
        self.b_proj = nn.Linear(config.hidden_size, self.value_heads, bias=False)
        self.g_proj = nn.Linear(config.hidden_size, value_width, bias=False)
        self.o_norm = DenseK3GatedRMSNorm(
            self.value_dim,
            config.rms_norm_eps,
            activation=config.kda_output_gate_activation,
            alpha=config.kda_output_gate_alpha,
        )
        self.o_proj = nn.Linear(value_width, config.hidden_size, bias=False)

    def _operator(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        raw_decay: torch.Tensor,
        beta_logits: torch.Tensor,
        initial_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        backend = self.config.kda_backend
        if backend == "fla":
            options = FLABackendOptions(gate_mode=self.config.kda_gate_mode)
            operator = fla_fused_recurrent_kda if q.shape[1] == 1 and initial_state is not None else fla_chunk_kda
            output, final_state = operator(
                q,
                k,
                v,
                raw_decay,
                beta_logits,
                self.A_log,
                self.dt_bias,
                initial_state=initial_state,
                output_final_state=True,
                options=options,
            )
        elif backend == "reference":
            output, final_state = kda_reference_from_raw(
                q,
                k,
                v,
                raw_decay,
                beta_logits,
                self.A_log,
                self.dt_bias,
                gate_mode=self.config.kda_gate_mode,
                initial_state=initial_state,
                output_final_state=True,
            )
        else:
            raise ValueError(f"Unknown KDA backend: {backend}")
        if final_state is None:
            raise RuntimeError("KDA backend did not return its requested final state")
        return output, final_state.float()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cache: DenseK3HybridCache | None,
        use_cache: bool,
    ) -> torch.Tensor:
        batch, sequence, _ = hidden_states.shape
        current_mask = None
        if attention_mask is not None:
            current_mask = attention_mask[:, -sequence:].to(hidden_states.dtype)
            hidden_states = hidden_states * current_mask.unsqueeze(-1)
        cache_layer: KDACacheLayer | None = cache.kda_layer(self.layer_idx) if cache is not None else None
        q, q_state = _causal_depthwise_conv(
            self.q_proj(hidden_states), self.q_conv1d, None if cache_layer is None else cache_layer.q_conv_state
        )
        k, k_state = _causal_depthwise_conv(
            self.k_proj(hidden_states), self.k_conv1d, None if cache_layer is None else cache_layer.k_conv_state
        )
        v, v_state = _causal_depthwise_conv(
            self.v_proj(hidden_states), self.v_conv1d, None if cache_layer is None else cache_layer.v_conv_state
        )
        q = q.view(batch, sequence, self.qk_heads, self.key_dim)
        k = k.view(batch, sequence, self.qk_heads, self.key_dim)
        v = v.view(batch, sequence, self.value_heads, self.value_dim)
        raw_decay = self.f_b_proj(self.f_a_proj(hidden_states)).view(
            batch, sequence, self.value_heads, self.key_dim
        )
        beta_logits = self.b_proj(hidden_states)
        initial_state = None if cache_layer is None else cache_layer.recurrent_state
        core, final_state = self._operator(q, k, v, raw_decay, beta_logits, initial_state)
        gate = self.g_proj(hidden_states).view(batch, sequence, self.value_heads, self.value_dim)
        output = self.o_norm(core, gate).reshape(batch, sequence, -1)
        output = self.o_proj(output)
        if current_mask is not None:
            output = output * current_mask.unsqueeze(-1)
        if cache_layer is not None:
            cache_layer.q_conv_state = q_state
            cache_layer.k_conv_state = k_state
            cache_layer.v_conv_state = v_state
            cache_layer.recurrent_state = final_state.detach()
        elif use_cache:
            raise ValueError("use_cache=True requires a DenseK3HybridCache")
        return output


class DenseK3DecoderLayer(nn.Module):
    """Unmodified Qwen PreNorm residual block around a selected P4 mixer."""

    def __init__(self, config: DenseK3Config, layer_idx: int):
        super().__init__()
        layer_type = config.mixer_types[layer_idx]
        self.layer_type = layer_type
        self.input_layernorm = DenseK3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = DenseK3RMSNorm(config.hidden_size, config.rms_norm_eps)
        if layer_type == "kda":
            self.self_attn = DenseK3KDA(config, layer_idx)
        elif layer_type == "gqa":
            self.self_attn = DenseK3GQA(config, layer_idx)
        else:
            self.self_attn = DenseK3MLA(config, layer_idx)
        self.mlp = DenseK3MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None,
        cache: DenseK3HybridCache | None,
        use_cache: bool,
    ) -> torch.Tensor:
        residual = hidden_states
        normalized = self.input_layernorm(hidden_states)
        if self.layer_type == "kda":
            mixed = self.self_attn(normalized, attention_mask, cache, use_cache)
        else:
            mixed = self.self_attn(normalized, position_ids, attention_mask, cache, use_cache)
        hidden_states = residual + mixed
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class DenseK3PreTrainedModel(PreTrainedModel):
    """Shared Hugging Face integration for P4 model classes."""

    config_class = DenseK3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DenseK3DecoderLayer"]

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, (DenseK3RMSNorm, DenseK3GatedRMSNorm)):
            nn.init.ones_(module.weight)


class DenseK3Model(DenseK3PreTrainedModel):
    """Text-only P4 hybrid backbone."""

    def __init__(self, config: DenseK3Config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [DenseK3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = DenseK3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: DenseK3HybridCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        hidden_state_layers: list[int] | tuple[int, ...] | None = None,
        return_dict: bool | None = None,
        **_: Any,
    ) -> DenseK3ModelOutput | tuple[Any, ...]:
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if self.training and self.gradient_checkpointing and use_cache:
            use_cache = False
        output_hidden_states = False if output_hidden_states is None else output_hidden_states
        selected_layers = tuple(hidden_state_layers or ())
        if output_hidden_states and selected_layers:
            raise ValueError("output_hidden_states and hidden_state_layers are mutually exclusive")
        if len(set(selected_layers)) != len(selected_layers) or any(
            index < 0 or index >= self.config.num_hidden_layers for index in selected_layers
        ):
            raise ValueError("hidden_state_layers must contain unique valid layer indices")
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        hidden_states = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        batch, sequence, _ = hidden_states.shape
        if use_cache and past_key_values is None:
            past_key_values = DenseK3HybridCache(self.config)
        past_length = 0 if past_key_values is None else past_key_values.seen_tokens
        if position_ids is None:
            position_ids = torch.arange(
                past_length, past_length + sequence, device=hidden_states.device, dtype=torch.long
            ).unsqueeze(0).expand(batch, -1)
        if attention_mask is None:
            attention_mask = torch.ones((batch, past_length + sequence), device=hidden_states.device)
        all_hidden: list[torch.Tensor] | None = [] if output_hidden_states or selected_layers else None
        if all_hidden is not None and output_hidden_states:
            all_hidden.append(hidden_states)
        for layer_index, layer in enumerate(self.layers):
            if self.training and self.gradient_checkpointing:
                if past_key_values is not None:
                    raise ValueError("Gradient-checkpointed training requires an empty cache")

                def custom_forward(
                    states: torch.Tensor,
                    positions: torch.Tensor,
                    mask: torch.Tensor,
                    current_layer: DenseK3DecoderLayer = layer,
                ) -> torch.Tensor:
                    return current_layer(states, positions, mask, None, False)

                hidden_states = checkpoint(
                    custom_forward,
                    hidden_states,
                    position_ids,
                    attention_mask,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(hidden_states, position_ids, attention_mask, past_key_values, use_cache)
            if all_hidden is not None and (output_hidden_states or layer_index in selected_layers):
                all_hidden.append(hidden_states)
        hidden_states = self.norm(hidden_states)
        if all_hidden is not None and output_hidden_states:
            all_hidden[-1] = hidden_states
        if use_cache and past_key_values is not None:
            past_key_values.advance(sequence)
        output = DenseK3ModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=None if all_hidden is None else tuple(all_hidden),
        )
        if return_dict:
            return output
        return output.to_tuple()


class DenseK3ForCausalLM(DenseK3PreTrainedModel):
    """P4 DenseK3 language model with a tied Qwen vocabulary head."""

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: DenseK3Config):
        super().__init__(config)
        self.model = DenseK3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        self.tie_weights()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Module) -> None:
        self.lm_head = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: DenseK3HybridCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | None = None,
        **kwargs: Any,
    ) -> DenseK3CausalLMOutput | tuple[Any, ...]:
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        backbone = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        hidden = backbone.last_hidden_state
        hidden_for_logits = hidden[:, -logits_to_keep:] if logits_to_keep is not None and logits_to_keep > 0 else hidden
        logits = self.lm_head(hidden_for_logits).float()
        loss = None
        if labels is not None:
            if logits.shape[1] != hidden.shape[1]:
                raise ValueError("labels require logits for the complete sequence")
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, self.config.vocab_size),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        output = DenseK3CausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=backbone.past_key_values,
            hidden_states=backbone.hidden_states,
        )
        if return_dict:
            return output
        return output.to_tuple()

    @torch.no_grad()
    def greedy_generate(
        self,
        input_ids: torch.LongTensor,
        *,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 16,
        eos_token_id: int | list[int] | None = None,
    ) -> torch.LongTensor:
        """Minimal deterministic greedy generation used by the P4.7 smoke gate."""
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        generated = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        cache: DenseK3HybridCache | None = None
        current = input_ids
        eos = self.config.eos_token_id if eos_token_id is None else eos_token_id
        eos_values = {eos} if isinstance(eos, int) else set(eos or [])
        for _ in range(max_new_tokens):
            output = self(
                input_ids=current,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            cache = output.past_key_values
            current = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, current), dim=-1)
            attention_mask = torch.cat((attention_mask, torch.ones_like(current)), dim=-1)
            if eos_values and all(int(token) in eos_values for token in current.flatten()):
                break
        return generated


__all__ = [
    "DenseK3CausalLMOutput",
    "DenseK3DecoderLayer",
    "DenseK3ForCausalLM",
    "DenseK3GQA",
    "DenseK3KDA",
    "DenseK3Model",
    "DenseK3ModelOutput",
    "DenseK3RMSNorm",
]
