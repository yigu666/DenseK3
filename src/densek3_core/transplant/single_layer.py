"""Auditable donor and target single-layer mixers for P3 parity."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from densek3_core.kda.reference import kda_reference, l2_normalize_kda, qwen_compat_decay
from densek3_core.transplant.gdn_to_kda import GDNBridgeConfig, audit_source_state, audit_target_state


@dataclass(frozen=True)
class MixerResult:
    """Mixer output plus the recurrence quantities compared by P3."""

    output: torch.Tensor
    final_state: torch.Tensor
    decay: torch.Tensor
    beta: torch.Tensor


@dataclass(frozen=True)
class MixerTrace:
    """Named tensors needed to isolate every P3.4 projection boundary."""

    q_projection: torch.Tensor
    k_projection: torch.Tensor
    v_projection: torch.Tensor
    q_conv: torch.Tensor
    k_conv: torch.Tensor
    v_conv: torch.Tensor
    normalized_q: torch.Tensor
    normalized_k: torch.Tensor
    raw_a_logits: torch.Tensor
    f_a_intermediate: torch.Tensor | None
    channel_a_logits: torch.Tensor
    decay: torch.Tensor
    beta_logits: torch.Tensor
    beta: torch.Tensor
    recurrent_core: torch.Tensor
    final_state: torch.Tensor
    z_logits: torch.Tensor
    gated_norm: torch.Tensor
    output: torch.Tensor

    def result(self) -> MixerResult:
        return MixerResult(
            output=self.output,
            final_state=self.final_state,
            decay=self.decay,
            beta=self.beta,
        )


def causal_depthwise_conv(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Apply the donor's causal depthwise convolution and SiLU."""
    sequence = hidden.shape[1]
    channels = hidden.shape[2]
    mixed = F.conv1d(
        hidden.transpose(1, 2).to(weight.dtype),
        weight,
        padding=weight.shape[-1] - 1,
        groups=channels,
    )[:, :, :sequence]
    return F.silu(mixed).transpose(1, 2).to(hidden.dtype)


def gated_rms_norm(
    hidden: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Match Qwen3.5 gated RMSNorm operation order."""
    input_dtype = hidden.dtype
    normalized = hidden.float()
    variance = normalized.square().mean(dim=-1, keepdim=True)
    normalized = normalized * torch.rsqrt(variance + eps)
    normalized = weight * normalized.to(input_dtype)
    normalized = normalized * F.silu(gate.float())
    return normalized.to(input_dtype)


def _state_on(state: dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    converted = {}
    for name, tensor in state.items():
        target_dtype = torch.float32 if name in {"A_log", "dt_bias"} else dtype
        converted[name] = tensor.to(device=device, dtype=target_dtype)
    return converted


def _reshape_qkv(
    config: GDNBridgeConfig,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = q.view(*q.shape[:2], config.qk_num_heads, config.qk_head_dim)
    k = k.view(*k.shape[:2], config.qk_num_heads, config.qk_head_dim)
    v = v.view(*v.shape[:2], config.value_num_heads, config.value_head_dim)
    return q, k, v


def _finish_mixer(
    config: GDNBridgeConfig,
    hidden_states: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    gate_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    output_weight: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    core, final_state = kda_reference(
        q,
        k,
        v,
        decay,
        beta,
        initial_state=initial_state,
        output_final_state=True,
    )
    z = F.linear(hidden_states, gate_weight)
    z = z.view(*z.shape[:2], config.value_num_heads, config.value_head_dim)
    normalized = gated_rms_norm(core, z, norm_weight, config.rms_norm_eps)
    output = F.linear(normalized.flatten(2), output_weight)
    return core, final_state, z, normalized, output


class QwenGDNReferenceMixer:
    """Exact Qwen3.5 GDN equations driven by a real layer state dict."""

    def __init__(self, config: GDNBridgeConfig, source_state: dict[str, torch.Tensor]):
        audit_source_state(source_state, config)
        self.config = config
        self.state = source_state

    def to(self, device: torch.device | str, dtype: torch.dtype) -> QwenGDNReferenceMixer:
        device = torch.device(device)
        return QwenGDNReferenceMixer(self.config, _state_on(self.state, device, dtype))

    def gates(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = F.linear(hidden_states, self.state["in_proj_a.weight"])
        b = F.linear(hidden_states, self.state["in_proj_b.weight"])
        scalar_decay = -self.state["A_log"].float().exp() * F.softplus(a.float() + self.state["dt_bias"])
        decay = scalar_decay.unsqueeze(-1).expand(-1, -1, -1, self.config.qk_head_dim)
        return decay, torch.sigmoid(b)

    def forward_from_recurrence_inputs(
        self,
        hidden_states: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        decay: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> MixerResult:
        _, final_state, _, _, output = _finish_mixer(
            self.config,
            hidden_states,
            q,
            k,
            v,
            decay,
            beta,
            self.state["in_proj_z.weight"],
            self.state["norm.weight"],
            self.state["out_proj.weight"],
            initial_state,
        )
        return MixerResult(output=output, final_state=final_state, decay=decay, beta=beta)

    def trace(
        self,
        hidden_states: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> MixerTrace:
        config = self.config
        fused_projection = F.linear(hidden_states, self.state["in_proj_qkv.weight"])
        q_projection, k_projection, v_projection = torch.split(
            fused_projection,
            [config.qk_dim, config.qk_dim, config.value_dim],
            dim=-1,
        )
        fused_conv = causal_depthwise_conv(fused_projection, self.state["conv1d.weight"])
        q_conv, k_conv, v_conv = torch.split(
            fused_conv,
            [config.qk_dim, config.qk_dim, config.value_dim],
            dim=-1,
        )
        q, k, v = _reshape_qkv(config, q_conv, k_conv, v_conv)
        raw_a = F.linear(hidden_states, self.state["in_proj_a.weight"])
        channel_a = raw_a.unsqueeze(-1).expand(-1, -1, -1, config.qk_head_dim)
        decay, beta = self.gates(hidden_states)
        beta_logits = F.linear(hidden_states, self.state["in_proj_b.weight"])
        core, final_state, z, normalized, output = _finish_mixer(
            config,
            hidden_states,
            q,
            k,
            v,
            decay,
            beta,
            self.state["in_proj_z.weight"],
            self.state["norm.weight"],
            self.state["out_proj.weight"],
            initial_state,
        )
        return MixerTrace(
            q_projection=q_projection,
            k_projection=k_projection,
            v_projection=v_projection,
            q_conv=q_conv,
            k_conv=k_conv,
            v_conv=v_conv,
            normalized_q=l2_normalize_kda(q),
            normalized_k=l2_normalize_kda(k),
            raw_a_logits=raw_a,
            f_a_intermediate=None,
            channel_a_logits=channel_a,
            decay=decay,
            beta_logits=beta_logits,
            beta=beta,
            recurrent_core=core,
            final_state=final_state,
            z_logits=z,
            gated_norm=normalized,
            output=output,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> MixerResult:
        return self.trace(hidden_states, initial_state).result()


class DenseK3QwenCompatMixer:
    """DenseK3 KDA mixer restricted to the P4 qwen_compat bridge."""

    def __init__(self, config: GDNBridgeConfig, target_state: dict[str, torch.Tensor]):
        audit_target_state(target_state, config)
        self.config = config
        self.state = target_state

    def to(self, device: torch.device | str, dtype: torch.dtype) -> DenseK3QwenCompatMixer:
        device = torch.device(device)
        return DenseK3QwenCompatMixer(self.config, _state_on(self.state, device, dtype))

    def gates(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = F.linear(hidden_states, self.state["f_a_proj.weight"])
        raw_decay = F.linear(latent, self.state["f_b_proj.weight"])
        raw_decay = raw_decay.view(*raw_decay.shape[:2], self.config.value_num_heads, self.config.qk_head_dim)
        decay = qwen_compat_decay(raw_decay, self.state["A_log"], self.state["dt_bias"])
        beta = torch.sigmoid(F.linear(hidden_states, self.state["b_proj.weight"]))
        return decay, beta

    def convolve_projections(
        self,
        q_projection: torch.Tensor,
        k_projection: torch.Tensor,
        v_projection: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_conv = causal_depthwise_conv(q_projection, self.state["q_conv1d.weight"])
        k_conv = causal_depthwise_conv(k_projection, self.state["k_conv1d.weight"])
        v_conv = causal_depthwise_conv(v_projection, self.state["v_conv1d.weight"])
        return _reshape_qkv(self.config, q_conv, k_conv, v_conv)

    def forward_from_recurrence_inputs(
        self,
        hidden_states: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        decay: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> MixerResult:
        _, final_state, _, _, output = _finish_mixer(
            self.config,
            hidden_states,
            q,
            k,
            v,
            decay,
            beta,
            self.state["g_proj.weight"],
            self.state["o_norm.weight"],
            self.state["o_proj.weight"],
            initial_state,
        )
        return MixerResult(output=output, final_state=final_state, decay=decay, beta=beta)

    def trace(
        self,
        hidden_states: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> MixerTrace:
        config = self.config
        q_projection = F.linear(hidden_states, self.state["q_proj.weight"])
        k_projection = F.linear(hidden_states, self.state["k_proj.weight"])
        v_projection = F.linear(hidden_states, self.state["v_proj.weight"])
        q_conv = causal_depthwise_conv(q_projection, self.state["q_conv1d.weight"])
        k_conv = causal_depthwise_conv(k_projection, self.state["k_conv1d.weight"])
        v_conv = causal_depthwise_conv(v_projection, self.state["v_conv1d.weight"])
        q, k, v = _reshape_qkv(config, q_conv, k_conv, v_conv)
        f_a = F.linear(hidden_states, self.state["f_a_proj.weight"])
        channel_a_flat = F.linear(f_a, self.state["f_b_proj.weight"])
        channel_a = channel_a_flat.view(*channel_a_flat.shape[:2], config.value_num_heads, config.qk_head_dim)
        decay = qwen_compat_decay(channel_a, self.state["A_log"], self.state["dt_bias"])
        beta_logits = F.linear(hidden_states, self.state["b_proj.weight"])
        beta = torch.sigmoid(beta_logits)
        core, final_state, z, normalized, output = _finish_mixer(
            config,
            hidden_states,
            q,
            k,
            v,
            decay,
            beta,
            self.state["g_proj.weight"],
            self.state["o_norm.weight"],
            self.state["o_proj.weight"],
            initial_state,
        )
        return MixerTrace(
            q_projection=q_projection,
            k_projection=k_projection,
            v_projection=v_projection,
            q_conv=q_conv,
            k_conv=k_conv,
            v_conv=v_conv,
            normalized_q=l2_normalize_kda(q),
            normalized_k=l2_normalize_kda(k),
            raw_a_logits=f_a[..., : config.value_num_heads],
            f_a_intermediate=f_a,
            channel_a_logits=channel_a,
            decay=decay,
            beta_logits=beta_logits,
            beta=beta,
            recurrent_core=core,
            final_state=final_state,
            z_logits=z,
            gated_norm=normalized,
            output=output,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> MixerResult:
        return self.trace(hidden_states, initial_state).result()
