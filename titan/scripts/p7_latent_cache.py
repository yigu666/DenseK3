"""Titan-only true persistent latent cache for the frozen P6 MLA model.

The P6 checkpoint and canonical DenseK3 sources remain immutable.  This module
patches only the current Python process: selected MLA layers persist the
post-latent-RMSNorm rank-512 sequence and reconstruct temporary K/V tensors for
reference attention.  Expanded K/V tensors never enter the persistent cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from densek3_core.modeling.hybrid_cache import GQACacheLayer, KDACacheLayer


@dataclass
class LatentCacheLayer:
    """Persistent MLA cache containing only `[B,T,R]` normalized latents."""

    latent: torch.Tensor | None = None
    latent_dim: int = 512
    append_lengths: list[int] = field(default_factory=list)

    @property
    def initialized(self) -> bool:
        return self.latent is not None

    @property
    def sequence_length(self) -> int:
        return 0 if self.latent is None else int(self.latent.shape[1])

    def update(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 3 or latent.shape[-1] != self.latent_dim:
            raise ValueError(
                "MLA latent cache updates must use [B,T,R] with "
                f"R={self.latent_dim}; observed {tuple(latent.shape)}"
            )
        detached = latent.detach()
        if self.latent is None:
            self.latent = detached
        else:
            if (
                self.latent.shape[0] != detached.shape[0]
                or self.latent.shape[2] != detached.shape[2]
                or self.latent.device != detached.device
                or self.latent.dtype != detached.dtype
            ):
                raise ValueError("MLA latent cache update is incompatible with existing state")
            self.latent = torch.cat((self.latent, detached), dim=1)
        self.append_lengths.append(int(detached.shape[1]))
        return self.latent

    def reorder(self, beam_idx: torch.LongTensor) -> None:
        if self.latent is not None:
            self.latent = self.latent.index_select(0, beam_idx.to(self.latent.device))

    def repeat_interleave(self, repeats: int) -> None:
        if self.latent is not None:
            self.latent = self.latent.repeat_interleave(repeats, dim=0)

    def reset(self) -> None:
        self.latent = None
        self.append_lengths.clear()

    def clone(self) -> LatentCacheLayer:
        return LatentCacheLayer(
            latent=None if self.latent is None else self.latent.clone(),
            latent_dim=self.latent_dim,
            append_lengths=list(self.append_lengths),
        )

    def segments(self) -> tuple[torch.Tensor, ...]:
        """Return latent views using the original cache-update boundaries."""
        if self.latent is None:
            return ()
        if sum(self.append_lengths) != self.sequence_length:
            raise RuntimeError("MLA latent append boundaries do not cover the cache")
        segments = []
        start = 0
        for length in self.append_lengths:
            segments.append(self.latent[:, start : start + length])
            start += length
        return tuple(segments)

    def bytes(self) -> int:
        if self.latent is None:
            return 0
        return self.latent.numel() * self.latent.element_size()


class TitanP7HybridCache:
    """Hybrid cache with KDA recurrent states and selected MLA latent histories."""

    is_compileable = False
    latent_layer_indices: tuple[int, ...] = ()

    def __init__(self, config: Any):
        selected = set(self.latent_layer_indices or tuple(config.mla_indices))
        if not selected.issubset(set(config.mla_indices)):
            raise ValueError("P7 latent layers must be a subset of P6 MLA layers")
        self.mixer_types = tuple(config.mixer_types)
        self.mla_indices = tuple(config.mla_indices)
        self.layers: list[KDACacheLayer | GQACacheLayer | LatentCacheLayer] = []
        for layer_idx, layer_type in enumerate(self.mixer_types):
            if layer_type == "kda":
                layer: KDACacheLayer | GQACacheLayer | LatentCacheLayer = (
                    KDACacheLayer(
                        recurrent_state_layout=config.recurrent_state_layout
                    )
                )
            elif layer_type == "mla" and layer_idx in selected:
                layer = LatentCacheLayer(latent_dim=config.mla_kv_lora_rank)
            else:
                layer = GQACacheLayer()
            self.layers.append(layer)
        self._seen_tokens = 0

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, layer_idx: int):
        return self.layers[layer_idx]

    @property
    def seen_tokens(self) -> int:
        return self._seen_tokens

    def get_seq_length(self, layer_idx: int = 0) -> int:
        del layer_idx
        return self._seen_tokens

    def get_max_cache_shape(self) -> None:
        return None

    def get_max_length(self) -> None:
        return None

    def has_previous_state(self, layer_idx: int | None = None) -> bool:
        if layer_idx is None:
            return self._seen_tokens > 0
        return self.layers[layer_idx].initialized

    def kda_layer(self, layer_idx: int) -> KDACacheLayer:
        layer = self.layers[layer_idx]
        if not isinstance(layer, KDACacheLayer):
            raise TypeError(f"Layer {layer_idx} is not a KDA cache layer")
        return layer

    def gqa_layer(self, layer_idx: int) -> GQACacheLayer:
        layer = self.layers[layer_idx]
        if not isinstance(layer, GQACacheLayer):
            raise TypeError(f"Layer {layer_idx} is not an expanded cache layer")
        return layer

    def latent_layer(self, layer_idx: int) -> LatentCacheLayer:
        layer = self.layers[layer_idx]
        if not isinstance(layer, LatentCacheLayer):
            raise TypeError(f"Layer {layer_idx} is not a latent cache layer")
        return layer

    def update_gqa(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.gqa_layer(layer_idx).update(key, value)

    def update_latent(self, latent: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return self.latent_layer(layer_idx).update(latent)

    def advance(self, token_count: int) -> None:
        if token_count < 0:
            raise ValueError("token_count must be non-negative")
        self._seen_tokens += token_count

    def reorder_cache(self, beam_idx: torch.LongTensor) -> TitanP7HybridCache:
        for layer in self.layers:
            layer.reorder(beam_idx)
        return self

    def batch_repeat_interleave(self, repeats: int) -> None:
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        for layer in self.layers:
            if isinstance(layer, KDACacheLayer):
                names = (
                    "q_conv_state",
                    "k_conv_state",
                    "v_conv_state",
                    "recurrent_state",
                )
                for name in names:
                    tensor = getattr(layer, name)
                    if tensor is not None:
                        setattr(
                            layer,
                            name,
                            tensor.repeat_interleave(repeats, dim=0),
                        )
            elif isinstance(layer, LatentCacheLayer):
                layer.repeat_interleave(repeats)
            else:
                for name in ("key", "value"):
                    tensor = getattr(layer, name)
                    if tensor is not None:
                        setattr(
                            layer,
                            name,
                            tensor.repeat_interleave(repeats, dim=0),
                        )

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        self.reorder_cache(indices)

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()
        self._seen_tokens = 0

    def crop(self, max_length: int) -> None:
        raise NotImplementedError(
            f"DenseK3 recurrent cache cannot be cropped (requested max_length={max_length})"
        )

    def clone(self) -> TitanP7HybridCache:
        copied = object.__new__(TitanP7HybridCache)
        copied.mixer_types = self.mixer_types
        copied.mla_indices = self.mla_indices
        copied._seen_tokens = self._seen_tokens
        copied.layers = []
        for layer in self.layers:
            if isinstance(layer, LatentCacheLayer):
                copied.layers.append(layer.clone())
            elif isinstance(layer, KDACacheLayer):
                copied.layers.append(
                    KDACacheLayer(
                        **{
                            name: (
                                getattr(layer, name)
                                if not isinstance(getattr(layer, name), torch.Tensor)
                                else getattr(layer, name).clone()
                            )
                            for name in (
                                "q_conv_state",
                                "k_conv_state",
                                "v_conv_state",
                                "recurrent_state",
                                "recurrent_state_layout",
                            )
                        }
                    )
                )
            else:
                copied.layers.append(
                    GQACacheLayer(
                        key=None if layer.key is None else layer.key.clone(),
                        value=None if layer.value is None else layer.value.clone(),
                    )
                )
        return copied

    def cache_bytes(self) -> dict[str, int]:
        kda = 0
        expanded_k = 0
        expanded_v = 0
        expanded_k_numel = 0
        expanded_v_numel = 0
        latent = 0
        for layer in self.layers:
            if isinstance(layer, KDACacheLayer):
                for name in (
                    "q_conv_state",
                    "k_conv_state",
                    "v_conv_state",
                    "recurrent_state",
                ):
                    tensor = getattr(layer, name)
                    if tensor is not None:
                        kda += tensor.numel() * tensor.element_size()
            elif isinstance(layer, LatentCacheLayer):
                latent += layer.bytes()
            else:
                if layer.key is not None:
                    expanded_k_numel += layer.key.numel()
                    expanded_k += layer.key.numel() * layer.key.element_size()
                if layer.value is not None:
                    expanded_v_numel += layer.value.numel()
                    expanded_v += layer.value.numel() * layer.value.element_size()
        total = kda + expanded_k + expanded_v + latent
        return {
            "kda": kda,
            "expanded_k": expanded_k,
            "expanded_v": expanded_v,
            "latent": latent,
            "persistent_expanded_k_numel": expanded_k_numel,
            "persistent_expanded_v_numel": expanded_v_numel,
            "total": total,
        }

    def latent_contract(self) -> dict[str, Any]:
        latent_layers = [
            index
            for index, layer in enumerate(self.layers)
            if isinstance(layer, LatentCacheLayer)
        ]
        return {
            "latent_cache_layers": latent_layers,
            "latent_layer_count": len(latent_layers),
            "latent_dims": {
                str(index): self.latent_layer(index).latent_dim
                for index in latent_layers
            },
            "latent_shapes": {
                str(index): (
                    None
                    if self.latent_layer(index).latent is None
                    else list(self.latent_layer(index).latent.shape)
                )
                for index in latent_layers
            },
            "latent_append_lengths": {
                str(index): list(self.latent_layer(index).append_lengths)
                for index in latent_layers
            },
            "latent_finite": {
                str(index): (
                    True
                    if self.latent_layer(index).latent is None
                    else bool(torch.isfinite(self.latent_layer(index).latent).all())
                )
                for index in latent_layers
            },
            "seen_tokens": self.seen_tokens,
            "bytes": self.cache_bytes(),
        }


def reconstruct_kv_from_latent(
    mla: Any,
    latent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct temporary normalized K and V from cached normalized latent."""
    batch, sequence, _ = latent.shape
    kv = mla.kv_b_proj(latent).view(
        batch,
        sequence,
        mla.num_heads,
        mla.head_dim + mla.value_head_dim,
    )
    key, value = torch.split(kv, (mla.head_dim, mla.value_head_dim), dim=-1)
    if mla.apply_qk_norm:
        key = mla.k_norm(key)
    return key.transpose(1, 2), value.transpose(1, 2)


def reconstruct_kv_from_cache_layer(
    mla: Any,
    layer: LatentCacheLayer,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct K/V with the same GEMM decomposition as cache writes.

    The persistent payload remains one normalized latent tensor. Retaining
    only integer append boundaries lets the reference runtime reproduce the
    expanded-cache projection shapes without storing K or V.
    """
    reconstructed = [
        reconstruct_kv_from_latent(mla, segment) for segment in layer.segments()
    ]
    if not reconstructed:
        raise RuntimeError("Cannot reconstruct K/V from an empty latent cache")
    keys, values = zip(*reconstructed, strict=True)
    return torch.cat(keys, dim=2), torch.cat(values, dim=2)


def latent_mla_forward(
    mla: Any,
    hidden_states: torch.Tensor,
    position_ids: torch.LongTensor,
    attention_mask: torch.Tensor | None,
    cache: Any,
    use_cache: bool,
) -> torch.Tensor:
    """Reference MLA forward that persists only normalized latent history."""
    del position_ids
    from densek3_core.modeling.modeling_densek3 import _expanded_nope_attention

    if cache is None or not isinstance(cache[mla.layer_idx], LatentCacheLayer):
        return _ORIGINAL_MLA_FORWARD(
            mla,
            hidden_states,
            torch.empty(0, dtype=torch.long, device=hidden_states.device),
            attention_mask,
            cache,
            use_cache,
        )
    batch, sequence, _ = hidden_states.shape
    query = mla.q_proj(hidden_states).view(
        batch,
        sequence,
        mla.num_heads,
        mla.head_dim,
    )
    if mla.apply_qk_norm:
        query = mla.q_norm(query)
    query = query.transpose(1, 2)
    gate = mla.g_proj(hidden_states)
    current_latent = mla.kv_a_layernorm(mla.kv_a_proj(hidden_states))
    full_latent = cache.update_latent(current_latent, mla.layer_idx)
    key, value = reconstruct_kv_from_cache_layer(
        mla,
        cache.latent_layer(mla.layer_idx),
    )
    past_length = full_latent.shape[1] - sequence
    output = _expanded_nope_attention(
        query,
        key,
        value,
        attention_mask,
        past_length=past_length,
        dropout=mla.dropout,
        training=mla.training,
    )
    output = output.transpose(1, 2).contiguous().reshape(batch, sequence, -1)
    return mla.o_proj(output * torch.sigmoid(gate))


_ORIGINAL_CACHE_CLASS: Any = None
_ORIGINAL_MLA_FORWARD: Any = None
_RUNTIME_ENABLED = False


def enable_p7_latent_cache_runtime(
    latent_layer_indices: tuple[int, ...] | list[int] | None = None,
) -> dict[str, Any]:
    """Enable the Titan P7 cache in this process without modifying checkpoint code."""
    global _ORIGINAL_CACHE_CLASS, _ORIGINAL_MLA_FORWARD, _RUNTIME_ENABLED
    import densek3_core.modeling.modeling_densek3 as modeling_densek3

    if not _RUNTIME_ENABLED:
        _ORIGINAL_CACHE_CLASS = modeling_densek3.DenseK3HybridCache
        _ORIGINAL_MLA_FORWARD = modeling_densek3.DenseK3MLA.forward
    selected = tuple(int(index) for index in (latent_layer_indices or ()))
    TitanP7HybridCache.latent_layer_indices = selected
    modeling_densek3.DenseK3HybridCache = TitanP7HybridCache
    modeling_densek3.DenseK3MLA.forward = latent_mla_forward
    _RUNTIME_ENABLED = True
    return {
        "status": "PASS",
        "runtime": "TITAN_P7_TRUE_PERSISTENT_LATENT_CACHE_REFERENCE",
        "selected_latent_layers": list(selected),
        "post_latent_rmsnorm_cached": True,
        "temporary_kv_reconstruction": True,
        "persistent_expanded_kv": False,
        "weights_modified": False,
        "canonical_source_modified": False,
        "formal_bf16_reclassified": False,
    }


def disable_p7_latent_cache_runtime() -> None:
    """Restore the process-local canonical P6 expanded-cache runtime."""
    global _RUNTIME_ENABLED
    if not _RUNTIME_ENABLED:
        return
    import densek3_core.modeling.modeling_densek3 as modeling_densek3

    modeling_densek3.DenseK3HybridCache = _ORIGINAL_CACHE_CLASS
    modeling_densek3.DenseK3MLA.forward = _ORIGINAL_MLA_FORWARD
    TitanP7HybridCache.latent_layer_indices = ()
    _RUNTIME_ENABLED = False
