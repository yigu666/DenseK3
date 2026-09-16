"""Hybrid recurrent/KV cache for the P4 DenseK3 model."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from densek3_core.modeling.configuration_densek3 import DenseK3Config


@dataclass
class KDACacheLayer:
    """Three convolution states plus one FP32 V-first recurrence state."""

    q_conv_state: torch.Tensor | None = None
    k_conv_state: torch.Tensor | None = None
    v_conv_state: torch.Tensor | None = None
    recurrent_state: torch.Tensor | None = None
    recurrent_state_layout: str = "v_first"

    @property
    def initialized(self) -> bool:
        return self.recurrent_state is not None

    def reorder(self, beam_idx: torch.LongTensor) -> None:
        for name in ("q_conv_state", "k_conv_state", "v_conv_state", "recurrent_state"):
            tensor = getattr(self, name)
            if tensor is not None:
                setattr(self, name, tensor.index_select(0, beam_idx.to(tensor.device)))

    def reset(self) -> None:
        self.q_conv_state = None
        self.k_conv_state = None
        self.v_conv_state = None
        self.recurrent_state = None


@dataclass
class GQACacheLayer:
    """Expanded full-attention key/value cache in `[B,H,T,K]`.

    P4 GQA uses four KV heads; the P6 expanded reference MLA path uses 16.
    P7 true latent caching deliberately does not reuse this class.
    """

    key: torch.Tensor | None = None
    value: torch.Tensor | None = None

    @property
    def initialized(self) -> bool:
        return self.key is not None

    @property
    def sequence_length(self) -> int:
        return 0 if self.key is None else self.key.shape[2]

    def update(self, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if key.shape != value.shape:
            raise ValueError("GQA key and value cache updates must have matching shapes")
        if self.key is None:
            self.key = key
            self.value = value
        else:
            self.key = torch.cat((self.key, key), dim=2)
            self.value = torch.cat((self.value, value), dim=2)
        return self.key, self.value

    def reorder(self, beam_idx: torch.LongTensor) -> None:
        if self.key is not None:
            self.key = self.key.index_select(0, beam_idx.to(self.key.device))
            self.value = self.value.index_select(0, beam_idx.to(self.value.device))

    def reset(self) -> None:
        self.key = None
        self.value = None


class DenseK3HybridCache:
    """Model-wide cache containing KDA states and growing expanded GQA/MLA KV pairs."""

    is_compileable = False

    def __init__(self, config: DenseK3Config):
        self.mixer_types = tuple(config.mixer_types)
        self.layers: list[KDACacheLayer | GQACacheLayer] = [
            KDACacheLayer(recurrent_state_layout=config.recurrent_state_layout)
            if layer_type == "kda"
            else GQACacheLayer()
            for layer_type in self.mixer_types
        ]
        self._seen_tokens = 0

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, layer_idx: int) -> KDACacheLayer | GQACacheLayer:
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
            raise TypeError(f"Layer {layer_idx} is not a GQA cache layer")
        return layer

    def update_gqa(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.gqa_layer(layer_idx).update(key, value)

    def advance(self, token_count: int) -> None:
        if token_count < 0:
            raise ValueError("token_count must be non-negative")
        self._seen_tokens += token_count

    def reorder_cache(self, beam_idx: torch.LongTensor) -> DenseK3HybridCache:
        for layer in self.layers:
            layer.reorder(beam_idx)
        return self

    def batch_repeat_interleave(self, repeats: int) -> None:
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        for layer in self.layers:
            for name in (
                ("q_conv_state", "k_conv_state", "v_conv_state", "recurrent_state")
                if isinstance(layer, KDACacheLayer)
                else ("key", "value")
            ):
                tensor = getattr(layer, name)
                if tensor is not None:
                    setattr(layer, name, tensor.repeat_interleave(repeats, dim=0))

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

    def cache_bytes(self) -> dict[str, int]:
        kda = 0
        gqa = 0
        for layer in self.layers:
            names = (
                ("q_conv_state", "k_conv_state", "v_conv_state", "recurrent_state")
                if isinstance(layer, KDACacheLayer)
                else ("key", "value")
            )
            for name in names:
                tensor = getattr(layer, name)
                if tensor is not None:
                    size = tensor.numel() * tensor.element_size()
                    if isinstance(layer, KDACacheLayer):
                        kda += size
                    else:
                        gqa += size
        return {"kda": kda, "gqa": gqa, "total": kda + gqa}
