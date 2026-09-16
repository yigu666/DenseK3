from __future__ import annotations

import torch

from densek3_core.transplant.gdn_to_kda import GDNBridgeConfig


def small_bridge_config() -> GDNBridgeConfig:
    return GDNBridgeConfig(
        layer_index=0,
        hidden_size=8,
        qk_num_heads=2,
        value_num_heads=4,
        qk_head_dim=3,
        value_head_dim=5,
        conv_kernel_size=4,
        decay_projection_rank=6,
        rms_norm_eps=1e-6,
    )


def deterministic_tensor(shape: tuple[int, ...], *, offset: int = 0, scale: float = 0.01) -> torch.Tensor:
    count = 1
    for size in shape:
        count *= size
    values = torch.arange(offset, offset + count, dtype=torch.float32)
    values = ((values % 29) - 14) * scale
    return values.reshape(shape)


def small_source_state() -> dict[str, torch.Tensor]:
    config = small_bridge_config()
    shapes = config.expected_source_shapes()
    state = {}
    for index, (name, shape) in enumerate(shapes.items()):
        state[name] = deterministic_tensor(shape, offset=index * 7)
    state["A_log"] = torch.linspace(-1.0, 0.5, config.value_num_heads)
    state["dt_bias"] = torch.linspace(-0.5, 0.5, config.value_num_heads)
    state["norm.weight"] = torch.linspace(0.8, 1.2, config.value_head_dim)
    return state
