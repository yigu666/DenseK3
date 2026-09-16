"""Frozen structural constants for the Titan P8 Block AttnRes probe."""

from __future__ import annotations

TRANSFORMER_LAYERS = 32
RESIDUAL_SITES = 64
FINAL_SITE = 64
TOTAL_ROUTING_SITES = 65
FORMAL_BLOCK_COUNT = 8
FORMAL_BLOCK_SIZE_RESIDUALS = 8
FORMAL_BLOCK_SIZE_TRANSFORMERS = 4
BLOCK_BOUNDARY_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31)
SOURCE_TENSOR_COUNT = 562
SOURCE_PARAMETER_COUNT = 4_226_431_232
MLA_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31)


def residual_site(layer_index: int, branch: str) -> int:
    if layer_index < 0 or layer_index >= TRANSFORMER_LAYERS:
        raise ValueError(f"Invalid transformer layer: {layer_index}")
    if branch == "mixer":
        return layer_index * 2
    if branch == "mlp":
        return layer_index * 2 + 1
    raise ValueError(f"Unknown residual branch: {branch}")
