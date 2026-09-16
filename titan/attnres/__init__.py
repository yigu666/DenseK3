"""Titan-only Block AttnRes reference runtime."""

from .block_reference import (
    BlockAttnResReference,
    disable_block_attnres_runtime,
    enable_block_attnres_runtime,
)
from .contracts import BLOCK_BOUNDARY_LAYERS, FINAL_SITE, RESIDUAL_SITES
from .state import DepthBlockState

__all__ = [
    "BLOCK_BOUNDARY_LAYERS",
    "FINAL_SITE",
    "RESIDUAL_SITES",
    "BlockAttnResReference",
    "DepthBlockState",
    "disable_block_attnres_runtime",
    "enable_block_attnres_runtime",
]
