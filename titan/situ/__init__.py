"""Titan-only SiTU-GLU overlay for P9."""

from .activation import SITU_BETA, SITU_LINEAR_BETA, situ_glu, swiglu
from .calibration import SiTUCalibrationCollector
from .runtime import (
    disable_situ_glu_runtime,
    enable_situ_glu_runtime,
    fold_down_projection_scales,
    tensor_sha256,
)

__all__ = [
    "SITU_BETA",
    "SITU_LINEAR_BETA",
    "SiTUCalibrationCollector",
    "disable_situ_glu_runtime",
    "enable_situ_glu_runtime",
    "fold_down_projection_scales",
    "situ_glu",
    "swiglu",
    "tensor_sha256",
]
