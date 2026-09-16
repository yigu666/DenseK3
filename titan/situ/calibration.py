"""Streaming, teacher-forced SiTU initialization calibration for Titan P9."""

from __future__ import annotations

import math
import types
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from .activation import situ_glu, swiglu


class PairAccumulator:
    """Exact scalar error moments without retaining activation tensors."""

    def __init__(self) -> None:
        self.numel = 0
        self.error_sq = 0.0
        self.reference_sq = 0.0
        self.actual_sq = 0.0
        self.dot = 0.0
        self.abs_sum = 0.0
        self.max_abs = 0.0
        self.finite = True

    @torch.no_grad()
    def update(self, actual: torch.Tensor, reference: torch.Tensor) -> None:
        if actual.shape != reference.shape:
            raise ValueError(f"P9 metric shape mismatch: {actual.shape} != {reference.shape}")
        left = actual.detach().float()
        right = reference.detach().float()
        diff = left - right
        self.numel += diff.numel()
        self.error_sq += float(diff.square().sum())
        self.reference_sq += float(right.square().sum())
        self.actual_sq += float(left.square().sum())
        self.dot += float((left * right).sum())
        self.abs_sum += float(diff.abs().sum())
        self.max_abs = max(self.max_abs, float(diff.abs().max()))
        self.finite = self.finite and bool(torch.isfinite(left).all()) and bool(
            torch.isfinite(right).all()
        )

    def report(self) -> dict[str, Any]:
        denominator = max(self.reference_sq, 1.0e-30)
        cosine_denominator = max(math.sqrt(self.actual_sq * self.reference_sq), 1.0e-30)
        return {
            "numel": self.numel,
            "relative_l2": math.sqrt(self.error_sq / denominator),
            "cosine": self.dot / cosine_denominator,
            "rms_ratio": math.sqrt(self.actual_sq / denominator),
            "mean_abs": self.abs_sum / max(self.numel, 1),
            "max_abs": self.max_abs,
            "finite": self.finite,
        }


class DistributionAccumulator:
    def __init__(self, threshold: float, sample_per_update: int = 4096) -> None:
        self.threshold = float(threshold)
        self.sample_per_update = int(sample_per_update)
        self.numel = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.over_threshold = 0
        self.finite = True
        self.samples: list[torch.Tensor] = []

    @torch.no_grad()
    def update(self, value: torch.Tensor) -> None:
        flat = value.detach().float().flatten()
        self.numel += flat.numel()
        self.total += float(flat.sum())
        self.total_sq += float(flat.square().sum())
        self.over_threshold += int((flat.abs() > self.threshold).sum())
        self.finite = self.finite and bool(torch.isfinite(flat).all())
        stride = max(flat.numel() // self.sample_per_update, 1)
        self.samples.append(flat[::stride][: self.sample_per_update].cpu())

    def report(self) -> dict[str, Any]:
        mean = self.total / max(self.numel, 1)
        variance = max(self.total_sq / max(self.numel, 1) - mean * mean, 0.0)
        sample = torch.cat(self.samples).double().abs() if self.samples else torch.zeros(1)
        return {
            "numel": self.numel,
            "mean": mean,
            "std": math.sqrt(variance),
            "abs_p95": float(torch.quantile(sample, 0.95)),
            "abs_p99": float(torch.quantile(sample, 0.99)),
            "abs_p99_9": float(torch.quantile(sample, 0.999)),
            "threshold": self.threshold if math.isfinite(self.threshold) else None,
            "over_threshold_fraction": self.over_threshold / max(self.numel, 1),
            "sample_count": sample.numel(),
            "finite": self.finite,
        }


@dataclass
class LayerCalibration:
    intermediate_size: int
    gate: DistributionAccumulator = field(default_factory=lambda: DistributionAccumulator(4.0))
    up: DistributionAccumulator = field(default_factory=lambda: DistributionAccumulator(25.0))
    swiglu_activation: DistributionAccumulator = field(
        default_factory=lambda: DistributionAccumulator(float("inf"))
    )
    situ_activation: DistributionAccumulator = field(
        default_factory=lambda: DistributionAccumulator(float("inf"))
    )
    fit_raw_activation: PairAccumulator = field(default_factory=PairAccumulator)
    validation_raw_activation: PairAccumulator = field(default_factory=PairAccumulator)
    validation_matched_activation: PairAccumulator = field(default_factory=PairAccumulator)
    validation_raw_output: PairAccumulator = field(default_factory=PairAccumulator)
    validation_matched_output: PairAccumulator = field(default_factory=PairAccumulator)
    validation_raw_post: PairAccumulator = field(default_factory=PairAccumulator)
    validation_matched_post: PairAccumulator = field(default_factory=PairAccumulator)

    def __post_init__(self) -> None:
        self.numerator = torch.zeros(self.intermediate_size, dtype=torch.float64)
        self.denominator = torch.zeros(self.intermediate_size, dtype=torch.float64)


class SiTUCalibrationCollector:
    """Patch MLPs while returning the original SwiGLU branch output."""

    def __init__(self, model: nn.Module, phase: str) -> None:
        if phase not in {"fit", "validation"}:
            raise ValueError(phase)
        self.model = model
        self.phase = phase
        self.layers = {
            index: LayerCalibration(layer.mlp.gate_proj.out_features)
            for index, layer in enumerate(model.model.layers)
        }
        self.scales: dict[int, torch.Tensor] = {}
        self.pending: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def set_scales(self, scales: dict[int, torch.Tensor]) -> None:
        self.scales = {index: value.detach().cpu().float() for index, value in scales.items()}

    @torch.no_grad()
    def _forward(self, layer_index: int, mlp: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
        state = self.layers[layer_index]
        gate = mlp.gate_proj(hidden)
        up = mlp.up_proj(hidden)
        old = swiglu(gate, up)
        raw = situ_glu(gate, up)
        state.gate.update(gate)
        state.up.update(up)
        state.swiglu_activation.update(old)
        state.situ_activation.update(raw)
        state.fit_raw_activation.update(raw, old)
        if self.phase == "fit":
            raw32 = raw.float()
            old32 = old.float()
            state.numerator += (raw32 * old32).sum(dim=(0, 1)).cpu().double()
            state.denominator += raw32.square().sum(dim=(0, 1)).cpu().double()
            return mlp.down_proj(old)
        if layer_index not in self.scales:
            raise RuntimeError(f"Missing P9 S1 scale for layer {layer_index}")
        scale = self.scales[layer_index].to(device=raw.device, dtype=torch.float32)
        matched = (raw.float() * scale.view(1, 1, -1)).to(raw.dtype)
        old_output = mlp.down_proj(old)
        raw_output = mlp.down_proj(raw)
        matched_output = mlp.down_proj(matched)
        state.validation_raw_activation.update(raw, old)
        state.validation_matched_activation.update(matched, old)
        state.validation_raw_output.update(raw_output, old_output)
        state.validation_matched_output.update(matched_output, old_output)
        self.pending[layer_index] = (old_output, raw_output, matched_output)
        return old_output

    def observe_branch(
        self,
        *,
        layer_index: int,
        branch_name: str,
        branch_output: torch.Tensor,
        post_residual: torch.Tensor,
        **_: Any,
    ) -> None:
        if branch_name != "mlp" or self.phase != "validation":
            return
        old_output, raw_output, matched_output = self.pending.pop(layer_index)
        if branch_output.data_ptr() != old_output.data_ptr() and not torch.equal(
            branch_output, old_output
        ):
            raise RuntimeError(f"P9 observer branch mismatch at layer {layer_index}")
        state = self.layers[layer_index]
        state.validation_raw_post.update(post_residual + raw_output - old_output, post_residual)
        state.validation_matched_post.update(
            post_residual + matched_output - old_output, post_residual
        )

    def enable(self) -> None:
        for layer_index, layer in enumerate(self.model.model.layers):
            mlp = layer.mlp
            if hasattr(mlp, "_titan_p9_calibration_forward"):
                raise RuntimeError("P9 calibration collector already enabled")
            mlp._titan_p9_calibration_forward = mlp.forward  # type: ignore[attr-defined]

            def patched(
                this: nn.Module,
                hidden: torch.Tensor,
                current_index: int = layer_index,
            ) -> torch.Tensor:
                return self._forward(current_index, this, hidden)

            mlp.forward = types.MethodType(patched, mlp)

    def disable(self) -> None:
        for layer in self.model.model.layers:
            mlp = layer.mlp
            original = getattr(mlp, "_titan_p9_calibration_forward", None)
            if original is not None:
                mlp.forward = original
                delattr(mlp, "_titan_p9_calibration_forward")
        if self.pending:
            raise RuntimeError(f"Unobserved P9 MLP outputs: {sorted(self.pending)}")

    def fitted_scales(self, epsilon: float = 1.0e-12) -> dict[int, torch.Tensor]:
        if self.phase != "fit":
            raise RuntimeError("Scales can only be fitted from the fit phase")
        return {
            index: (state.numerator / (state.denominator + epsilon)).float()
            for index, state in self.layers.items()
        }

    def fit_report(self, scales: dict[int, torch.Tensor]) -> list[dict[str, Any]]:
        records = []
        for index, state in self.layers.items():
            scale = scales[index].double()
            records.append(
                {
                    "layer": index,
                    "gate": state.gate.report(),
                    "up": state.up.report(),
                    "swiglu_activation": state.swiglu_activation.report(),
                    "situ_activation": state.situ_activation.report(),
                    "raw_situ_vs_swiglu": state.fit_raw_activation.report(),
                    "s1_scale": {
                        "min": float(scale.min()),
                        "median": float(scale.median()),
                        "max": float(scale.max()),
                        "mean": float(scale.mean()),
                        "finite": bool(torch.isfinite(scale).all()),
                    },
                }
            )
        return records

    def validation_report(self) -> list[dict[str, Any]]:
        records = []
        for index, state in self.layers.items():
            raw_post = state.validation_raw_post.report()
            matched_post = state.validation_matched_post.report()
            select_s1 = matched_post["finite"] and (
                matched_post["relative_l2"],
                state.validation_matched_output.report()["relative_l2"],
            ) < (
                raw_post["relative_l2"],
                state.validation_raw_output.report()["relative_l2"],
            )
            records.append(
                {
                    "layer": index,
                    "selection": "S1_ACTIVATION_MATCHED" if select_s1 else "S0_RAW",
                    "raw": {
                        "activation": state.validation_raw_activation.report(),
                        "mlp_output": state.validation_raw_output.report(),
                        "post_residual": raw_post,
                    },
                    "matched": {
                        "activation": state.validation_matched_activation.report(),
                        "mlp_output": state.validation_matched_output.report(),
                        "post_residual": matched_post,
                    },
                }
            )
        return records
