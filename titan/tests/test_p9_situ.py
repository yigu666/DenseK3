from __future__ import annotations

import json
from types import SimpleNamespace

import torch
from torch import nn

from titan.situ import (
    SiTUCalibrationCollector,
    disable_situ_glu_runtime,
    enable_situ_glu_runtime,
    fold_down_projection_scales,
    situ_glu,
    swiglu,
)


class TinyMLP(nn.Module):
    def __init__(self, hidden: int = 8, intermediate: int = 12) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.down_proj(swiglu(self.gate_proj(value), self.up_proj(value)))


class TinyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = TinyMLP()


class TinyModel(nn.Module):
    def __init__(self, layers: int = 2) -> None:
        super().__init__()
        self.model = SimpleNamespace(layers=nn.ModuleList([TinyLayer() for _ in range(layers)]))


def test_situ_uses_uncapped_gate_for_sigmoid_and_restores_dtype() -> None:
    gate = torch.tensor([[-12.0, -1.0, 1.0, 12.0]], dtype=torch.float16)
    up = torch.tensor([[30.0, -2.0, 2.0, -30.0]], dtype=torch.float16)
    actual = situ_glu(gate, up)
    gate32 = gate.float()
    expected = (
        4.0
        * torch.tanh(gate32 / 4.0)
        * torch.sigmoid(gate32)
        * 25.0
        * torch.tanh(up.float() / 25.0)
    ).half()
    wrong = (
        4.0
        * torch.tanh(gate32 / 4.0)
        * torch.sigmoid(4.0 * torch.tanh(gate32 / 4.0))
        * 25.0
        * torch.tanh(up.float() / 25.0)
    ).half()
    assert actual.dtype == gate.dtype
    assert torch.equal(actual, expected)
    assert not torch.equal(actual, wrong)


def test_situ_backward_is_finite() -> None:
    gate = torch.randn(2, 3, 7, requires_grad=True)
    up = torch.randn(2, 3, 7, requires_grad=True)
    situ_glu(gate, up).square().mean().backward()
    assert gate.grad is not None and torch.isfinite(gate.grad).all()
    assert up.grad is not None and torch.isfinite(up.grad).all()


def test_situ_fp32_reference_is_strictly_close() -> None:
    gate = torch.linspace(-8.0, 8.0, 257)
    up = torch.linspace(30.0, -30.0, 257)
    actual = situ_glu(gate, up)
    reference = (
        4.0
        * torch.tanh(gate / 4.0)
        * torch.sigmoid(gate)
        * 25.0
        * torch.tanh(up / 25.0)
    )
    torch.testing.assert_close(actual, reference, atol=1.0e-6, rtol=1.0e-6)


def test_instance_runtime_is_reversible() -> None:
    torch.manual_seed(1)
    model = TinyModel()
    hidden = torch.randn(1, 3, 8)
    baseline = model.model.layers[0].mlp(hidden)
    enable_situ_glu_runtime(model)
    situ = model.model.layers[0].mlp(hidden)
    disable_situ_glu_runtime(model)
    restored = model.model.layers[0].mlp(hidden)
    assert not torch.equal(situ, baseline)
    assert torch.equal(restored, baseline)


def test_down_projection_scale_folding_is_deterministic() -> None:
    torch.manual_seed(2)
    model = TinyModel()
    layer = model.model.layers[0]
    activation = torch.randn(1, 4, 12)
    scale = torch.linspace(0.8, 1.2, 12)
    expected = layer.mlp.down_proj(activation * scale)
    report = fold_down_projection_scales(model, {0: scale}, {0})
    actual = layer.mlp.down_proj(activation)
    assert report["passed"]
    assert report["changed_layers"] == [0]
    torch.testing.assert_close(actual, expected, atol=2.0e-6, rtol=2.0e-6)


def test_calibration_fits_scales_and_selects_valid_candidate() -> None:
    torch.manual_seed(3)
    model = TinyModel()
    hidden = torch.randn(1, 5, 8)
    fit = SiTUCalibrationCollector(model, "fit")
    fit.enable()
    try:
        for layer in model.model.layers:
            layer.mlp(hidden)
    finally:
        fit.disable()
    scales = fit.fitted_scales()
    assert set(scales) == {0, 1}
    assert all(torch.isfinite(value).all() for value in scales.values())
    json.dumps(fit.fit_report(scales), allow_nan=False)

    validation = SiTUCalibrationCollector(model, "validation")
    validation.set_scales(scales)
    validation.enable()
    try:
        for index, layer in enumerate(model.model.layers):
            output = layer.mlp(hidden)
            validation.observe_branch(
                layer_index=index,
                branch_name="mlp",
                branch_output=output,
                post_residual=hidden + output,
            )
    finally:
        validation.disable()
    report = validation.validation_report()
    assert len(report) == 2
    assert all(item["selection"] in {"S0_RAW", "S1_ACTIVATION_MATCHED"} for item in report)
    assert all(item["raw"]["post_residual"]["finite"] for item in report)
