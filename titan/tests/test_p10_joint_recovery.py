from __future__ import annotations

# ruff: noqa: E402, I001 -- Titan overlay imports are deliberate.

import sys
from pathlib import Path

import torch

TITAN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TITAN))

from attnres import BlockAttnResReference
from densek3_core.modeling import DenseK3Config, DenseK3ForCausalLM
from p10 import (
    FP32MasterParameters,
    MergeableLoRALinear,
    audit_gradient_coverage,
    build_p10_titan_reduced_scope,
    build_p10_trainable_scope,
    cosine_learning_rate,
    merge_all_lora,
)


def production_topology_tiny_model() -> DenseK3ForCausalLM:
    config = DenseK3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        mixer_types=["kda", "kda", "kda", "mla"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        mla_kv_lora_rank=8,
        mla_qk_nope_head_dim=4,
        mla_value_head_dim=4,
        mla_qk_norm=True,
        partial_rotary_factor=0.5,
        kda_qk_num_heads=2,
        kda_value_num_heads=4,
        kda_qk_head_dim=4,
        kda_value_head_dim=4,
        kda_decay_projection_rank=4,
        kda_gate_mode="kimi_k3",
        kda_gate_lower_bound=-5.0,
        kda_output_gate_activation="silu",
        kda_backend="reference",
        mrope_section=(1, 0, 0),
    )
    models = [DenseK3ForCausalLM(config) for _ in range(8)]
    model = models[0]
    model.model.layers = torch.nn.ModuleList(
        [layer for current in models for layer in current.model.layers]
    )
    return model


def test_lora_zero_function_and_dense_merge() -> None:
    torch.manual_seed(10)
    base = torch.nn.Linear(7, 5, bias=False)
    original = base.weight.detach().clone()
    module = MergeableLoRALinear(base, rank=2, alpha=4)
    value = torch.randn(3, 7)
    assert torch.equal(module(value), torch.nn.functional.linear(value, original))
    with torch.no_grad():
        module.lora_B.normal_()
    expected = module(value)
    merged = module.merge(row_chunk=2)
    assert torch.allclose(merged(value), expected, atol=1e-6, rtol=1e-6)
    assert not any("lora" in name for name, _ in merged.named_parameters())


def test_semantic_scope_is_exact_and_merge_removes_runtime_lora() -> None:
    model = production_topology_tiny_model()
    runtime = BlockAttnResReference(model.config.hidden_size, model.config.rms_norm_eps)
    scope = build_p10_trainable_scope(model, runtime, rank=8, alpha=16)
    assert scope.manifest["passed"]
    assert scope.manifest["observed"] == scope.manifest["expected"]
    assert not model.model.embed_tokens.weight.requires_grad
    assert not model.lm_head.weight.requires_grad
    assert all(handle.module.lora_B.count_nonzero() == 0 for handle in scope.lora)
    with torch.no_grad():
        scope.lora[0].module.lora_B.fill_(0.01)
    report = merge_all_lora(scope.lora)
    assert len(report) == 232
    assert report[0]["delta_norm"] > 0
    assert not any("lora" in name.lower() for name, _ in model.named_parameters())


def test_fp32_master_copies_scaled_gradient_and_update_back() -> None:
    source = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float16))
    masters = FP32MasterParameters({"x": source})
    source.grad = torch.tensor([4.0, -2.0], dtype=torch.float16)
    masters.copy_scaled_grads_from_model()
    master = masters.optimizer_parameters()[0]
    assert master.dtype == torch.float32
    assert torch.equal(master.grad, torch.tensor([4.0, -2.0]))
    with torch.no_grad():
        master.add_(torch.tensor([0.25, -0.5]))
    masters.copy_to_model()
    assert torch.equal(source, torch.tensor([1.25, 1.5], dtype=torch.float16))


def test_schedule_is_one_continuous_one_million_token_trajectory() -> None:
    total_steps = 512
    values = [
        cosine_learning_rate(
            step,
            total_steps,
            2e-5,
            warmup_ratio=0.1,
            minimum_ratio=0.1,
        )
        for step in (0, 50, 127, 128, 255, 511)
    ]
    assert values[0] < values[1]
    assert values[2] > values[3] > values[4] > values[5]
    assert values[-1] >= 2e-6


def test_gradient_coverage_rejects_silent_missing_trainables() -> None:
    present = torch.nn.Parameter(torch.ones(1))
    missing = torch.nn.Parameter(torch.ones(1))
    present.grad = torch.ones_like(present)
    parameters = [present, missing]
    report = audit_gradient_coverage(
        parameters,
        {id(present): "present", id(missing): "missing"},
    )
    assert report == {
        "complete": False,
        "optimizer_tensors": 2,
        "gradient_tensors": 1,
        "missing_gradient_count": 1,
        "missing_gradient_names": ["missing"],
    }


def test_titan_reduced_scope_excludes_exactly_264_undifferentiable_tensors() -> None:
    model = production_topology_tiny_model()
    runtime = BlockAttnResReference(model.config.hidden_size, model.config.rms_norm_eps)
    scope = build_p10_titan_reduced_scope(model, runtime, rank=8, alpha=16)
    observed = scope.manifest["observed"]
    # The helper expands the production 24 KDA / 8 MLA topology.
    assert observed["lora_modules"] == 160
    assert observed["lora_tensors"] == 320
    assert observed["direct_model_tensors"] == 113
    assert observed["attnres_tensors"] == 130
    assert scope.manifest["excluded_optimizer_tensor_count"] == 264
    assert len(scope.trainable_parameters()) == 563
    assert scope.manifest["scope_policy"] == "P10_TITAN_REDUCED_SCOPE_2026_08_21"
    assert scope.manifest["passed"]
    assert all(
        handle.name.endswith("o_proj")
        for handle in scope.lora
        if handle.family == "KDA"
    )
