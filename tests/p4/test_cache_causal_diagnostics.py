from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest
import torch

from densek3_core.evaluation.p4_cache_diagnostics import (
    convolution_partition_probe,
    gqa_append_partition_probe,
    hidden_error_curve,
    projection_partition_probe,
    recurrence_partition_probe,
    segment_ranges,
)
from densek3_core.evaluation.p4_decay_diagnostics import _exact_quantiles
from densek3_core.modeling import DenseK3Config, DenseK3ForCausalLM
from densek3_core.modeling.modeling_densek3 import _causal_depthwise_conv


def small_config() -> DenseK3Config:
    return DenseK3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        mixer_types=["kda", "gqa", "kda", "gqa"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        partial_rotary_factor=0.5,
        kda_qk_num_heads=2,
        kda_value_num_heads=4,
        kda_qk_head_dim=4,
        kda_value_head_dim=4,
        kda_decay_projection_rank=8,
        kda_conv_kernel_size=3,
        kda_backend="reference",
    )


def test_segment_ranges_match_prefill_then_decode() -> None:
    assert segment_ranges(5, 2) == [(0, 2), (2, 3), (3, 4), (4, 5)]
    assert segment_ranges(1, 1) == [(0, 1)]


def test_exact_quantile_uses_scalable_numpy_partition() -> None:
    values = torch.tensor([9.0, 0.0, 4.0, 1.0, 7.0, 3.0, 8.0, 2.0, 6.0, 5.0], dtype=torch.float64)
    expected = torch.quantile(values, torch.tensor([0.5, 0.9], dtype=torch.float64))
    actual, backend = _exact_quantiles(values.clone(), (0.5, 0.9), torch_max_elements=4)
    assert backend == "numpy_exact_partition"
    assert torch.equal(actual, expected)


def test_resume_cases_require_frozen_split_and_token_hash() -> None:
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/diagnose_p4_cache.py"))
    tokens = torch.tensor([[1, 2, 3]])
    expected = [(1, 2, tokens)]
    saved = [
        {
            "prompt_length": 1,
            "decode_length": 2,
            "sequence_length": 3,
            "token_sha256": namespace["tensor_sha256"](tokens),
        }
    ]
    namespace["_validate_resumed_cases"](saved, expected, integrated=True)
    saved[0]["token_sha256"] = "wrong"
    with pytest.raises(ValueError, match="frozen tokens"):
        namespace["_validate_resumed_cases"](saved, expected, integrated=True)


def test_resume_report_preserves_previous_exception_audit(tmp_path: Path) -> None:
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/diagnose_p4_cache.py"))
    path = tmp_path / "partial.json"
    partial = {
        "schema_version": 1,
        "stage": "P4.6-CACHE-DIAG",
        "status": "EXCEPTION",
        "token_seed": 20260811,
        "p4_6_formal_status": "FAIL_UNCHANGED",
        "numerics_amendment_approved": False,
        "last_completed_phase": "integrated_error_propagation",
        "exception_type": "RuntimeError",
        "exception": "quantile() input tensor is too large",
        "traceback": "trace",
    }
    path.write_text(json.dumps(partial), encoding="utf-8")
    report = namespace["_load_resume_report"](path)
    assert report["status"] == "RESUMING"
    assert report["resume"]["enabled"] is True
    assert all(report["resume"]["safety_checks"].values())
    assert report["resume_history"][-1]["previous_exception_type"] == "RuntimeError"
    assert "exception" not in report


def test_canonical_projection_conv_and_recurrence_partition() -> None:
    torch.manual_seed(20260811)
    model = DenseK3ForCausalLM(small_config()).eval()
    layer = model.model.layers[0].self_attn
    hidden = torch.randn((1, 5, 16))
    projection = projection_partition_probe(layer, hidden, prompt_length=2)
    assert projection["all_finite"] is True
    assert all(item["passed"] for item in projection["forced_fp32"].values())

    projected = {
        "q": (layer.q_proj(hidden), layer.q_conv1d),
        "k": (layer.k_proj(hidden), layer.k_conv1d),
        "v": (layer.v_proj(hidden), layer.v_conv1d),
    }
    convolution = {
        name: convolution_partition_probe(value, conv, prompt_length=2)
        for name, (value, conv) in projected.items()
    }
    assert all(item["output"]["passed"] for item in convolution.values())
    assert all(item["final_state_exact"] for item in convolution.values())

    q, _ = _causal_depthwise_conv(projected["q"][0], layer.q_conv1d, None)
    k, _ = _causal_depthwise_conv(projected["k"][0], layer.k_conv1d, None)
    v, _ = _causal_depthwise_conv(projected["v"][0], layer.v_conv1d, None)
    q = q.view(1, 5, layer.qk_heads, layer.key_dim)
    k = k.view(1, 5, layer.qk_heads, layer.key_dim)
    v = v.view(1, 5, layer.value_heads, layer.value_dim)
    raw_decay = layer.f_b_proj(layer.f_a_proj(hidden)).view(
        1,
        5,
        layer.value_heads,
        layer.key_dim,
    )
    beta = layer.b_proj(hidden)
    recurrence = recurrence_partition_probe(layer, q, k, v, raw_decay, beta, prompt_length=2)
    assert recurrence["reference"]["output_exact"] is True
    assert recurrence["reference"]["final_state_exact"] is True
    assert recurrence["production_fla"]["output"]["passed"] is True
    assert recurrence["production_fla"]["final_state"]["passed"] is True


def test_gqa_append_and_hidden_curve_are_machine_readable() -> None:
    key = torch.randn((1, 2, 7, 4))
    value = torch.randn_like(key)
    append = gqa_append_partition_probe(key, value, prompt_length=3)
    assert append == {
        "key_exact": True,
        "value_exact": True,
        "sequence_length": 7,
        "expected_sequence_length": 7,
        "passed": True,
    }

    full = (torch.zeros((1, 3, 4), dtype=torch.bfloat16),) * 3
    segmented = list(full)
    segmented[1] = segmented[1].clone()
    segmented[1][0, 0, 0] = 0.125
    curve = hidden_error_curve(full, tuple(segmented))
    assert [item["boundary"] for item in curve] == ["embedding", "after_layer_0", "final_norm"]
    assert curve[0]["passed"] is True
    assert curve[1]["passed"] is False
