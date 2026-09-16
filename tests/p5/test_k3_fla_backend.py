from __future__ import annotations

import importlib.util

import pytest
import torch

from densek3_core.kda.fla_backend import FLABackendOptions, fla_chunk_kda, fla_fused_recurrent_kda
from densek3_core.kda.metrics import optimized_output_tolerance, optimized_state_tolerance
from densek3_core.kda.reference import kda_reference_from_raw

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.fla,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for FLA KDA"),
    pytest.mark.skipif(importlib.util.find_spec("fla") is None, reason="FLA is not installed"),
]


def assert_optimized_close(
    actual_output: torch.Tensor,
    actual_state: torch.Tensor,
    reference_output: torch.Tensor,
    reference_state: torch.Tensor,
) -> None:
    output_rtol, output_atol = optimized_output_tolerance(torch.bfloat16)
    state_rtol, state_atol = optimized_state_tolerance(torch.bfloat16)
    torch.testing.assert_close(actual_output, reference_output, rtol=output_rtol, atol=output_atol)
    torch.testing.assert_close(actual_state, reference_state, rtol=state_rtol, atol=state_atol)


def test_kimi_k3_chunk_and_recurrent_paths_match_reference() -> None:
    torch.manual_seed(20260811)
    device = torch.device("cuda")
    batch, sequence, qk_heads, value_heads, key_dim, value_dim = 1, 7, 16, 32, 128, 128
    q = torch.randn(batch, sequence, qk_heads, key_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn_like(q)
    v = torch.randn(batch, sequence, value_heads, value_dim, dtype=torch.bfloat16, device=device)
    raw_decay = torch.randn(batch, sequence, value_heads, key_dim, dtype=torch.bfloat16, device=device)
    beta_logits = torch.randn(batch, sequence, value_heads, dtype=torch.bfloat16, device=device)
    a_log = torch.zeros(value_heads, dtype=torch.float32, device=device)
    dt_bias = torch.zeros(value_heads * key_dim, dtype=torch.float32, device=device)
    options = FLABackendOptions(gate_mode="kimi_k3")

    reference_output, reference_state = kda_reference_from_raw(
        q,
        k,
        v,
        raw_decay,
        beta_logits,
        a_log,
        dt_bias,
        gate_mode="kimi_k3",
        output_final_state=True,
    )
    actual_output, actual_state = fla_chunk_kda(
        q,
        k,
        v,
        raw_decay,
        beta_logits,
        a_log,
        dt_bias,
        options=options,
    )
    assert_optimized_close(actual_output, actual_state, reference_output, reference_state)

    initial_state = torch.randn_like(reference_state) * 0.01
    reference_token, reference_token_state = kda_reference_from_raw(
        q[:, :1],
        k[:, :1],
        v[:, :1],
        raw_decay[:, :1],
        beta_logits[:, :1],
        a_log,
        dt_bias,
        gate_mode="kimi_k3",
        initial_state=initial_state,
        output_final_state=True,
    )
    actual_token, actual_token_state = fla_fused_recurrent_kda(
        q[:, :1],
        k[:, :1],
        v[:, :1],
        raw_decay[:, :1],
        beta_logits[:, :1],
        a_log,
        dt_bias,
        options=options,
        initial_state=initial_state,
    )
    assert_optimized_close(
        actual_token,
        actual_token_state,
        reference_token,
        reference_token_state,
    )
