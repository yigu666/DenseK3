import importlib.util

import pytest
import torch

from densek3_core.kda.fla_backend import fla_fused_recurrent_kda
from densek3_core.kda.metrics import optimized_output_tolerance, optimized_state_tolerance
from densek3_core.kda.reference import kda_reference_from_raw

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.fla,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for FLA KDA"),
    pytest.mark.skipif(importlib.util.find_spec("fla") is None, reason="FLA is not installed"),
]


def test_fused_recurrent_adapter_smoke() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda")
    q = torch.randn(1, 1, 16, 128, dtype=torch.bfloat16, device=device)
    k = torch.randn_like(q)
    v = torch.randn(1, 1, 32, 128, dtype=torch.bfloat16, device=device)
    raw_decay = torch.randn(1, 1, 32, 128, dtype=torch.bfloat16, device=device)
    beta_logits = torch.randn(1, 1, 32, dtype=torch.bfloat16, device=device)
    a_log = torch.zeros(32, dtype=torch.float32, device=device)
    dt_bias = torch.zeros(32 * 128, dtype=torch.float32, device=device)
    state = torch.zeros(1, 32, 128, 128, dtype=torch.float32, device=device)
    output, final_state = fla_fused_recurrent_kda(
        q, k, v, raw_decay, beta_logits, a_log, dt_bias, initial_state=state
    )
    assert output.shape == v.shape
    assert final_state.shape == state.shape
    assert final_state.dtype == torch.float32


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_reference_matches_fla_fused_single_token(dtype: torch.dtype) -> None:
    torch.manual_seed(42)
    device = torch.device("cuda")
    batch, qk_heads, value_heads, key_dim, value_dim = 2, 16, 32, 128, 128
    q = torch.randn(batch, 1, qk_heads, key_dim, dtype=dtype, device=device)
    k = torch.randn_like(q)
    v = torch.randn(batch, 1, value_heads, value_dim, dtype=dtype, device=device)
    raw_decay = torch.randn(batch, 1, value_heads, key_dim, dtype=dtype, device=device)
    beta_logits = torch.randn(batch, 1, value_heads, dtype=dtype, device=device)
    a_log = torch.zeros(value_heads, dtype=torch.float32, device=device)
    dt_bias = torch.zeros(value_heads * key_dim, dtype=torch.float32, device=device)
    state = torch.randn(batch, value_heads, value_dim, key_dim, dtype=torch.float32, device=device) * 0.02
    reference_output, reference_state = kda_reference_from_raw(
        q,
        k,
        v,
        raw_decay,
        beta_logits,
        a_log,
        dt_bias,
        initial_state=state,
        output_final_state=True,
    )
    actual_output, actual_state = fla_fused_recurrent_kda(
        q, k, v, raw_decay, beta_logits, a_log, dt_bias, initial_state=state
    )
    output_rtol, output_atol = optimized_output_tolerance(dtype)
    state_rtol, state_atol = optimized_state_tolerance(dtype)
    torch.testing.assert_close(actual_output, reference_output, rtol=output_rtol, atol=output_atol)
    torch.testing.assert_close(actual_state, reference_state, rtol=state_rtol, atol=state_atol)


def test_fused_recurrent_decode_trajectory() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda")
    sequence, qk_heads, value_heads, key_dim, value_dim = 7, 16, 32, 128, 128
    q = torch.randn(1, sequence, qk_heads, key_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn_like(q)
    v = torch.randn(1, sequence, value_heads, value_dim, dtype=torch.bfloat16, device=device)
    raw_decay = torch.randn(1, sequence, value_heads, key_dim, dtype=torch.bfloat16, device=device)
    beta_logits = torch.randn(1, sequence, value_heads, dtype=torch.bfloat16, device=device)
    a_log = torch.zeros(value_heads, dtype=torch.float32, device=device)
    dt_bias = torch.zeros(value_heads * key_dim, dtype=torch.float32, device=device)
    reference_output, reference_state = kda_reference_from_raw(
        q, k, v, raw_decay, beta_logits, a_log, dt_bias, output_final_state=True
    )
    state = None
    outputs = []
    for index in range(sequence):
        output, state = fla_fused_recurrent_kda(
            q[:, index : index + 1],
            k[:, index : index + 1],
            v[:, index : index + 1],
            raw_decay[:, index : index + 1],
            beta_logits[:, index : index + 1],
            a_log,
            dt_bias,
            initial_state=state,
        )
        outputs.append(output)
    actual_output = torch.cat(outputs, dim=1)
    output_rtol, output_atol = optimized_output_tolerance(torch.bfloat16)
    state_rtol, state_atol = optimized_state_tolerance(torch.bfloat16)
    torch.testing.assert_close(actual_output, reference_output, rtol=output_rtol, atol=output_atol)
    torch.testing.assert_close(state, reference_state, rtol=state_rtol, atol=state_atol)
