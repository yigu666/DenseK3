import importlib.util

import pytest
import torch

from densek3_core.kda.fla_backend import FLABackendOptions, fla_chunk_kda
from densek3_core.kda.metrics import optimized_output_tolerance, optimized_state_tolerance
from densek3_core.kda.reference import kda_reference_from_raw

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.fla,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for FLA KDA"),
    pytest.mark.skipif(importlib.util.find_spec("fla") is None, reason="FLA is not installed"),
]


def test_fla_chunk_adapter_smoke() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda")
    q = torch.randn(1, 2, 16, 128, dtype=torch.bfloat16, device=device)
    k = torch.randn_like(q)
    v = torch.randn(1, 2, 32, 128, dtype=torch.bfloat16, device=device)
    raw_decay = torch.randn(1, 2, 32, 128, dtype=torch.bfloat16, device=device)
    beta_logits = torch.randn(1, 2, 32, dtype=torch.bfloat16, device=device)
    a_log = torch.zeros(32, dtype=torch.float32, device=device)
    dt_bias = torch.zeros(32 * 128, dtype=torch.float32, device=device)
    output, final_state = fla_chunk_kda(q, k, v, raw_decay, beta_logits, a_log, dt_bias)
    assert output.shape == v.shape
    assert final_state.shape == (1, 32, 128, 128)
    assert final_state.dtype == torch.float32


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("initial_state_mode", ["none", "random"])
@pytest.mark.parametrize("sequence", [1, 7, 65, 129])
def test_reference_matches_fla_chunk(dtype: torch.dtype, initial_state_mode: str, sequence: int) -> None:
    torch.manual_seed(42)
    device = torch.device("cuda")
    batch, qk_heads, value_heads, key_dim, value_dim = 1, 16, 32, 128, 128
    q = torch.randn(batch, sequence, qk_heads, key_dim, dtype=dtype, device=device)
    k = torch.randn_like(q)
    v = torch.randn(batch, sequence, value_heads, value_dim, dtype=dtype, device=device)
    raw_decay = torch.randn(batch, sequence, value_heads, key_dim, dtype=dtype, device=device)
    beta_logits = torch.randn(batch, sequence, value_heads, dtype=dtype, device=device)
    a_log = torch.zeros(value_heads, dtype=torch.float32, device=device)
    dt_bias = torch.zeros(value_heads * key_dim, dtype=torch.float32, device=device)
    initial_state = None
    if initial_state_mode == "random":
        initial_state = torch.randn(batch, value_heads, value_dim, key_dim, dtype=torch.float32, device=device) * 0.02
    reference_output, reference_state = kda_reference_from_raw(
        q,
        k,
        v,
        raw_decay,
        beta_logits,
        a_log,
        dt_bias,
        initial_state=initial_state,
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
        initial_state=initial_state,
        options=FLABackendOptions(gate_mode="qwen_compat"),
    )
    output_rtol, output_atol = optimized_output_tolerance(dtype)
    state_rtol, state_atol = optimized_state_tolerance(dtype)
    torch.testing.assert_close(actual_output, reference_output, rtol=output_rtol, atol=output_atol)
    torch.testing.assert_close(actual_state, reference_state, rtol=state_rtol, atol=state_atol)


def test_fla_chunk_supports_non_square_v_first_state() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda")
    q = torch.randn(1, 7, 2, 64, device=device)
    k = torch.randn_like(q)
    v = torch.randn(1, 7, 4, 128, device=device)
    raw_decay = torch.randn(1, 7, 4, 64, device=device)
    beta_logits = torch.randn(1, 7, 4, device=device)
    a_log = torch.zeros(4, device=device)
    dt_bias = torch.zeros(4 * 64, device=device)
    state = torch.randn(1, 4, 128, 64, dtype=torch.float32, device=device) * 0.02
    output, final_state = fla_chunk_kda(
        q, k, v, raw_decay, beta_logits, a_log, dt_bias, initial_state=state
    )
    assert output.shape == v.shape
    assert final_state.shape == state.shape
    assert final_state.dtype == torch.float32
