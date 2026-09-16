import importlib.util

import pytest
import torch

from densek3_core.kda.fla_backend import fla_chunk_kda
from densek3_core.kda.reference import kda_reference_from_raw

REFERENCE_CASES = [
    (1, 1, 2, 4, 32, 32, torch.float32, False),
    (2, 2, 2, 4, 64, 64, torch.bfloat16, True),
    (1, 3, 2, 2, 32, 32, torch.float32, False),
    (2, 4, 2, 4, 64, 128, torch.bfloat16, True),
    (1, 7, 16, 32, 128, 128, torch.bfloat16, False),
    (1, 31, 2, 4, 32, 32, torch.float32, True),
    (1, 63, 2, 4, 32, 32, torch.bfloat16, False),
    (1, 64, 2, 4, 32, 32, torch.bfloat16, True),
    (1, 65, 2, 4, 32, 32, torch.bfloat16, False),
    (1, 127, 2, 4, 32, 32, torch.float32, True),
    (1, 128, 2, 4, 32, 32, torch.bfloat16, False),
    (1, 129, 2, 4, 32, 32, torch.bfloat16, True),
    (1, 1024, 2, 4, 32, 32, torch.bfloat16, False),
]


@pytest.mark.parametrize("batch,sequence,qk_heads,value_heads,key_dim,value_dim,dtype,random_state", REFERENCE_CASES)
def test_reference_numerical_matrix(
    batch: int,
    sequence: int,
    qk_heads: int,
    value_heads: int,
    key_dim: int,
    value_dim: int,
    dtype: torch.dtype,
    random_state: bool,
    compute_device: torch.device,
) -> None:
    torch.manual_seed(42)
    q = torch.randn(batch, sequence, qk_heads, key_dim, dtype=dtype, device=compute_device)
    k = torch.randn_like(q)
    v = torch.randn(batch, sequence, value_heads, value_dim, dtype=dtype, device=compute_device)
    raw_decay = torch.randn(batch, sequence, value_heads, key_dim, dtype=dtype, device=compute_device)
    beta_logits = torch.randn(batch, sequence, value_heads, dtype=dtype, device=compute_device)
    a_log = torch.zeros(value_heads, dtype=torch.float32, device=compute_device)
    dt_bias = torch.zeros(value_heads * key_dim, dtype=torch.float32, device=compute_device)
    state = None
    if random_state:
        state = torch.randn(batch, value_heads, value_dim, key_dim, dtype=torch.float32, device=compute_device) * 0.02
    output, final_state = kda_reference_from_raw(
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
    assert output.shape == v.shape
    assert final_state.shape == (batch, value_heads, value_dim, key_dim)
    assert final_state.dtype == torch.float32
    assert torch.isfinite(output).all()
    assert torch.isfinite(final_state).all()


@pytest.mark.gpu
@pytest.mark.fla
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for FLA KDA")
@pytest.mark.skipif(importlib.util.find_spec("fla") is None, reason="FLA is not installed")
@pytest.mark.parametrize("sequence", [1, 2, 7, 32, 65, 129])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fla_target_numerical_matrix(sequence: int, dtype: torch.dtype) -> None:
    torch.manual_seed(42)
    device = torch.device("cuda")
    q = torch.randn(1, sequence, 16, 128, dtype=dtype, device=device)
    k = torch.randn_like(q)
    v = torch.randn(1, sequence, 32, 128, dtype=dtype, device=device)
    raw_decay = torch.randn(1, sequence, 32, 128, dtype=dtype, device=device)
    beta_logits = torch.randn(1, sequence, 32, dtype=dtype, device=device)
    a_log = torch.zeros(32, dtype=torch.float32, device=device)
    dt_bias = torch.zeros(32 * 128, dtype=torch.float32, device=device)
    output, final_state = fla_chunk_kda(q, k, v, raw_decay, beta_logits, a_log, dt_bias)
    assert output.shape == v.shape
    assert final_state.shape == (1, 32, 128, 128)
    assert final_state.dtype == torch.float32
    assert torch.isfinite(output).all()
    assert torch.isfinite(final_state).all()
