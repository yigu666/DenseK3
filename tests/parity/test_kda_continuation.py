import importlib.util

import pytest
import torch

from densek3_core.kda.fla_backend import fla_chunk_kda, fla_fused_recurrent_kda
from densek3_core.kda.metrics import optimized_output_tolerance, optimized_state_tolerance
from densek3_core.kda.reference import kda_reference


@pytest.mark.parametrize("split", [1, 17, 47, 128])
def test_reference_continuation_invariant(split: int, compute_device: torch.device) -> None:
    torch.manual_seed(42)
    batch, sequence, qk_heads, value_heads, key_dim, value_dim = 1, 129, 2, 4, 8, 12
    q = torch.randn(batch, sequence, qk_heads, key_dim, device=compute_device)
    k = torch.randn_like(q)
    v = torch.randn(batch, sequence, value_heads, value_dim, device=compute_device)
    decay = -torch.rand(batch, sequence, value_heads, key_dim, device=compute_device)
    beta = torch.rand(batch, sequence, value_heads, device=compute_device)
    one_shot_output, one_shot_state = kda_reference(q, k, v, decay, beta, output_final_state=True)
    prefix_output, prefix_state = kda_reference(
        q[:, :split], k[:, :split], v[:, :split], decay[:, :split], beta[:, :split], output_final_state=True
    )
    suffix_output, resumed_state = kda_reference(
        q[:, split:],
        k[:, split:],
        v[:, split:],
        decay[:, split:],
        beta[:, split:],
        initial_state=prefix_state,
        output_final_state=True,
    )
    resumed_output = torch.cat((prefix_output, suffix_output), dim=1)
    torch.testing.assert_close(one_shot_output, resumed_output, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(one_shot_state, resumed_state, rtol=1e-5, atol=1e-6)


@pytest.mark.gpu
@pytest.mark.fla
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for FLA KDA")
@pytest.mark.skipif(importlib.util.find_spec("fla") is None, reason="FLA is not installed")
@pytest.mark.parametrize("split", [1, 17, 47, 128])
def test_fla_chunk_to_recurrent_continuation(split: int) -> None:
    torch.manual_seed(42)
    device = torch.device("cuda")
    q = torch.randn(1, 129, 16, 128, dtype=torch.bfloat16, device=device)
    k = torch.randn_like(q)
    v = torch.randn(1, 129, 32, 128, dtype=torch.bfloat16, device=device)
    raw_decay = torch.randn(1, 129, 32, 128, dtype=torch.bfloat16, device=device)
    beta_logits = torch.randn(1, 129, 32, dtype=torch.bfloat16, device=device)
    a_log = torch.zeros(32, dtype=torch.float32, device=device)
    dt_bias = torch.zeros(32 * 128, dtype=torch.float32, device=device)
    one_shot_output, one_shot_state = fla_chunk_kda(
        q, k, v, raw_decay, beta_logits, a_log, dt_bias
    )
    prefix_output, state = fla_chunk_kda(
        q[:, :split],
        k[:, :split],
        v[:, :split],
        raw_decay[:, :split],
        beta_logits[:, :split],
        a_log,
        dt_bias,
    )
    suffix_outputs = []
    for index in range(split, 129):
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
        suffix_outputs.append(output)
    resumed_output = torch.cat((prefix_output, *suffix_outputs), dim=1)
    output_rtol, output_atol = optimized_output_tolerance(torch.bfloat16)
    state_rtol, state_atol = optimized_state_tolerance(torch.bfloat16)
    torch.testing.assert_close(resumed_output, one_shot_output, rtol=output_rtol, atol=output_atol)
    torch.testing.assert_close(state, one_shot_state, rtol=state_rtol, atol=state_atol)
