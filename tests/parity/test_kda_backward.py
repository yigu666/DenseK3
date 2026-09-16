import importlib.util

import pytest
import torch

from densek3_core.kda.fla_backend import fla_chunk_kda
from densek3_core.kda.reference import kda_reference_from_raw

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.fla,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for FLA KDA"),
    pytest.mark.skipif(importlib.util.find_spec("fla") is None, reason="FLA is not installed"),
]


def _run(backend: str):
    torch.manual_seed(42)
    device = torch.device("cuda")
    q = torch.randn(1, 7, 16, 32, dtype=torch.bfloat16, device=device, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn(1, 7, 32, 32, dtype=torch.bfloat16, device=device, requires_grad=True)
    raw_decay = torch.randn(1, 7, 32, 32, dtype=torch.bfloat16, device=device, requires_grad=True)
    beta_logits = torch.randn(1, 7, 32, dtype=torch.bfloat16, device=device, requires_grad=True)
    a_log = torch.zeros(32, dtype=torch.float32, device=device, requires_grad=True)
    dt_bias = torch.zeros(32 * 32, dtype=torch.float32, device=device, requires_grad=True)
    state = torch.randn(1, 32, 32, 32, dtype=torch.float32, device=device) * 0.02
    state.requires_grad_(True)
    if backend == "reference":
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
    else:
        output, final_state = fla_chunk_kda(
            q, k, v, raw_decay, beta_logits, a_log, dt_bias, initial_state=state
        )
    output_weight = torch.linspace(-0.5, 0.5, output.numel(), device=device).reshape_as(output)
    state_weight = torch.linspace(-0.1, 0.1, final_state.numel(), device=device).reshape_as(final_state)
    loss = (output.float() * output_weight).sum() + (final_state * state_weight).sum()
    gradients = torch.autograd.grad(loss, (q, k, v, raw_decay, beta_logits, a_log, dt_bias, state))
    return gradients


def test_reference_matches_fla_chunk_backward() -> None:
    reference_gradients = _run("reference")
    actual_gradients = _run("fla")
    names = ("dq", "dk", "dv", "ddecay", "dbeta", "dA_log", "ddt_bias", "d_initial_state")
    for name, reference, actual in zip(names, reference_gradients, actual_gradients, strict=True):
        assert torch.isfinite(actual).all(), name
        tolerance = 1e-2 if name in {"dA_log", "ddt_bias"} else 8e-3
        torch.testing.assert_close(actual, reference, rtol=tolerance, atol=tolerance, msg=name)
