import pytest
import torch

from densek3_core.kda.reference import kda_recurrent_step, kda_reference


def _inputs(batch: int, sequence: int, dtype: torch.dtype, device: torch.device):
    torch.manual_seed(42)
    q = torch.randn(batch, sequence, 2, 8, dtype=dtype, device=device)
    k = torch.randn_like(q)
    v = torch.randn(batch, sequence, 4, 12, dtype=dtype, device=device)
    decay = -torch.rand(batch, sequence, 4, 8, dtype=torch.float32, device=device)
    beta = torch.rand(batch, sequence, 4, dtype=dtype, device=device)
    return q, k, v, decay, beta


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("sequence", [1, 2, 7, 32, 129])
@pytest.mark.parametrize("activation_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("initial_state_mode", ["none", "random"])
def test_full_matches_token_recurrent(
    batch: int,
    sequence: int,
    activation_dtype: torch.dtype,
    initial_state_mode: str,
    compute_device: torch.device,
) -> None:
    q, k, v, decay, beta = _inputs(batch, sequence, activation_dtype, compute_device)
    initial_state = None
    if initial_state_mode == "random":
        initial_state = torch.randn(batch, 4, 12, 8, dtype=torch.float32, device=compute_device) * 0.05
    full_output, full_state = kda_reference(
        q, k, v, decay, beta, initial_state=initial_state, output_final_state=True
    )
    state = torch.zeros(batch, 4, 12, 8, dtype=torch.float32, device=compute_device)
    if initial_state is not None:
        state = initial_state.clone()
    recurrent_outputs = []
    for token_index in range(sequence):
        output_t, state = kda_recurrent_step(
            q[:, token_index], k[:, token_index], v[:, token_index], decay[:, token_index], beta[:, token_index], state
        )
        recurrent_outputs.append(output_t)
    recurrent_output = torch.stack(recurrent_outputs, dim=1)
    torch.testing.assert_close(full_output, recurrent_output, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(full_state, state, rtol=1e-5, atol=1e-6)


def test_full_supports_target_gva_shape(compute_device: torch.device) -> None:
    torch.manual_seed(42)
    q = torch.randn(1, 2, 16, 128, device=compute_device)
    k = torch.randn_like(q)
    v = torch.randn(1, 2, 32, 128, device=compute_device)
    decay = -torch.rand(1, 2, 32, 128, device=compute_device)
    beta = torch.rand(1, 2, 32, device=compute_device)
    output, state = kda_reference(q, k, v, decay, beta, output_final_state=True)
    assert output.shape == (1, 2, 32, 128)
    assert state.shape == (1, 32, 128, 128)
    assert state.dtype == torch.float32

