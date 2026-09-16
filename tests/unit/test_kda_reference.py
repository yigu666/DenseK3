import pytest
import torch

from densek3_core.kda.reference import (
    kda_recurrent_step,
    kda_reference_packed,
    kda_reference_padded,
    l2_normalize_kda,
    qwen_compat_decay,
)


@pytest.mark.parametrize("activation_dtype", [torch.float32, torch.bfloat16])
def test_single_step_shape_dtype_finite_and_non_mutating(activation_dtype: torch.dtype, compute_device: torch.device) -> None:
    torch.manual_seed(42)
    batch, qk_heads, value_heads, key_dim, value_dim = 2, 2, 4, 8, 12
    q = torch.randn(batch, qk_heads, key_dim, dtype=activation_dtype, device=compute_device)
    k = torch.randn_like(q)
    v = torch.randn(batch, value_heads, value_dim, dtype=activation_dtype, device=compute_device)
    decay = -torch.rand(batch, value_heads, key_dim, dtype=torch.float32, device=compute_device)
    beta = torch.rand(batch, value_heads, dtype=activation_dtype, device=compute_device)
    state = torch.randn(batch, value_heads, value_dim, key_dim, dtype=torch.float32, device=compute_device) * 0.05
    original = state.clone()
    output, next_state = kda_recurrent_step(q, k, v, decay, beta, state)
    assert output.shape == v.shape
    assert output.dtype == activation_dtype
    assert next_state.shape == state.shape
    assert next_state.dtype == torch.float32
    assert torch.isfinite(output).all()
    assert torch.isfinite(next_state).all()
    assert torch.equal(state, original)
    assert not torch.equal(next_state, state)


def test_two_to_one_gva_mapping_is_semantic() -> None:
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    k = q.clone()
    v = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    decay = torch.zeros(1, 4, 2)
    beta = torch.ones(1, 4)
    state = torch.zeros(1, 4, 1, 2)
    output, _ = kda_recurrent_step(q, k, v, decay, beta, state, scale=1.0)
    torch.testing.assert_close(output.flatten(), torch.tensor([1.0, 2.0, 3.0, 4.0]), rtol=1e-6, atol=1e-6)


def test_l2_normalization_uses_sum_of_squares() -> None:
    value = torch.tensor([[3.0, 4.0]])
    normalized = l2_normalize_kda(value, eps=0.0)
    torch.testing.assert_close(normalized, torch.tensor([[0.6, 0.8]]), rtol=0, atol=0)


def test_qwen_compat_decay_shape_and_range() -> None:
    raw = torch.zeros(2, 3, 4, 5)
    a_log = torch.zeros(4)
    bias = torch.zeros(20)
    decay = qwen_compat_decay(raw, a_log, bias)
    assert decay.shape == raw.shape
    assert torch.all(decay < 0)


def test_packed_sequences_do_not_share_state() -> None:
    torch.manual_seed(42)
    q = torch.randn(1, 11, 2, 8)
    k = torch.randn_like(q)
    v = torch.randn(1, 11, 4, 12)
    decay = -torch.rand(1, 11, 4, 8)
    beta = torch.rand(1, 11, 4)
    offsets = torch.tensor([0, 1, 4, 11], dtype=torch.int32)
    packed_output, packed_state = kda_reference_packed(
        q, k, v, decay, beta, offsets, output_final_state=True
    )
    for index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        isolated_output, isolated_state = kda_reference_packed(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            decay[:, start:end],
            beta[:, start:end],
            torch.tensor([0, int(end - start)], dtype=torch.int32),
            output_final_state=True,
        )
        torch.testing.assert_close(packed_output[:, start:end], isolated_output)
        torch.testing.assert_close(packed_state[index : index + 1], isolated_state)


def test_left_padding_is_zero_and_matches_unpadded() -> None:
    torch.manual_seed(42)
    q = torch.randn(2, 7, 2, 8)
    k = torch.randn_like(q)
    v = torch.randn(2, 7, 4, 12)
    decay = -torch.rand(2, 7, 4, 8)
    beta = torch.rand(2, 7, 4)
    mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]])
    output, state = kda_reference_padded(q, k, v, decay, beta, mask, output_final_state=True)
    assert torch.count_nonzero(output[0, :3]) == 0
    isolated_output, isolated_state = kda_reference_padded(
        q[0:1, 3:], k[0:1, 3:], v[0:1, 3:], decay[0:1, 3:], beta[0:1, 3:], torch.ones(1, 4), output_final_state=True
    )
    torch.testing.assert_close(output[0:1, 3:], isolated_output)
    torch.testing.assert_close(state[0:1], isolated_state)
