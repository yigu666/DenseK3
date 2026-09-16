import pytest
import torch

from densek3_core.config import load_p1_config
from densek3_core.kda.contracts import KDAContract
from densek3_core.kda.state import KDAState, validate_state_tensor


def test_zero_state_matches_p1_contract(compute_device: torch.device) -> None:
    contract = load_p1_config().kda
    state = KDAState.zeros(contract, batch_size=2, device=compute_device)
    state.validate_for(contract, batch_size=2)
    assert state.tensor.shape == (2, 32, 128, 128)
    assert state.tensor.dtype == torch.float32
    assert state.tensor.device == compute_device
    assert torch.count_nonzero(state.tensor) == 0


def test_random_state_clone_is_independent(compute_device: torch.device) -> None:
    contract = load_p1_config().kda
    generator = torch.Generator(device=compute_device).manual_seed(42)
    state = KDAState.random(contract, 1, compute_device, generator=generator)
    cloned = state.clone()
    cloned.tensor.add_(1)
    assert not torch.equal(state.tensor, cloned.tensor)


def test_v_first_layout_is_observable_for_non_square_state() -> None:
    contract = KDAContract(
        qk_num_heads=2,
        value_num_heads=4,
        qk_head_dim=64,
        value_head_dim=128,
        short_conv_kernel_size=4,
        decay_projection_rank=64,
        qk_l2norm=True,
        beta_activation="sigmoid",
        allow_negative_eigenvalues=False,
        recurrent_state_dtype="float32",
        recurrent_state_layout="v_first",
        training_backend="fla_chunk",
        short_inference_backend="fla_fused_recurrent",
    )
    contract.validate()
    state = KDAState.zeros(contract, 2, "cpu")
    assert state.tensor.shape == (2, 4, 128, 64)
    validate_state_tensor(state.tensor, 2, 4, 128, 64, torch.device("cpu"))


def test_state_rejects_wrong_dtype() -> None:
    with pytest.raises(TypeError, match="float32"):
        KDAState(torch.zeros(1, 2, 4, 3, dtype=torch.bfloat16))

