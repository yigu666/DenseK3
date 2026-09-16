"""Explicit DenseK3 recurrent-state abstraction."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from densek3_core.kda.contracts import KDAContract


@dataclass(frozen=True)
class KDAState:
    """An FP32 V-first recurrent state with physical shape `[B, HV, V, K]`."""

    tensor: torch.Tensor
    layout: str = "v_first"

    def __post_init__(self) -> None:
        if self.tensor.ndim != 4:
            raise ValueError("KDA state must have rank 4")
        if self.tensor.dtype != torch.float32:
            raise TypeError(f"KDA state must be float32, got {self.tensor.dtype}")
        if self.layout != "v_first":
            raise ValueError("DenseK3 P1 only supports V-first state layout")

    @property
    def batch_size(self) -> int:
        return self.tensor.shape[0]

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def validate_for(self, contract: KDAContract, batch_size: int | None = None) -> None:
        expected_batch = self.batch_size if batch_size is None else batch_size
        expected = contract.state_shape(expected_batch)
        if tuple(self.tensor.shape) != expected:
            raise ValueError(f"Expected V-first state shape {expected}, got {tuple(self.tensor.shape)}")

    def clone(self) -> KDAState:
        return KDAState(self.tensor.clone(), self.layout)

    def to(self, device: torch.device | str) -> KDAState:
        return KDAState(self.tensor.to(device=device, dtype=torch.float32), self.layout)

    @classmethod
    def zeros(cls, contract: KDAContract, batch_size: int, device: torch.device | str) -> KDAState:
        tensor = torch.zeros(contract.state_shape(batch_size), dtype=contract.torch_state_dtype, device=device)
        return cls(tensor=tensor, layout=contract.recurrent_state_layout)

    @classmethod
    def random(
        cls,
        contract: KDAContract,
        batch_size: int,
        device: torch.device | str,
        generator: torch.Generator | None = None,
        scale: float = 0.05,
    ) -> KDAState:
        tensor = torch.randn(
            contract.state_shape(batch_size),
            dtype=contract.torch_state_dtype,
            device=device,
            generator=generator,
        )
        return cls(tensor=tensor * scale, layout=contract.recurrent_state_layout)


def validate_state_tensor(
    state: torch.Tensor,
    batch_size: int,
    value_heads: int,
    value_dim: int,
    key_dim: int,
    device: torch.device,
) -> None:
    expected = (batch_size, value_heads, value_dim, key_dim)
    if tuple(state.shape) != expected:
        raise ValueError(f"Expected V-first state shape {expected}, got {tuple(state.shape)}")
    if state.dtype != torch.float32:
        raise TypeError(f"KDA recurrent state must be float32, got {state.dtype}")
    if state.device != device:
        raise ValueError(f"KDA recurrent state must be on {device}, got {state.device}")
