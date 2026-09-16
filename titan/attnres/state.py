"""Depth-state decomposition for Titan P8 Block AttnRes."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class DepthBlockState:
    """Embedding source, completed block sums, and current partial block."""

    embedding: torch.Tensor
    block_size_residuals: int
    completed: list[torch.Tensor] = field(default_factory=list)
    partial: torch.Tensor | None = None
    residual_count: int = 0

    def __post_init__(self) -> None:
        if self.block_size_residuals <= 0:
            raise ValueError("Block size must be positive")
        if not self.completed:
            self.completed = [self.embedding]

    def sources(self) -> tuple[torch.Tensor, ...]:
        values = list(self.completed)
        if self.partial is not None:
            values.append(self.partial)
        return tuple(values)

    def reconstruct(self) -> torch.Tensor:
        values = self.sources()
        if not values:
            raise RuntimeError("AttnRes state has no depth sources")
        result = values[0]
        for value in values[1:]:
            result = result + value
        return result

    def update(self, branch_output: torch.Tensor) -> bool:
        if branch_output.shape != self.embedding.shape:
            raise ValueError("Residual branch output shape changed")
        self.partial = (
            branch_output
            if self.partial is None
            else self.partial + branch_output
        )
        self.residual_count += 1
        boundary = self.residual_count % self.block_size_residuals == 0
        if boundary:
            if self.partial is None:
                raise RuntimeError("Cannot close an empty residual block")
            self.completed.append(self.partial)
            self.partial = None
        return boundary

    @property
    def completed_residual_blocks(self) -> int:
        return len(self.completed) - 1
