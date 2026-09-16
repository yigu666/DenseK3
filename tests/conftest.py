"""Shared deterministic KDA test fixtures."""

from __future__ import annotations

import importlib.util

import pytest
import torch


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: requires a CUDA device")
    config.addinivalue_line("markers", "fla: requires the pinned FLA runtime")


@pytest.fixture(scope="session")
def compute_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def fla_runtime_available() -> bool:
    return torch.cuda.is_available() and importlib.util.find_spec("fla") is not None
