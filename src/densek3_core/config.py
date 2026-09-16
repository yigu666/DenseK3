"""P1 architecture configuration loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from densek3_core.kda.contracts import KDAContract

P1_CONFIG_RELATIVE_PATH = Path("configs/architecture/densek3-4b-p1.yaml")


@dataclass(frozen=True)
class DenseK3P1Config:
    """Validated subset of the locked P1 architecture configuration."""

    schema_version: int
    status: str
    model: dict[str, Any]
    layers: dict[str, Any]
    kda: KDAContract
    raw: dict[str, Any]
    source_path: Path


def find_project_root(start: Path | None = None) -> Path:
    """Find the project root containing the locked P1 configuration."""
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / P1_CONFIG_RELATIVE_PATH).is_file():
            return candidate
    raise FileNotFoundError(f"Could not find {P1_CONFIG_RELATIVE_PATH} above {current}")


def load_p1_config(path: str | Path | None = None) -> DenseK3P1Config:
    """Load and validate the P1 YAML without redefining architecture values."""
    source_path = Path(path).resolve() if path is not None else find_project_root() / P1_CONFIG_RELATIVE_PATH
    with source_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("P1 configuration must be a mapping")
    if raw.get("schema_version") != 1:
        raise ValueError(f"Unsupported P1 schema_version: {raw.get('schema_version')!r}")
    if raw.get("status") != "p1_architecture_locked":
        raise ValueError(f"P1 architecture is not locked: {raw.get('status')!r}")
    for section in ("model", "layers", "kda"):
        if not isinstance(raw.get(section), dict):
            raise ValueError(f"Missing mapping section: {section}")
    contract = KDAContract.from_mapping(raw["kda"])
    layer_types = raw["layers"].get("layer_types")
    if len(layer_types) != raw["model"].get("num_hidden_layers"):
        raise ValueError("P1 layer_types length does not match num_hidden_layers")
    return DenseK3P1Config(
        schema_version=raw["schema_version"],
        status=raw["status"],
        model=raw["model"],
        layers=raw["layers"],
        kda=contract,
        raw=raw,
        source_path=source_path,
    )

