"""Low-memory indexed checkpoint loading for the P4 server runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from densek3_core.modeling import DenseK3Config, DenseK3ForCausalLM


def _dependencies():
    try:
        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("P4 indexed model loading requires accelerate and safetensors") from exc
    return init_empty_weights, set_module_tensor_to_device, safe_open


def _load_indexed_parameters(
    model: torch.nn.Module,
    checkpoint_dir: Path,
    name_map: dict[str, str],
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    _, set_module_tensor_to_device, safe_open = _dependencies()
    index = json.loads((checkpoint_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    target_to_source = {target: source for source, target in name_map.items()}
    model_parameters = dict(model.named_parameters(remove_duplicate=False))
    missing_targets = sorted(set(target_to_source) - set(model_parameters))
    if missing_targets:
        raise ValueError(f"Runtime target parameters do not exist in model: {missing_targets}")
    by_shard: dict[str, list[str]] = {}
    for source_name in name_map:
        by_shard.setdefault(weight_map[source_name], []).append(source_name)
    for shard_name in sorted(by_shard):
        with safe_open(checkpoint_dir / shard_name, framework="pt", device="cpu") as handle:
            for source_name in sorted(by_shard[shard_name]):
                target_name = name_map[source_name]
                value = handle.get_tensor(source_name)
                parameter_dtype = (
                    torch.float32 if target_name.endswith((".A_log", ".dt_bias")) else dtype
                )
                set_module_tensor_to_device(
                    model,
                    target_name,
                    device,
                    value=value,
                    dtype=parameter_dtype,
                )
    loaded = set(target_to_source)
    allowed_tied_missing = {"lm_head.weight"}
    unexpected_missing = sorted(set(model_parameters) - loaded - allowed_tied_missing)
    if unexpected_missing:
        raise ValueError(f"Runtime checkpoint left model parameters uninitialized: {unexpected_missing}")


def _move_buffers(model: torch.nn.Module, device: torch.device) -> None:
    _, set_module_tensor_to_device, _ = _dependencies()
    for name, value in list(model.named_buffers()):
        if value.device.type == "meta":
            raise ValueError(f"Cannot materialize meta buffer without a value: {name}")
        set_module_tensor_to_device(model, name, device, value=value)


def _assert_materialized(model: torch.nn.Module) -> None:
    meta = [name for name, value in model.named_parameters() if value.device.type == "meta"]
    if meta:
        raise ValueError(f"Model still contains meta parameters after indexed loading: {meta}")


def load_densek3_model(
    checkpoint_dir: str | Path,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    config: DenseK3Config | None = None,
) -> DenseK3ForCausalLM:
    """Load the 4.2B artifact directly to its authoritative GPU device."""
    init_empty_weights, _, _ = _dependencies()
    checkpoint_dir = Path(checkpoint_dir).resolve()
    config = config or DenseK3Config.from_pretrained(checkpoint_dir)
    with init_empty_weights(include_buffers=False):
        model = DenseK3ForCausalLM(config)
    index = json.loads((checkpoint_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    names = sorted(index["weight_map"])
    _load_indexed_parameters(model, checkpoint_dir, {name: name for name in names}, torch.device(device), dtype)
    model.tie_weights()
    _move_buffers(model, torch.device(device))
    _assert_materialized(model)
    model.eval()
    return model


def load_densek3_model_from_pretrained(
    checkpoint_dir: str | Path,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> DenseK3ForCausalLM:
    """Exercise the public HF loader, then restore the P1-required FP32 gate parameters."""
    _, set_module_tensor_to_device, safe_open = _dependencies()
    checkpoint_dir = Path(checkpoint_dir).resolve()
    load_kwargs: dict[str, Any] = {
        "device_map": {"": str(device)},
        "low_cpu_mem_usage": True,
    }
    try:
        model = DenseK3ForCausalLM.from_pretrained(checkpoint_dir, dtype=dtype, **load_kwargs)
    except TypeError:
        # Compatibility for the local Transformers 4.x test environment. The
        # authoritative server uses the v5 `dtype` spelling.
        model = DenseK3ForCausalLM.from_pretrained(checkpoint_dir, torch_dtype=dtype, **load_kwargs)
    index = json.loads((checkpoint_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    selected = [name for name in index["weight_map"] if name.endswith((".A_log", ".dt_bias"))]
    by_shard: dict[str, list[str]] = {}
    for name in selected:
        by_shard.setdefault(index["weight_map"][name], []).append(name)
    for shard, names in sorted(by_shard.items()):
        with safe_open(checkpoint_dir / shard, framework="pt", device="cpu") as handle:
            for name in sorted(names):
                set_module_tensor_to_device(
                    model,
                    name,
                    torch.device(device),
                    value=handle.get_tensor(name),
                    dtype=torch.float32,
                )
    model.tie_weights()
    _assert_materialized(model)
    model.eval()
    return model


def _qwen_text_classes() -> tuple[type[Any], type[Any]]:
    try:
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    except ImportError as exc:
        raise RuntimeError("P4 donor evaluation requires Transformers with Qwen3.5 text classes") from exc
    return Qwen3_5TextConfig, Qwen3_5ForCausalLM


def load_qwen_text_model(
    source_dir: str | Path,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.nn.Module:
    """Load only the original Qwen text backbone, excluding vision and MTP."""
    init_empty_weights, _, _ = _dependencies()
    source_dir = Path(source_dir).resolve()
    full_config = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
    config_class, model_class = _qwen_text_classes()
    config = config_class.from_dict(full_config["text_config"])
    with init_empty_weights(include_buffers=False):
        model = model_class(config)
    index = json.loads((source_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    prefix = "model.language_model."
    name_map = {
        source_name: "model." + source_name.removeprefix(prefix)
        for source_name in sorted(index["weight_map"])
        if source_name.startswith(prefix)
    }
    _load_indexed_parameters(model, source_dir, name_map, torch.device(device), dtype)
    model.tie_weights()
    _move_buffers(model, torch.device(device))
    _assert_materialized(model)
    model.eval()
    return model


__all__ = ["load_densek3_model", "load_densek3_model_from_pretrained", "load_qwen_text_model"]
