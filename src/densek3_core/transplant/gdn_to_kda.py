"""Deterministic single-layer Qwen GDN to DenseK3 KDA conversion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

SOURCE_TENSOR_NAMES = (
    "A_log",
    "conv1d.weight",
    "dt_bias",
    "in_proj_a.weight",
    "in_proj_b.weight",
    "in_proj_qkv.weight",
    "in_proj_z.weight",
    "norm.weight",
    "out_proj.weight",
)
ALLOWED_OPERATIONS = {"COPY", "SLICE", "REPEAT", "RESHAPE", "CONSTRUCT", "EXCLUDE"}
FORBIDDEN_OPERATIONS = {"UNKNOWN", "UNMAPPED", "SILENT_INIT"}


@dataclass(frozen=True)
class GDNBridgeConfig:
    """Dimensions needed by the P3 single-layer bridge."""

    layer_index: int
    hidden_size: int
    qk_num_heads: int
    value_num_heads: int
    qk_head_dim: int
    value_head_dim: int
    conv_kernel_size: int
    decay_projection_rank: int
    rms_norm_eps: float

    @property
    def qk_dim(self) -> int:
        return self.qk_num_heads * self.qk_head_dim

    @property
    def value_dim(self) -> int:
        return self.value_num_heads * self.value_head_dim

    @property
    def qkv_dim(self) -> int:
        return 2 * self.qk_dim + self.value_dim

    @property
    def source_prefix(self) -> str:
        return f"model.language_model.layers.{self.layer_index}.linear_attn."

    @property
    def target_prefix(self) -> str:
        return f"model.layers.{self.layer_index}.self_attn."

    def validate(self) -> None:
        if self.layer_index < 0:
            raise ValueError("layer_index must be non-negative")
        if self.value_num_heads % self.qk_num_heads:
            raise ValueError("value heads must be divisible by Q/K heads")
        if self.value_num_heads // self.qk_num_heads != 2:
            raise ValueError("P3 requires the locked 2:1 GVA mapping")
        if self.decay_projection_rank < self.value_num_heads:
            raise ValueError("decay projection rank must embed every scalar decay head")
        if min(self.hidden_size, self.qk_head_dim, self.value_head_dim, self.conv_kernel_size) <= 0:
            raise ValueError("all bridge dimensions must be positive")

    def expected_source_shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "A_log": (self.value_num_heads,),
            "conv1d.weight": (self.qkv_dim, 1, self.conv_kernel_size),
            "dt_bias": (self.value_num_heads,),
            "in_proj_a.weight": (self.value_num_heads, self.hidden_size),
            "in_proj_b.weight": (self.value_num_heads, self.hidden_size),
            "in_proj_qkv.weight": (self.qkv_dim, self.hidden_size),
            "in_proj_z.weight": (self.value_dim, self.hidden_size),
            "norm.weight": (self.value_head_dim,),
            "out_proj.weight": (self.hidden_size, self.value_dim),
        }

    def expected_target_shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "A_log": (self.value_num_heads,),
            "dt_bias": (self.value_num_heads * self.qk_head_dim,),
            "q_proj.weight": (self.qk_dim, self.hidden_size),
            "k_proj.weight": (self.qk_dim, self.hidden_size),
            "v_proj.weight": (self.value_dim, self.hidden_size),
            "q_conv1d.weight": (self.qk_dim, 1, self.conv_kernel_size),
            "k_conv1d.weight": (self.qk_dim, 1, self.conv_kernel_size),
            "v_conv1d.weight": (self.value_dim, 1, self.conv_kernel_size),
            "f_a_proj.weight": (self.decay_projection_rank, self.hidden_size),
            "f_b_proj.weight": (self.value_num_heads * self.qk_head_dim, self.decay_projection_rank),
            "b_proj.weight": (self.value_num_heads, self.hidden_size),
            "g_proj.weight": (self.value_dim, self.hidden_size),
            "o_norm.weight": (self.value_head_dim,),
            "o_proj.weight": (self.hidden_size, self.value_dim),
        }

    @classmethod
    def from_model_directory(cls, model_dir: Path | str, layer_index: int = 0) -> GDNBridgeConfig:
        model_dir = Path(model_dir)
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        text = config["text_config"]
        layer_types = text["layer_types"]
        if layer_index >= len(layer_types) or layer_types[layer_index] != "linear_attention":
            raise ValueError(f"Qwen layer {layer_index} is not a linear-attention layer")
        bridge = cls(
            layer_index=layer_index,
            hidden_size=int(text["hidden_size"]),
            qk_num_heads=int(text["linear_num_key_heads"]),
            value_num_heads=int(text["linear_num_value_heads"]),
            qk_head_dim=int(text["linear_key_head_dim"]),
            value_head_dim=int(text["linear_value_head_dim"]),
            conv_kernel_size=int(text["linear_conv_kernel_dim"]),
            decay_projection_rank=128,
            rms_norm_eps=float(text["rms_norm_eps"]),
        )
        bridge.validate()
        return bridge


@dataclass(frozen=True)
class ConversionResult:
    """Converted tensors and their complete deterministic manifest."""

    config: GDNBridgeConfig
    target_state: dict[str, torch.Tensor]
    manifest: dict[str, Any]

    @property
    def parameter_hash(self) -> str:
        return hash_state_dict(self.target_state)

    @property
    def manifest_hash(self) -> str:
        return hash_json(self.manifest)


def hash_json(value: Any) -> str:
    """Hash JSON after canonical key and separator normalization."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def hash_state_dict(state: dict[str, torch.Tensor]) -> str:
    """Hash tensor names, dtypes, shapes, and exact bytes in lexical key order."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def audit_source_state(source_state: dict[str, torch.Tensor], config: GDNBridgeConfig) -> dict[str, Any]:
    """Require the exact P1-locked source key set and shapes."""
    expected = config.expected_source_shapes()
    missing = sorted(set(expected) - set(source_state))
    unexpected = sorted(set(source_state) - set(expected))
    shape_mismatches = {
        name: {"expected": list(expected[name]), "actual": list(source_state[name].shape)}
        for name in sorted(set(expected) & set(source_state))
        if tuple(source_state[name].shape) != expected[name]
    }
    if missing or unexpected or shape_mismatches:
        raise ValueError(
            f"Qwen GDN source audit failed: missing={missing}, unexpected={unexpected}, shapes={shape_mismatches}"
        )
    tensors = [
        {
            "name": name,
            "shape": list(source_state[name].shape),
            "dtype": str(source_state[name].dtype).removeprefix("torch."),
            "numel": source_state[name].numel(),
        }
        for name in SOURCE_TENSOR_NAMES
    ]
    return {
        "status": "PASS",
        "layer_index": config.layer_index,
        "source_prefix": config.source_prefix,
        "tensor_count": len(tensors),
        "total_parameters": sum(item["numel"] for item in tensors),
        "source_hash": hash_state_dict(source_state),
        "tensors": tensors,
    }


def audit_target_state(target_state: dict[str, torch.Tensor], config: GDNBridgeConfig) -> None:
    """Require the exact single-layer target key set and shapes."""
    expected = config.expected_target_shapes()
    missing = sorted(set(expected) - set(target_state))
    unexpected = sorted(set(target_state) - set(expected))
    shape_mismatches = {
        name: {"expected": list(expected[name]), "actual": list(target_state[name].shape)}
        for name in sorted(set(expected) & set(target_state))
        if tuple(target_state[name].shape) != expected[name]
    }
    if missing or unexpected or shape_mismatches:
        raise ValueError(
            f"DenseK3 KDA target audit failed: missing={missing}, unexpected={unexpected}, shapes={shape_mismatches}"
        )


def load_qwen_gdn_layer(
    model_dir: Path | str,
    layer_index: int = 0,
    device: str = "cpu",
) -> tuple[GDNBridgeConfig, dict[str, torch.Tensor], dict[str, Any]]:
    """Load only one real GDN layer from the indexed Qwen checkpoint."""
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("P3 checkpoint loading requires safetensors") from exc

    model_dir = Path(model_dir).resolve()
    config = GDNBridgeConfig.from_model_directory(model_dir, layer_index)
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    full_names = [config.source_prefix + name for name in SOURCE_TENSOR_NAMES]
    indexed_layer_names = sorted(name for name in weight_map if name.startswith(config.source_prefix))
    if indexed_layer_names != sorted(full_names):
        missing = sorted(set(full_names) - set(indexed_layer_names))
        unexpected = sorted(set(indexed_layer_names) - set(full_names))
        raise ValueError(f"Checkpoint index GDN key mismatch: missing={missing}, unexpected={unexpected}")

    source_state: dict[str, torch.Tensor] = {}
    by_shard: dict[str, list[str]] = {}
    for full_name in full_names:
        by_shard.setdefault(weight_map[full_name], []).append(full_name)
    for shard_name in sorted(by_shard):
        shard_path = model_dir / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing Qwen checkpoint shard: {shard_path}")
        with safe_open(shard_path, framework="pt", device=device) as handle:
            for full_name in sorted(by_shard[shard_name]):
                source_state[full_name.removeprefix(config.source_prefix)] = handle.get_tensor(full_name)
    audit = audit_source_state(source_state, config)
    audit.update(
        {
            "model_directory": str(model_dir),
            "index_path": str(index_path),
            "checkpoint_shards": sorted(by_shard),
        }
    )
    return config, source_state, audit


def _manifest_entry(
    operation: str,
    target_key: str,
    target: torch.Tensor,
    source_keys: list[str],
    source_state: dict[str, torch.Tensor],
    transform: str,
) -> dict[str, Any]:
    return {
        "operation": operation,
        "source_keys": source_keys,
        "target_key": target_key,
        "source_shapes": [list(source_state[name].shape) for name in source_keys],
        "target_shape": list(target.shape),
        "target_dtype": str(target.dtype).removeprefix("torch."),
        "target_numel": target.numel(),
        "transform": transform,
    }


def _validate_manifest(
    manifest: dict[str, Any],
    source_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
) -> None:
    operations = {entry["operation"] for entry in manifest["entries"]}
    if operations & FORBIDDEN_OPERATIONS or not operations <= ALLOWED_OPERATIONS:
        raise ValueError(f"Invalid parameter operations: {sorted(operations)}")
    covered_sources = {name for entry in manifest["entries"] for name in entry["source_keys"]}
    covered_targets = {entry["target_key"] for entry in manifest["entries"]}
    if covered_sources != set(source_state):
        raise ValueError(f"Source manifest coverage mismatch: {sorted(set(source_state) - covered_sources)}")
    if covered_targets != set(target_state):
        raise ValueError(f"Target manifest coverage mismatch: {sorted(set(target_state) - covered_targets)}")
    if manifest["source_total_parameters"] != sum(tensor.numel() for tensor in source_state.values()):
        raise ValueError("Source parameter total is inconsistent")
    if manifest["target_total_parameters"] != sum(tensor.numel() for tensor in target_state.values()):
        raise ValueError("Target parameter total is inconsistent")


def convert_gdn_to_kda(
    source_state: dict[str, torch.Tensor],
    config: GDNBridgeConfig,
) -> ConversionResult:
    """Apply the P1 bridge without random initialization or implicit parameters."""
    audit_source_state(source_state, config)
    qk_dim = config.qk_dim
    value_dim = config.value_dim
    qkv = source_state["in_proj_qkv.weight"]
    conv = source_state["conv1d.weight"]
    a_weight = source_state["in_proj_a.weight"]

    target: dict[str, torch.Tensor] = {
        "A_log": source_state["A_log"].clone(),
        "dt_bias": source_state["dt_bias"].repeat_interleave(config.qk_head_dim).clone(),
        "q_proj.weight": qkv[:qk_dim].clone(),
        "k_proj.weight": qkv[qk_dim : 2 * qk_dim].clone(),
        "v_proj.weight": qkv[2 * qk_dim : 2 * qk_dim + value_dim].clone(),
        "q_conv1d.weight": conv[:qk_dim].clone(),
        "k_conv1d.weight": conv[qk_dim : 2 * qk_dim].clone(),
        "v_conv1d.weight": conv[2 * qk_dim : 2 * qk_dim + value_dim].clone(),
        "b_proj.weight": source_state["in_proj_b.weight"].clone(),
        "g_proj.weight": source_state["in_proj_z.weight"].clone(),
        "o_norm.weight": source_state["norm.weight"].clone(),
        "o_proj.weight": source_state["out_proj.weight"].clone(),
    }
    f_a = a_weight.new_zeros((config.decay_projection_rank, config.hidden_size))
    f_a[: config.value_num_heads].copy_(a_weight)
    f_b = a_weight.new_zeros((config.value_num_heads * config.qk_head_dim, config.decay_projection_rank))
    f_b_view = f_b.view(config.value_num_heads, config.qk_head_dim, config.decay_projection_rank)
    heads = torch.arange(config.value_num_heads, device=f_b.device).view(-1, 1)
    channels = torch.arange(config.qk_head_dim, device=f_b.device).view(1, -1)
    f_b_view[heads, channels, heads.expand(-1, config.qk_head_dim)] = 1
    target["f_a_proj.weight"] = f_a
    target["f_b_proj.weight"] = f_b
    audit_target_state(target, config)

    entries = [
        _manifest_entry("COPY", "A_log", target["A_log"], ["A_log"], source_state, "copy"),
        _manifest_entry(
            "REPEAT",
            "dt_bias",
            target["dt_bias"],
            ["dt_bias"],
            source_state,
            "repeat_interleave_qk_channels",
        ),
        _manifest_entry(
            "SLICE", "q_proj.weight", target["q_proj.weight"], ["in_proj_qkv.weight"], source_state, "q_rows"
        ),
        _manifest_entry(
            "SLICE", "k_proj.weight", target["k_proj.weight"], ["in_proj_qkv.weight"], source_state, "k_rows"
        ),
        _manifest_entry(
            "SLICE", "v_proj.weight", target["v_proj.weight"], ["in_proj_qkv.weight"], source_state, "v_rows"
        ),
        _manifest_entry(
            "SLICE", "q_conv1d.weight", target["q_conv1d.weight"], ["conv1d.weight"], source_state, "q_rows"
        ),
        _manifest_entry(
            "SLICE", "k_conv1d.weight", target["k_conv1d.weight"], ["conv1d.weight"], source_state, "k_rows"
        ),
        _manifest_entry(
            "SLICE", "v_conv1d.weight", target["v_conv1d.weight"], ["conv1d.weight"], source_state, "v_rows"
        ),
        _manifest_entry(
            "CONSTRUCT",
            "f_a_proj.weight",
            target["f_a_proj.weight"],
            ["in_proj_a.weight"],
            source_state,
            "copy_scalar_decay_rows_then_zero_remaining_rank",
        ),
        _manifest_entry(
            "CONSTRUCT",
            "f_b_proj.weight",
            target["f_b_proj.weight"],
            [],
            source_state,
            "head_channel_identity_embedding",
        ),
        _manifest_entry(
            "COPY", "b_proj.weight", target["b_proj.weight"], ["in_proj_b.weight"], source_state, "copy"
        ),
        _manifest_entry(
            "COPY", "g_proj.weight", target["g_proj.weight"], ["in_proj_z.weight"], source_state, "copy"
        ),
        _manifest_entry(
            "COPY", "o_norm.weight", target["o_norm.weight"], ["norm.weight"], source_state, "copy"
        ),
        _manifest_entry(
            "COPY", "o_proj.weight", target["o_proj.weight"], ["out_proj.weight"], source_state, "copy"
        ),
    ]
    manifest = {
        "schema_version": 1,
        "stage": "P3",
        "layer_index": config.layer_index,
        "source_prefix": config.source_prefix,
        "target_prefix": config.target_prefix,
        "config": asdict(config),
        "allowed_operations": sorted(ALLOWED_OPERATIONS),
        "forbidden_operations": sorted(FORBIDDEN_OPERATIONS),
        "source_tensor_count": len(source_state),
        "target_tensor_count": len(target),
        "source_total_parameters": sum(tensor.numel() for tensor in source_state.values()),
        "target_total_parameters": sum(tensor.numel() for tensor in target.values()),
        "source_hash": hash_state_dict(source_state),
        "target_hash": hash_state_dict(target),
        "entries": entries,
    }
    _validate_manifest(manifest, source_state, target)
    return ConversionResult(config=config, target_state=target, manifest=manifest)


def save_conversion(result: ConversionResult, output_dir: Path | str) -> dict[str, Any]:
    """Save one converted layer and return reproducibility metadata."""
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("P3 conversion saving requires safetensors") from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "model.safetensors"
    manifest_path = output_dir / "parameter-manifest.json"
    report_path = output_dir / "conversion-report.json"
    save_file(
        {name: tensor.detach().cpu().contiguous() for name, tensor in result.target_state.items()},
        weights_path,
        metadata={"format": "densek3_p3_single_layer", "layer_index": str(result.config.layer_index)},
    )
    manifest_path.write_text(json.dumps(result.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "stage": "P3",
        "layer_index": result.config.layer_index,
        "parameter_hash": result.parameter_hash,
        "manifest_hash": result.manifest_hash,
        "weights_file": weights_path.name,
        "manifest_file": manifest_path.name,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def load_saved_conversion(output_dir: Path | str) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    """Reload a P3 artifact and verify both recorded hashes."""
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("P3 conversion loading requires safetensors") from exc

    output_dir = Path(output_dir)
    state = load_file(output_dir / "model.safetensors", device="cpu")
    manifest = json.loads((output_dir / "parameter-manifest.json").read_text(encoding="utf-8"))
    report = json.loads((output_dir / "conversion-report.json").read_text(encoding="utf-8"))
    if hash_state_dict(state) != report["parameter_hash"]:
        raise ValueError("Reloaded P3 parameter hash differs from the saved report")
    if hash_json(manifest) != report["manifest_hash"]:
        raise ValueError("Reloaded P3 manifest hash differs from the saved report")
    return state, manifest, report
