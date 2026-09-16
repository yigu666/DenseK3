"""Deterministic P4 conversion of the complete Qwen text backbone."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from densek3_core.modeling.configuration_densek3 import DenseK3Config
from densek3_core.transplant.gdn_to_kda import (
    ALLOWED_OPERATIONS,
    FORBIDDEN_OPERATIONS,
    GDNBridgeConfig,
    convert_gdn_to_kda,
    hash_json,
)

P4_EXPECTED_SOURCE_COUNTS = {"text": 426, "vision": 297, "mtp": 15, "total": 738}
P4_EXPECTED_TARGET_TENSORS = 546
P4_EXPECTED_UNIQUE_PARAMETERS = 4_224_329_984
P4_MANIFEST_NAME = "p4-parameter-manifest.json"


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _canonical_parameter_hash(entries: Iterable[dict[str, Any]]) -> str:
    records = [
        {
            "name": entry["target_key"],
            "dtype": entry["dtype"],
            "shape": entry["shape"],
            "sha256": entry["sha256"],
        }
        for entry in entries
    ]
    return hash_json(sorted(records, key=lambda item: item["name"]))


def load_checkpoint_index(model_dir: str | Path) -> dict[str, Any]:
    """Load and minimally validate the donor safetensors index."""
    path = Path(model_dir) / "model.safetensors.index.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(index.get("weight_map"), dict):
        raise ValueError("Donor index does not contain a weight_map")
    return index


def classify_source_key(name: str) -> str:
    """Classify every donor tensor into the three P4 inventory domains."""
    if name.startswith("model.language_model."):
        return "text"
    if name.startswith("model.visual."):
        return "vision"
    if name.startswith("mtp."):
        return "mtp"
    raise ValueError(f"Unknown donor parameter domain: {name}")


def audit_source_inventory(model_dir: str | Path) -> dict[str, Any]:
    """Replay the P1 source inventory from the real checkpoint index."""
    weight_map = load_checkpoint_index(model_dir)["weight_map"]
    counts = Counter(classify_source_key(name) for name in weight_map)
    observed = {**counts, "total": len(weight_map)}
    if observed != P4_EXPECTED_SOURCE_COUNTS:
        raise ValueError(f"P4 source inventory differs from P1: {observed}")
    config = DenseK3Config.from_qwen_model_directory(model_dir)
    return {
        "stage": "P4.0",
        "status": "PASS",
        "counts": observed,
        "num_hidden_layers": config.num_hidden_layers,
        "kda_layers": len(config.kda_indices),
        "gqa_layers": len(config.gqa_indices),
        "kda_indices": config.kda_indices,
        "gqa_indices": config.gqa_indices,
        "marker": "P4_00_SOURCE_INVENTORY=PASS",
    }


class IndexedTensorReader:
    """Read selected tensors from an indexed checkpoint without loading vision/MTP."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir).resolve()
        self.weight_map: dict[str, str] = load_checkpoint_index(self.model_dir)["weight_map"]

    def load(self, names: Iterable[str], device: str = "cpu") -> dict[str, torch.Tensor]:
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise RuntimeError("P4 conversion requires safetensors") from exc
        requested = list(names)
        missing = sorted(set(requested) - set(self.weight_map))
        if missing:
            raise KeyError(f"Donor tensors absent from checkpoint index: {missing}")
        by_shard: dict[str, list[str]] = {}
        for name in requested:
            by_shard.setdefault(self.weight_map[name], []).append(name)
        result: dict[str, torch.Tensor] = {}
        for shard_name in sorted(by_shard):
            with safe_open(self.model_dir / shard_name, framework="pt", device=device) as handle:
                for name in sorted(by_shard[shard_name]):
                    result[name] = handle.get_tensor(name)
        return result


class DeterministicShardWriter:
    """Write lexical, size-bounded safetensors shards and their HF index."""

    def __init__(
        self,
        output_dir: str | Path,
        max_shard_size: int = 4_000_000_000,
        *,
        stage: str = "P4",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_shard_size = max_shard_size
        self.stage = stage
        self.pending: dict[str, torch.Tensor] = {}
        self.pending_bytes = 0
        self.temporary_files: list[Path] = []
        self.tensor_to_temporary: dict[str, int] = {}
        self.total_size = 0

    def _flush(self) -> None:
        if not self.pending:
            return
        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise RuntimeError("P4 conversion requires safetensors") from exc
        index = len(self.temporary_files) + 1
        path = self.output_dir / f".p4-shard-{index:05d}.safetensors"
        save_file(
            {name: tensor.detach().cpu().contiguous() for name, tensor in sorted(self.pending.items())},
            path,
            metadata={"format": "pt", "stage": self.stage},
        )
        self.temporary_files.append(path)
        for name in self.pending:
            self.tensor_to_temporary[name] = index - 1
        self.pending = {}
        self.pending_bytes = 0

    def add(self, name: str, tensor: torch.Tensor) -> None:
        if name in self.pending or name in self.tensor_to_temporary:
            raise ValueError(f"Duplicate target tensor: {name}")
        size = tensor.numel() * tensor.element_size()
        if self.pending and self.pending_bytes + size > self.max_shard_size:
            self._flush()
        self.pending[name] = tensor.detach().cpu().contiguous()
        self.pending_bytes += size
        self.total_size += size

    def finish(self) -> dict[str, Any]:
        self._flush()
        shard_count = len(self.temporary_files)
        weight_map: dict[str, str] = {}
        for temporary_index, temporary in enumerate(self.temporary_files):
            final_name = f"model-{temporary_index + 1:05d}-of-{shard_count:05d}.safetensors"
            final_path = self.output_dir / final_name
            temporary.replace(final_path)
            for tensor_name, tensor_index in self.tensor_to_temporary.items():
                if tensor_index == temporary_index:
                    weight_map[tensor_name] = final_name
        index = {"metadata": {"total_size": self.total_size}, "weight_map": dict(sorted(weight_map.items()))}
        (self.output_dir / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return index


@dataclass(frozen=True)
class P4ConversionReport:
    """Paths and canonical hashes for one complete conversion."""

    output_dir: Path
    parameter_hash: str
    manifest_hash: str
    target_tensor_count: int
    target_unique_parameters: int


def _source_entry(
    source_key: str,
    operation: str,
    target_keys: list[str],
    transform: str,
) -> dict[str, Any]:
    return {
        "source_key": source_key,
        "operation": operation,
        "target_keys": target_keys,
        "transform": transform,
    }


def _target_entry(
    target_key: str,
    tensor: torch.Tensor,
    operation: str,
    source_keys: list[str],
    transform: str,
    layer_index: int | None,
) -> dict[str, Any]:
    return {
        "target_key": target_key,
        "operation": operation,
        "source_keys": source_keys,
        "transform": transform,
        "layer_index": layer_index,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "numel": tensor.numel(),
        "sha256": _tensor_sha256(tensor),
    }


def _copy_metadata(source_dir: Path, output_dir: Path) -> None:
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "merges.txt",
        "vocab.json",
        "chat_template.json",
        "LICENSE",
        "README.md",
    )
    for name in names:
        source = source_dir / name
        if source.is_file():
            shutil.copy2(source, output_dir / name)


def _direct_layer_mappings(layer_index: int, layer_type: str) -> list[tuple[str, str, str, str]]:
    source = f"model.language_model.layers.{layer_index}."
    target = f"model.layers.{layer_index}."
    mappings = [
        (source + "input_layernorm.weight", target + "input_layernorm.weight", "CONSTRUCT", "add_one"),
        (
            source + "post_attention_layernorm.weight",
            target + "post_attention_layernorm.weight",
            "CONSTRUCT",
            "add_one",
        ),
        (source + "mlp.gate_proj.weight", target + "mlp.gate_proj.weight", "COPY", "copy"),
        (source + "mlp.up_proj.weight", target + "mlp.up_proj.weight", "COPY", "copy"),
        (source + "mlp.down_proj.weight", target + "mlp.down_proj.weight", "COPY", "copy"),
    ]
    if layer_type == "gqa":
        mappings.extend(
            [
                (source + "self_attn.q_proj.weight", target + "self_attn.q_proj.weight", "COPY", "copy"),
                (source + "self_attn.k_proj.weight", target + "self_attn.k_proj.weight", "COPY", "copy"),
                (source + "self_attn.v_proj.weight", target + "self_attn.v_proj.weight", "COPY", "copy"),
                (source + "self_attn.o_proj.weight", target + "self_attn.o_proj.weight", "COPY", "copy"),
                (
                    source + "self_attn.q_norm.weight",
                    target + "self_attn.q_norm.weight",
                    "CONSTRUCT",
                    "add_one",
                ),
                (
                    source + "self_attn.k_norm.weight",
                    target + "self_attn.k_norm.weight",
                    "CONSTRUCT",
                    "add_one",
                ),
            ]
        )
    return mappings


def convert_full_text_checkpoint(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    max_shard_size: int = 4_000_000_000,
) -> P4ConversionReport:
    """Convert all 32 text layers, invoking the frozen P3 bridge exactly 24 times."""
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    inventory = audit_source_inventory(source_dir)
    config = DenseK3Config.from_qwen_model_directory(source_dir)
    reader = IndexedTensorReader(source_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("model-*-of-*.safetensors"):
        path.unlink()
    for path in output_dir.glob(".p4-shard-*.safetensors"):
        path.unlink()
    for name in ("model.safetensors.index.json", P4_MANIFEST_NAME, "p4-conversion-report.json"):
        path = output_dir / name
        if path.exists():
            path.unlink()
    writer = DeterministicShardWriter(output_dir, max_shard_size=max_shard_size)
    source_parameters: list[dict[str, Any]] = []
    target_parameters: list[dict[str, Any]] = []
    layer_reports: list[dict[str, Any]] = []

    base_mappings = [
        ("model.language_model.embed_tokens.weight", "model.embed_tokens.weight", "COPY", "copy"),
        ("model.language_model.norm.weight", "model.norm.weight", "CONSTRUCT", "add_one"),
    ]
    base = reader.load(item[0] for item in base_mappings)
    for source_key, target_key, operation, transform in base_mappings:
        tensor = base[source_key].clone()
        if transform == "add_one":
            tensor = tensor + 1
        writer.add(target_key, tensor)
        source_parameters.append(_source_entry(source_key, operation, [target_key], transform))
        target_parameters.append(_target_entry(target_key, tensor, operation, [source_key], transform, None))
    del base

    for layer_index, layer_type in enumerate(config.mixer_types):
        direct_mappings = _direct_layer_mappings(layer_index, layer_type)
        direct = reader.load(item[0] for item in direct_mappings)
        for source_key, target_key, operation, transform in direct_mappings:
            tensor = direct[source_key].clone()
            if transform == "add_one":
                tensor = tensor + 1
            writer.add(target_key, tensor)
            source_parameters.append(_source_entry(source_key, operation, [target_key], transform))
            target_parameters.append(
                _target_entry(target_key, tensor, operation, [source_key], transform, layer_index)
            )
        del direct

        if layer_type == "kda":
            bridge = GDNBridgeConfig.from_model_directory(source_dir, layer_index)
            full_source_names = [bridge.source_prefix + name for name in bridge.expected_source_shapes()]
            loaded = reader.load(full_source_names)
            local_source = {name.removeprefix(bridge.source_prefix): tensor for name, tensor in loaded.items()}
            converted = convert_gdn_to_kda(local_source, bridge)
            p3_source_targets: dict[str, list[str]] = {name: [] for name in local_source}
            for entry in converted.manifest["entries"]:
                target_key = bridge.target_prefix + entry["target_key"]
                tensor = converted.target_state[entry["target_key"]]
                writer.add(target_key, tensor)
                full_sources = [bridge.source_prefix + name for name in entry["source_keys"]]
                target_parameters.append(
                    _target_entry(
                        target_key,
                        tensor,
                        entry["operation"],
                        full_sources,
                        entry["transform"],
                        layer_index,
                    )
                )
                for local_name in entry["source_keys"]:
                    p3_source_targets[local_name].append(target_key)
            operation_by_source = {
                "A_log": ("COPY", "copy"),
                "conv1d.weight": ("SLICE", "split_fused_qkv_channels"),
                "dt_bias": ("REPEAT", "repeat_interleave_qk_channels"),
                "in_proj_a.weight": ("CONSTRUCT", "factorized_channel_decay_embedding"),
                "in_proj_b.weight": ("COPY", "copy"),
                "in_proj_qkv.weight": ("SLICE", "split_fused_qkv_rows"),
                "in_proj_z.weight": ("COPY", "copy"),
                "norm.weight": ("COPY", "copy"),
                "out_proj.weight": ("COPY", "copy"),
            }
            for local_name in sorted(local_source):
                operation, transform = operation_by_source[local_name]
                source_parameters.append(
                    _source_entry(
                        bridge.source_prefix + local_name,
                        operation,
                        sorted(p3_source_targets[local_name]),
                        transform,
                    )
                )
            layer_reports.append(
                {
                    "layer_index": layer_index,
                    "source_hash": converted.manifest["source_hash"],
                    "target_hash": converted.manifest["target_hash"],
                    "manifest_hash": converted.manifest_hash,
                    "source_parameters": converted.manifest["source_total_parameters"],
                    "target_parameters": converted.manifest["target_total_parameters"],
                    "operation_counts": dict(
                        sorted(Counter(entry["operation"] for entry in converted.manifest["entries"]).items())
                    ),
                }
            )
            del loaded, local_source, converted

    indexed_names = set(reader.weight_map)
    text_names = {name for name in indexed_names if classify_source_key(name) == "text"}
    covered_text = {entry["source_key"] for entry in source_parameters}
    if covered_text != text_names:
        raise ValueError(
            f"Text parameter coverage mismatch: missing={sorted(text_names - covered_text)}, "
            f"unexpected={sorted(covered_text - text_names)}"
        )
    for name in sorted(indexed_names - text_names):
        source_parameters.append(_source_entry(name, "EXCLUDE", [], f"exclude_{classify_source_key(name)}"))

    index = writer.finish()
    operations = {entry["operation"] for entry in source_parameters + target_parameters}
    if operations & FORBIDDEN_OPERATIONS or not operations <= ALLOWED_OPERATIONS:
        raise ValueError(f"Forbidden P4 manifest operation: {sorted(operations)}")
    if len(source_parameters) != P4_EXPECTED_SOURCE_COUNTS["total"]:
        raise ValueError("P4 source manifest does not cover all 738 checkpoint tensors exactly once")
    if len({entry["source_key"] for entry in source_parameters}) != len(source_parameters):
        raise ValueError("P4 source manifest contains duplicate source classifications")
    if len(target_parameters) != P4_EXPECTED_TARGET_TENSORS:
        raise ValueError(f"Unexpected P4 target tensor count: {len(target_parameters)}")
    unique_parameters = sum(entry["numel"] for entry in target_parameters)
    if unique_parameters != P4_EXPECTED_UNIQUE_PARAMETERS:
        raise ValueError(f"Unexpected P4 unique parameter count: {unique_parameters}")
    if len(layer_reports) != 24:
        raise ValueError("The frozen P3 converter was not applied to exactly 24 KDA layers")

    manifest = {
        "schema_version": 1,
        "stage": "P4",
        "status": "PASS",
        "source_model": str(source_dir),
        "source_revision": config.source_revision,
        "inventory": inventory,
        "topology": {
            "num_hidden_layers": 32,
            "kda_indices": config.kda_indices,
            "gqa_indices": config.gqa_indices,
            "kda_count": 24,
            "gqa_count": 8,
            "mlp_count": 32,
        },
        "tied_parameters": {"lm_head.weight": "model.embed_tokens.weight"},
        "excluded_domains": ["vision", "mtp"],
        "source_tensor_count": len(source_parameters),
        "target_tensor_count": len(target_parameters),
        "target_unique_parameters": unique_parameters,
        "operation_counts": dict(sorted(Counter(item["operation"] for item in source_parameters).items())),
        "parameter_hash": _canonical_parameter_hash(target_parameters),
        "layer_conversions": layer_reports,
        "source_parameters": sorted(source_parameters, key=lambda item: item["source_key"]),
        "target_parameters": sorted(target_parameters, key=lambda item: item["target_key"]),
        "weight_index": index,
        "p1_contract_modified": False,
        "p2_frozen": True,
        "p3_frozen": True,
        "p3_numerics_amendment": config.p3_numerics_amendment,
    }
    manifest_hash = hash_json(manifest)
    manifest["manifest_hash_without_self"] = manifest_hash
    (output_dir / P4_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    config.save_pretrained(output_dir)
    if not (output_dir / "generation_config.json").is_file():
        from transformers import GenerationConfig

        GenerationConfig.from_model_config(config).save_pretrained(output_dir)
    _copy_metadata(source_dir, output_dir)
    report = {
        "stage": "P4.1-P4.2",
        "status": "PASS",
        "parameter_hash": manifest["parameter_hash"],
        "manifest_hash": manifest_hash,
        "target_tensor_count": len(target_parameters),
        "target_unique_parameters": unique_parameters,
        "checkpoint_shards": len(set(index["weight_map"].values())),
        "markers": ["P4_01_24_LAYER_CONVERSION=PASS", "P4_02_PARAMETER_ACCOUNTING=PASS"],
    }
    (output_dir / "p4-conversion-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return P4ConversionReport(
        output_dir=output_dir,
        parameter_hash=manifest["parameter_hash"],
        manifest_hash=manifest_hash,
        target_tensor_count=len(target_parameters),
        target_unique_parameters=unique_parameters,
    )


def verify_p4_artifact(checkpoint_dir: str | Path, *, verify_tensor_hashes: bool = False) -> dict[str, Any]:
    """Verify manifest accounting and optionally every saved tensor byte hash."""
    checkpoint_dir = Path(checkpoint_dir)
    required_files = (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        P4_MANIFEST_NAME,
        "p4-conversion-report.json",
    )
    missing_files = [name for name in required_files if not (checkpoint_dir / name).is_file()]
    if missing_files:
        raise FileNotFoundError(f"P4 artifact is missing required files: {missing_files}")
    manifest = json.loads((checkpoint_dir / P4_MANIFEST_NAME).read_text(encoding="utf-8"))
    self_hash = manifest.pop("manifest_hash_without_self")
    if hash_json(manifest) != self_hash:
        raise ValueError("P4 manifest hash mismatch")
    if manifest["target_tensor_count"] != P4_EXPECTED_TARGET_TENSORS:
        raise ValueError("P4 target tensor count mismatch")
    if manifest["target_unique_parameters"] != P4_EXPECTED_UNIQUE_PARAMETERS:
        raise ValueError("P4 target parameter count mismatch")
    if _canonical_parameter_hash(manifest["target_parameters"]) != manifest["parameter_hash"]:
        raise ValueError("P4 canonical parameter hash mismatch")
    verified = 0
    if verify_tensor_hashes:
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise RuntimeError("P4 artifact verification requires safetensors") from exc
        index = load_checkpoint_index(checkpoint_dir)
        by_name = {item["target_key"]: item for item in manifest["target_parameters"]}
        by_shard: dict[str, list[str]] = {}
        for name, shard in index["weight_map"].items():
            by_shard.setdefault(shard, []).append(name)
        for shard, names in sorted(by_shard.items()):
            with safe_open(checkpoint_dir / shard, framework="pt", device="cpu") as handle:
                for name in sorted(names):
                    if _tensor_sha256(handle.get_tensor(name)) != by_name[name]["sha256"]:
                        raise ValueError(f"Saved tensor hash mismatch: {name}")
                    verified += 1
    return {
        "status": "PASS",
        "parameter_hash": manifest["parameter_hash"],
        "manifest_hash": self_hash,
        "tensor_hashes_verified": verified,
    }


__all__ = [
    "P4ConversionReport",
    "P4_EXPECTED_SOURCE_COUNTS",
    "P4_EXPECTED_TARGET_TENSORS",
    "P4_EXPECTED_UNIQUE_PARAMETERS",
    "audit_source_inventory",
    "classify_source_key",
    "convert_full_text_checkpoint",
    "verify_p4_artifact",
]
