"""Pure contracts and storage verification for the P6.3/P6.4 closeout."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from densek3_core.recovery.p6_mla_probe import tensor_sha256

P6_MLA_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31)
P6_SELECTED_STEPS = {3: 128, 7: 128, 11: 0, 15: 0, 19: 256, 23: 96, 27: 128, 31: 256}
P6_EXPECTED_TENSORS = 562
P6_EXPECTED_PARAMETERS = 4_226_431_232


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def checkpoint_identity(checkpoint_dir: Path) -> dict[str, Any]:
    """Hash every immutable checkpoint file before capability evaluation."""
    checkpoint_dir = checkpoint_dir.resolve()
    index = json.loads((checkpoint_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    candidate = json.loads((checkpoint_dir / "p6-candidate-manifest.json").read_text(encoding="utf-8"))
    shards = sorted(set(index["weight_map"].values()))
    immutable_names = [
        "config.json",
        "model.safetensors.index.json",
        "p6-candidate-manifest.json",
        *shards,
    ]
    files = {
        name: {
            "size": (checkpoint_dir / name).stat().st_size,
            "sha256": sha256_file(checkpoint_dir / name),
        }
        for name in immutable_names
    }
    core = {
        "directory": str(checkpoint_dir),
        "files": files,
        "shards": shards,
        "tensor_count": len(index["weight_map"]),
        "total_size": int(index["metadata"]["total_size"]),
        "candidate_manifest_sha256": files["p6-candidate-manifest.json"]["sha256"],
        "config_sha256": files["config.json"]["sha256"],
        "index_sha256": files["model.safetensors.index.json"]["sha256"],
        "candidate_status": candidate["status"],
    }
    return {**core, "fingerprint": object_sha256(core)}


def capability_gate(
    *,
    p5_ce: float,
    p6_ce: float,
    maximum_delta: float,
) -> dict[str, Any]:
    delta = p6_ce - p5_ce
    checks = {
        "finite": math.isfinite(p5_ce) and math.isfinite(p6_ce),
        "delta": delta <= maximum_delta,
    }
    return {
        "p5_ce": p5_ce,
        "p6_ce": p6_ce,
        "delta": delta,
        "maximum_delta": maximum_delta,
        "maximum_ce": p5_ce + maximum_delta,
        "checks": checks,
        "passed": all(checks.values()),
    }


def verify_checkpoint_storage(
    checkpoint_dir: Path,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Verify names/shapes and all MLA tensor hashes without loading 8 GB at once."""
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("P6 closeout storage verification requires safetensors") from exc

    checkpoint_dir = checkpoint_dir.resolve()
    index = json.loads((checkpoint_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(name)
    observed_names: set[str] = set()
    parameter_count = 0
    mla_hashes: dict[str, str] = {}
    shape_manifest: dict[str, list[int]] = {}
    for shard, expected_names in sorted(by_shard.items()):
        with safe_open(checkpoint_dir / shard, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if keys != set(expected_names):
                raise ValueError(f"P6 shard key mismatch: {shard}")
            for name in sorted(keys):
                shape = list(handle.get_slice(name).get_shape())
                shape_manifest[name] = shape
                parameter_count += math.prod(shape)
                observed_names.add(name)
                if ".self_attn." in name and any(
                    name.startswith(f"model.layers.{index}.") for index in P6_MLA_LAYERS
                ):
                    mla_hashes[name] = tensor_sha256(handle.get_tensor(name))
    expected_mla_hashes = candidate["mla_parameter_hashes"]
    checks = {
        "tensor_names": observed_names == set(weight_map),
        "tensor_count": len(observed_names) == P6_EXPECTED_TENSORS == candidate["tensor_count"],
        "parameter_count": parameter_count
        == P6_EXPECTED_PARAMETERS
        == candidate["unique_parameter_count"],
        "index_exact": index == candidate["weight_index"],
        "mla_tensor_names": set(mla_hashes) == set(expected_mla_hashes),
        "mla_tensor_hashes": mla_hashes == expected_mla_hashes,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "tensor_count": len(observed_names),
        "parameter_count": parameter_count,
        "shape_manifest_sha256": object_sha256(shape_manifest),
        "mla_tensor_hashes": mla_hashes,
    }


def verify_local_artifact_binding(
    local_artifact: Path,
    storage_mla_hashes: dict[str, str],
) -> dict[str, Any]:
    from safetensors.torch import load_file

    state = load_file(str(local_artifact), device="cpu")
    expected_names = {
        f"model.layers.{layer}.self_attn.{name}"
        for layer in P6_MLA_LAYERS
        for name in ("kv_a_proj.weight", "kv_a_layernorm.weight", "kv_b_proj.weight")
    }
    hashes = {name: tensor_sha256(value) for name, value in state.items()}
    checks = {
        "tensor_names": set(hashes) == expected_names,
        "tensor_count": len(hashes) == 24,
        "candidate_tensor_names": set(hashes) <= set(storage_mla_hashes),
        "candidate_tensor_hashes": all(
            storage_mla_hashes.get(name) == value for name, value in hashes.items()
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "artifact_sha256": sha256_file(local_artifact),
        "tensor_hashes": hashes,
    }


__all__ = [
    "P6_EXPECTED_PARAMETERS",
    "P6_EXPECTED_TENSORS",
    "P6_MLA_LAYERS",
    "P6_SELECTED_STEPS",
    "capability_gate",
    "checkpoint_identity",
    "object_sha256",
    "sha256_file",
    "verify_checkpoint_storage",
    "verify_local_artifact_binding",
]
