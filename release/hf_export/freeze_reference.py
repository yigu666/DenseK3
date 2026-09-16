"""Capture a redacted, machine-readable canonical P11.6 reference snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from reference_loader import load_canonical_densek3_reference


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_config(config: Any) -> dict[str, Any]:
    values = config.to_dict()
    for key in ("_name_or_path", "name_or_path", "transformers_version"):
        values.pop(key, None)
    # The runtime snapshot can contain local source paths; the config itself must not.
    for key in tuple(values):
        if "path" in key.lower() and isinstance(values[key], str):
            values[key] = "REDACTED_LOCAL_PATH"
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    model, attnres, tokenizer, loader_metadata, _ = load_canonical_densek3_reference(checkpoint)
    names: dict[str, list[int]] = {}
    dtypes: dict[str, int] = {}
    model_count = 0
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        names[name] = list(parameter.shape)
        if id(parameter) not in seen:
            model_count += parameter.numel()
            seen.add(id(parameter))
        dtypes[str(parameter.dtype)] = dtypes.get(str(parameter.dtype), 0) + 1
    attnres_count = 0
    for name, parameter in attnres.named_parameters():
        names[f"attnres.{name}"] = list(parameter.shape)
        attnres_count += parameter.numel()
        dtypes[str(parameter.dtype)] = dtypes.get(str(parameter.dtype), 0) + 1

    files = {}
    for name in (
        "model-dense-overrides.safetensors",
        "attnres.safetensors",
        "p11-6-fast-candidate-manifest.json",
        "p11-6-fast-runtime-config.json",
    ):
        path = checkpoint / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}

    config = _safe_config(model.config)
    result = {
        "schema_version": 1,
        "public_model_name": "DenseK3-4B",
        "canonical_internal_stage": "P11.6",
        "donor": {
            "model": "Qwen/Qwen3.5-4B-Base",
            "revision": "1001bb4d826a52d1f399e183466143f4da7b741b",
        },
        "source_artifacts": files,
        "loader_metadata": {
            "loader": "evaluation.model_loader.load_p11",
            "training_performed": False,
            "backward_performed": False,
            "optimizer_constructed": False,
            "weights_modified": False,
        },
        "parameter_count": {
            "model_unique_parameters": model_count,
            "attnres_parameters": attnres_count,
            "effective_total_unique_parameters": model_count + attnres_count,
        },
        "parameter_shapes": dict(sorted(names.items())),
        "parameter_dtype_histogram": dtypes,
        "model_dtype": str(next(model.parameters()).dtype),
        "precision_boundaries": {
            "model": "float16 (Titan-compatible release path)",
            "kda_recurrent_state": "float32",
            "kda_reduction": "canonical operator-defined FP32 boundaries",
            "mla_latent_cache": "model projection dtype; detached persistent [B,T,512]",
        },
        "architecture": {
            "num_hidden_layers": 32,
            "hidden_size": 2560,
            "intermediate_size": 9216,
            "vocab_size": 248320,
            "mixer_pattern": ["kda", "kda", "kda", "mla"] * 8,
            "kda_layers": 24,
            "mla_layers": [3, 7, 11, 15, 19, 23, 27, 31],
            "kda_q_heads": 16,
            "kda_k_heads": 16,
            "kda_v_heads": 32,
            "kda_head_dim": 128,
            "kda_conv_kernel": 4,
            "kda_decay_rank": 128,
            "kda_recurrent_state_shape": ["B", 32, 128, 128],
            "mla_heads": 16,
            "mla_kv_lora_rank": 512,
            "mla_qk_nope_head_dim": 256,
            "mla_value_head_dim": 256,
            "mla_strict_nope": True,
            "attnres_depth_blocks": 8,
            "attnres_sites": 65,
            "situ": {"enabled": True, "beta": 4.0, "linear_beta": 25.0},
            "tied_embedding_lm_head": True,
        },
        "cache": {
            "implementation": "TitanP7HybridCache",
            "latent_layers": [3, 7, 11, 15, 19, 23, 27, 31],
            "latent_shape": ["B", "T", 512],
            "persistent_expanded_k": False,
            "persistent_expanded_v": False,
        },
        "config_snapshot": config,
        "tokenizer": {
            "source": "Qwen3.5-4B-Base donor tokenizer",
            "local_files_only": True,
            "trust_remote_code": False,
            "vocab_size": int(len(tokenizer)),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "REFERENCE_CAPTURED", "output": str(output), "parameters": model_count + attnres_count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

