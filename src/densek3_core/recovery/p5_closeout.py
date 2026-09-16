"""Pure contracts and artifact helpers for P5 K3-decay + SiLU closeout."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import torch

from densek3_core.modeling import DenseK3Config, DenseK3ForCausalLM
from densek3_core.modeling.modeling_densek3 import DenseK3KDA
from densek3_core.transplant.full_model import DeterministicShardWriter

P5_ARCHITECTURE = "K3_DECAY_PLUS_SILU"
P5_EXPECTED_TENSOR_COUNT = 546
P5_EXPECTED_UNIQUE_PARAMETERS = 4_224_329_984


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    data = value.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(data.numpy().tobytes()).hexdigest()


def safetensor_hashes(path: str | Path) -> dict[str, str]:
    """Hash every tensor through the safetensors safe_open mapping interface."""
    from safetensors import safe_open

    observed = {}
    with safe_open(Path(path), framework="pt", device="cpu") as handle:
        # Unlike a dict, safe_open has keys() but is not itself iterable.
        for name in handle.keys():  # noqa: SIM118
            observed[name] = tensor_sha256(handle.get_tensor(name))
    return observed


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def apply_native_k3_decay_silu(model: DenseK3ForCausalLM) -> dict[str, Any]:
    """Select the approved native P5 architecture without a homotopy field."""
    model.config.kda_gate_mode = "kimi_k3"
    model.config.kda_gate_lower_bound = -5.0
    model.config.kda_output_gate_activation = "silu"
    model.config.kda_output_gate_alpha = None
    kda_layers = 0
    for layer in model.model.layers:
        if isinstance(layer.self_attn, DenseK3KDA):
            layer.self_attn.o_norm.activation = "silu"
            layer.self_attn.o_norm.alpha = None
            kda_layers += 1
    model.config.validate_architecture()
    checks = {
        "native_parameterization": model.config.kda_parameterization == "p5_k3_decay_silu",
        "k3_decay": model.config.kda_gate_mode == "kimi_k3"
        and model.config.kda_gate_lower_bound == -5.0,
        "silu_gate": model.config.kda_output_gate_activation == "silu",
        "no_alpha": model.config.kda_output_gate_alpha is None,
        "all_24_kda_layers": kda_layers == len(model.config.kda_indices) == 24,
        "module_contract": all(
            layer.self_attn.o_norm.activation == "silu" and layer.self_attn.o_norm.alpha is None
            for layer in model.model.layers
            if isinstance(layer.self_attn, DenseK3KDA)
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"Native P5 architecture contract failed: {checks}")
    return {"architecture": P5_ARCHITECTURE, "checks": checks}


def compact_parameterization(layers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only requested P5.4 decay-retention/gate statistics."""
    result = []
    for layer in layers:
        result.append(
            {
                "layer_index": layer["layer_index"],
                "mode": layer["mode"],
                "lower_bound": layer["lower_bound"],
                "output_gate_activation": layer["output_gate_activation"],
                "output_gate_alpha": layer["output_gate_alpha"],
                "decay": {
                    key: layer["log_decay"][key]
                    for key in ("mean", "rms", "p01", "p50", "p99", "min", "max", "nan_count", "inf_count")
                },
                "retention": {
                    key: layer["retention"][key]
                    for key in ("mean", "rms", "p01", "p50", "p99", "min", "max", "nan_count", "inf_count")
                },
                "output_gate": {
                    key: layer["output_gate"][key]
                    for key in (
                        "mean",
                        "rms",
                        "p01",
                        "p50",
                        "p99",
                        "min",
                        "max",
                        "negative_fraction",
                        "near_zero_fraction",
                        "nan_count",
                        "inf_count",
                    )
                },
            }
        )
    return result


def evaluate_closeout_gate(report: dict[str, Any]) -> dict[str, Any]:
    """Apply the approved practical P5.4/P5.5 capability and artifact Gate."""
    evaluation = report["evaluation"]
    p4 = evaluation["models"]["p4_qwen_compat"]
    p5 = evaluation["models"]["p5_k3_decay_silu"]
    heldout = evaluation["heldout"]
    checks = {
        "finite": p5["finite"] is True,
        "generation_not_collapsed": p5["generation_not_collapsed"] is True,
        "canonical_ce": p5["canonical_ce"] <= p4["canonical_ce"] + 0.15,
        "heldout_no_major_regression": heldout["gate"]["passed"] is True,
        "final_hidden_cosine": evaluation["p4_to_p5"]["final_hidden"]["cosine_similarity"] >= 0.85,
        "logits_cosine": evaluation["p4_to_p5"]["logits"]["cosine_similarity"] >= 0.90,
        "cache": report["artifact"]["cache"]["passed"] is True,
        "save_reload": report["artifact"]["save_reload"]["passed"] is True,
        "native_architecture": report["artifact"]["architecture"]["passed"] is True,
        "checkpoint_verification": report["artifact"]["verification"]["status"] == "PASS",
        "checkpoint_source_binding": report["artifact"]["source_binding"]["passed"] is True,
        "checkpoint_generation": report["artifact"]["generation"]["not_collapsed"] is True,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "thresholds": {
            "canonical_ce_max_delta_from_p4": 0.15,
            "heldout_major_regression_max_delta_from_p4": heldout["gate"]["threshold"],
            "final_hidden_cosine_min": 0.85,
            "logits_cosine_min": 0.90,
        },
    }


def checkpoint_architecture_checks(config: DenseK3Config, state: dict[str, torch.Tensor]) -> dict[str, bool]:
    return {
        "parameter_count_exact": sum(value.numel() for value in state.values())
        == P5_EXPECTED_UNIQUE_PARAMETERS,
        "tensor_count_exact": len(state) == P5_EXPECTED_TENSOR_COUNT,
        "topology_exact": config.num_hidden_layers == 32,
        "kda_24": len(config.kda_indices) == 24,
        "gqa_8": len(config.gqa_indices) == 8,
        "k3_decay": config.kda_gate_mode == "kimi_k3" and config.kda_gate_lower_bound == -5.0,
        "silu_gate": config.kda_output_gate_activation == "silu",
        "no_alpha": config.kda_output_gate_alpha is None,
        "native_parameterization": config.kda_parameterization == "p5_k3_decay_silu",
    }


def save_native_checkpoint(
    model: DenseK3ForCausalLM,
    tokenizer_source: str | Path,
    output_dir: str | Path,
    *,
    source_p4_parameter_hash: str,
    probe_a_manifest: dict[str, Any] | None = None,
    initialization_provenance: dict[str, Any] | None = None,
    max_shard_size: int = 4_000_000_000,
) -> dict[str, Any]:
    """Write a standalone indexed HF checkpoint with deterministic lexical shards."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("model-*-of-*.safetensors"):
        path.unlink()
    for path in output_dir.glob(".p4-shard-*.safetensors"):
        path.unlink()
    for name in (
        "model.safetensors.index.json",
        "p5-checkpoint-manifest.json",
        "p5-checkpoint-report.json",
    ):
        path = output_dir / name
        if path.exists():
            path.unlink()

    state = {name: value.detach().cpu().contiguous() for name, value in model.named_parameters()}
    if "lm_head.weight" in state:
        del state["lm_head.weight"]
    checks = checkpoint_architecture_checks(model.config, state)
    if not all(checks.values()):
        raise ValueError(f"P5 checkpoint architecture/accounting failed: {checks}")
    writer = DeterministicShardWriter(output_dir, max_shard_size=max_shard_size, stage="P5")
    entries = []
    for name, value in sorted(state.items()):
        writer.add(name, value)
        entries.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype).removeprefix("torch."),
                "numel": value.numel(),
                "sha256": tensor_sha256(value),
            }
        )
    index = writer.finish()
    model.config.save_pretrained(output_dir)
    config_path = output_dir / "config.json"
    serialized_config = json.loads(config_path.read_text(encoding="utf-8"))
    serialized_config.pop("kda_output_gate_alpha", None)
    config_path.write_text(
        json.dumps(serialized_config, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    source = Path(tokenizer_source)
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "merges.txt",
        "vocab.json",
        "chat_template.json",
        "LICENSE",
        "README.md",
    ):
        if (source / name).is_file():
            shutil.copy2(source / name, output_dir / name)
    if (probe_a_manifest is None) == (initialization_provenance is None):
        raise ValueError(
            "P5 checkpoint requires exactly one trained or analytic initialization provenance"
        )
    if initialization_provenance is None:
        assert probe_a_manifest is not None
        initialization_provenance = {
            "type": "PROBE_A_TRAINED_RECOVERY",
            "stage": probe_a_manifest["stage"],
            "step": probe_a_manifest["step"],
            "effective_tokens": probe_a_manifest["effective_tokens"],
            "trainable_sha256": probe_a_manifest["trainable_sha256"],
        }
    core = {
        "schema_version": 1,
        "stage": "P5.5",
        "architecture": P5_ARCHITECTURE,
        "source_p4_parameter_hash": source_p4_parameter_hash,
        "initialization_provenance": initialization_provenance,
        "target_tensor_count": len(entries),
        "target_unique_parameters": sum(item["numel"] for item in entries),
        "parameter_hash": canonical_json_hash(entries),
        "architecture_checks": checks,
        "tied_parameters": {"lm_head.weight": "model.embed_tokens.weight"},
        "weight_index": index,
        "parameters": entries,
        "no_optimizer_state": True,
        "no_probe_fields": not any("probe" in key.lower() for key in serialized_config),
        "no_homotopy": True,
        "alpha_field_absent_from_config": "kda_output_gate_alpha" not in serialized_config,
    }
    if probe_a_manifest is not None:
        core["probe_a_source"] = {
            key: initialization_provenance[key]
            for key in ("stage", "step", "effective_tokens", "trainable_sha256")
        }
        core["probe_b_parameters_applied"] = False
    manifest = {**core, "manifest_hash_without_self": canonical_json_hash(core)}
    (output_dir / "p5-checkpoint-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def verify_native_checkpoint(checkpoint_dir: str | Path, *, verify_tensors: bool) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir).resolve()
    manifest = json.loads((checkpoint_dir / "p5-checkpoint-manifest.json").read_text(encoding="utf-8"))
    config = DenseK3Config.from_pretrained(checkpoint_dir)
    index = json.loads((checkpoint_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    manifest_core = {key: value for key, value in manifest.items() if key != "manifest_hash_without_self"}
    checks = {
        "architecture": manifest["architecture"] == P5_ARCHITECTURE,
        "parameter_count": manifest["target_unique_parameters"] == P5_EXPECTED_UNIQUE_PARAMETERS,
        "tensor_count": manifest["target_tensor_count"] == P5_EXPECTED_TENSOR_COUNT,
        "index_names": set(index["weight_map"]) == {item["name"] for item in manifest["parameters"]},
        "native_config": config.kda_parameterization == "p5_k3_decay_silu",
        "no_alpha": config.kda_output_gate_alpha is None,
        "no_optimizer": not (checkpoint_dir / "optimizer.pt").exists(),
        "no_probe_config_fields": not any("probe" in key.lower() for key in config.to_dict()),
        "alpha_field_absent": "kda_output_gate_alpha"
        not in json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8")),
        "manifest_no_homotopy": manifest["no_homotopy"] is True,
        "manifest_no_probe_fields": manifest["no_probe_fields"] is True,
        "no_probe_b_parameters": manifest.get("probe_b_parameters_applied") is not True,
        "parameter_manifest_hash": canonical_json_hash(manifest["parameters"])
        == manifest["parameter_hash"],
        "manifest_hash": canonical_json_hash(manifest_core) == manifest["manifest_hash_without_self"],
        "weight_index": manifest["weight_index"] == index,
        "all_shards_exist": all(
            (checkpoint_dir / shard).is_file() for shard in set(index["weight_map"].values())
        ),
    }
    tensor_hashes = None
    if verify_tensors:
        observed = {}
        for shard in sorted(set(index["weight_map"].values())):
            observed.update(safetensor_hashes(checkpoint_dir / shard))
        expected = {item["name"]: item["sha256"] for item in manifest["parameters"]}
        tensor_hashes = observed == expected
        checks["tensor_hashes"] = tensor_hashes
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "parameter_hash": manifest["parameter_hash"],
        "manifest_hash": manifest["manifest_hash_without_self"],
        "tensor_hashes_verified": tensor_hashes,
    }


def verify_probe_a_checkpoint_binding(
    checkpoint_manifest: dict[str, Any],
    *,
    expected_p4_parameter_hash: str,
    probe_a_checkpoint: str | Path,
    probe_a_report: dict[str, Any],
) -> dict[str, Any]:
    """Bind a resumed P5.5 artifact to frozen P4 and Probe-A step-000123 tensors."""
    probe_a_checkpoint = Path(probe_a_checkpoint).resolve()
    probe_manifest = json.loads((probe_a_checkpoint / "manifest.json").read_text(encoding="utf-8"))
    trainable_path = probe_a_checkpoint / "trainable.safetensors"
    trainable_hashes = safetensor_hashes(trainable_path)
    checkpoint_hashes = {
        item["name"]: item["sha256"] for item in checkpoint_manifest["parameters"]
    }
    best = probe_a_report["best_checkpoint"]
    source = checkpoint_manifest["probe_a_source"]
    checks = {
        "p4_parameter_hash": checkpoint_manifest["source_p4_parameter_hash"]
        == expected_p4_parameter_hash,
        "probe_stage": source["stage"] == probe_manifest["stage"] == "P5.3-Probe-A",
        "probe_step": source["step"] == probe_manifest["step"] == best["step"] == 123,
        "probe_effective_tokens": source["effective_tokens"]
        == probe_manifest["effective_tokens"]
        == best["effective_tokens"]
        == 251_904,
        "probe_trainable_file_hash": sha256_file(trainable_path)
        == source["trainable_sha256"]
        == probe_manifest["trainable_sha256"]
        == best["trainable_sha256"],
        "trainable_names_in_checkpoint": set(trainable_hashes).issubset(checkpoint_hashes),
        "trainable_tensor_hashes": all(
            checkpoint_hashes.get(name) == digest for name, digest in trainable_hashes.items()
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "probe_trainable_tensor_count": len(trainable_hashes),
        "probe_trainable_sha256": source["trainable_sha256"],
    }


def heldout_gate(p4_ce: float, p5_ce: float, *, threshold: float = 0.15) -> dict[str, Any]:
    values = (float(p4_ce), float(p5_ce), float(threshold))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Heldout Gate inputs must be finite")
    delta = p5_ce - p4_ce
    return {"signed_delta": delta, "threshold": threshold, "passed": delta <= threshold}


def formal_probe_b_review(report: dict[str, Any]) -> dict[str, Any]:
    """Convert completed negative-recovery Probe-B evidence into the approved decision."""
    evaluations = report["evaluations"]
    assessments = report["probe_b_gate"]["checkpoint_assessments"]
    metrics = [
        {
            "effective_tokens": item["effective_tokens"],
            "canonical_ce": item["canonical"]["cross_entropy"],
            "recovery_fraction": assessment["recovery_fraction"],
            "finite": item["finite"],
            "generation_not_collapsed": item["generation"]["not_collapsed"],
        }
        for item, assessment in zip(evaluations, assessments, strict=True)
    ]
    trained = metrics[1:]
    checks = {
        "prior_or_rescored_status": report["probe_b_status"]
        in {"REVIEW_REQUIRED", "NO_GO_AFTER_FORMAL_REVIEW"},
        "technical_execution_pass": report["status"] == "PASS" and report["budget_respected"] is True,
        "full_budget_completed": report["effective_training_tokens"] == report["budget_cap_tokens"] == 499_712,
        "fresh_optimizer": all(report["fresh_optimizer_checks"].values()),
        "source_probe_a_step_123": report["source"]["manifest"]["step"] == 123,
        "four_evaluations": [item["effective_tokens"] for item in metrics]
        == [0, 100_352, 251_904, 499_712],
        "all_trained_recovery_negative": all(item["recovery_fraction"] < 0 for item in trained),
        "all_finite": all(item["finite"] for item in metrics),
        "all_generation_not_collapsed": all(item["generation_not_collapsed"] for item in metrics),
        "probe_c_not_run": report.get("probe_c_allowed") is not True
        and report["next_action"]
        in {"DO_NOT_PROCEED_TO_PROBE_C", "P5_CLOSEOUT_K3_DECAY_PLUS_SILU"},
    }
    if not all(checks.values()):
        raise ValueError(f"Probe-B formal review evidence failed: {checks}")
    return {
        "review_type": "NO_TRAINING_METRICS_ONLY_FORMAL_REVIEW",
        "technical_execution": "VALID",
        "training_failure": False,
        "budget_insufficient_claimed": False,
        "optimizer_rerun": False,
        "training_data_loaded": False,
        "model_loaded": False,
        "switch_ce": report["switch"]["ce_immediately_after_switch"],
        "metrics": metrics,
        "causal_basis": [
            "all post-switch trained recovery fractions are negative",
            "all evaluations are finite",
            "generation remains non-collapsed",
            "the approved 499712-token budget completed",
            "the technical execution and fresh optimizer contracts passed",
        ],
        "checks": checks,
        "decision": "NO_GO_AFTER_FORMAL_REVIEW",
        "architecture_decision": "KEEP_K3_DECAY_SILU",
    }


__all__ = [
    "P5_ARCHITECTURE",
    "apply_native_k3_decay_silu",
    "checkpoint_architecture_checks",
    "compact_parameterization",
    "evaluate_closeout_gate",
    "formal_probe_b_review",
    "heldout_gate",
    "save_native_checkpoint",
    "safetensor_hashes",
    "sha256_file",
    "tensor_sha256",
    "verify_native_checkpoint",
    "verify_probe_a_checkpoint_binding",
]
