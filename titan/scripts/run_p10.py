"""Run the frozen Titan P10 joint-recovery recipe and dense closeout."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
TITAN = ROOT / "titan"
sys.path.insert(0, str(TITAN))

from attnres import BlockAttnResReference, enable_block_attnres_runtime  # noqa: E402
from kda_sm75_backend import enable_sm75_fused_recurrent_fallback  # noqa: E402
from p7_latent_cache import enable_p7_latent_cache_runtime  # noqa: E402
from p10 import (  # noqa: E402
    FP32MasterParameters,
    audit_gradient_coverage,
    build_p10_trainable_scope,
    cosine_learning_rate,
    merge_all_lora,
)
from run_p8_reference import cache_equivalence, corpus_ce, generation, gpu_start_state  # noqa: E402
from run_p9 import apply_candidate_delta  # noqa: E402
from situ import enable_situ_glu_runtime  # noqa: E402

from densek3_core.evaluation.p4_runtime import load_densek3_model  # noqa: E402
from densek3_core.modeling.modeling_densek3 import DenseK3KDA, DenseK3MLA  # noqa: E402
from densek3_core.recovery.p5_corpus import (  # noqa: E402
    P5_SEQUENCE_LENGTH,
    PackedTokenDataset,
    verify_frozen_corpus,
)
from densek3_core.recovery.p5_probe import chunked_causal_lm_loss  # noqa: E402
from densek3_core.recovery.p6_closeout import sha256_file  # noqa: E402

CONFIG = TITAN / "configs/p10-titan-joint-recovery.yaml"
CACHE_CONTRACT = TITAN / "configs/p6-titan-fp16-cache-compat-2026-08-19-01.yaml"
P6_SOURCE = (
    ROOT
    / "outputs/densek3-core/checkpoints/p6/"
    "densek3-4b-8mla-c3-qk-norm-calibrated-candidate"
)
P9_SOURCE = ROOT / "outputs/densek3-core/checkpoints/p9/densek3-4b-core-k3-situ-titan"
P9_FINAL = TITAN / "manifests/reproduction/p9/p9-final-check.json"
TOKENIZER = ROOT / "outputs/densek3-core/checkpoints/p5/densek3-4b-k3-decay-silu-i1"
DATA = ROOT / "data/p5-probe"
REPORT_DIR = TITAN / "manifests/reproduction/p10"
CHECKPOINT_DIR = ROOT / "outputs/densek3-core/checkpoints/p10/joint-recovery"
CANDIDATE = ROOT / "outputs/densek3-core/checkpoints/p10/densek3-4b-core-k3-joint-titan"
PHASE1_FINAL = REPORT_DIR / "p10-phase1-final.json"
EXTENSION_FINAL = REPORT_DIR / "p10-extension-final.json"
FINAL = REPORT_DIR / "p10-final-check.json"
SMOKE_FINAL = REPORT_DIR / "p10-one-step-smoke.json"
RUN_LABEL = "P10"
RUN_STAGE_PREFIX = "TITAN_P10"
RUN_MODE = "FULL_SCOPE_CORE_K3_JOINT_RECOVERY"
CANDIDATE_STAGE = "TITAN_FP16_PROVISIONAL_P10_CANDIDATE"
CANDIDATE_ARCHITECTURE = "DENSEK3_CORE_KDA_MLA_ATTNRES_SITU_JOINT_RECOVERED"
RUNTIME_CONFIG_FILE = "p10-runtime-config.json"
CANDIDATE_MANIFEST_FILE = "p10-candidate-manifest.json"
SCOPE_BUILDER = build_p10_trainable_scope
COMPENSATION_AUDIT_ENABLED = False
COMPENSATION_BASELINE = CHECKPOINT_DIR / "kda-compensation-step-zero.safetensors"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def tensor_dict_sha256(values: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def p9_identity() -> dict[str, Any]:
    files = {
        path.name: sha256_file(path)
        for path in sorted(P9_SOURCE.iterdir())
        if path.is_file()
    }
    return {
        "path": str(P9_SOURCE.resolve()),
        "files": files,
        "fingerprint": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()
        ).hexdigest(),
    }


def architecture_contract(model: torch.nn.Module, runtime: BlockAttnResReference) -> dict[str, Any]:
    kda = sum(isinstance(layer.self_attn, DenseK3KDA) for layer in model.model.layers)
    mla = sum(isinstance(layer.self_attn, DenseK3MLA) for layer in model.model.layers)
    checks = {
        "kda_24": kda == 24,
        "mla_8": mla == 8,
        "mla_rank_512": all(
            layer.self_attn.kv_lora_rank == 512
            for layer in model.model.layers
            if isinstance(layer.self_attn, DenseK3MLA)
        ),
        "nope": all(
            layer.self_attn.position_encoding == "none"
            for layer in model.model.layers
            if isinstance(layer.self_attn, DenseK3MLA)
        ),
        "qk_norm": all(
            layer.self_attn.apply_qk_norm
            for layer in model.model.layers
            if isinstance(layer.self_attn, DenseK3MLA)
        ),
        "attnres_sites_65": len(runtime.sites) == 65,
        "situ_config": True,
    }
    return {"checks": checks, "passed": all(checks.values()), "kda_layers": kda, "mla_layers": mla}


def load_p9_stack() -> tuple[torch.nn.Module, BlockAttnResReference, dict[str, Any]]:
    enable_sm75_fused_recurrent_fallback()
    enable_p7_latent_cache_runtime()
    model = load_densek3_model(P6_SOURCE, device="cuda", dtype=torch.float16)
    delta = apply_candidate_delta(model)
    runtime = BlockAttnResReference(
        model.config.hidden_size,
        model.config.rms_norm_eps,
        block_size_residuals=8,
    ).cuda()
    runtime.load_state_dict(load_file(str(P9_SOURCE / "attnres-zero.safetensors")))
    enable_situ_glu_runtime(model)
    enable_block_attnres_runtime(model, runtime)
    architecture = architecture_contract(model, runtime)
    if not architecture["passed"]:
        raise ValueError(f"P10 architecture contract failed: {architecture}")
    return model, runtime, {"p9_delta": delta, "architecture": architecture}


def split_documents(corpus_manifest: dict[str, Any], split: str, start: int, end: int) -> set[str]:
    sequence_length = int(corpus_manifest["sequence_length"])
    token_start = start * sequence_length
    token_end = end * sequence_length
    return {
        item["document_id"]
        for item in corpus_manifest["splits"][split]["memberships"]
        if item["token_start"] < token_end and item["eos_position"] >= token_start
    }


def data_contract(config: dict[str, Any]) -> dict[str, Any]:
    verified = verify_frozen_corpus(DATA)
    manifest = load_json(DATA / "packed/corpus-manifest.json")
    train_start, train_end = (int(value) for value in config["data"]["train_sequence_range"])
    val_start, val_end = (
        int(value) for value in config["data"]["recovery_validation_sequence_range"]
    )
    train_documents = split_documents(manifest, "train", train_start, train_end)
    validation_documents = split_documents(manifest, "dev", val_start, val_end)
    overlap = sorted(train_documents & validation_documents)
    checks = {
        "corpus_frozen": verified["p5_probe_corpus_frozen"] is True,
        "sequence_length": manifest["sequence_length"] == P5_SEQUENCE_LENGTH,
        "train_range_1m": train_end - train_start == 512,
        "recovery_validation_8": val_end - val_start == 8,
        "document_disjoint": not overlap,
        "heldout_not_accessed": True,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "train_document_count": len(train_documents),
        "validation_document_count": len(validation_documents),
        "document_overlap": overlap,
        "heldout_not_accessed": True,
    }


@torch.inference_mode()
def subset_ce(
    model: torch.nn.Module,
    dataset: PackedTokenDataset,
    sequence_range: tuple[int, int],
) -> float:
    total_loss = 0.0
    total_tokens = 0
    for index in range(*sequence_range):
        tokens = torch.from_numpy(dataset[index].copy()).unsqueeze(0).cuda()
        hidden = model.model(tokens, use_cache=False, return_dict=True).last_hidden_state[:, :-1]
        labels = tokens[:, 1:]
        for start in range(0, hidden.shape[1], 64):
            end = min(start + 64, hidden.shape[1])
            logits = F.linear(hidden[:, start:end].float(), model.lm_head.weight.float())
            total_loss += float(
                F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels[:, start:end].reshape(-1),
                    reduction="sum",
                )
            )
            total_tokens += labels[:, start:end].numel()
        del tokens, hidden, labels
    return total_loss / total_tokens


@torch.inference_mode()
def routing_diagnostic(
    model: torch.nn.Module,
    runtime: BlockAttnResReference,
    tokens: torch.Tensor,
) -> dict[str, Any]:
    runtime.clear_diagnostics()
    runtime.capture_diagnostics = True
    try:
        hidden = model.model(tokens, use_cache=False, return_dict=True).last_hidden_state
    finally:
        runtime.capture_diagnostics = False
    records = list(runtime.routing_records)
    if len(records) != 65:
        raise RuntimeError(f"Expected 65 P10 routing sites, observed {len(records)}")
    query = runtime.query_report()

    def depth_summary(start: int, end: int) -> dict[str, Any]:
        subset = records[start:end]
        return {
            "site_range": [start, end],
            "count": len(subset),
            "entropy_gap_mean": float(np.mean([item["entropy_gap_mean"] for item in subset])),
            "alpha_max_mean": float(np.mean([item["alpha_max_mean"] for item in subset])),
            "content_std_mean": float(np.mean([item["content_std"] for item in subset])),
            "nonuniform_sites": sum(item["entropy_gap_mean"] > 1.0e-6 for item in subset),
            "content_dependent_sites": sum(item["content_std"] > 1.0e-7 for item in subset),
        }

    report = {
        "query": query,
        "nonuniform_sites": sum(item["entropy_gap_mean"] > 1.0e-6 for item in records),
        "content_dependent_sites": sum(item["content_std"] > 1.0e-7 for item in records),
        "entropy_gap_distribution": [float(item["entropy_gap_mean"]) for item in records],
        "alpha_max_distribution": [float(item["alpha_max_mean"]) for item in records],
        "embedding_weight_distribution": [float(item["embedding_weight_mean"]) for item in records],
        "recent_weight_distribution": [float(item["recent_weight_mean"]) for item in records],
        "embedding_weight_mean": float(np.mean([item["embedding_weight_mean"] for item in records])),
        "recent_weight_mean": float(np.mean([item["recent_weight_mean"] for item in records])),
        "by_depth": {
            "early": depth_summary(0, 22),
            "middle": depth_summary(22, 44),
            "late": depth_summary(44, 65),
        },
        "finite": bool(torch.isfinite(hidden).all()),
    }
    report["active"] = (
        report["query"]["max"] > 0.0
        and report["nonuniform_sites"] > 0
        and report["content_dependent_sites"] > 0
    )
    runtime.clear_diagnostics()
    return report


@torch.inference_mode()
def capture_kda_compensation_tensors(
    model: torch.nn.Module,
    tokens: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Capture the four frozen-KDA compensation boundaries on a fixed input."""
    captured: dict[str, torch.Tensor] = {}
    hooks = []

    def save(name: str, value: torch.Tensor) -> None:
        captured[name] = value.detach().cpu().contiguous()

    for layer_index, layer in enumerate(model.model.layers):
        mixer = layer.self_attn
        if not isinstance(mixer, DenseK3KDA):
            continue
        hooks.append(
            mixer.register_forward_pre_hook(
                lambda _module, args, index=layer_index: save(
                    f"layer_{index:02d}.input_hidden", args[0]
                )
            )
        )
        hooks.append(
            mixer.o_norm.register_forward_pre_hook(
                lambda _module, args, index=layer_index: save(
                    f"layer_{index:02d}.recurrent_core", args[0]
                )
            )
        )
        hooks.append(
            mixer.o_norm.register_forward_hook(
                lambda _module, _args, output, index=layer_index: save(
                    f"layer_{index:02d}.post_o_norm", output
                )
            )
        )
        hooks.append(
            mixer.o_proj.register_forward_hook(
                lambda _module, _args, output, index=layer_index: save(
                    f"layer_{index:02d}.post_o_proj", output
                )
            )
        )
    try:
        model.model(tokens, use_cache=False, return_dict=True)
    finally:
        for hook in hooks:
            hook.remove()
    if len(captured) != 24 * 4:
        raise RuntimeError(f"Expected 96 KDA compensation tensors, observed {len(captured)}")
    return captured


def tensor_drift(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    left = reference.double().reshape(-1)
    right = actual.double().reshape(-1)
    difference = right - left
    denominator = max(float(torch.linalg.vector_norm(left)), 1.0e-30)
    return {
        "relative_l2": float(torch.linalg.vector_norm(difference)) / denominator,
        "cosine": float(F.cosine_similarity(left, right, dim=0)),
        "rms_ratio": float(right.square().mean().sqrt())
        / max(float(left.square().mean().sqrt()), 1.0e-30),
    }


def compensation_drift_report(
    baseline: dict[str, torch.Tensor],
    actual: dict[str, torch.Tensor],
) -> dict[str, Any]:
    if set(baseline) != set(actual):
        raise ValueError(f"KDA compensation tensor keys changed: {set(baseline) ^ set(actual)}")
    per_layer: dict[str, dict[str, Any]] = {}
    for layer_index in range(32):
        prefix = f"layer_{layer_index:02d}."
        keys = sorted(name for name in baseline if name.startswith(prefix))
        if not keys:
            continue
        per_layer[str(layer_index)] = {
            name.split(".")[-1]: tensor_drift(baseline[name], actual[name])
            for name in keys
        }
    return {
        "status": "MEASURED",
        "baseline_path": str(COMPENSATION_BASELINE.resolve()),
        "per_kda_layer": per_layer,
        "worst_relative_l2": {
            boundary: max(
                values[boundary]["relative_l2"]
                for values in per_layer.values()
            )
            for boundary in ("input_hidden", "recurrent_core", "post_o_norm", "post_o_proj")
        },
    }


@torch.inference_mode()
def evaluate_checkpoint(
    model: torch.nn.Module,
    runtime: BlockAttnResReference,
    dev: PackedTokenDataset,
    diagnostic_tokens: torch.Tensor,
    validation_range: tuple[int, int],
    *,
    step: int,
) -> dict[str, Any]:
    model.eval()
    runtime.eval()
    ce = subset_ce(model, dev, validation_range)
    routing = routing_diagnostic(model, runtime, diagnostic_tokens)
    compensation: dict[str, Any] | None = None
    if COMPENSATION_AUDIT_ENABLED:
        actual = capture_kda_compensation_tensors(model, diagnostic_tokens)
        if step == 0:
            COMPENSATION_BASELINE.parent.mkdir(parents=True, exist_ok=True)
            save_file(actual, str(COMPENSATION_BASELINE))
            compensation = {
                "status": "BASELINE_FROZEN",
                "path": str(COMPENSATION_BASELINE.resolve()),
                "sha256": sha256_file(COMPENSATION_BASELINE),
                "tensor_count": len(actual),
            }
        else:
            if not COMPENSATION_BASELINE.is_file():
                raise FileNotFoundError("P10-T KDA compensation baseline is missing")
            baseline = load_file(str(COMPENSATION_BASELINE))
            compensation = compensation_drift_report(baseline, actual)
            compensation["baseline_sha256"] = sha256_file(COMPENSATION_BASELINE)
    return {
        "step": step,
        "effective_tokens": step * P5_SEQUENCE_LENGTH,
        "recovery_validation_ce": ce,
        "routing": routing,
        "frozen_kda_compensation": compensation,
        "finite": math.isfinite(ce) and routing["finite"],
    }


def trainable_state(scope: Any) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    for handle in scope.lora:
        values[f"lora.{handle.name}.A"] = handle.module.lora_A.detach().cpu()
        values[f"lora.{handle.name}.B"] = handle.module.lora_B.detach().cpu()
    for name, parameter in scope.direct_model.items():
        values[f"direct.model.{name}"] = parameter.detach().cpu()
    for name, parameter in scope.direct_attnres.items():
        values[f"direct.attnres.{name}"] = parameter.detach().cpu()
    return values


def trainable_family_by_key(scope: Any) -> dict[str, str]:
    families: dict[str, str] = {}
    for handle in scope.lora:
        family = "KDA_OUTPUT" if handle.family == "KDA" else handle.family
        families[f"lora.{handle.name}.A"] = family
        families[f"lora.{handle.name}.B"] = family
    for name in scope.direct_model:
        key = f"direct.model.{name}"
        if ".self_attn.o_norm." in name:
            families[key] = "KDA_OUTPUT"
        elif any(
            marker in name
            for marker in (".self_attn.q_norm.", ".self_attn.k_norm.", ".self_attn.kv_a_layernorm.")
        ):
            families[key] = "MLA"
        else:
            families[key] = "NORM"
    for name in scope.direct_attnres:
        families[f"direct.attnres.{name}"] = "ATTNRES"
    return families


def parameter_update_by_family(
    scope: Any,
    initial: dict[str, torch.Tensor],
) -> dict[str, Any]:
    current = trainable_state(scope)
    if set(initial) != set(current):
        raise ValueError(f"P10 trainable keys changed during update audit: {set(initial) ^ set(current)}")
    families = trainable_family_by_key(scope)
    accumulator: dict[str, dict[str, float | int]] = {}
    for name, after in current.items():
        family = families[name]
        before = initial[name].double()
        difference = after.double() - before
        values = accumulator.setdefault(
            family,
            {"delta_squared": 0.0, "reference_squared": 0.0, "changed_tensors": 0, "tensor_count": 0},
        )
        values["delta_squared"] += float(difference.square().sum())
        values["reference_squared"] += float(before.square().sum())
        values["changed_tensors"] += int(not torch.equal(before, after.double()))
        values["tensor_count"] += 1
    report = {}
    for family, values in sorted(accumulator.items()):
        delta_norm = math.sqrt(float(values.pop("delta_squared")))
        reference_norm = math.sqrt(float(values.pop("reference_squared")))
        report[family] = {
            **values,
            "delta_norm": delta_norm,
            "reference_norm": reference_norm,
            "relative_parameter_delta": delta_norm / max(reference_norm, 1.0e-30),
        }
    return report


def frozen_parameter_audit(model: torch.nn.Module, names: list[str]) -> dict[str, Any]:
    parameters = dict(model.named_parameters())
    missing = sorted(set(names) - set(parameters))
    values = {name: parameters[name] for name in names if name in parameters}
    return {
        "parameter_count": len(values),
        "missing": missing,
        "all_requires_grad_false": all(not value.requires_grad for value in values.values()),
        "tensor_hash": tensor_dict_sha256(values),
    }


def frozen_scope_audit(model: torch.nn.Module, scope: Any) -> dict[str, Any]:
    names = list(scope.manifest.get("frozen_audit_parameter_names", []))
    return frozen_parameter_audit(model, names)


def unexpected_gradient_names(
    model: torch.nn.Module,
    runtime: torch.nn.Module,
    scope: Any,
) -> list[str]:
    approved = {id(parameter) for parameter in scope.trainable_parameters()}
    values = [
        (f"model.{name}", parameter)
        for name, parameter in model.named_parameters()
    ] + [
        (f"attnres.{name}", parameter)
        for name, parameter in runtime.named_parameters()
    ]
    return sorted(
        name
        for name, parameter in values
        if parameter.grad is not None and id(parameter) not in approved
    )


@torch.no_grad()
def load_trainable_state(scope: Any, path: Path) -> None:
    values = load_file(str(path))
    expected: set[str] = set()
    for handle in scope.lora:
        for suffix, parameter in (("A", handle.module.lora_A), ("B", handle.module.lora_B)):
            key = f"lora.{handle.name}.{suffix}"
            parameter.copy_(values[key].to(parameter.device, parameter.dtype))
            expected.add(key)
    for prefix, mapping in (
        ("direct.model", scope.direct_model),
        ("direct.attnres", scope.direct_attnres),
    ):
        for name, parameter in mapping.items():
            key = f"{prefix}.{name}"
            parameter.copy_(values[key].to(parameter.device, parameter.dtype))
            expected.add(key)
    if set(values) != expected:
        raise ValueError(f"P10 trainable-state key mismatch: {set(values) ^ expected}")


def save_checkpoint(
    step: int,
    scope: Any,
    masters: FP32MasterParameters,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    evaluation: dict[str, Any],
    source_identity: dict[str, Any],
) -> dict[str, Any]:
    directory = CHECKPOINT_DIR / f"step-{step:06d}"
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    weights = directory / "trainable.safetensors"
    values = trainable_state(scope)
    save_file(values, str(weights))
    training_state = directory / "training-state.pt"
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "master_names": [item.name for item in masters.bindings],
        },
        training_state,
    )
    manifest = {
        "stage": f"{RUN_STAGE_PREFIX}_CORE_K3_JOINT_RECOVERY_CHECKPOINT",
        "run_label": RUN_LABEL,
        "run_mode": RUN_MODE,
        "step": step,
        "effective_tokens": step * P5_SEQUENCE_LENGTH,
        "path": str(directory.resolve()),
        "trainable_sha256": sha256_file(weights),
        "trainable_tensor_hash": tensor_dict_sha256(values),
        "training_state_sha256": sha256_file(training_state),
        "source_fingerprint": source_identity["fingerprint"],
        "evaluation": evaluation,
        "lora_present_in_training_checkpoint": True,
        "heldout_accessed": False,
    }
    atomic_json(directory / "checkpoint-manifest.json", manifest)
    return manifest


def checkpoint_manifests() -> list[dict[str, Any]]:
    if not CHECKPOINT_DIR.exists():
        return []
    return [
        load_json(path)
        for path in sorted(CHECKPOINT_DIR.glob("step-*/checkpoint-manifest.json"))
    ]


def restore_checkpoint(
    manifest: dict[str, Any],
    scope: Any,
    masters: FP32MasterParameters,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> None:
    directory = Path(manifest["path"])
    if sha256_file(directory / "trainable.safetensors") != manifest["trainable_sha256"]:
        raise ValueError("P10 trainable checkpoint hash mismatch")
    load_trainable_state(scope, directory / "trainable.safetensors")
    masters.refresh_from_model()
    state = torch.load(directory / "training-state.pt", map_location="cpu", weights_only=False)
    if state["step"] != manifest["step"]:
        raise ValueError("P10 optimizer checkpoint step mismatch")
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])


def phase_decision(evaluations: list[dict[str, Any]], target_step: int) -> dict[str, Any]:
    zero = next(item for item in evaluations if item["step"] == 0)
    trained = [item for item in evaluations if 0 < item["step"] <= target_step]
    best = min(trained, key=lambda item: item["recovery_validation_ce"])
    final = next(item for item in evaluations if item["step"] == target_step)
    improvement = zero["recovery_validation_ce"] - best["recovery_validation_ce"]
    ordered = sorted(trained, key=lambda item: item["step"])
    checkpoint_deltas = [
        left["recovery_validation_ce"] - right["recovery_validation_ce"]
        for left, right in zip(ordered, ordered[1:], strict=False)
    ]
    clearly_improving = bool(checkpoint_deltas) and all(value > 0.0 for value in checkpoint_deltas)
    healthy = (
        all(item["finite"] for item in evaluations)
        and final["routing"]["active"]
    )
    return {
        "zero_ce": zero["recovery_validation_ce"],
        "best_trained": best,
        "final": final,
        "best_improvement": improvement,
        "all_finite": all(item["finite"] for item in evaluations),
        "attnres_active": any(item["routing"]["active"] for item in trained),
        "checkpoint_validation_improvements": checkpoint_deltas,
        "validation_trajectory_clearly_improving": clearly_improving,
        "extension_review": (
            "ELIGIBLE_PENDING_FULL_DEV"
            if target_step == 128 and clearly_improving and healthy
            else "NOT_ELIGIBLE"
            if target_step == 128
            else "FINAL_BUDGET_REACHED"
        ),
    }


def build_optimizer(scope: Any, config: dict[str, Any]) -> tuple[Any, Any, Any]:
    direct_all = {
        **{f"model.{name}": value for name, value in scope.direct_model.items()},
        **{f"attnres.{name}": value for name, value in scope.direct_attnres.items()},
    }
    masters = FP32MasterParameters(direct_all)
    lora_parameters = [
        parameter
        for handle in scope.lora
        for parameter in (handle.module.lora_A, handle.module.lora_B)
    ]
    recipe = config["optimization"]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": lora_parameters,
                "lr": float(recipe["lora_learning_rate"]),
                "weight_decay": float(recipe["lora_weight_decay"]),
                "group_name": "mergeable_lora",
            },
            {
                "params": masters.optimizer_parameters(),
                "lr": float(recipe["direct_learning_rate"]),
                "weight_decay": float(recipe["direct_weight_decay"]),
                "group_name": "direct_fp32_master",
            },
        ],
        fused=True,
    )
    scaler = torch.amp.GradScaler("cuda")
    return masters, optimizer, scaler


def optimizer_parameter_names(scope: Any, masters: FP32MasterParameters) -> dict[int, str]:
    names: dict[int, str] = {}
    for handle in scope.lora:
        names[id(handle.module.lora_A)] = f"lora.{handle.name}.A"
        names[id(handle.module.lora_B)] = f"lora.{handle.name}.B"
    for item in masters.bindings:
        names[id(item.master)] = f"direct_master.{item.name}"
    return names


def optimizer_gradient_norm_by_family(
    scope: Any,
    masters: FP32MasterParameters,
) -> dict[str, float]:
    family_by_key = trainable_family_by_key(scope)
    grouped: dict[str, float] = {}
    for handle in scope.lora:
        family = "KDA_OUTPUT" if handle.family == "KDA" else handle.family
        for parameter in (handle.module.lora_A, handle.module.lora_B):
            if parameter.grad is not None:
                grouped[family] = grouped.get(family, 0.0) + float(
                    parameter.grad.detach().double().square().sum()
                )
    for binding in masters.bindings:
        prefix, name = binding.name.split(".", 1)
        key = f"direct.{prefix}.{name}"
        family = family_by_key[key]
        if binding.master.grad is not None:
            grouped[family] = grouped.get(family, 0.0) + float(
                binding.master.grad.detach().double().square().sum()
            )
    return {family: math.sqrt(value) for family, value in sorted(grouped.items())}


def compensation_warning(
    updates: dict[str, Any],
    routing: dict[str, Any],
) -> dict[str, Any]:
    reference = max(
        float(updates.get("MLP", {}).get("relative_parameter_delta", 0.0)),
        float(updates.get("NORM", {}).get("relative_parameter_delta", 0.0)),
        1.0e-30,
    )
    kda_ratio = float(updates.get("KDA_OUTPUT", {}).get("relative_parameter_delta", 0.0)) / reference
    mla_ratio = float(updates.get("MLA", {}).get("relative_parameter_delta", 0.0)) / reference
    high_reasons = []
    if kda_ratio > 4.0:
        high_reasons.append("KDA_OUTPUT_RELATIVE_DELTA_GT_4X_MLP_NORM_REFERENCE")
    if mla_ratio > 4.0:
        high_reasons.append("MLA_RELATIVE_DELTA_GT_4X_MLP_NORM_REFERENCE")
    return {
        "status": "HIGH" if high_reasons else "LOW",
        "high_reasons": high_reasons,
        "kda_output_delta_ratio": kda_ratio,
        "mla_delta_ratio": mla_ratio,
        "embedding_weight_mean": routing.get("embedding_weight_mean"),
        "recent_weight_mean": routing.get("recent_weight_mean"),
        "assessment_scope": "UPDATE_IMBALANCE_AND_ATTNRES_ENDPOINT_WEIGHTS",
    }


def one_step_smoke() -> dict[str, Any]:
    """Exercise the exact P10 backward/update path without persisting model weights."""

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    state = gpu_start_state()
    report: dict[str, Any] = {
        "stage": f"{RUN_STAGE_PREFIX}_ONE_STEP_MEMORY_AND_BACKWARD_SMOKE",
        "run_label": RUN_LABEL,
        "run_mode": RUN_MODE,
        "status": "RUNNING",
        "started_at": now(),
        "gpu": state,
        "persistent_checkpoint_written": False,
        "heldout_accessed": False,
    }
    atomic_json(SMOKE_FINAL, report)
    if not state["idle"]:
        report.update({"status": "DEFERRED_GPU_NOT_IDLE", "completed_at": now()})
        atomic_json(SMOKE_FINAL, report)
        return report
    seed = int(config["optimization"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    model, runtime, load_report = load_p9_stack()
    scope = SCOPE_BUILDER(
        model,
        runtime,
        rank=int(config["optimization"]["lora_rank"]),
        alpha=float(config["optimization"]["lora_alpha"]),
    )
    frozen_before = frozen_scope_audit(model, scope)
    masters, optimizer, scaler = build_optimizer(scope, config)
    model.train()
    runtime.train()
    model.model.gradient_checkpointing = True
    train = PackedTokenDataset(DATA / "packed", "train")
    tokens = torch.from_numpy(train[0].copy()).unsqueeze(0).cuda()
    optimizer.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    runtime.zero_grad(set_to_none=True)
    hidden = model.model(tokens, use_cache=False, return_dict=True).last_hidden_state
    loss = chunked_causal_lm_loss(
        hidden,
        tokens,
        model.lm_head.weight,
        chunk_size=64,
        checkpoint_chunks=True,
    )
    scaler.scale(loss).backward()
    masters.copy_scaled_grads_from_model()
    scaler.unscale_(optimizer)
    parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    parameter_names = optimizer_parameter_names(scope, masters)
    coverage = audit_gradient_coverage(parameters, parameter_names)
    with_grad = [parameter for parameter in parameters if parameter.grad is not None]
    missing_gradient_names = coverage["missing_gradient_names"]
    unexpected = unexpected_gradient_names(model, runtime, scope)
    finite = bool(torch.isfinite(loss)) and all(
        bool(torch.isfinite(parameter.grad).all()) for parameter in with_grad
    )
    gradient_norm = torch.nn.utils.clip_grad_norm_(with_grad, 1.0)
    scaler.step(optimizer)
    scaler.update()
    masters.copy_to_model()
    frozen_after = frozen_scope_audit(model, scope)
    frozen_checks = {
        "names_present": not frozen_before["missing"] and not frozen_after["missing"],
        "requires_grad_false": frozen_before["all_requires_grad_false"]
        and frozen_after["all_requires_grad_false"],
        "hash_unchanged": frozen_before["tensor_hash"] == frozen_after["tensor_hash"],
    }
    preflight_passed = (
        finite
        and coverage["complete"]
        and not unexpected
        and all(frozen_checks.values())
    )
    report.update(
        {
            "status": (
                "PASS"
                if preflight_passed
                else "BLOCKED_MISSING_GRADIENTS"
                if finite and not coverage["complete"]
                else "BLOCKED_SCOPE_AUDIT"
                if finite
                else "FAIL"
            ),
            "completed_at": now(),
            "loss": float(loss.detach()),
            "gradient_norm_before_clip": float(gradient_norm),
            "gradient_tensors": len(with_grad),
            "optimizer_tensors": len(parameters),
            "gradient_coverage_complete": coverage["complete"],
            "unexpected_gradient_count": len(unexpected),
            "unexpected_gradient_names": unexpected,
            "frozen_scope_audit_before": frozen_before,
            "frozen_scope_audit_after": frozen_after,
            "frozen_scope_checks": frozen_checks,
            "missing_gradient_names": missing_gradient_names,
            "missing_gradient_count_by_family": {
                family: sum(name.startswith(family) for name in missing_gradient_names)
                for family in ("lora", "direct_master.model", "direct_master.attnres")
            },
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "query_norm_after_step": runtime.query_report(),
            "trainable_scope": scope.manifest,
            "load_report": load_report,
            "persistent_checkpoint_written": False,
            "source_files_modified": False,
            "heldout_accessed": False,
        }
    )
    atomic_json(SMOKE_FINAL, report)
    print(f"{RUN_STAGE_PREFIX}_ONE_STEP_SMOKE={report['status']}", flush=True)
    print(f"{RUN_LABEL}_PEAK_ALLOCATED_BYTES={report['peak_allocated_bytes']}", flush=True)
    return report


def train_to_step(phase: str, target_step: int, evaluation_steps: set[int]) -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    state = gpu_start_state()
    report_path = PHASE1_FINAL if phase == "phase1" else EXTENSION_FINAL
    envelope: dict[str, Any] = {
        "stage": f"{RUN_STAGE_PREFIX}_{phase.upper()}",
        "run_label": RUN_LABEL,
        "run_mode": RUN_MODE,
        "status": "RUNNING",
        "started_at": now(),
        "gpu": state,
        "heldout_accessed": False,
        "architecture_modified": False,
        "p11_titan_migration_allowed": False,
    }
    atomic_json(report_path, envelope)
    if not state["idle"]:
        envelope.update({"status": "DEFERRED_GPU_NOT_IDLE", "completed_at": now()})
        atomic_json(report_path, envelope)
        return envelope

    p9 = load_json(P9_FINAL)
    source_identity = p9_identity()
    corpus = data_contract(config)
    prerequisites = {
        "p9_pass": p9["status"] == "PASS",
        "p9_titan_allowed": p9["p10_titan_migration_allowed"] is True,
        "p9_ce_frozen": abs(
            float(p9["p9_full_dev_ce"]) - float(config["source"]["p9_full_dev_ce"])
        ) < 1.0e-12,
        "source_present": P9_SOURCE.is_dir(),
        "corpus": corpus["passed"],
        "heldout_forbidden": config["forbidden"]["heldout"] is True,
        "architecture_frozen": config["architecture"]["modification_allowed"] is False,
    }
    if not all(prerequisites.values()):
        raise ValueError(f"P10 prerequisites failed: {prerequisites}")

    seed = int(config["optimization"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, runtime, load_report = load_p9_stack()
    scope = SCOPE_BUILDER(
        model,
        runtime,
        rank=int(config["optimization"]["lora_rank"]),
        alpha=float(config["optimization"]["lora_alpha"]),
    )
    frozen_phase_start = frozen_scope_audit(model, scope)
    masters, optimizer, scaler = build_optimizer(scope, config)
    train = PackedTokenDataset(DATA / "packed", "train")
    dev = PackedTokenDataset(DATA / "packed", "dev")
    validation_range = tuple(config["data"]["recovery_validation_sequence_range"])
    diagnostic_tokens = torch.from_numpy(dev[0][:128].copy()).unsqueeze(0).cuda()

    existing = checkpoint_manifests()
    if phase == "extension" and (not existing or max(item["step"] for item in existing) < 128):
        raise ValueError("P10 extension requires the exact phase-1 step-128 checkpoint")
    if existing:
        latest = max(
            (item for item in existing if item["step"] <= target_step),
            key=lambda item: item["step"],
        )
        if latest["source_fingerprint"] != source_identity["fingerprint"]:
            raise ValueError("P10 checkpoint source fingerprint changed")
        restore_checkpoint(latest, scope, masters, optimizer, scaler)
        start_step = int(latest["step"])
    else:
        start_step = 0
        evaluation = evaluate_checkpoint(
            model,
            runtime,
            dev,
            diagnostic_tokens,
            validation_range,
            step=0,
        )
        zero_values = trainable_state(scope)
        evaluation["parameter_update_by_family"] = parameter_update_by_family(
            scope, zero_values
        )
        evaluation["kda_freeze_compensation_warning"] = {
            "status": "BASELINE",
            "high_reasons": [],
        }
        save_checkpoint(0, scope, masters, optimizer, scaler, evaluation, source_identity)

    max_steps = int(config["budget"]["maximum_effective_tokens"]) // P5_SEQUENCE_LENGTH
    recipe = config["optimization"]
    training_curve = []
    model.model.gradient_checkpointing = True
    started = time.perf_counter()
    for step in range(start_step, target_step):
        model.train()
        runtime.train()
        lora_lr = cosine_learning_rate(
            step,
            max_steps,
            float(recipe["lora_learning_rate"]),
            warmup_ratio=float(recipe["warmup_ratio"]),
            minimum_ratio=float(recipe["minimum_learning_rate_ratio"]),
        )
        direct_lr = cosine_learning_rate(
            step,
            max_steps,
            float(recipe["direct_learning_rate"]),
            warmup_ratio=float(recipe["warmup_ratio"]),
            minimum_ratio=float(recipe["minimum_learning_rate_ratio"]),
        )
        optimizer.param_groups[0]["lr"] = lora_lr
        optimizer.param_groups[1]["lr"] = direct_lr
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        runtime.zero_grad(set_to_none=True)
        tokens = torch.from_numpy(train[step].copy()).unsqueeze(0).cuda()
        hidden = model.model(tokens, use_cache=False, return_dict=True).last_hidden_state
        loss = chunked_causal_lm_loss(
            hidden,
            tokens,
            model.lm_head.weight,
            chunk_size=64,
            checkpoint_chunks=True,
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite P10 loss at step {step + 1}")
        scaler.scale(loss).backward()
        masters.copy_scaled_grads_from_model()
        scaler.unscale_(optimizer)
        optimizer_parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        gradients_with_values = [
            parameter for parameter in optimizer_parameters if parameter.grad is not None
        ]
        gradient_coverage = audit_gradient_coverage(
            optimizer_parameters,
            optimizer_parameter_names(scope, masters),
        )
        if not gradient_coverage["complete"]:
            raise RuntimeError(
                "P10 gradient coverage incomplete before optimizer step: "
                f"{gradient_coverage['gradient_tensors']}/"
                f"{gradient_coverage['optimizer_tensors']} tensors; "
                f"missing={gradient_coverage['missing_gradient_count']}"
            )
        unexpected = unexpected_gradient_names(model, runtime, scope)
        if unexpected:
            raise RuntimeError(
                f"{RUN_LABEL} unexpected gradients outside approved scope: "
                f"count={len(unexpected)} first={unexpected[:8]}"
            )
        gradients_finite = all(
            bool(torch.isfinite(parameter.grad).all()) for parameter in gradients_with_values
        )
        gradient_norm_by_family = optimizer_gradient_norm_by_family(scope, masters)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            gradients_with_values,
            float(recipe["gradient_clip_norm"]),
        )
        if not gradients_finite or not math.isfinite(float(gradient_norm)):
            raise FloatingPointError(f"Invalid P10 gradients at step {step + 1}")
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        masters.copy_to_model()
        completed = step + 1
        training_curve.append(
            {
                "step": completed,
                "effective_tokens": completed * P5_SEQUENCE_LENGTH,
                "sequence_index": step,
                "causal_lm_ce": float(loss.detach()),
                "lora_learning_rate": lora_lr,
                "direct_learning_rate": direct_lr,
                "gradient_norm_before_clip": float(gradient_norm),
                "gradient_norm_by_family_before_clip": gradient_norm_by_family,
                "gradient_tensors": len(gradients_with_values),
                "optimizer_tensors": len(optimizer_parameters),
                "grad_scaler_before": old_scale,
                "grad_scaler_after": scaler.get_scale(),
            }
        )
        del tokens, hidden, loss
        if completed % 8 == 0:
            print(
                f"{RUN_STAGE_PREFIX}_TRAIN STEP={completed}/{target_step} "
                f"TOKENS={completed * P5_SEQUENCE_LENGTH} "
                f"CE={training_curve[-1]['causal_lm_ce']:.6f}",
                flush=True,
            )
        if completed in evaluation_steps:
            gc.collect()
            torch.cuda.empty_cache()
            evaluation = evaluate_checkpoint(
                model,
                runtime,
                dev,
                diagnostic_tokens,
                validation_range,
                step=completed,
            )
            zero_manifest = next(
                item for item in checkpoint_manifests() if int(item["step"]) == 0
            )
            zero_values = load_file(
                str(Path(zero_manifest["path"]) / "trainable.safetensors")
            )
            updates = parameter_update_by_family(scope, zero_values)
            evaluation["parameter_update_by_family"] = updates
            evaluation["gradient_norm_by_family_before_clip"] = gradient_norm_by_family
            evaluation["kda_freeze_compensation_warning"] = compensation_warning(
                updates, evaluation["routing"]
            )
            save_checkpoint(
                completed,
                scope,
                masters,
                optimizer,
                scaler,
                evaluation,
                source_identity,
            )
            print(
                f"{RUN_STAGE_PREFIX}_EVAL TOKENS={completed * P5_SEQUENCE_LENGTH} "
                f"VAL_CE={evaluation['recovery_validation_ce']:.9f} "
                f"QUERY_MAX={evaluation['routing']['query']['max']:.6g} "
                f"NONUNIFORM={evaluation['routing']['nonuniform_sites']}",
                flush=True,
            )

    manifests = [item for item in checkpoint_manifests() if item["step"] <= target_step]
    evaluations = [item["evaluation"] for item in manifests]
    decision = phase_decision(evaluations, target_step)
    frozen_phase_end = frozen_scope_audit(model, scope)
    frozen_scope_checks = {
        "requires_grad_false": frozen_phase_start["all_requires_grad_false"]
        and frozen_phase_end["all_requires_grad_false"],
        "hash_unchanged": frozen_phase_start["tensor_hash"]
        == frozen_phase_end["tensor_hash"],
        "names_present": not frozen_phase_start["missing"] and not frozen_phase_end["missing"],
    }
    if not all(frozen_scope_checks.values()):
        raise RuntimeError(f"{RUN_LABEL} frozen scope changed during training: {frozen_scope_checks}")
    result = {
        **envelope,
        "status": "PASS_PHASE_COMPLETE",
        "completed_at": now(),
        "wall_seconds": time.perf_counter() - started,
        "prerequisites": prerequisites,
        "source_identity": source_identity,
        "load_report": load_report,
        "data_contract": corpus,
        "trainable_scope": scope.manifest,
        "frozen_scope_audit_start": frozen_phase_start,
        "frozen_scope_audit_end": frozen_phase_end,
        "frozen_scope_checks": frozen_scope_checks,
        "optimizer_contract": {
            "lora_lr": recipe["lora_learning_rate"],
            "direct_lr": recipe["direct_learning_rate"],
            "fp32_master_parameters": True,
            "schedule_total_steps": max_steps,
            "schedule_not_restarted_at_phase_boundary": True,
        },
        "evaluations": evaluations,
        "decision": decision,
        "training_curve": training_curve,
        "heldout_accessed": False,
        "architecture_modified": False,
        "p10_lora_used_for_optimization": True,
        "p10_lora_present_at_runtime": None,
        "p11_titan_migration_allowed": False,
    }
    atomic_json(report_path, result)
    print(f"{RUN_STAGE_PREFIX}_{phase.upper()}=PASS_PHASE_COMPLETE", flush=True)
    print(f"{RUN_LABEL}_EXTENSION_REVIEW={decision['extension_review']}", flush=True)
    print("HELDOUT_ACCESSED=false", flush=True)
    return result


def candidate_override_state(model: torch.nn.Module, scope: Any) -> dict[str, torch.Tensor]:
    parameter_map = dict(model.named_parameters())
    names = {f"{handle.name}.weight" for handle in scope.lora} | set(scope.direct_model)
    missing = names - set(parameter_map)
    if missing:
        raise KeyError(f"Merged P10 dense parameters missing: {sorted(missing)[:8]}")
    return {name: parameter_map[name].detach().cpu() for name in sorted(names)}


@torch.no_grad()
def apply_dense_overrides(model: torch.nn.Module, path: Path) -> None:
    values = load_file(str(path))
    parameters = dict(model.named_parameters())
    for name, value in values.items():
        target = parameters[name]
        target.copy_(value.to(target.device, target.dtype))


def closeout() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    state = gpu_start_state()
    envelope: dict[str, Any] = {
        "stage": f"{RUN_STAGE_PREFIX}_CORE_K3_JOINT_RECOVERY_CLOSEOUT",
        "run_label": RUN_LABEL,
        "run_mode": RUN_MODE,
        "status": "RUNNING",
        "started_at": now(),
        "gpu": state,
        "heldout_accessed": False,
        "p11_titan_migration_allowed": False,
    }
    atomic_json(FINAL, envelope)
    if not state["idle"]:
        envelope.update({"status": "DEFERRED_GPU_NOT_IDLE", "completed_at": now()})
        atomic_json(FINAL, envelope)
        return envelope
    manifests = checkpoint_manifests()
    if not manifests:
        raise ValueError("P10 closeout has no training checkpoints")
    trained = [item for item in manifests if item["step"] > 0]
    best_trained = min(
        trained,
        key=lambda item: item["evaluation"]["recovery_validation_ce"],
    )
    step_zero = next(item for item in manifests if item["step"] == 0)
    safety_best = min(
        manifests,
        key=lambda item: item["evaluation"]["recovery_validation_ce"],
    )
    if safety_best["step"] == 0:
        result = {
            **envelope,
            "status": "NO_GO_STEP_ZERO_SAFETY_FALLBACK",
            "completed_at": now(),
            "best_checkpoint": safety_best,
            "best_trained_checkpoint": best_trained,
            "p10_joint_recovery": "NO_GO",
            "p10_architecture_retained": True,
            "heldout_accessed": False,
        }
        atomic_json(FINAL, result)
        return result

    model, runtime, load_report = load_p9_stack()
    scope = SCOPE_BUILDER(
        model,
        runtime,
        rank=int(config["optimization"]["lora_rank"]),
        alpha=float(config["optimization"]["lora_alpha"]),
    )
    frozen_candidate_before = frozen_scope_audit(model, scope)
    frozen_audit_names = list(scope.manifest.get("frozen_audit_parameter_names", []))
    load_trainable_state(scope, Path(safety_best["path"]) / "trainable.safetensors")
    initial_values = load_file(str(Path(step_zero["path"]) / "trainable.safetensors"))
    direct_delta = {}
    for prefix, mapping, family in (
        ("direct.model", scope.direct_model, "MODEL_DIRECT"),
        ("direct.attnres", scope.direct_attnres, "ATTNRES"),
    ):
        squared = 0.0
        changed = 0
        for name, parameter in mapping.items():
            before = initial_values[f"{prefix}.{name}"].float()
            after = parameter.detach().cpu().float()
            difference = after - before
            squared += float(difference.double().square().sum())
            changed += not torch.equal(before, after)
        direct_delta[family] = {"delta_norm": math.sqrt(squared), "changed_tensors": changed}
    lora_merge = merge_all_lora(scope.lora)
    if any("lora" in name.lower() for name, _ in model.named_parameters()):
        raise RuntimeError(f"{RUN_LABEL} LoRA parameters remain after merge")

    if CANDIDATE.exists():
        raise FileExistsError(f"{RUN_LABEL} candidate exists; refusing to overwrite")
    staging = CANDIDATE.with_name(CANDIDATE.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    overrides = candidate_override_state(model, scope)
    save_file(overrides, str(staging / "model-dense-overrides.safetensors"))
    attnres_state = {name: value.detach().cpu() for name, value in runtime.state_dict().items()}
    save_file(attnres_state, str(staging / "attnres.safetensors"))
    runtime_config = {
        "architecture": CANDIDATE_ARCHITECTURE,
        "run_label": RUN_LABEL,
        "run_mode": RUN_MODE,
        "source_p9": str(P9_SOURCE.resolve()),
        "effective_tokens": safety_best["effective_tokens"],
        "lora_used_for_optimization": True,
        "lora_present_at_runtime": False,
        "dense_override_tensor_count": len(overrides),
        "attnres_tensor_count": len(attnres_state),
        "titan_dtype": "fp16",
        "canonical_bf16_replay": "PENDING",
    }
    atomic_json(staging / RUNTIME_CONFIG_FILE, runtime_config)
    files = {
        path.name: sha256_file(path)
        for path in staging.iterdir()
        if path.is_file()
    }
    atomic_json(
        staging / CANDIDATE_MANIFEST_FILE,
        {
            "stage": CANDIDATE_STAGE,
            "status": "PENDING_RELOAD_EVALUATION",
            "source_identity": p9_identity(),
            "selected_checkpoint": safety_best,
            "files": files,
            "lora_merge": lora_merge,
            "direct_delta": direct_delta,
            "frozen_scope_audit_source": frozen_candidate_before,
            "kda_freeze_compensation": safety_best["evaluation"].get(
                "kda_freeze_compensation_warning"
            ),
            "heldout_accessed": False,
        },
    )
    staging.replace(CANDIDATE)

    del overrides, initial_values, model, runtime, scope
    gc.collect()
    torch.cuda.empty_cache()
    reloaded, reloaded_runtime, reloaded_load = load_p9_stack()
    apply_dense_overrides(reloaded, CANDIDATE / "model-dense-overrides.safetensors")
    reloaded_runtime.load_state_dict(load_file(str(CANDIDATE / "attnres.safetensors")))
    if any("lora" in name.lower() for name, _ in reloaded.named_parameters()):
        raise RuntimeError(f"Reloaded {RUN_LABEL} runtime contains LoRA")
    dev = PackedTokenDataset(DATA / "packed", "dev")
    full_dev_ce = corpus_ce(reloaded, dev, label=f"{RUN_LABEL}_CLOSEOUT")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=False)
    cache_limits = yaml.safe_load(CACHE_CONTRACT.read_text(encoding="utf-8"))[
        "evaluation"
    ]["fallback_requires_all"]
    # Helpers own the temporary AttnRes enable/disable boundary.
    from attnres import disable_block_attnres_runtime

    disable_block_attnres_runtime(reloaded)
    generated = generation(reloaded, tokenizer, reloaded_runtime)
    cache = cache_equivalence(reloaded, reloaded_runtime, cache_limits)
    enable_block_attnres_runtime(reloaded, reloaded_runtime)
    diagnostic_tokens = torch.from_numpy(dev[0][:128].copy()).unsqueeze(0).cuda()
    routing = routing_diagnostic(reloaded, reloaded_runtime, diagnostic_tokens)
    frozen_candidate_after = frozen_parameter_audit(reloaded, frozen_audit_names)
    source_ce = float(config["source"]["p9_full_dev_ce"])
    strong_ce = float(config["source"]["p8_zero_full_dev_ce"])
    safety = {
        "effective_tokens_gt_zero": safety_best["effective_tokens"] > 0,
        "attnres_active": routing["active"],
        "full_dev_finite": math.isfinite(full_dev_ce),
        "generation_finite": all(item["finite"] for item in generated),
        "cache_pass": cache["status"] == "PASS",
        "lora_absent_runtime": not any(
            "lora" in name.lower() for name, _ in reloaded.named_parameters()
        ),
        "embedding_lm_tied": reloaded.model.embed_tokens.weight.data_ptr()
        == reloaded.lm_head.weight.data_ptr(),
        "heldout_not_accessed": True,
        "frozen_kda_exact": frozen_candidate_before["tensor_hash"]
        == frozen_candidate_after["tensor_hash"],
    }
    if full_dev_ce <= strong_ce and all(safety.values()):
        decision = "STRONG_PASS"
    elif full_dev_ce <= source_ce and all(safety.values()):
        decision = "PASS"
    else:
        decision = "NO_GO"
    candidate_manifest = load_json(CANDIDATE / CANDIDATE_MANIFEST_FILE)
    candidate_manifest.update(
        {
            "status": "PASS" if decision in {"PASS", "STRONG_PASS"} else "NO_GO",
            "full_dev_ce": full_dev_ce,
            "routing": routing,
            "generation": generated,
            "cache": cache,
            "safety": safety,
            "decision": decision,
            "lora_present_at_runtime": False,
            "frozen_scope_audit_candidate": frozen_candidate_after,
        }
    )
    atomic_json(CANDIDATE / CANDIDATE_MANIFEST_FILE, candidate_manifest)
    result = {
        **envelope,
        "status": "PASS" if decision in {"PASS", "STRONG_PASS"} else "NO_GO",
        "completed_at": now(),
        "decision": decision,
        "selected_checkpoint": safety_best,
        "best_trained_checkpoint": best_trained,
        "p9_full_dev_ce": source_ce,
        "p8_zero_full_dev_ce": strong_ce,
        "p10_full_dev_ce": full_dev_ce,
        "delta_ce_vs_p9": full_dev_ce - source_ce,
        "full_architecture_shock_recovered": full_dev_ce <= strong_ce,
        "routing": routing,
        "generation": generated,
        "cache": cache,
        "safety": safety,
        "candidate": str(CANDIDATE.resolve()),
        "load_report": load_report,
        "reload_report": reloaded_load,
        "frozen_scope_audit_source": frozen_candidate_before,
        "frozen_scope_audit_candidate": frozen_candidate_after,
        "p10_lora_used_for_optimization": True,
        "p10_lora_present_at_runtime": False,
        "p10_formal_status": "TITAN_FP16_PROVISIONAL_PENDING_NATIVE_BF16_REPLAY",
        "canonical_p10_status": "PENDING_NATIVE_BF16_FULL_SCOPE_RETRAIN",
        "p10_titan_is_canonical": False,
        "kda_freeze_compensation": safety_best["evaluation"].get(
            "kda_freeze_compensation_warning"
        ),
        "p11_titan_migration_allowed": decision in {"PASS", "STRONG_PASS"},
        "p11_titan_source": (
            str(CANDIDATE.resolve()) if decision in {"PASS", "STRONG_PASS"} else None
        ),
        "heldout_accessed": False,
    }
    atomic_json(FINAL, result)
    print(f"{RUN_STAGE_PREFIX}_JOINT_RECOVERY={decision}", flush=True)
    print(f"{RUN_LABEL}_FULL_DEV_CE={full_dev_ce:.12f}", flush=True)
    print(f"{RUN_LABEL}_LORA_PRESENT_AT_RUNTIME={str(False).lower()}", flush=True)
    print(f"P11_TITAN_MIGRATION_ALLOWED={str(result['p11_titan_migration_allowed']).lower()}", flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("smoke", "phase1", "extension", "closeout"),
        default="phase1",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_path = {
        "smoke": SMOKE_FINAL,
        "phase1": PHASE1_FINAL,
        "extension": EXTENSION_FINAL,
        "closeout": FINAL,
    }[args.phase]
    try:
        if args.phase == "smoke":
            print(f"{RUN_STAGE_PREFIX}_ONE_STEP_SMOKE=RUNNING", flush=True)
            result = one_step_smoke()
        elif args.phase == "phase1":
            print(f"{RUN_STAGE_PREFIX}_PHASE1=RUNNING", flush=True)
            result = train_to_step("phase1", 128, {32, 64, 128})
        elif args.phase == "extension":
            print(f"{RUN_STAGE_PREFIX}_EXTENSION=RUNNING", flush=True)
            result = train_to_step("extension", 512, {256, 512})
        else:
            print(f"{RUN_STAGE_PREFIX}_CLOSEOUT=RUNNING", flush=True)
            result = closeout()
        return 0 if result["status"] not in {
            "ERROR",
            "DEFERRED_GPU_NOT_IDLE",
            "BLOCKED_MISSING_GRADIENTS",
        } else 1
    except Exception as error:  # noqa: BLE001
        failure = {
            "stage": f"{RUN_STAGE_PREFIX}_{args.phase.upper()}",
            "run_label": RUN_LABEL,
            "run_mode": RUN_MODE,
            "status": "ERROR",
            "exception_type": type(error).__name__,
            "exception": str(error),
            "traceback": traceback.format_exc(),
            "completed_at": now(),
            "heldout_accessed": False,
            "architecture_modified": False,
            "p11_titan_migration_allowed": False,
        }
        atomic_json(report_path, failure)
        print(f"{RUN_STAGE_PREFIX}_EXCEPTION={type(error).__name__}: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
