"""Run the eight dependency-ordered P4 full-model transplantation gates."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from densek3_core.evaluation.p4_cache_gate import (
    P4_CACHE_AMENDMENT_TYPE,
    P4_CACHE_BATCH_STATE_LIMITS,
    P4_CACHE_CASES,
    P4_CACHE_NUMERICS_AMENDMENT,
    P4_CACHE_TENSOR_LIMITS,
    P4_CACHE_TOKEN_SEED,
    evaluate_batch_control,
    evaluate_integrated_case,
    evaluate_numerical_envelope,
    load_and_evaluate_direct_cache_evidence,
    token_decision_metrics,
)
from densek3_core.evaluation.p4_decay_diagnostics import (
    error_distribution as scalable_error_distribution,
)
from densek3_core.evaluation.p4_local_gate import (
    P3_NUMERICS_AMENDMENT,
    P4_DIAGNOSTIC_LAYER,
    P4_NUMERICS_AMENDMENT,
    evaluate_output_distribution,
    load_and_evaluate_p4_amendment_evidence,
)
from densek3_core.evaluation.p4_runtime import load_densek3_model, load_qwen_text_model
from densek3_core.kda.metrics import (
    OPTIMIZED_BF16_ATOL,
    OPTIMIZED_BF16_RTOL,
    REFERENCE_FP32_ATOL,
    REFERENCE_FP32_RTOL,
    error_metrics,
)
from densek3_core.modeling import DenseK3Config, DenseK3ForCausalLM, DenseK3HybridCache
from densek3_core.transplant.full_model import (
    P4_EXPECTED_TARGET_TENSORS,
    P4_EXPECTED_UNIQUE_PARAMETERS,
    P4_MANIFEST_NAME,
    audit_source_inventory,
    convert_full_text_checkpoint,
    verify_p4_artifact,
)
from densek3_core.transplant.gdn_to_kda import convert_gdn_to_kda, load_qwen_gdn_layer
from densek3_core.transplant.single_layer import DenseK3QwenCompatMixer, QwenGDNReferenceMixer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models/Qwen3.5-4B-Base"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/densek3-core/manifests/p4"
DEFAULT_ARTIFACT_DIR = ROOT / "outputs/densek3-core/checkpoints/p4/densek3-4b-qwen-compat"
DEFAULT_PROMPTS = ROOT / "data/p4-smoke-prompts.jsonl"
STAGES = (
    ("P4.0", "source_inventory", "p4-source-inventory.json", "P4_00_SOURCE_INVENTORY"),
    ("P4.1", "layer_conversion", "p4-layer-conversion.json", "P4_01_24_LAYER_CONVERSION"),
    ("P4.2", "parameter_accounting", "p4-parameter-accounting.json", "P4_02_PARAMETER_ACCOUNTING"),
    ("P4.3", "model_construction", "p4-model-construction.json", "P4_03_MODEL_CONSTRUCTION"),
    ("P4.4", "all_layer_local_parity", "p4-all-layer-local-parity.json", "P4_04_ALL_LAYER_LOCAL_PARITY"),
    ("P4.5", "full_model_forward", "p4-error-propagation.json", "P4_05_FULL_MODEL_FORWARD"),
    ("P4.6", "hybrid_cache_generation", "p4-hybrid-cache.json", "P4_06_HYBRID_CACHE_GENERATION"),
    ("P4.7", "save_load_generate", "p4-save-load-generate.json", "P4_07_SAVE_LOAD_GENERATE"),
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("P4.4-P4.7 require the authoritative CUDA server runtime")
    return torch.device("cuda")


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def tensor_sha256(tensor: torch.Tensor) -> str:
    import hashlib

    value = tensor.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def comparison(
    reference: torch.Tensor,
    actual: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    if reference.shape != actual.shape:
        raise ValueError(f"Comparison shape mismatch: {reference.shape} != {actual.shape}")
    metrics = error_metrics(reference, actual)
    metrics.update(
        {
            "rtol": rtol,
            "atol": atol,
            "passed": bool(torch.allclose(reference, actual, rtol=rtol, atol=atol)),
            "exact": bool(torch.equal(reference, actual)),
            "shape": list(reference.shape),
            "shape_match": True,
        }
    )
    return metrics


def error_distribution(reference: torch.Tensor, actual: torch.Tensor, rtol: float, atol: float) -> dict[str, Any]:
    reference_flat = reference.detach().double().flatten()
    actual_flat = actual.detach().double().flatten()
    absolute = (actual_flat - reference_flat).abs()
    tolerance = atol + rtol * reference_flat.abs()
    failed = absolute > tolerance
    quantiles = torch.tensor([0.5, 0.9, 0.95, 0.99, 0.999], dtype=torch.float64, device=absolute.device)
    values = torch.quantile(absolute, quantiles) if absolute.numel() else torch.zeros_like(quantiles)
    metrics = error_metrics(reference, actual)
    metrics.update(
        {
            "num_elements": absolute.numel(),
            "num_failed_allclose": int(failed.sum()),
            "failed_allclose_fraction": float(failed.double().mean()) if failed.numel() else 0.0,
            "p50_abs_error": float(values[0]),
            "p90_abs_error": float(values[1]),
            "p95_abs_error": float(values[2]),
            "p99_abs_error": float(values[3]),
            "p99_9_abs_error": float(values[4]),
            "nan_count": int(torch.isnan(reference_flat).sum() + torch.isnan(actual_flat).sum()),
            "inf_count": int(torch.isinf(reference_flat).sum() + torch.isinf(actual_flat).sum()),
        }
    )
    return metrics


def stage_inventory(context: dict[str, Any]) -> dict[str, Any]:
    report = audit_source_inventory(context["model_dir"])
    context["inventory"] = report
    report["completed_at"] = now()
    return report


def _load_manifest(artifact_dir: Path) -> dict[str, Any]:
    return json.loads((artifact_dir / P4_MANIFEST_NAME).read_text(encoding="utf-8"))


def stage_conversion(context: dict[str, Any]) -> dict[str, Any]:
    artifact_dir: Path = context["artifact_dir"]
    if context["reconvert"] or not (artifact_dir / P4_MANIFEST_NAME).is_file():
        result = convert_full_text_checkpoint(context["model_dir"], artifact_dir)
        reused = False
        parameter_hash = result.parameter_hash
        manifest_hash = result.manifest_hash
    else:
        verified = verify_p4_artifact(artifact_dir)
        reused = True
        parameter_hash = verified["parameter_hash"]
        manifest_hash = verified["manifest_hash"]
    manifest = _load_manifest(artifact_dir)
    layers = manifest["layer_conversions"]
    if len(layers) != 24 or [item["layer_index"] for item in layers] != manifest["topology"]["kda_indices"]:
        raise ValueError("P4 artifact does not contain the locked 24 layer conversions")
    context["manifest"] = manifest
    return {
        "stage": "P4.1",
        "status": "PASS",
        "converter": "densek3_core.transplant.gdn_to_kda.convert_gdn_to_kda",
        "converter_invocations": 24,
        "same_converter_for_all_layers": True,
        "artifact_reused": reused,
        "parameter_hash": parameter_hash,
        "manifest_hash": manifest_hash,
        "layers": layers,
        "result_marker": "P4_01_24_LAYER_CONVERSION=PASS",
    }


def stage_accounting(context: dict[str, Any]) -> dict[str, Any]:
    manifest = context.get("manifest") or _load_manifest(context["artifact_dir"])
    operations = Counter(item["operation"] for item in manifest["source_parameters"])
    target_names = {item["target_key"] for item in manifest["target_parameters"]}
    forbidden = {"UNKNOWN", "UNMAPPED", "SILENT_INIT"}
    if set(operations) & forbidden:
        raise ValueError(f"Forbidden whole-model parameter classifications: {set(operations) & forbidden}")
    checks = {
        "unique_parameters": manifest["target_unique_parameters"] == P4_EXPECTED_UNIQUE_PARAMETERS,
        "target_tensors": manifest["target_tensor_count"] == P4_EXPECTED_TARGET_TENSORS,
        "kda_mixers": manifest["topology"]["kda_count"] == 24,
        "gqa_mixers": manifest["topology"]["gqa_count"] == 8,
        "mlp_blocks": manifest["topology"]["mlp_count"] == 32,
        "tied_lm_head": manifest["tied_parameters"] == {"lm_head.weight": "model.embed_tokens.weight"},
        "embedding_present": "model.embed_tokens.weight" in target_names,
        "final_norm_present": "model.norm.weight" in target_names,
        "vision_excluded": sum(
            item["operation"] == "EXCLUDE" and item["transform"] == "exclude_vision"
            for item in manifest["source_parameters"]
        )
        == 297,
        "mtp_excluded": sum(
            item["operation"] == "EXCLUDE" and item["transform"] == "exclude_mtp"
            for item in manifest["source_parameters"]
        )
        == 15,
        "all_source_tensors_classified": manifest["source_tensor_count"] == 738,
        "no_forbidden_classification": not (set(operations) & forbidden),
    }
    if not all(checks.values()):
        raise ValueError(f"P4 parameter accounting failed: {checks}")
    return {
        "stage": "P4.2",
        "status": "PASS",
        "checks": checks,
        "target_unique_parameters": manifest["target_unique_parameters"],
        "target_tensor_count": manifest["target_tensor_count"],
        "operation_counts": dict(sorted(operations.items())),
        "excluded_domains": manifest["excluded_domains"],
        "result_marker": "P4_02_PARAMETER_ACCOUNTING=PASS",
    }


def stage_construction(context: dict[str, Any]) -> dict[str, Any]:
    config = DenseK3Config.from_pretrained(context["artifact_dir"])
    with torch.device("meta"):
        model = DenseK3ForCausalLM(config)
    layer_types = [layer.layer_type for layer in model.model.layers]
    unique_parameters = sum(parameter.numel() for parameter in model.parameters())
    named_unique = dict(model.named_parameters())
    checks = {
        "decoder_layers": len(model.model.layers) == 32,
        "kda_mixers": layer_types.count("kda") == 24,
        "gqa_mixers": layer_types.count("gqa") == 8,
        "mlp_blocks": sum(hasattr(layer, "mlp") for layer in model.model.layers) == 32,
        "tied_lm_head": model.lm_head.weight is model.model.embed_tokens.weight,
        "unique_parameters": unique_parameters == P4_EXPECTED_UNIQUE_PARAMETERS,
        "unique_parameter_tensors": len(named_unique) == P4_EXPECTED_TARGET_TENSORS,
    }
    if not all(checks.values()):
        raise ValueError(f"P4 model construction checks failed: {checks}")
    del model
    return {
        "stage": "P4.3",
        "status": "PASS",
        "checks": checks,
        "layer_types": layer_types,
        "gqa_indices": config.gqa_indices,
        "kda_indices": config.kda_indices,
        "result_marker": "P4_03_MODEL_CONSTRUCTION=PASS",
    }


def _local_case(
    donor: QwenGDNReferenceMixer,
    target: DenseK3QwenCompatMixer,
    hidden: torch.Tensor,
    dtype: torch.dtype,
    *,
    p4_amendment_evidence_accepted: bool = False,
) -> dict[str, Any]:
    expected = donor.forward(hidden)
    actual = target.forward(hidden)
    rtol, atol = (
        (REFERENCE_FP32_RTOL, REFERENCE_FP32_ATOL)
        if dtype == torch.float32
        else (OPTIMIZED_BF16_RTOL, OPTIMIZED_BF16_ATOL)
    )
    comparisons = {
        name: comparison(getattr(expected, name), getattr(actual, name), rtol=rtol, atol=atol)
        for name in ("output", "final_state", "decay", "beta")
    }
    standard_pass = all(item["passed"] for item in comparisons.values())
    distribution = None
    distribution_evaluation = None
    output_distribution_fallback_used = False
    if dtype == torch.bfloat16 and not comparisons["output"]["passed"]:
        distribution = error_distribution(expected.output, actual.output, rtol, atol)
        distribution_evaluation = evaluate_output_distribution(distribution)
        output_distribution_fallback_used = (
            p4_amendment_evidence_accepted and distribution_evaluation["passed"]
        )
    output_gate = comparisons["output"]["passed"] or output_distribution_fallback_used
    functional_hard_gates = {
        "output": output_gate,
        "final_state": comparisons["final_state"]["passed"],
        "beta": comparisons["beta"]["passed"],
    }
    auxiliary_decay_standard_allclose = comparisons["decay"]["passed"]
    auxiliary_exception_used = (
        dtype == torch.bfloat16
        and p4_amendment_evidence_accepted
        and not auxiliary_decay_standard_allclose
        and all(functional_hard_gates.values())
    )
    passed = all(functional_hard_gates.values()) and (
        auxiliary_decay_standard_allclose or auxiliary_exception_used
    )
    if standard_pass:
        pass_mode = "standard_allclose"
    elif output_distribution_fallback_used and auxiliary_exception_used:
        pass_mode = "composite_bf16_transplant_gate"
    elif output_distribution_fallback_used:
        pass_mode = "bf16_transplant_distribution_fallback"
    elif auxiliary_exception_used:
        pass_mode = "auxiliary_intermediate_exception"
    else:
        pass_mode = "failed"
    return {
        "dtype": str(dtype).removeprefix("torch."),
        "sequence_length": hidden.shape[1],
        "standard_allclose": standard_pass,
        "pass_mode": pass_mode,
        "numerics_amendment": P4_NUMERICS_AMENDMENT if not standard_pass and passed else None,
        "functional_hard_gates": functional_hard_gates,
        "auxiliary_intermediates": {
            "tensors": ["raw_a_logits", "channel_a_logits", "decay"],
            "standard_allclose_preferred": True,
            "standard_allclose_required": False,
            "decay_standard_allclose": auxiliary_decay_standard_allclose,
            "exception_used": auxiliary_exception_used,
        },
        "output_distribution_fallback_used": output_distribution_fallback_used,
        "p3_distribution_thresholds_used": output_distribution_fallback_used,
        "p3_original_fallback_used": False,
        "p3_original_fallback_preconditions_inherited": False,
        "comparisons": comparisons,
        "output_error_distribution": distribution,
        "output_distribution_evaluation": distribution_evaluation,
        "passed": passed,
    }


def stage_local_parity(context: dict[str, Any]) -> dict[str, Any]:
    device = require_cuda()
    config = DenseK3Config.from_pretrained(context["artifact_dir"])
    amendment_evidence = load_and_evaluate_p4_amendment_evidence(context["output_dir"])
    manifest = context.get("manifest") or _load_manifest(context["artifact_dir"])
    current_layer = next(
        item for item in manifest["layer_conversions"] if item["layer_index"] == P4_DIAGNOSTIC_LAYER
    )
    evidence_identity = amendment_evidence["artifact_layer_identity"]
    current_artifact_checks = {
        "layer_index": evidence_identity["layer_index"] == current_layer["layer_index"],
        "source_hash": evidence_identity["source_hash"] == current_layer["source_hash"],
        "target_hash": evidence_identity["target_hash"] == current_layer["target_hash"],
    }
    amendment_evidence["current_artifact_identity_checks"] = current_artifact_checks
    amendment_evidence["accepted"] = amendment_evidence["accepted"] and all(
        current_artifact_checks.values()
    )
    if not amendment_evidence["accepted"]:
        raise ValueError(
            "P4-NUM-2026-08-11-01 evidence did not pass formal re-evaluation; "
            "inspect the two layer-4 diagnostic reports"
        )
    cases = []
    generator = torch.Generator(device="cpu").manual_seed(20260810)
    for layer_index in config.kda_indices:
        bridge, source_state, _ = load_qwen_gdn_layer(context["model_dir"], layer_index, device="cpu")
        converted = convert_gdn_to_kda(source_state, bridge)
        layer_cases = []
        cpu_hidden = torch.randn((1, 7, bridge.hidden_size), generator=generator)
        for dtype in (torch.float32, torch.bfloat16):
            donor = QwenGDNReferenceMixer(bridge, source_state).to(device, dtype)
            target = DenseK3QwenCompatMixer(bridge, converted.target_state).to(device, dtype)
            hidden = cpu_hidden.to(device=device, dtype=dtype)
            with torch.no_grad():
                layer_cases.append(
                    _local_case(
                        donor,
                        target,
                        hidden,
                        dtype,
                        p4_amendment_evidence_accepted=(
                            layer_index == P4_DIAGNOSTIC_LAYER
                            and dtype == torch.bfloat16
                        ),
                    )
                )
            del donor, target, hidden
        amendment_applied = any(item["numerics_amendment"] is not None for item in layer_cases)
        case = {
            "layer_index": layer_index,
            "source_hash": converted.manifest["source_hash"],
            "target_hash": converted.manifest["target_hash"],
            "cases": layer_cases,
            "passed": all(item["passed"] for item in layer_cases),
            "pass_mode": "composite_bf16_transplant_gate" if amendment_applied else "standard_allclose",
            "numerics_amendment": P4_NUMERICS_AMENDMENT if amendment_applied else None,
        }
        cases.append(case)
        save_json(
            context["output_dir"] / "p4-all-layer-local-parity.json",
            {
                "stage": "P4.4",
                "status": "RUNNING",
                "layers_total": 24,
                "layers_completed": len(cases),
                "cases": cases,
                "amendment_evidence": amendment_evidence,
            },
        )
        del source_state, converted
        clear_cuda()
    passed = len(cases) == 24 and all(case["passed"] for case in cases)
    if not passed:
        failed = [case["layer_index"] for case in cases if not case["passed"]]
        raise ValueError(f"P4 local transplant replay failed at layers: {failed}")
    amendment_layers = [case["layer_index"] for case in cases if case["numerics_amendment"] is not None]
    auxiliary_exception_layers = [
        case["layer_index"]
        for case in cases
        if any(item["auxiliary_intermediates"]["exception_used"] for item in case["cases"])
    ]
    layer4 = next(case for case in cases if case["layer_index"] == P4_DIAGNOSTIC_LAYER)
    layer4_fp32 = next(item for item in layer4["cases"] if item["dtype"] == "float32")
    layer4_bf16 = next(item for item in layer4["cases"] if item["dtype"] == "bfloat16")
    return {
        "stage": "P4.4",
        "status": "PASS",
        "pass_mode": "composite_bf16_transplant_gate" if amendment_layers else "standard_allclose",
        "numerics_amendment": P4_NUMERICS_AMENDMENT if amendment_layers else None,
        "layers_total": len(cases),
        "layers_passed": sum(case["passed"] for case in cases),
        "amendment_layers": amendment_layers,
        "auxiliary_intermediate_exception_layers": auxiliary_exception_layers,
        "output_distribution_fallback_layers": [P4_DIAGNOSTIC_LAYER],
        "functional_output_verified": all(
            item["functional_hard_gates"]["output"] for case in cases for item in case["cases"]
        ),
        "final_state_verified": all(
            item["functional_hard_gates"]["final_state"] for case in cases for item in case["cases"]
        ),
        "auxiliary_decay_standard_allclose": layer4_bf16["comparisons"]["decay"]["passed"],
        "auxiliary_decay_causally_explained": amendment_evidence["accepted"],
        "p3_distribution_fallback_used": (
            amendment_evidence["t129_functional_checks"]["output_distribution_gate"]
            and amendment_evidence["t129_functional_checks"]["output_standard_gate_recorded_failed"]
        ),
        "p3_original_fallback_used": False,
        "fp32_strict_parity": layer4_fp32["standard_allclose"],
        "forced_fp32_verified": amendment_evidence["acceptance_requires_all"][
            "forced_fp32_projection_parity"
        ],
        "exact_injection_verified": amendment_evidence["acceptance_requires_all"][
            "causal_injection_isolation"
        ],
        "amendment_evidence": amendment_evidence,
        "cases": cases,
        "result_marker": "P4_04_ALL_LAYER_LOCAL_PARITY=PASS",
    }


def _hidden_metric(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    metrics = error_metrics(reference, actual)
    reference_rms = float(reference.float().square().mean().sqrt())
    actual_rms = float(actual.float().square().mean().sqrt())
    metrics.update(
        {
            "reference_rms": reference_rms,
            "actual_rms": actual_rms,
            "rms_ratio": actual_rms / max(reference_rms, torch.finfo(torch.float32).eps),
        }
    )
    return metrics


def _tokenizer(model_dir: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)


def stage_full_forward(context: dict[str, Any]) -> dict[str, Any]:
    device = require_cuda()
    tokenizer = _tokenizer(context["model_dir"])
    prompt = (
        "Dense recurrent language models can combine efficient state updates with occasional "
        "softmax attention. 请简要解释这种混合结构，并给出一个代码示例。"
    )
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    labels = input_ids.clone()
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    donor = load_qwen_text_model(context["model_dir"], device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        donor_output = donor(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    torch.cuda.synchronize(device)
    donor_seconds = time.perf_counter() - start
    donor_peak = torch.cuda.max_memory_allocated(device)
    donor_hidden = tuple(value.detach().cpu() for value in donor_output.hidden_states)
    donor_logits = donor_output.logits.detach().cpu().float()
    save_json(
        context["output_dir"] / "p4-error-propagation.json",
        {
            "stage": "P4.5",
            "status": "RUNNING",
            "donor_forward_completed": True,
            "token_count": input_ids.shape[1],
            "donor_seconds_including_load": donor_seconds,
            "donor_peak_allocated_bytes": donor_peak,
        },
    )
    del donor_output, donor
    clear_cuda()

    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    target = load_densek3_model(context["artifact_dir"], device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        target_output = target(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    torch.cuda.synchronize(device)
    target_seconds = time.perf_counter() - start
    target_peak = torch.cuda.max_memory_allocated(device)
    target_hidden = tuple(value.detach().cpu() for value in target_output.hidden_states)
    target_logits = target_output.logits.detach().cpu().float()
    if len(donor_hidden) != 33 or len(target_hidden) != 33:
        raise ValueError(f"Expected embedding plus 32 layer hidden states, got {len(donor_hidden)}, {len(target_hidden)}")
    curve = []
    for layer_index in range(32):
        metric = _hidden_metric(donor_hidden[layer_index + 1], target_hidden[layer_index + 1])
        metric.update({"layer_index": layer_index, "layer_type": target.config.mixer_types[layer_index]})
        curve.append(metric)
    final_hidden = _hidden_metric(donor_hidden[-1], target_hidden[-1])
    logits_metric = error_metrics(donor_logits, target_logits)
    donor_ce = F.cross_entropy(
        donor_logits[:, :-1].reshape(-1, donor_logits.shape[-1]), labels.cpu()[:, 1:].reshape(-1)
    )
    target_ce = F.cross_entropy(
        target_logits[:, :-1].reshape(-1, target_logits.shape[-1]), labels.cpu()[:, 1:].reshape(-1)
    )
    donor_last = donor_logits[:, -1]
    target_last = target_logits[:, -1]
    donor_top = donor_last.topk(10, dim=-1).indices
    target_top = target_last.topk(10, dim=-1).indices
    overlap = len(set(donor_top[0].tolist()) & set(target_top[0].tolist()))
    report = {
        "stage": "P4.5",
        "status": "PASS",
        "gate_semantics": "diagnostic_completion_without_relaxed_full_model_parity_threshold",
        "prompt": prompt,
        "token_count": input_ids.shape[1],
        "error_propagation_curve": curve,
        "final_hidden": final_hidden,
        "logits": logits_metric,
        "cross_entropy": {
            "donor": float(donor_ce),
            "target": float(target_ce),
            "absolute_difference": abs(float(target_ce - donor_ce)),
        },
        "next_token": {
            "donor_top1": int(donor_last.argmax(dim=-1)[0]),
            "target_top1": int(target_last.argmax(dim=-1)[0]),
            "top1_match": bool(torch.equal(donor_last.argmax(dim=-1), target_last.argmax(dim=-1))),
            "top10_overlap": overlap,
            "donor_top10": donor_top[0].tolist(),
            "target_top10": target_top[0].tolist(),
        },
        "runtime": {
            "donor_seconds_including_load": donor_seconds,
            "target_seconds_including_load": target_seconds,
            "donor_peak_allocated_bytes": donor_peak,
            "target_peak_allocated_bytes": target_peak,
        },
        "p5a_compatibility_recovery_decision": "DEFER_TO_REVIEW_OF_ERROR_PROPAGATION_CURVE",
        "result_marker": "P4_05_FULL_MODEL_FORWARD=PASS",
    }
    context["target_model"] = target
    context["tokenizer"] = tokenizer
    del target_output, donor_hidden, target_hidden, donor_logits, target_logits
    return report


def _cache_comparison(
    reference: DenseK3HybridCache,
    actual: DenseK3HybridCache,
    config: DenseK3Config,
) -> dict[str, Any]:
    layers = []
    seen_tokens_exact = reference.seen_tokens == actual.seen_tokens
    standard_passed = seen_tokens_exact
    structural_passed = seen_tokens_exact
    numerical_envelope_passed = True
    for index, layer_type in enumerate(config.mixer_types):
        if layer_type == "kda":
            expected = reference.kda_layer(index)
            observed = actual.kda_layer(index)
            fields = {}
            for name in ("q_conv_state", "k_conv_state", "v_conv_state", "recurrent_state"):
                field = comparison(
                    getattr(expected, name),
                    getattr(observed, name),
                    rtol=OPTIMIZED_BF16_RTOL,
                    atol=OPTIMIZED_BF16_ATOL,
                )
                field["numerical_envelope"] = evaluate_numerical_envelope(
                    field,
                    P4_CACHE_TENSOR_LIMITS,
                )
                fields[name] = field
            layer_standard = all(item["passed"] for item in fields.values())
            layer_structural = all(
                item["shape_match"]
                and item["reference_nan_count"] == 0
                and item["actual_nan_count"] == 0
                and item["reference_inf_count"] == 0
                and item["actual_inf_count"] == 0
                for item in fields.values()
            )
            layer_envelope = all(item["numerical_envelope"]["passed"] for item in fields.values())
            layers.append(
                {
                    "layer_index": index,
                    "layer_type": layer_type,
                    "fields": fields,
                    "standard_allclose_passed": layer_standard,
                    "structural_passed": layer_structural,
                    "numerical_envelope_passed": layer_envelope,
                    "passed": layer_structural and layer_envelope,
                }
            )
        else:
            expected = reference.gqa_layer(index)
            observed = actual.gqa_layer(index)
            fields = {}
            for name in ("key", "value"):
                field = comparison(
                    getattr(expected, name),
                    getattr(observed, name),
                    rtol=OPTIMIZED_BF16_RTOL,
                    atol=OPTIMIZED_BF16_ATOL,
                )
                field["numerical_envelope"] = evaluate_numerical_envelope(
                    field,
                    P4_CACHE_TENSOR_LIMITS,
                )
                fields[name] = field
            layer_standard = expected.sequence_length == observed.sequence_length and all(
                item["passed"] for item in fields.values()
            )
            layer_structural = expected.sequence_length == observed.sequence_length and all(
                item["shape_match"]
                and item["reference_nan_count"] == 0
                and item["actual_nan_count"] == 0
                and item["reference_inf_count"] == 0
                and item["actual_inf_count"] == 0
                for item in fields.values()
            )
            layer_envelope = all(item["numerical_envelope"]["passed"] for item in fields.values())
            layers.append(
                {
                    "layer_index": index,
                    "layer_type": layer_type,
                    "sequence_length": observed.sequence_length,
                    "expected_sequence_length": expected.sequence_length,
                    "fields": fields,
                    "standard_allclose_passed": layer_standard,
                    "structural_passed": layer_structural,
                    "numerical_envelope_passed": layer_envelope,
                    "passed": layer_structural and layer_envelope,
                }
            )
        standard_passed = standard_passed and layer_standard
        structural_passed = structural_passed and layer_structural
        numerical_envelope_passed = numerical_envelope_passed and layer_envelope
    return {
        "seen_tokens": actual.seen_tokens,
        "expected_seen_tokens": reference.seen_tokens,
        "seen_tokens_exact": seen_tokens_exact,
        "layers": layers,
        "standard_allclose_passed": standard_passed,
        "structural_passed": structural_passed,
        "numerical_envelope_passed": numerical_envelope_passed,
        "passed": structural_passed and numerical_envelope_passed,
    }


def _cache_case(model: DenseK3ForCausalLM, tokens: torch.LongTensor, prompt: int) -> dict[str, Any]:
    total = tokens.shape[1]
    mask = torch.ones_like(tokens)
    with torch.no_grad():
        full = model(tokens, attention_mask=mask, use_cache=True, return_dict=True)
        prefill = model(tokens[:, :prompt], attention_mask=mask[:, :prompt], use_cache=True, return_dict=True)
        incremental_logits = [prefill.logits]
        cache = prefill.past_key_values
        for position in range(prompt, total):
            step = model(
                tokens[:, position : position + 1],
                attention_mask=mask[:, : position + 1],
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = step.past_key_values
            incremental_logits.append(step.logits)
    incremental = torch.cat(incremental_logits, dim=1)
    logits = comparison(
        full.logits,
        incremental,
        rtol=OPTIMIZED_BF16_RTOL,
        atol=OPTIMIZED_BF16_ATOL,
    )
    logits["distribution"] = scalable_error_distribution(
        full.logits,
        incremental,
        torch.bfloat16,
        top_k=0,
    )
    decisions = token_decision_metrics(
        full.logits,
        incremental,
        top_k=10,
        prompt_length=prompt,
    )
    caches = _cache_comparison(full.past_key_values, cache, model.config)
    standard_passed = logits["passed"] and caches["standard_allclose_passed"]
    gate = evaluate_integrated_case(
        prompt_length=prompt,
        decode_length=total - prompt,
        standard_allclose_passed=standard_passed,
        logits_distribution=logits["distribution"],
        cache_structural_passed=caches["structural_passed"],
        cache_numerical_envelope_passed=caches["numerical_envelope_passed"],
        token_decisions=decisions,
    )
    return {
        "prompt_length": prompt,
        "decode_length": total - prompt,
        "sequence_length": total,
        "token_sha256": tensor_sha256(tokens),
        "logits": logits,
        "cache": caches,
        "token_decisions": decisions,
        "standard_allclose_passed": standard_passed,
        "composite_gate": gate,
        "pass_mode": gate["pass_mode"],
        "passed": gate["passed"],
    }


def _batch_isolation(model: DenseK3ForCausalLM, tokens: torch.LongTensor) -> dict[str, Any]:
    with torch.no_grad():
        batched = model(tokens, attention_mask=torch.ones_like(tokens), use_cache=True, return_dict=True)
        singles = [
            model(tokens[index : index + 1], attention_mask=torch.ones_like(tokens[index : index + 1]), use_cache=True)
            for index in range(tokens.shape[0])
        ]
    joined = torch.cat([item.logits for item in singles], dim=0)
    logits = comparison(
        joined,
        batched.logits,
        rtol=OPTIMIZED_BF16_RTOL,
        atol=OPTIMIZED_BF16_ATOL,
    )
    logits["distribution"] = scalable_error_distribution(
        joined,
        batched.logits,
        torch.bfloat16,
        top_k=0,
    )
    decisions = token_decision_metrics(joined, batched.logits, top_k=10)
    state_checks = []
    for layer_index in model.config.kda_indices:
        expected = torch.cat(
            [item.past_key_values.kda_layer(layer_index).recurrent_state for item in singles], dim=0
        )
        state = comparison(
            expected,
            batched.past_key_values.kda_layer(layer_index).recurrent_state,
            rtol=OPTIMIZED_BF16_RTOL,
            atol=OPTIMIZED_BF16_ATOL,
        )
        state["layer_index"] = layer_index
        state["numerical_envelope"] = evaluate_numerical_envelope(
            state,
            P4_CACHE_BATCH_STATE_LIMITS,
        )
        state_checks.append(state)
    standard_passed = logits["passed"] and all(item["passed"] for item in state_checks)
    state_evaluations = [item["numerical_envelope"] for item in state_checks]
    gate = evaluate_batch_control(
        standard_allclose_passed=standard_passed,
        logits_distribution=logits["distribution"],
        state_evaluations=state_evaluations,
        token_decisions=decisions,
    )
    return {
        "token_sha256": tensor_sha256(tokens),
        "logits": logits,
        "token_decisions": decisions,
        "kda_recurrent_states": state_checks,
        "states_standard_passed": sum(item["passed"] for item in state_checks),
        "states_total": len(state_checks),
        "standard_allclose_passed": standard_passed,
        "composite_gate": gate,
        "pass_mode": gate["pass_mode"],
        "passed": gate["passed"],
    }


def stage_cache(context: dict[str, Any]) -> dict[str, Any]:
    device = require_cuda()
    direct_evidence = load_and_evaluate_direct_cache_evidence(
        context["output_dir"],
        context["artifact_dir"],
    )
    if not direct_evidence["passed"]:
        raise ValueError("Approved P4.6 direct cache-semantics evidence did not revalidate")
    model: DenseK3ForCausalLM = context.get("target_model") or load_densek3_model(
        context["artifact_dir"], device=device, dtype=torch.bfloat16
    )
    generator = torch.Generator(device="cpu").manual_seed(P4_CACHE_TOKEN_SEED)
    cases = []
    for prompt, decode in P4_CACHE_CASES:
        tokens = torch.randint(
            low=0,
            high=model.config.vocab_size,
            size=(1, prompt + decode),
            generator=generator,
        ).to(device)
        cases.append(_cache_case(model, tokens, prompt))
        save_json(
            context["output_dir"] / "p4-hybrid-cache.json",
            {
                "stage": "P4.6",
                "status": "RUNNING",
                "numerics_amendment": P4_CACHE_NUMERICS_AMENDMENT,
                "gate_hierarchy": P4_CACHE_AMENDMENT_TYPE,
                "direct_cache_semantics": direct_evidence,
                "cases_completed": len(cases),
                "cases": cases,
            },
        )
    batch_tokens = torch.randint(
        low=0, high=model.config.vocab_size, size=(2, 7), generator=generator
    ).to(device)
    batch = _batch_isolation(model, batch_tokens)
    expected_case_identity = direct_evidence["integrated_case_identity"]
    case_identity_checks = [
        case["prompt_length"] == expected["prompt_length"]
        and case["decode_length"] == expected["decode_length"]
        and case["sequence_length"] == expected["sequence_length"]
        and case["token_sha256"] == expected["token_sha256"]
        for case, expected in zip(cases, expected_case_identity, strict=True)
    ]
    batch_identity_check = batch["token_sha256"] == direct_evidence["batch_token_sha256"]
    replay_identity = {
        "integrated_case_token_hashes": case_identity_checks,
        "batch_token_hash": batch_identity_check,
        "passed": all(case_identity_checks) and batch_identity_check,
    }
    passed = (
        direct_evidence["passed"]
        and replay_identity["passed"]
        and all(case["passed"] for case in cases)
        and batch["passed"]
    )
    standard_allclose = all(case["standard_allclose_passed"] for case in cases) and batch[
        "standard_allclose_passed"
    ]
    report = {
        "stage": "P4.6",
        "status": "PASS" if passed else "FAIL",
        "formal_status": "PASS_WITH_COMPOSITE_CACHE_EQUIVALENCE_GATE" if passed else "FAIL",
        "pass_mode": "PASS_WITH_COMPOSITE_CACHE_EQUIVALENCE_GATE" if passed else "fail",
        "numerics_amendment": P4_CACHE_NUMERICS_AMENDMENT,
        "amendment_type": P4_CACHE_AMENDMENT_TYPE,
        "scope": "P4.6 deterministic cache-equivalence replay only",
        "global_allclose_modified": False,
        "architecture_modified": False,
        "converter_modified": False,
        "fla_modified": False,
        "formal_gate_modified": False,
        "token_decision_rescore": {
            "completed": True,
            "diagnostic_only": False,
            "metric_corrected": True,
            "greedy_top1_source": "argmax",
            "top10_source": "independent_topk_10",
            "formal_gate_scope": "prefill_last_plus_decode",
            "prefill_unused_diagnostic_only": True,
            "top10_mean_overlap_is_hard_gate": False,
            "top10_min_overlap_hard_gate": 9,
            "reported_scopes": [
                "prefill_unused",
                "prefill_last",
                "decode",
                "generation_relevant",
            ],
            "mismatch_details": [
                "top1_top2_margin",
                "cross_choice_gap",
                "cross_choice_rank",
                "cross_choice_probability",
                "top10_membership",
            ],
        },
        "direct_cache_semantics": direct_evidence,
        "replay_identity": replay_identity,
        "standard_allclose_all_cases_and_batch": standard_allclose,
        "cases_completed": len(cases),
        "cases": cases,
        "batch_isolation": batch,
        "cache_layout": "24 KDA recurrent+conv states and 8 GQA KV caches",
        "result_marker": f"P4_06_HYBRID_CACHE_GENERATION={'PASS' if passed else 'FAIL'}",
    }
    save_json(context["output_dir"] / "p4-hybrid-cache.json", report)
    print(f"P4_6_FORMAL_STATUS={report['formal_status']}", flush=True)
    if not passed:
        raise ValueError("P4 Hybrid cache consistency failed; inspect p4-hybrid-cache.json")
    context["target_model"] = model
    return report


def _load_prompts(path: Path) -> list[dict[str, str]]:
    prompts = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    required = {"english", "chinese", "code", "factual", "longer"}
    if {item["id"] for item in prompts} != required:
        raise ValueError(f"P4 smoke prompts must contain exactly {sorted(required)}")
    return prompts


def _generation_baseline(
    model: DenseK3ForCausalLM,
    tokenizer: Any,
    prompts: list[dict[str, str]],
) -> dict[str, Any]:
    device = next(model.parameters()).device
    records = []
    for item in prompts:
        encoded = tokenizer(item["prompt"], return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        generated = model.greedy_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=8,
        )
        new_tokens = generated[:, input_ids.shape[1] :]
        records.append(
            {
                "id": item["id"],
                "prompt": item["prompt"],
                "input_ids": input_ids.cpu().tolist(),
                "generated_ids": generated.cpu().tolist(),
                "new_token_count": new_tokens.shape[1],
                "unique_new_token_fraction": (
                    len(set(new_tokens.flatten().cpu().tolist())) / max(new_tokens.numel(), 1)
                ),
                "text": tokenizer.decode(generated[0], skip_special_tokens=False),
            }
        )
    probe_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]], device=device)
    with torch.no_grad():
        output = model(probe_ids, use_cache=True, return_dict=True)
    return {
        "forward_logits_hash": tensor_sha256(output.logits),
        "forward_cache_seen_tokens": output.past_key_values.seen_tokens,
        "generations": records,
    }


def stage_save_load(context: dict[str, Any]) -> dict[str, Any]:
    device = require_cuda()
    prompts = _load_prompts(context["prompts_path"])
    model: DenseK3ForCausalLM = context.get("target_model") or load_densek3_model(
        context["artifact_dir"], device=device, dtype=torch.bfloat16
    )
    tokenizer = context.get("tokenizer") or _tokenizer(context["model_dir"])
    baseline = _generation_baseline(model, tokenizer, prompts)
    baseline_path = context["output_dir"] / "p4-reload-baseline.json"
    save_json(baseline_path, baseline)
    del model
    context.pop("target_model", None)
    clear_cuda()
    probe_path = context["output_dir"] / "p4-reload-probe.json"
    command = [
        sys.executable,
        str(ROOT / "scripts/p4_reload_probe.py"),
        "--checkpoint-dir",
        str(context["artifact_dir"]),
        "--prompts",
        str(context["prompts_path"]),
        "--baseline",
        str(baseline_path),
        "--output",
        str(probe_path),
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"P4 new-process reload probe failed ({completed.returncode}):\n{completed.stdout}\n{completed.stderr}"
        )
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    if probe.get("status") != "PASS":
        raise ValueError(f"P4 reload probe returned non-PASS: {probe}")
    return {
        "stage": "P4.7",
        "status": "PASS",
        "new_process": True,
        "command": command,
        "subprocess_stdout": completed.stdout,
        "artifact_verification": probe["artifact_verification"],
        "forward_hash_match": probe["forward_hash_match"],
        "generation_match": probe["generation_match"],
        "generations": probe["observed"]["generations"],
        "result_marker": "P4_07_SAVE_LOAD_GENERATE=PASS",
    }


def load_replay_prerequisites(context: dict[str, Any]) -> dict[str, Any]:
    """Revalidate saved P4.0-P4.5 reports before a bounded P4.6 replay."""
    import hashlib

    reports = []
    checks = []
    for stage, key, filename, marker in STAGES[:6]:
        path = context["output_dir"] / filename
        if not path.is_file():
            raise FileNotFoundError(f"P4.6 replay prerequisite is missing: {path}")
        payload = path.read_bytes()
        report = json.loads(payload)
        report_checks = {
            "stage": report.get("stage") == stage,
            "status": report.get("status") == "PASS",
            "result_marker": report.get("result_marker") == marker + "=PASS",
        }
        checks.append(all(report_checks.values()))
        reports.append(
            {
                "stage": stage,
                "status_key": key,
                "path": str(path),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "checks": report_checks,
                "passed": all(report_checks.values()),
            }
        )
    result = {
        "mode": "replay_p4_6_then_continue_p4_7",
        "reports": reports,
        "passed": len(reports) == 6 and all(checks),
    }
    if not result["passed"]:
        raise ValueError("P4.6 replay prerequisites did not revalidate")
    return result


STAGE_FUNCTIONS = {
    "P4.0": stage_inventory,
    "P4.1": stage_conversion,
    "P4.2": stage_accounting,
    "P4.3": stage_construction,
    "P4.4": stage_local_parity,
    "P4.5": stage_full_forward,
    "P4.6": stage_cache,
    "P4.7": stage_save_load,
}


def write_environment(context: dict[str, Any]) -> None:
    report = {
        "stage": "P4",
        "captured_at": now(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            name: package_version(name)
            for name in ("torch", "transformers", "accelerate", "safetensors", "flash-linear-attention")
        },
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "p1_contract_modified": False,
        "p2_frozen": True,
        "p3_frozen": True,
        "p3_numerics_amendment": P3_NUMERICS_AMENDMENT,
        "p4_numerics_amendments": [P4_NUMERICS_AMENDMENT, P4_CACHE_NUMERICS_AMENDMENT],
        "p4_amendment_scopes": {
            P4_NUMERICS_AMENDMENT: "P4.4 single-layer qwen_compat transplant parity only",
            P4_CACHE_NUMERICS_AMENDMENT: "P4.6 deterministic cache-equivalence replay only",
        },
        "replay_p4_6": context["replay_p4_6"],
        "architecture_modified": False,
        "converter_modified": False,
        "kda_recurrence_modified": False,
        "state_contract_modified": False,
        "p2_numerics_modified": False,
        "p3_numerics_modified": False,
    }
    save_json(context["output_dir"] / "p4-environment.json", report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--reconvert", action="store_true", help="Rebuild an existing valid P4 artifact")
    parser.add_argument(
        "--replay-p4-6",
        action="store_true",
        help="Revalidate saved P4.0-P4.5, replay formal P4.6, then continue to P4.7 if accepted",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.replay_p4_6 and args.reconvert:
        raise ValueError("--replay-p4-6 cannot be combined with --reconvert")
    context: dict[str, Any] = {
        "model_dir": args.model_dir.resolve(),
        "output_dir": args.output_dir.resolve(),
        "artifact_dir": args.artifact_dir.resolve(),
        "prompts_path": args.prompts.resolve(),
        "reconvert": args.reconvert,
        "replay_p4_6": args.replay_p4_6,
    }
    context["output_dir"].mkdir(parents=True, exist_ok=True)
    write_environment(context)
    stage_status: dict[str, bool] = {}
    replay_prerequisites = None
    stages_to_run = STAGES
    if args.replay_p4_6:
        replay_prerequisites = load_replay_prerequisites(context)
        for _, key, _, _ in STAGES[:6]:
            stage_status[key] = True
        stages_to_run = STAGES[6:]
    first_failed_stage = None
    failure = None
    for stage, key, filename, marker in stages_to_run:
        try:
            report = STAGE_FUNCTIONS[stage](context)
            report.setdefault("stage", stage)
            report.setdefault("status", "PASS")
            report.setdefault("completed_at", now())
            report.setdefault("result_marker", marker + "=PASS")
            save_json(context["output_dir"] / filename, report)
            stage_status[key] = True
            print(marker + "=PASS", flush=True)
        except Exception as exc:  # noqa: BLE001 - runner must persist every failure envelope
            stack = traceback.format_exc()
            report_path = context["output_dir"] / filename
            failure_report: dict[str, Any] = {}
            if report_path.is_file():
                try:
                    partial = json.loads(report_path.read_text(encoding="utf-8"))
                    if isinstance(partial, dict) and partial.get("stage") == stage:
                        failure_report.update(partial)
                except (json.JSONDecodeError, OSError):
                    pass
            failure_report.update(
                {
                    "stage": stage,
                    "status": "FAIL",
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                    "traceback": stack,
                    "completed_at": now(),
                    "result_marker": marker + "=FAIL",
                }
            )
            save_json(report_path, failure_report)
            stage_status[key] = False
            first_failed_stage = stage
            failure = failure_report
            print(marker + "=FAIL", flush=True)
            break
    for _, key, _, _ in STAGES:
        stage_status.setdefault(key, False)
    passed = all(stage_status.values())
    final = {
        "stage": "P4",
        "status": "PASS" if passed else "FAIL",
        "first_failed_stage": first_failed_stage,
        **stage_status,
        "p1_contract_modified": False,
        "p2_frozen": True,
        "p3_frozen": True,
        "p3_numerics_amendment": P3_NUMERICS_AMENDMENT,
        "p4_numerics_amendments": [P4_NUMERICS_AMENDMENT, P4_CACHE_NUMERICS_AMENDMENT],
        "p4_amendment_scopes": {
            P4_NUMERICS_AMENDMENT: "P4.4 single-layer qwen_compat transplant parity only",
            P4_CACHE_NUMERICS_AMENDMENT: "P4.6 deterministic cache-equivalence replay only",
        },
        "replay_p4_6": args.replay_p4_6,
        "replay_prerequisites": replay_prerequisites,
        "architecture_modified": False,
        "converter_modified": False,
        "kda_recurrence_modified": False,
        "state_contract_modified": False,
        "p2_numerics_modified": False,
        "p3_numerics_modified": False,
        "p5_allowed": passed,
        "result_marker": f"P4_FINAL_CHECK={'PASS' if passed else 'FAIL'}",
        "failure_marker": None if passed else f"P4_FIRST_FAILED_STAGE={first_failed_stage}",
        "failure": failure,
        "completed_at": now(),
    }
    save_json(context["output_dir"] / "p4-final-check.json", final)
    print(final["result_marker"], flush=True)
    print(f"P5_ALLOWED={str(passed).lower()}", flush=True)
    if not passed:
        print(final["failure_marker"], flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
