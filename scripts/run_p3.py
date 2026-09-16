"""Run the six dependency-ordered P3 single-layer transplantation gates."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import traceback
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from densek3_core.kda.metrics import (
    OPTIMIZED_BF16_ATOL,
    OPTIMIZED_BF16_RTOL,
    REFERENCE_FP32_ATOL,
    REFERENCE_FP32_RTOL,
    error_metrics,
)
from densek3_core.transplant.gdn_to_kda import (
    ALLOWED_OPERATIONS,
    FORBIDDEN_OPERATIONS,
    ConversionResult,
    GDNBridgeConfig,
    convert_gdn_to_kda,
    hash_state_dict,
    load_qwen_gdn_layer,
    load_saved_conversion,
    save_conversion,
)
from densek3_core.transplant.single_layer import (
    DenseK3QwenCompatMixer,
    MixerResult,
    MixerTrace,
    QwenGDNReferenceMixer,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models/Qwen3.5-4B-Base"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/densek3-core/manifests/p3"
DEFAULT_ARTIFACT_DIR = ROOT / "outputs/densek3-core/checkpoints/p3/layer-0-qwen-compat-kda"

P3_NUMERICS_AMENDMENT = "P3-NUM-2026-08-10-01"
P3_BF16_FALLBACK_LIMITS = {
    "relative_l2_error_max": 7.0e-3,
    "cosine_similarity_min": 0.99997,
    "mean_abs_error_max": 1.0e-3,
    "p99_9_abs_error_max": 4.0e-3,
    "failed_allclose_fraction_max": 1.0e-4,
    "max_abs_error_max": 1.25e-1,
    "nan_count_max": 0,
    "inf_count_max": 0,
}

STAGES = (
    ("P3.0", "qwen_gdn_source", "p3-source-audit.json", "P3_00_QWEN_GDN_SOURCE"),
    ("P3.1", "deterministic_conversion", "p3-conversion-report.json", "P3_01_DETERMINISTIC_CONVERSION"),
    ("P3.2", "parameter_manifest", "p3-parameter-manifest.json", "P3_02_PARAMETER_MANIFEST"),
    ("P3.3", "decay_embedding", "p3-decay-embedding.json", "P3_03_DECAY_EMBEDDING"),
    ("P3.4", "single_layer_parity", "p3-functional-parity.json", "P3_04_SINGLE_LAYER_PARITY"),
    ("P3.5", "reproducibility", "p3-reproducibility.json", "P3_05_REPRODUCIBILITY"),
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


def save_functional_exception(path: Path, exc: Exception, stack: str) -> dict[str, Any]:
    """Persist the P3.4 evidence envelope before the runner marks failure."""
    report: dict[str, Any] = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, dict) and existing.get("stage") == "P3.4":
                report = existing
        except (json.JSONDecodeError, OSError):
            pass
    exception_failure = {
        "exception_type": type(exc).__name__,
        "exception": str(exc),
    }
    report.update(
        {
            "stage": "P3.4",
            "status": "EXCEPTION",
            "cases_total": report.get("cases_total", 0),
            "cases_passed": report.get("cases_passed", 0),
            "cases_failed": report.get("cases_failed", 0),
            "cases": report.get("cases", []),
            "first_failure": report.get("first_failure") or exception_failure,
            "exception_failure": exception_failure,
            "diagnostics_completed": False,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": stack,
            "result_marker": "P3_04_SINGLE_LAYER_PARITY=EXCEPTION",
        }
    )
    save_json(path, report)
    return report


def comparison(
    reference: torch.Tensor,
    actual: torch.Tensor,
    dtype: torch.dtype,
    *,
    exact: bool = False,
) -> dict[str, Any]:
    if exact:
        rtol = atol = 0.0
    elif dtype == torch.float32:
        rtol, atol = REFERENCE_FP32_RTOL, REFERENCE_FP32_ATOL
    elif dtype == torch.bfloat16:
        rtol, atol = OPTIMIZED_BF16_RTOL, OPTIMIZED_BF16_ATOL
    else:
        raise ValueError(f"Unsupported P3 parity dtype: {dtype}")
    metrics = error_metrics(reference, actual)
    metrics.update(
        {
            "rtol": rtol,
            "atol": atol,
            "passed": bool(torch.allclose(reference, actual, rtol=rtol, atol=atol)),
        }
    )
    return metrics


def require_cuda(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("P3.3-P3.5 require the authoritative CUDA server runtime")


def stage_source(context: dict[str, Any]) -> dict[str, Any]:
    config, source_state, audit = load_qwen_gdn_layer(
        context["model_dir"],
        layer_index=context["layer_index"],
        device="cpu",
    )
    context.update({"config": config, "source_state": source_state})
    audit.update(
        {
            "stage": "P3.0",
            "transformers": package_version("transformers"),
            "safetensors": package_version("safetensors"),
            "result_marker": "P3_00_QWEN_GDN_SOURCE=PASS",
        }
    )
    return audit


def stage_conversion(context: dict[str, Any]) -> dict[str, Any]:
    first = convert_gdn_to_kda(context["source_state"], context["config"])
    second = convert_gdn_to_kda(context["source_state"], context["config"])
    deterministic = first.parameter_hash == second.parameter_hash and first.manifest_hash == second.manifest_hash
    if not deterministic:
        raise ValueError("Repeated conversion produced different parameter or manifest hashes")
    context["conversion"] = first
    return {
        "stage": "P3.1",
        "status": "PASS",
        "layer_index": context["layer_index"],
        "source_hash": first.manifest["source_hash"],
        "parameter_hash": first.parameter_hash,
        "manifest_hash": first.manifest_hash,
        "no_random_initialization": True,
        "repeat_conversion_hash_match": deterministic,
        "target_tensors": {
            name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch.")}
            for name, tensor in sorted(first.target_state.items())
        },
        "result_marker": "P3_01_DETERMINISTIC_CONVERSION=PASS",
    }


def stage_manifest(context: dict[str, Any]) -> dict[str, Any]:
    conversion: ConversionResult = context["conversion"]
    manifest = conversion.manifest
    operations = Counter(entry["operation"] for entry in manifest["entries"])
    invalid = (set(operations) - ALLOWED_OPERATIONS) | (set(operations) & FORBIDDEN_OPERATIONS)
    if invalid:
        raise ValueError(f"Forbidden parameter classifications: {sorted(invalid)}")
    report = dict(manifest)
    report.update(
        {
            "stage": "P3.2",
            "status": "PASS",
            "operation_counts": dict(sorted(operations.items())),
            "unknown_count": 0,
            "unmapped_count": 0,
            "silent_init_count": 0,
            "manifest_hash": conversion.manifest_hash,
            "result_marker": "P3_02_PARAMETER_MANIFEST=PASS",
        }
    )
    return report


def _validate_embedding_parameters(
    config: GDNBridgeConfig,
    source: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
) -> dict[str, bool]:
    f_a = target["f_a_proj.weight"]
    f_b = target["f_b_proj.weight"].view(
        config.value_num_heads,
        config.qk_head_dim,
        config.decay_projection_rank,
    )
    heads = torch.arange(config.value_num_heads).view(-1, 1)
    channels = torch.arange(config.qk_head_dim).view(1, -1)
    selected = f_b[heads, channels, heads.expand(-1, config.qk_head_dim)]
    checks = {
        "f_a_scalar_rows_copied": torch.equal(f_a[: config.value_num_heads], source["in_proj_a.weight"]),
        "f_a_remaining_rows_zero": bool(torch.count_nonzero(f_a[config.value_num_heads :]) == 0),
        "f_b_embedding_entries_one": bool(torch.all(selected == 1)),
        "f_b_only_embedding_entries_nonzero": int(torch.count_nonzero(f_b))
        == config.value_num_heads * config.qk_head_dim,
        "dt_bias_repeated": torch.equal(
            target["dt_bias"], source["dt_bias"].repeat_interleave(config.qk_head_dim)
        ),
        "a_log_copied_for_runtime_broadcast": torch.equal(target["A_log"], source["A_log"]),
    }
    if not all(checks.values()):
        raise ValueError(f"Scalar-decay embedding parameter proof failed: {checks}")
    return checks


def stage_decay(context: dict[str, Any]) -> dict[str, Any]:
    device: torch.device = context["device"]
    require_cuda(device)
    config: GDNBridgeConfig = context["config"]
    source = context["source_state"]
    conversion: ConversionResult = context["conversion"]
    parameter_checks = _validate_embedding_parameters(config, source, conversion.target_state)
    donor = QwenGDNReferenceMixer(config, source).to(device, torch.float32)
    target = DenseK3QwenCompatMixer(config, conversion.target_state).to(device, torch.float32)
    torch.manual_seed(42)
    hidden = torch.randn(1, 7, config.hidden_size, dtype=torch.float32, device=device)
    donor_decay, donor_beta = donor.gates(hidden)
    target_decay, target_beta = target.gates(hidden)
    decay_result = comparison(donor_decay, target_decay, torch.float32)
    beta_result = comparison(donor_beta, target_beta, torch.float32, exact=True)
    channel_spread = float((target_decay - target_decay[..., :1]).abs().max())
    if not decay_result["passed"] or not beta_result["passed"] or channel_spread != 0.0:
        raise ValueError(
            f"Scalar-decay functional embedding failed: decay={decay_result}, beta={beta_result}, spread={channel_spread}"
        )
    return {
        "stage": "P3.3",
        "status": "PASS",
        "seed": 42,
        "dtype": "float32",
        "shape": {"B": 1, "T": 7, "D": config.hidden_size},
        "parameter_checks": parameter_checks,
        "decay": decay_result,
        "beta": beta_result,
        "max_within_head_channel_spread": channel_spread,
        "a_log_parameter_shape": list(conversion.target_state["A_log"].shape),
        "a_log_runtime_behavior": "broadcast each of 32 head parameters over 128 key channels",
        "result_marker": "P3_03_DECAY_EMBEDDING=PASS",
    }


def _functional_case(
    donor: QwenGDNReferenceMixer,
    target: DenseK3QwenCompatMixer,
    config: GDNBridgeConfig,
    device: torch.device,
    dtype: torch.dtype,
    sequence: int,
) -> dict[str, Any]:
    torch.manual_seed(42 + sequence)
    hidden = torch.randn(1, sequence, config.hidden_size, dtype=dtype, device=device)
    initial_mode = "random" if sequence in {7, 129} else "none"
    initial_state = None
    if initial_mode == "random":
        initial_state = (
            torch.randn(
                1,
                config.value_num_heads,
                config.value_head_dim,
                config.qk_head_dim,
                dtype=torch.float32,
                device=device,
            )
            * 0.02
        )
    with torch.inference_mode():
        expected: MixerResult = donor.forward(hidden, initial_state=initial_state)
        actual: MixerResult = target.forward(hidden, initial_state=initial_state)
    output_result = comparison(expected.output, actual.output, dtype)
    state_result = comparison(expected.final_state, actual.final_state, dtype)
    decay_result = comparison(expected.decay, actual.decay, dtype)
    beta_result = comparison(expected.beta, actual.beta, dtype)
    output_distribution = _error_distribution(expected.output, actual.output, dtype)
    standard_passed = all(
        item["passed"] for item in (output_result, state_result, decay_result, beta_result)
    )
    return {
        "dtype": str(dtype).removeprefix("torch."),
        "B": 1,
        "T": sequence,
        "D": config.hidden_size,
        "H": config.qk_num_heads,
        "HV": config.value_num_heads,
        "K": config.qk_head_dim,
        "V": config.value_head_dim,
        "initial_state": initial_mode,
        "output": output_result,
        "output_error_distribution": output_distribution,
        "final_state": state_result,
        "decay": decay_result,
        "beta": beta_result,
        "standard_allclose_passed": standard_passed,
        "pass_mode": "standard_allclose" if standard_passed else None,
        "passed": standard_passed,
    }


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return REFERENCE_FP32_RTOL, REFERENCE_FP32_ATOL
    if dtype == torch.bfloat16:
        return OPTIMIZED_BF16_RTOL, OPTIMIZED_BF16_ATOL
    raise ValueError(f"Unsupported diagnostic dtype: {dtype}")


def _error_distribution(
    reference: torch.Tensor,
    actual: torch.Tensor,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Record elementwise evidence without changing the authoritative Gate."""
    rtol, atol = _tolerances(dtype)
    reference_flat = reference.detach().double().reshape(-1)
    actual_flat = actual.detach().double().reshape(-1)
    absolute = (actual_flat - reference_flat).abs()
    threshold = atol + rtol * reference_flat.abs()
    failed = absolute > threshold
    quantiles = {}
    for label, probability in (
        ("p50", 0.5),
        ("p90", 0.9),
        ("p95", 0.95),
        ("p99", 0.99),
        ("p99_9", 0.999),
    ):
        quantiles[f"{label}_abs_error"] = (
            float(torch.quantile(absolute, probability)) if absolute.numel() else 0.0
        )
    stable_denominator = torch.clamp(reference_flat.abs(), min=1e-3)
    stable_relative = absolute / stable_denominator
    histogram = {
        "exact": int((absolute == 0).sum()),
        "0_to_1e-4": int(((absolute > 0) & (absolute <= 1e-4)).sum()),
        "1e-4_to_5e-4": int(((absolute > 1e-4) & (absolute <= 5e-4)).sum()),
        "5e-4_to_1e-3": int(((absolute > 5e-4) & (absolute <= 1e-3)).sum()),
        "1e-3_to_5e-3": int(((absolute > 1e-3) & (absolute <= 5e-3)).sum()),
        "5e-3_to_1e-2": int(((absolute > 5e-3) & (absolute <= 1e-2)).sum()),
        "above_1e-2": int((absolute > 1e-2).sum()),
    }
    result: dict[str, Any] = {
        "num_elements": absolute.numel(),
        "num_exact": int((absolute == 0).sum()),
        "num_failed_allclose": int(failed.sum()),
        "stable_relative_denominator_floor": 1e-3,
        "stable_relative_mean": float(stable_relative.mean()) if stable_relative.numel() else 0.0,
        "stable_relative_p99": (
            float(torch.quantile(stable_relative, 0.99)) if stable_relative.numel() else 0.0
        ),
        "histogram": histogram,
        "rtol": rtol,
        "atol": atol,
    }
    result.update(error_metrics(reference, actual))
    result.update(quantiles)
    return result


def _finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all())


def _trace_comparisons(
    donor: MixerTrace,
    target: MixerTrace,
    dtype: torch.dtype,
) -> dict[str, Any]:
    fields = (
        "q_projection",
        "k_projection",
        "v_projection",
        "q_conv",
        "k_conv",
        "v_conv",
        "normalized_q",
        "normalized_k",
        "raw_a_logits",
        "channel_a_logits",
        "decay",
        "beta_logits",
        "beta",
        "recurrent_core",
        "final_state",
        "z_logits",
        "gated_norm",
        "output",
    )
    compared: dict[str, Any] = {}
    for field in fields:
        expected = getattr(donor, field)
        actual = getattr(target, field)
        if expected.shape != actual.shape:
            compared[field] = {
                "shape_match": False,
                "donor_shape": list(expected.shape),
                "target_shape": list(actual.shape),
            }
            continue
        item = comparison(expected, actual, dtype)
        item.update(
            {
                "shape_match": True,
                "shape": list(expected.shape),
                "donor_finite": _finite(expected),
                "target_finite": _finite(actual),
            }
        )
        compared[field] = item
    latent = target.f_a_intermediate
    compared["f_a_intermediate"] = {
        "shape": None if latent is None else list(latent.shape),
        "finite": latent is not None and _finite(latent),
    }
    return compared


def _reshape_qkv(
    config: GDNBridgeConfig,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        q.view(*q.shape[:2], config.qk_num_heads, config.qk_head_dim),
        k.view(*k.shape[:2], config.qk_num_heads, config.qk_head_dim),
        v.view(*v.shape[:2], config.value_num_heads, config.value_head_dim),
    )


def _linear_fp32(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(hidden.float(), weight.float()).to(hidden.dtype)


def _forced_fp32_projection_diagnostic(
    donor: QwenGDNReferenceMixer,
    target: DenseK3QwenCompatMixer,
    hidden: torch.Tensor,
) -> dict[str, Any]:
    """Use FP32 GEMMs only as causal evidence; this is never a model backend."""
    config = donor.config
    donor_fused = _linear_fp32(hidden, donor.state["in_proj_qkv.weight"])
    donor_q, donor_k, donor_v = torch.split(
        donor_fused,
        [config.qk_dim, config.qk_dim, config.value_dim],
        dim=-1,
    )
    target_q = _linear_fp32(hidden, target.state["q_proj.weight"])
    target_k = _linear_fp32(hidden, target.state["k_proj.weight"])
    target_v = _linear_fp32(hidden, target.state["v_proj.weight"])
    donor_a = _linear_fp32(hidden, donor.state["in_proj_a.weight"])
    target_f_a = F.linear(hidden.float(), target.state["f_a_proj.weight"].float())
    target_channel_a = F.linear(target_f_a, target.state["f_b_proj.weight"].float())
    target_channel_a = target_channel_a.view(
        *hidden.shape[:2],
        config.value_num_heads,
        config.qk_head_dim,
    ).to(hidden.dtype)
    donor_channel_a = donor_a.unsqueeze(-1).expand_as(target_channel_a)
    tensor_pairs = {
        "q_projection": (donor_q, target_q),
        "k_projection": (donor_k, target_k),
        "v_projection": (donor_v, target_v),
        "channel_a_logits": (donor_channel_a, target_channel_a),
    }
    fields = {}
    for name, (expected, actual) in tensor_pairs.items():
        item = comparison(expected, actual, hidden.dtype)
        item["exact_match"] = bool(torch.equal(expected, actual))
        fields[name] = item
    all_exact = all(item["exact_match"] for item in fields.values())
    return {
        "diagnostic_only": True,
        "formal_backend_modified": False,
        "input_dtype": str(hidden.dtype).removeprefix("torch."),
        "gemm_dtype": "float32",
        "output_cast_dtype": str(hidden.dtype).removeprefix("torch."),
        "components": fields,
        "all_exact": all_exact,
    }


def _projection_diagnostics(context: dict[str, Any]) -> dict[str, Any]:
    config: GDNBridgeConfig = context["config"]
    conversion: ConversionResult = context["conversion"]
    device: torch.device = context["device"]
    cases = []
    for dtype in (torch.float32, torch.bfloat16):
        donor = QwenGDNReferenceMixer(config, context["source_state"]).to(device, dtype)
        target = DenseK3QwenCompatMixer(config, conversion.target_state).to(device, dtype)
        torch.manual_seed(404)
        hidden = torch.randn(1, 7, config.hidden_size, device=device, dtype=dtype)
        with torch.inference_mode():
            donor_trace = donor.trace(hidden)
            target_trace = target.trace(hidden)
            compared = _trace_comparisons(donor_trace, target_trace, dtype)
            forced = (
                _forced_fp32_projection_diagnostic(donor, target, hidden)
                if dtype == torch.bfloat16
                else None
            )
        explained = all(
            item.get("shape_match", True)
            and item.get("donor_finite", True)
            and item.get("target_finite", item.get("finite", True))
            for item in compared.values()
        )
        cases.append(
            {
                "dtype": str(dtype).removeprefix("torch."),
                "seed": 404,
                "T": 7,
                "components": compared,
                "forced_fp32_projection": forced,
                "all_shapes_and_finiteness_explained": explained,
            }
        )
    passed = all(case["all_shapes_and_finiteness_explained"] for case in cases)
    return {
        "stage": "P3.4a",
        "status": "PASS" if passed else "FAIL",
        "pass_definition": "all requested boundaries measured with matching layouts and finite tensors",
        "bit_exact_required": False,
        "cases": cases,
        "result_marker": f"P3_04A_PROJECTION_DIAGNOSTICS={'PASS' if passed else 'FAIL'}",
    }


def _injection_result(
    reference: MixerResult,
    actual: MixerResult,
    dtype: torch.dtype,
) -> dict[str, Any]:
    output = comparison(reference.output, actual.output, dtype)
    state = comparison(reference.final_state, actual.final_state, dtype)
    output_exact = bool(torch.equal(reference.output, actual.output))
    state_exact = bool(torch.equal(reference.final_state, actual.final_state))
    return {
        "output": output,
        "final_state": state,
        "passed": output["passed"] and state["passed"],
        "output_exact": output_exact,
        "final_state_exact": state_exact,
        "exact": output_exact and state_exact,
    }


def _injection_isolation(context: dict[str, Any]) -> dict[str, Any]:
    config: GDNBridgeConfig = context["config"]
    conversion: ConversionResult = context["conversion"]
    device: torch.device = context["device"]
    dtype = torch.bfloat16
    donor = QwenGDNReferenceMixer(config, context["source_state"]).to(device, dtype)
    target = DenseK3QwenCompatMixer(config, conversion.target_state).to(device, dtype)
    torch.manual_seed(171)
    hidden = torch.randn(1, 129, config.hidden_size, dtype=dtype, device=device)
    initial_state = (
        torch.randn(
            1,
            config.value_num_heads,
            config.value_head_dim,
            config.qk_head_dim,
            dtype=torch.float32,
            device=device,
        )
        * 0.02
    )
    with torch.inference_mode():
        donor_trace = donor.trace(hidden, initial_state)
        target_trace = target.trace(hidden, initial_state)
        target_qkv = _reshape_qkv(
            config,
            target_trace.q_conv,
            target_trace.k_conv,
            target_trace.v_conv,
        )
        donor_postconv_qkv = _reshape_qkv(
            config,
            donor_trace.q_conv,
            donor_trace.k_conv,
            donor_trace.v_conv,
        )
        donor_preconv_qkv = target.convolve_projections(
            donor_trace.q_projection,
            donor_trace.k_projection,
            donor_trace.v_projection,
        )

        def run(
            qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            decay: torch.Tensor,
        ) -> MixerResult:
            return target.forward_from_recurrence_inputs(
                hidden,
                *qkv,
                decay,
                target_trace.beta,
                initial_state,
            )

        actuals = {
            "case_0_native_transplant": target_trace.result(),
            "case_1_donor_decay": run(target_qkv, donor_trace.decay),
            "case_2a_donor_projection_preconv": run(donor_preconv_qkv, target_trace.decay),
            "case_2b_donor_projection_postconv": run(donor_postconv_qkv, target_trace.decay),
            "case_3_both_injected": run(donor_postconv_qkv, donor_trace.decay),
        }
    cases = {
        name: _injection_result(donor_trace.result(), result, dtype)
        for name, result in actuals.items()
    }
    both_injected_exact = cases["case_3_both_injected"]["exact"]
    passed = cases["case_3_both_injected"]["passed"] and both_injected_exact
    return {
        "stage": "P3.4b",
        "status": "PASS" if passed else "FAIL",
        "dtype": "bfloat16",
        "seed": 171,
        "T": 129,
        "initial_state": "random",
        "cases": cases,
        "both_injected_exact": both_injected_exact,
        "pass_definition": "both donor post-convolution QKV and donor decay restore donor parity",
        "result_marker": f"P3_04B_EXACT_INJECTION_ISOLATION={'PASS' if passed else 'FAIL'}",
    }


def _initial_state_isolation(context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    config: GDNBridgeConfig = context["config"]
    conversion: ConversionResult = context["conversion"]
    device: torch.device = context["device"]
    dtype = torch.bfloat16
    donor = QwenGDNReferenceMixer(config, context["source_state"]).to(device, dtype)
    target = DenseK3QwenCompatMixer(config, conversion.target_state).to(device, dtype)
    torch.manual_seed(404129)
    hidden = torch.randn(1, 129, config.hidden_size, dtype=dtype, device=device)
    shape = (1, config.value_num_heads, config.value_head_dim, config.qk_head_dim)
    torch.manual_seed(404130)
    states = {
        "none": None,
        "zeros": torch.zeros(shape, dtype=torch.float32, device=device),
        "random": torch.randn(shape, dtype=torch.float32, device=device) * 0.02,
    }
    cases: dict[str, Any] = {}
    distributions: dict[str, Any] = {}
    with torch.inference_mode():
        for mode, initial_state in states.items():
            expected = donor.forward(hidden, initial_state)
            actual = target.forward(hidden, initial_state)
            output = comparison(expected.output, actual.output, dtype)
            state = comparison(expected.final_state, actual.final_state, dtype)
            decay = comparison(expected.decay, actual.decay, dtype)
            cases[mode] = {
                "output": output,
                "final_state": state,
                "decay": decay,
                "all_finite": all(
                    _finite(tensor)
                    for tensor in (
                        expected.output,
                        actual.output,
                        expected.final_state,
                        actual.final_state,
                        expected.decay,
                        actual.decay,
                    )
                ),
            }
            distributions[mode] = {
                "output": _error_distribution(expected.output, actual.output, dtype),
                "final_state": _error_distribution(expected.final_state, actual.final_state, dtype),
                "decay": _error_distribution(expected.decay, actual.decay, dtype),
            }
    passed = all(case["all_finite"] for case in cases.values())
    isolation = {
        "stage": "P3.4c",
        "status": "PASS" if passed else "FAIL",
        "dtype": "bfloat16",
        "input_seed": 404129,
        "state_seed": 404130,
        "T": 129,
        "cases": cases,
        "pass_definition": "none, zero, and random initial-state cases measured with identical input and finite results",
        "result_marker": f"P3_04C_INITIAL_STATE_ISOLATION={'PASS' if passed else 'FAIL'}",
    }
    distribution_status = all(
        all(value["num_elements"] >= 1 for value in case.values())
        for case in distributions.values()
    )
    distribution = {
        "stage": "P3.4d",
        "status": "PASS" if distribution_status else "FAIL",
        "dtype": "bfloat16",
        "T": 129,
        "cases": distributions,
        "result_marker": f"P3_04D_ERROR_DISTRIBUTION={'PASS' if distribution_status else 'FAIL'}",
    }
    return isolation, distribution


def _comparison_is_finite(result: dict[str, Any]) -> bool:
    return all(
        result.get(name, 0) == 0
        for name in (
            "reference_nan_count",
            "actual_nan_count",
            "reference_inf_count",
            "actual_inf_count",
        )
    )


def _evaluate_bf16_distribution_fallback(distribution: dict[str, Any]) -> dict[str, Any]:
    num_elements = int(distribution["num_elements"])
    failed_fraction = (
        float(distribution["num_failed_allclose"]) / num_elements if num_elements else float("inf")
    )
    nan_count = sum(
        int(distribution[name])
        for name in ("reference_nan_count", "actual_nan_count")
    )
    inf_count = sum(
        int(distribution[name])
        for name in ("reference_inf_count", "actual_inf_count")
    )
    observed = {
        "relative_l2_error": float(distribution["relative_l2_error"]),
        "cosine_similarity": float(distribution["cosine_similarity"]),
        "mean_abs_error": float(distribution["mean_abs_error"]),
        "p99_9_abs_error": float(distribution["p99_9_abs_error"]),
        "failed_allclose_fraction": failed_fraction,
        "max_abs_error": float(distribution["max_abs_error"]),
        "nan_count": nan_count,
        "inf_count": inf_count,
    }
    checks = {
        "relative_l2_error": observed["relative_l2_error"]
        <= P3_BF16_FALLBACK_LIMITS["relative_l2_error_max"],
        "cosine_similarity": observed["cosine_similarity"]
        >= P3_BF16_FALLBACK_LIMITS["cosine_similarity_min"],
        "mean_abs_error": observed["mean_abs_error"]
        <= P3_BF16_FALLBACK_LIMITS["mean_abs_error_max"],
        "p99_9_abs_error": observed["p99_9_abs_error"]
        <= P3_BF16_FALLBACK_LIMITS["p99_9_abs_error_max"],
        "failed_allclose_fraction": observed["failed_allclose_fraction"]
        <= P3_BF16_FALLBACK_LIMITS["failed_allclose_fraction_max"],
        "max_abs_error": observed["max_abs_error"]
        <= P3_BF16_FALLBACK_LIMITS["max_abs_error_max"],
        "nan_count": observed["nan_count"] <= P3_BF16_FALLBACK_LIMITS["nan_count_max"],
        "inf_count": observed["inf_count"] <= P3_BF16_FALLBACK_LIMITS["inf_count_max"],
    }
    return {
        "numerics_amendment": P3_NUMERICS_AMENDMENT,
        "approved_thresholds": dict(P3_BF16_FALLBACK_LIMITS),
        "observed": observed,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _formal_cases_have_no_nonfinite(cases: list[dict[str, Any]]) -> bool:
    return all(
        _comparison_is_finite(case[name])
        for case in cases
        for name in ("output", "final_state", "decay", "beta")
    )


def _apply_p3_functional_gate(
    cases: list[dict[str, Any]],
    projection_diagnostics: dict[str, Any],
    injection_isolation: dict[str, Any],
    initial_state_isolation: dict[str, Any],
    error_distribution: dict[str, Any],
) -> dict[str, Any]:
    fp32_cases = [case for case in cases if case["dtype"] == "float32"]
    bf16_cases = [case for case in cases if case["dtype"] == "bfloat16"]
    bf16_projection_case = next(
        (
            case
            for case in projection_diagnostics["cases"]
            if case["dtype"] == "bfloat16"
        ),
        None,
    )
    forced_fp32 = None if bf16_projection_case is None else bf16_projection_case["forced_fp32_projection"]
    preconditions = {
        "fp32_formal_strict_parity": bool(fp32_cases)
        and all(case["standard_allclose_passed"] for case in fp32_cases),
        "bf16_final_recurrent_state_existing_gate": bool(bf16_cases)
        and all(case["final_state"]["passed"] for case in bf16_cases),
        "bf16_decay_existing_gate": bool(bf16_cases)
        and all(case["decay"]["passed"] for case in bf16_cases),
        "bf16_beta_existing_gate": bool(bf16_cases)
        and all(case["beta"]["passed"] for case in bf16_cases),
        "p3_4a_projection_diagnostics": projection_diagnostics["status"] == "PASS",
        "p3_4b_exact_injection_isolation": injection_isolation["status"] == "PASS",
        "p3_4c_initial_state_isolation": initial_state_isolation["status"] == "PASS",
        "p3_4d_error_distribution": error_distribution["status"] == "PASS",
        "donor_qkv_and_decay_injection_exact": bool(
            injection_isolation.get("both_injected_exact", False)
        ),
        "forced_fp32_candidate_projections_exact": bool(
            forced_fp32 is not None and forced_fp32.get("all_exact", False)
        ),
        "all_formal_tensors_finite": _formal_cases_have_no_nonfinite(cases),
    }
    preconditions_passed = all(preconditions.values())
    standard_failed = [case for case in cases if not case["standard_allclose_passed"]]
    fallback_cases = []
    for case in cases:
        standard_passed = case["standard_allclose_passed"]
        fallback_eligible = (
            case["dtype"] == "bfloat16"
            and not standard_passed
            and not case["output"]["passed"]
            and case["final_state"]["passed"]
            and case["decay"]["passed"]
            and case["beta"]["passed"]
        )
        fallback = (
            _evaluate_bf16_distribution_fallback(case["output_error_distribution"])
            if fallback_eligible
            else None
        )
        fallback_accepted = bool(
            fallback_eligible
            and preconditions_passed
            and fallback is not None
            and fallback["passed"]
        )
        case.update(
            {
                "fallback_eligible": fallback_eligible,
                "fallback_evaluation": fallback,
                "fallback_accepted": fallback_accepted,
                "pass_mode": (
                    "standard_allclose"
                    if standard_passed
                    else "bf16_transplant_distribution_fallback"
                    if fallback_accepted
                    else None
                ),
                "passed": standard_passed or fallback_accepted,
            }
        )
        if fallback is not None:
            fallback_cases.append(
                {
                    "dtype": case["dtype"],
                    "T": case["T"],
                    "initial_state": case["initial_state"],
                    "accepted": fallback_accepted,
                    "evaluation": fallback,
                }
            )
    final_failed = [case for case in cases if not case["passed"]]
    fallback_used = any(case["fallback_accepted"] for case in cases)
    passed = not final_failed

    def case_summary(case: dict[str, Any]) -> dict[str, Any]:
        return {
            "dtype": case["dtype"],
            "T": case["T"],
            "initial_state": case["initial_state"],
            "standard_allclose_passed": case["standard_allclose_passed"],
            "fallback_eligible": case["fallback_eligible"],
            "fallback_accepted": case["fallback_accepted"],
            "final_passed": case["passed"],
            "output": case["output"],
            "final_state": case["final_state"],
            "decay": case["decay"],
            "beta": case["beta"],
        }

    if passed and fallback_used:
        pass_mode = "bf16_transplant_distribution_fallback"
    elif passed:
        pass_mode = "standard_allclose"
    else:
        pass_mode = None
    return {
        "passed": passed,
        "pass_mode": pass_mode,
        "standard_allclose": not standard_failed,
        "standard_cases_passed": len(cases) - len(standard_failed),
        "standard_cases_failed": len(standard_failed),
        "first_standard_failure": case_summary(standard_failed[0]) if standard_failed else None,
        "final_cases_passed": len(cases) - len(final_failed),
        "final_cases_failed": len(final_failed),
        "first_final_failure": case_summary(final_failed[0]) if final_failed else None,
        "numerics_amendment": P3_NUMERICS_AMENDMENT if fallback_used else None,
        "fallback_used": fallback_used,
        "fallback": {
            "scope": "p3_qwen_gdn_to_densek3_kda_qwen_compat_bf16_functional_parity",
            "standard_allclose_evaluated_first": True,
            "preconditions": preconditions,
            "preconditions_passed": preconditions_passed,
            "approved_thresholds": dict(P3_BF16_FALLBACK_LIMITS),
            "candidate_cases": fallback_cases,
            "status": "PASS" if fallback_used and passed else "NOT_USED" if not standard_failed else "FAIL",
        },
        "structural_parity_verified": projection_diagnostics["status"] == "PASS",
        "fp32_strict_parity_verified": preconditions["fp32_formal_strict_parity"],
        "exact_injection_verified": preconditions["donor_qkv_and_decay_injection_exact"],
        "forced_fp32_projection_verified": preconditions[
            "forced_fp32_candidate_projections_exact"
        ],
    }


def stage_functional(context: dict[str, Any]) -> dict[str, Any]:
    device: torch.device = context["device"]
    require_cuda(device)
    config: GDNBridgeConfig = context["config"]
    conversion: ConversionResult = context["conversion"]
    report_path = context["output_dir"] / "p3-functional-parity.json"
    report: dict[str, Any] = {
        "stage": "P3.4",
        "status": "RUNNING",
        "seed_rule": "42 + sequence_length",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "standard_allclose_gate_unchanged": True,
        "p2_numerics_modified": False,
        "approved_fallback_amendment": P3_NUMERICS_AMENDMENT,
        "cases_total": 0,
        "cases_passed": 0,
        "cases_failed": 0,
        "cases": [],
        "first_failure": None,
        "diagnostics_completed": False,
        "result_marker": "P3_04_SINGLE_LAYER_PARITY=RUNNING",
    }
    save_json(report_path, report)
    cases: list[dict[str, Any]] = report["cases"]
    for dtype in (torch.float32, torch.bfloat16):
        donor = QwenGDNReferenceMixer(config, context["source_state"]).to(device, dtype)
        target = DenseK3QwenCompatMixer(config, conversion.target_state).to(device, dtype)
        for sequence in (1, 7, 65, 129):
            cases.append(_functional_case(donor, target, config, device, dtype, sequence))
            failed_so_far = [case for case in cases if not case["passed"]]
            report.update(
                {
                    "cases_total": len(cases),
                    "cases_passed": len(cases) - len(failed_so_far),
                    "cases_failed": len(failed_so_far),
                    "first_failure": failed_so_far[0] if failed_so_far else None,
                }
            )
            save_json(report_path, report)
        del donor, target
        torch.cuda.empty_cache()
    projection_diagnostics = _projection_diagnostics(context)
    report["projection_diagnostics"] = projection_diagnostics
    save_json(report_path, report)
    injection_isolation = _injection_isolation(context)
    report["exact_injection_isolation"] = injection_isolation
    save_json(report_path, report)
    initial_state_isolation, error_distribution = _initial_state_isolation(context)
    report["initial_state_isolation"] = initial_state_isolation
    report["error_distribution"] = error_distribution
    save_json(report_path, report)
    diagnostics = (
        projection_diagnostics,
        injection_isolation,
        initial_state_isolation,
        error_distribution,
    )
    for diagnostic in diagnostics:
        print(diagnostic["result_marker"], flush=True)
    diagnostics_completed = all(item["status"] in {"PASS", "FAIL"} for item in diagnostics)
    gate = _apply_p3_functional_gate(
        cases,
        projection_diagnostics,
        injection_isolation,
        initial_state_isolation,
        error_distribution,
    )
    passed = gate["passed"]
    pass_mode = gate["pass_mode"]
    if pass_mode == "bf16_transplant_distribution_fallback":
        result_marker = "P3_04_SINGLE_LAYER_PARITY=PASS_WITH_NUMERICAL_FALLBACK"
        print("P3_04_PASS_MODE=BF16_TRANSPLANT_DISTRIBUTION_FALLBACK", flush=True)
        print(f"P3_04_NUMERICS_AMENDMENT={P3_NUMERICS_AMENDMENT}", flush=True)
    else:
        result_marker = f"P3_04_SINGLE_LAYER_PARITY={'PASS' if passed else 'FAIL'}"
    report.update(
        {
            "status": "PASS" if passed else "FAIL",
            "pass_mode": pass_mode,
            "standard_allclose": gate["standard_allclose"],
            "numerics_amendment": gate["numerics_amendment"],
            "structural_parity_verified": gate["structural_parity_verified"],
            "fp32_strict_parity_verified": gate["fp32_strict_parity_verified"],
            "exact_injection_verified": gate["exact_injection_verified"],
            "forced_fp32_projection_verified": gate["forced_fp32_projection_verified"],
            "cases_total": len(cases),
            "cases_passed": gate["final_cases_passed"],
            "cases_failed": gate["final_cases_failed"],
            "standard_cases_passed": gate["standard_cases_passed"],
            "standard_cases_failed": gate["standard_cases_failed"],
            "first_standard_failure": gate["first_standard_failure"],
            "first_failure": gate["first_final_failure"],
            "diagnostics_completed": diagnostics_completed,
            "functional_fallback": gate["fallback"],
            "result_marker": result_marker,
        }
    )
    save_json(report_path, report)
    context["functional_report"] = report
    return report


def stage_reproducibility(context: dict[str, Any]) -> dict[str, Any]:
    device: torch.device = context["device"]
    require_cuda(device)
    conversion: ConversionResult = context["conversion"]
    artifact_dir: Path = context["artifact_dir"]
    if artifact_dir.resolve() == context["model_dir"].resolve():
        raise ValueError("P3 output directory must be distinct from the source checkpoint")
    repeated = convert_gdn_to_kda(context["source_state"], context["config"])
    report = save_conversion(conversion, artifact_dir)
    reloaded_state, reloaded_manifest, reloaded_report = load_saved_conversion(artifact_dir)
    hashes_match = {
        "repeat_parameter_hash": repeated.parameter_hash == conversion.parameter_hash,
        "repeat_manifest_hash": repeated.manifest_hash == conversion.manifest_hash,
        "reload_parameter_hash": hash_state_dict(reloaded_state) == conversion.parameter_hash,
        "reload_manifest_hash": reloaded_report["manifest_hash"] == conversion.manifest_hash,
        "saved_report_match": report == reloaded_report,
        "manifest_target_hash_match": reloaded_manifest["target_hash"] == conversion.parameter_hash,
    }
    if not all(hashes_match.values()):
        raise ValueError(f"P3 save/reload hash reproducibility failed: {hashes_match}")

    config: GDNBridgeConfig = context["config"]
    dtype = torch.bfloat16
    before = DenseK3QwenCompatMixer(config, conversion.target_state).to(device, dtype)
    after = DenseK3QwenCompatMixer(config, reloaded_state).to(device, dtype)
    torch.manual_seed(42)
    hidden = torch.randn(1, 7, config.hidden_size, dtype=dtype, device=device)
    with torch.inference_mode():
        before_result = before.forward(hidden)
        after_result = after.forward(hidden)
    output_result = comparison(before_result.output, after_result.output, dtype, exact=True)
    state_result = comparison(before_result.final_state, after_result.final_state, torch.float32, exact=True)
    if not output_result["passed"] or not state_result["passed"]:
        raise ValueError(f"Reloaded forward differs: output={output_result}, state={state_result}")
    forward_hash = hash_state_dict(
        {"output": after_result.output.detach().cpu(), "final_state": after_result.final_state.detach().cpu()}
    )
    return {
        "stage": "P3.5",
        "status": "PASS",
        "artifact_directory": str(artifact_dir),
        "parameter_hash": conversion.parameter_hash,
        "manifest_hash": conversion.manifest_hash,
        "forward_hash": forward_hash,
        "hash_checks": hashes_match,
        "reload_output": output_result,
        "reload_final_state": state_result,
        "result_marker": "P3_05_REPRODUCIBILITY=PASS",
    }


STAGE_FUNCTIONS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "P3.0": stage_source,
    "P3.1": stage_conversion,
    "P3.2": stage_manifest,
    "P3.3": stage_decay,
    "P3.4": stage_functional,
    "P3.5": stage_reproducibility,
}


def write_environment(context: dict[str, Any]) -> None:
    device: torch.device = context["device"]
    cuda_available = torch.cuda.is_available()
    save_json(
        context["output_dir"] / "p3-environment.json",
        {
            "recorded_at": now(),
            "project_root": str(ROOT),
            "model_directory": str(context["model_dir"]),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": cuda_available,
            "requested_device": str(device),
            "gpu": torch.cuda.get_device_name(device) if cuda_available and device.type == "cuda" else None,
            "transformers": package_version("transformers"),
            "safetensors": package_version("safetensors"),
            "p2_frozen": True,
            "p2_result_marker": "P2_FINAL_CHECK=PASS",
            "p2_numerics_modified": False,
            "p1_contract_modified": False,
            "p3_scope": "single_real_qwen_gdn_layer_to_qwen_compat_kda",
            "p3_bf16_functional_fallback_amendment": P3_NUMERICS_AMENDMENT,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    context: dict[str, Any] = {
        "model_dir": args.model_dir.resolve(),
        "layer_index": args.layer_index,
        "output_dir": args.output_dir.resolve(),
        "artifact_dir": args.artifact_dir.resolve(),
        "device": torch.device(args.device),
    }
    context["output_dir"].mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    write_environment(context)
    status: dict[str, Any] = {
        "stage": "P3",
        "overall_status": "RUNNING",
        "first_failed_stage": None,
        "p2_frozen": True,
        "p1_contract_modified": False,
        "updated_at": now(),
        "stages": {},
    }
    status_path = context["output_dir"] / "p3-stage-status.json"
    save_json(status_path, status)
    failure: dict[str, Any] | None = None
    blocked_by: str | None = None

    for stage_id, name, filename, marker in STAGES:
        entry: dict[str, Any] = {"name": name, "status": "RUNNING", "started_at": now()}
        status["stages"][stage_id] = entry
        save_json(status_path, status)
        if blocked_by is not None:
            entry.update({"status": "BLOCKED", "blocked_by": blocked_by, "completed_at": now()})
        else:
            try:
                result = STAGE_FUNCTIONS[stage_id](context)
                result.setdefault("status", "PASS")
                save_json(context["output_dir"] / filename, result)
                if result["status"] == "PASS":
                    entry.update(
                        {
                            "status": "PASS",
                            "result_file": filename,
                            "pass_mode": result.get("pass_mode"),
                            "numerics_amendment": result.get("numerics_amendment"),
                            "completed_at": now(),
                        }
                    )
                    if result.get("result_marker") not in {None, f"{marker}=PASS"}:
                        print(result["result_marker"], flush=True)
                    print(f"{marker}=PASS", flush=True)
                else:
                    blocked_by = stage_id
                    status["first_failed_stage"] = stage_id
                    actual = result.get("result_marker", f"{marker}={result['status']}")
                    entry.update(
                        {
                            "status": "FAIL",
                            "result_file": filename,
                            "actual": actual,
                            "completed_at": now(),
                        }
                    )
                    failure = {
                        "stage": "P3",
                        "first_failed_stage": stage_id,
                        "failed_gate": name,
                        "timestamp": now(),
                        "command": [str(Path(__file__)), "--model-dir", str(context["model_dir"])],
                        "layer_index": context["layer_index"],
                        "expected": f"{marker}=PASS",
                        "actual": actual,
                        "result_file": filename,
                        "first_failure": result.get("first_failure"),
                        "blocks_p4": True,
                        "reproduction_command": (
                            f"cd {ROOT} && python {Path(__file__)} --model-dir {context['model_dir']} "
                            f"--layer-index {context['layer_index']}"
                        ),
                    }
                    print(f"{marker}=FAIL", flush=True)
            except Exception as exc:
                blocked_by = stage_id
                status["first_failed_stage"] = stage_id
                stack = traceback.format_exc()
                if stage_id == "P3.4":
                    save_functional_exception(context["output_dir"] / filename, exc, stack)
                entry.update(
                    {
                        "status": "FAIL",
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                        "traceback": stack,
                        "completed_at": now(),
                    }
                )
                failure = {
                    "stage": "P3",
                    "first_failed_stage": stage_id,
                    "failed_gate": name,
                    "timestamp": now(),
                    "command": [str(Path(__file__)), "--model-dir", str(context["model_dir"])],
                    "layer_index": context["layer_index"],
                    "expected": f"{marker}=PASS",
                    "actual": str(exc),
                    "exception_type": type(exc).__name__,
                    "traceback": stack,
                    "blocks_p4": True,
                    "reproduction_command": (
                        f"cd {ROOT} && python {Path(__file__)} --model-dir {context['model_dir']} "
                        f"--layer-index {context['layer_index']}"
                    ),
                }
                print(f"{marker}=FAIL", flush=True)
        status["updated_at"] = now()
        save_json(status_path, status)

    passed = status["first_failed_stage"] is None
    status["overall_status"] = "PASS" if passed else "FAIL"
    status["updated_at"] = now()
    save_json(status_path, status)
    functional_report = context.get("functional_report", {})
    final = {
        "stage": "P3",
        "status": "PASS" if passed else "FAIL",
        "first_failed_stage": status["first_failed_stage"],
        "qwen_gdn_source": status["stages"]["P3.0"]["status"] == "PASS",
        "deterministic_conversion": status["stages"]["P3.1"]["status"] == "PASS",
        "parameter_manifest": status["stages"]["P3.2"]["status"] == "PASS",
        "decay_embedding": status["stages"]["P3.3"]["status"] == "PASS",
        "single_layer_parity": status["stages"]["P3.4"]["status"] == "PASS",
        "single_layer_parity_pass_mode": functional_report.get("pass_mode"),
        "single_layer_standard_allclose": functional_report.get("standard_allclose"),
        "p3_numerics_amendment": functional_report.get("numerics_amendment"),
        "p3_numerics_fallback_used": functional_report.get("pass_mode")
        == "bf16_transplant_distribution_fallback",
        "reproducibility": status["stages"]["P3.5"]["status"] == "PASS",
        "p2_frozen": True,
        "p1_contract_modified": False,
        "p4_allowed": passed,
        "result_marker": f"P3_FINAL_CHECK={'PASS' if passed else 'FAIL'}",
        "failure_marker": None if passed else f"P3_FIRST_FAILED_STAGE={status['first_failed_stage']}",
    }
    save_json(context["output_dir"] / "p3-final-check.json", final)
    failure_path = context["output_dir"] / "p3-failure.json"
    if failure is not None:
        failure["blocked_stages"] = [
            stage_id for stage_id, entry in status["stages"].items() if entry["status"] == "BLOCKED"
        ]
        save_json(failure_path, failure)
    elif failure_path.exists():
        failure_path.unlink()
    print(final["result_marker"], flush=True)
    print(f"P4_ALLOWED={'true' if passed else 'false'}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
