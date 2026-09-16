"""Read-only causal diagnostics for the P4 layer-4 BF16 decay mismatch."""

from __future__ import annotations

from contextlib import contextmanager
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
from densek3_core.kda.reference import qwen_compat_decay
from densek3_core.transplant.gdn_to_kda import GDNBridgeConfig
from densek3_core.transplant.single_layer import (
    DenseK3QwenCompatMixer,
    MixerResult,
    MixerTrace,
    QwenGDNReferenceMixer,
)

P4_LOCAL_SEED = 20260810
P4_LOCAL_SEQUENCE = 7
P4_DIAGNOSTIC_LAYER = 4
TORCH_QUANTILE_MAX_ELEMENTS = 16_000_000


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return REFERENCE_FP32_RTOL, REFERENCE_FP32_ATOL
    if dtype == torch.bfloat16:
        return OPTIMIZED_BF16_RTOL, OPTIMIZED_BF16_ATOL
    raise ValueError(f"Unsupported diagnostic dtype: {dtype}")


def compare(
    reference: torch.Tensor,
    actual: torch.Tensor,
    dtype: torch.dtype,
    *,
    exact: bool = False,
) -> dict[str, Any]:
    if reference.shape != actual.shape:
        raise ValueError(f"Comparison shape mismatch: {reference.shape} != {actual.shape}")
    rtol, atol = (0.0, 0.0) if exact else tolerances(dtype)
    metrics = error_metrics(reference, actual)
    metrics.update(
        {
            "rtol": rtol,
            "atol": atol,
            "passed": bool(torch.allclose(reference, actual, rtol=rtol, atol=atol)),
            "exact": bool(torch.equal(reference, actual)),
            "shape": list(reference.shape),
            "shape_match": tuple(reference.shape) == tuple(actual.shape),
        }
    )
    return metrics


def _unravel_index(flat_index: int, shape: tuple[int, ...]) -> list[int]:
    coordinates = []
    for size in reversed(shape):
        coordinates.append(flat_index % size)
        flat_index //= size
    return list(reversed(coordinates))


def _exact_quantiles(
    values: torch.Tensor,
    probabilities: tuple[float, ...],
    *,
    torch_max_elements: int = TORCH_QUANTILE_MAX_ELEMENTS,
) -> tuple[torch.Tensor, str]:
    """Compute exact linear quantiles beyond PyTorch's element-count limit."""
    if values.device.type != "cpu" or values.dtype != torch.float64 or values.ndim != 1:
        raise ValueError("Exact quantile input must be a flat CPU float64 tensor")
    if values.numel() <= torch_max_elements:
        quantiles = torch.tensor(probabilities, device=values.device, dtype=torch.float64)
        return torch.quantile(values, quantiles), "torch_exact"

    # torch.quantile rejects tensors above its internal indexing limit. NumPy's
    # partition-based linear quantile is exact for the same definition and can
    # operate in-place on this diagnostic-only finite-error copy.
    import numpy as np

    result = np.quantile(
        values.numpy(),
        np.asarray(probabilities, dtype=np.float64),
        method="linear",
        overwrite_input=True,
    )
    return torch.from_numpy(np.asarray(result, dtype=np.float64)), "numpy_exact_partition"


def error_distribution(
    reference: torch.Tensor,
    actual: torch.Tensor,
    dtype: torch.dtype,
    *,
    axis_names: tuple[str, ...] | None = None,
    top_k: int = 8,
) -> dict[str, Any]:
    """Summarize all errors while retaining a bounded list of concrete outliers."""
    if reference.shape != actual.shape:
        raise ValueError(f"Distribution shape mismatch: {reference.shape} != {actual.shape}")
    rtol, atol = tolerances(dtype)
    reference_flat = reference.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    actual_flat = actual.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    absolute = (actual_flat - reference_flat).abs()
    threshold = atol + rtol * reference_flat.abs()
    failed = absolute > threshold
    finite_error = absolute[torch.isfinite(absolute)]
    probabilities = (0.5, 0.9, 0.95, 0.99, 0.999)
    if finite_error.numel():
        values, quantile_backend = _exact_quantiles(finite_error, probabilities)
    else:
        values = torch.full((len(probabilities),), float("nan"), dtype=torch.float64)
        quantile_backend = "empty"
    histogram = {
        "exact": int((absolute == 0).sum()),
        "0_to_1e-4": int(((absolute > 0) & (absolute <= 1e-4)).sum()),
        "1e-4_to_5e-4": int(((absolute > 1e-4) & (absolute <= 5e-4)).sum()),
        "5e-4_to_1e-3": int(((absolute > 5e-4) & (absolute <= 1e-3)).sum()),
        "1e-3_to_5e-3": int(((absolute > 1e-3) & (absolute <= 5e-3)).sum()),
        "5e-3_to_1e-2": int(((absolute > 5e-3) & (absolute <= 1e-2)).sum()),
        "above_1e-2": int((absolute > 1e-2).sum()),
    }
    outliers = []
    count = min(top_k, absolute.numel())
    if count:
        ranked = torch.topk(torch.nan_to_num(absolute, nan=float("inf")), k=count).indices.tolist()
        shape = tuple(reference.shape)
        names = axis_names or tuple(f"dim_{index}" for index in range(len(shape)))
        if len(names) != len(shape):
            raise ValueError("axis_names must have one entry per tensor dimension")
        for flat_index in ranked:
            coordinates = _unravel_index(flat_index, shape)
            outliers.append(
                {
                    "flat_index": flat_index,
                    "index": dict(zip(names, coordinates, strict=True)),
                    "reference": float(reference_flat[flat_index]),
                    "actual": float(actual_flat[flat_index]),
                    "abs_error": float(absolute[flat_index]),
                    "allclose_limit": float(threshold[flat_index]),
                    "failed_allclose": bool(failed[flat_index]),
                }
            )
    metrics = error_metrics(reference_flat, actual_flat)
    metrics.update(
        {
            "num_elements": absolute.numel(),
            "num_exact": int((absolute == 0).sum()),
            "num_failed_allclose": int(failed.sum()),
            "failed_allclose_fraction": float(failed.double().mean()) if failed.numel() else 0.0,
            "p50_abs_error": float(values[0]),
            "p90_abs_error": float(values[1]),
            "p95_abs_error": float(values[2]),
            "p99_abs_error": float(values[3]),
            "p99_9_abs_error": float(values[4]),
            "quantile_backend": quantile_backend,
            "quantile_exact": True,
            "histogram": histogram,
            "top_outliers": outliers,
            "rtol": rtol,
            "atol": atol,
            "allclose_passed": bool(torch.allclose(reference, actual, rtol=rtol, atol=atol)),
        }
    )
    return metrics


def replay_p4_hidden(
    hidden_size: int,
    kda_indices: list[int],
    layer_index: int = P4_DIAGNOSTIC_LAYER,
    *,
    seed: int = P4_LOCAL_SEED,
    sequence: int = P4_LOCAL_SEQUENCE,
) -> tuple[torch.Tensor, int]:
    """Replay the exact CPU generator stream used by ``run_p4.stage_local_parity``."""
    if layer_index not in kda_indices:
        raise ValueError(f"Layer {layer_index} is not a KDA layer")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden = None
    draw_ordinal = 0
    for _draw_ordinal, current_layer in enumerate(kda_indices, start=1):
        draw_ordinal = _draw_ordinal
        hidden = torch.randn((1, sequence, hidden_size), generator=generator)
        if current_layer == layer_index:
            break
    if hidden is None:
        raise RuntimeError("P4 hidden replay did not produce a tensor")
    return hidden, draw_ordinal


def seeded_hidden(hidden_size: int, sequence: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((1, sequence, hidden_size), generator=generator)


def _finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all())


def _axis_summary(
    reference: torch.Tensor,
    actual: torch.Tensor,
    dtype: torch.dtype,
    axis: int,
    label: str,
) -> list[dict[str, Any]]:
    rtol, atol = tolerances(dtype)
    reference_cpu = reference.detach().to(device="cpu", dtype=torch.float64)
    actual_cpu = actual.detach().to(device="cpu", dtype=torch.float64)
    absolute = (actual_cpu - reference_cpu).abs()
    threshold = atol + rtol * reference_cpu.abs()
    reduce_dims = tuple(index for index in range(absolute.ndim) if index != axis)
    maximum = absolute.amax(dim=reduce_dims)
    mean = absolute.mean(dim=reduce_dims)
    failed_fraction = (absolute > threshold).double().mean(dim=reduce_dims)
    items = [
        {
            label: index,
            "max_abs_error": float(maximum[index]),
            "mean_abs_error": float(mean[index]),
            "failed_allclose_fraction": float(failed_fraction[index]),
        }
        for index in range(absolute.shape[axis])
    ]
    return sorted(items, key=lambda item: item["max_abs_error"], reverse=True)


def decay_path_diagnostics(
    donor: MixerTrace,
    target: MixerTrace,
    dtype: torch.dtype,
    *,
    donor_mixer: QwenGDNReferenceMixer | None = None,
    target_mixer: DenseK3QwenCompatMixer | None = None,
) -> dict[str, Any]:
    """Measure each boundary in the scalar-to-channel decay embedding."""
    axis_names = ("batch", "token", "value_head", "key_channel")
    raw = error_distribution(
        donor.raw_a_logits,
        target.raw_a_logits,
        dtype,
        axis_names=("batch", "token", "value_head"),
        top_k=16,
    )
    channel = error_distribution(
        donor.channel_a_logits,
        target.channel_a_logits,
        dtype,
        axis_names=axis_names,
        top_k=16,
    )
    decay = error_distribution(
        donor.decay,
        target.decay,
        dtype,
        axis_names=axis_names,
        top_k=16,
    )
    target_f_a_repeat = target.raw_a_logits.unsqueeze(-1).expand_as(target.channel_a_logits)
    f_b_replication = error_distribution(
        target_f_a_repeat,
        target.channel_a_logits,
        dtype,
        axis_names=axis_names,
        top_k=16,
    )
    target_channel_spread = target.channel_a_logits - target.channel_a_logits[..., :1]
    transform_replay = None
    if donor_mixer is not None and target_mixer is not None:
        config = donor_mixer.config
        donor_dt = donor_mixer.state["dt_bias"].repeat_interleave(config.qk_head_dim)
        donor_recomputed = qwen_compat_decay(
            donor.channel_a_logits,
            donor_mixer.state["A_log"],
            donor_dt,
        )
        target_recomputed = qwen_compat_decay(
            target.channel_a_logits,
            target_mixer.state["A_log"],
            target_mixer.state["dt_bias"],
        )
        transform_replay = {
            "donor_trace_vs_recomputed": compare(donor.decay, donor_recomputed, torch.float32),
            "target_trace_vs_recomputed": compare(target.decay, target_recomputed, torch.float32),
        }
    return {
        "raw_a_logits": raw,
        "target_f_b_vs_exact_repeat": f_b_replication,
        "channel_a_logits": channel,
        "decay": decay,
        "target_f_a_intermediate": {
            "shape": None if target.f_a_intermediate is None else list(target.f_a_intermediate.shape),
            "finite": target.f_a_intermediate is not None and _finite(target.f_a_intermediate),
        },
        "target_channel_copy_max_spread": float(target_channel_spread.abs().max()),
        "decay_transform_replay": transform_replay,
        "decay_error_by_value_head": _axis_summary(donor.decay, target.decay, dtype, 2, "value_head"),
        "decay_error_by_token": _axis_summary(donor.decay, target.decay, dtype, 1, "token"),
        "all_finite": all(
            _finite(tensor)
            for tensor in (
                donor.raw_a_logits,
                target.raw_a_logits,
                donor.channel_a_logits,
                target.channel_a_logits,
                donor.decay,
                target.decay,
            )
        ),
    }


def parameter_embedding_audit(
    config: GDNBridgeConfig,
    donor: QwenGDNReferenceMixer,
    target: DenseK3QwenCompatMixer,
) -> dict[str, Any]:
    """Recheck the deterministic scalar-to-channel parameter construction."""
    f_a = target.state["f_a_proj.weight"]
    f_b = target.state["f_b_proj.weight"].view(
        config.value_num_heads,
        config.qk_head_dim,
        config.decay_projection_rank,
    )
    expected_f_b = torch.zeros_like(f_b)
    heads = torch.arange(config.value_num_heads, device=f_b.device)
    expected_f_b[heads, :, heads] = 1
    checks = {
        "f_a_source_rows_exact": bool(
            torch.equal(f_a[: config.value_num_heads], donor.state["in_proj_a.weight"])
        ),
        "f_a_extra_rows_zero": int(torch.count_nonzero(f_a[config.value_num_heads :])) == 0,
        "f_b_one_hot_repeat_exact": bool(torch.equal(f_b, expected_f_b)),
        "a_log_exact": bool(torch.equal(target.state["A_log"], donor.state["A_log"])),
        "dt_bias_repeat_exact": bool(
            torch.equal(
                target.state["dt_bias"],
                donor.state["dt_bias"].repeat_interleave(config.qk_head_dim),
            )
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "f_a_shape": list(f_a.shape),
        "f_b_shape": list(f_b.shape),
    }


def trace_comparisons(donor: MixerTrace, target: MixerTrace, dtype: torch.dtype) -> dict[str, Any]:
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
    return {name: compare(getattr(donor, name), getattr(target, name), dtype) for name in fields}


@contextmanager
def ieee_fp32_matmul() -> Any:
    """Temporarily disable TF32 for a diagnostic-only projection replay."""
    backend = torch.backends.cuda.matmul
    try:
        old_backend_precision = backend.fp32_precision
    except (AttributeError, AssertionError):
        old_backend_precision = None
    if old_backend_precision is not None:
        backend.fp32_precision = "ieee"
        try:
            yield
        finally:
            backend.fp32_precision = old_backend_precision
    else:
        old_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        try:
            yield
        finally:
            torch.set_float32_matmul_precision(old_precision)


def _linear_fp32(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(hidden.float(), weight.float())


def forced_fp32_diagnostics(
    donor: QwenGDNReferenceMixer,
    target: DenseK3QwenCompatMixer,
    hidden: torch.Tensor,
    donor_native: MixerTrace,
    target_native: MixerTrace,
) -> dict[str, Any]:
    """Replay candidate GEMMs in IEEE FP32 without changing the formal backend."""
    config = donor.config
    with ieee_fp32_matmul():
        donor_fused = _linear_fp32(hidden, donor.state["in_proj_qkv.weight"])
        donor_q, donor_k, donor_v = torch.split(
            donor_fused,
            [config.qk_dim, config.qk_dim, config.value_dim],
            dim=-1,
        )
        target_q = _linear_fp32(hidden, target.state["q_proj.weight"])
        target_k = _linear_fp32(hidden, target.state["k_proj.weight"])
        target_v = _linear_fp32(hidden, target.state["v_proj.weight"])
        donor_raw = _linear_fp32(hidden, donor.state["in_proj_a.weight"])
        target_f_a = _linear_fp32(hidden, target.state["f_a_proj.weight"])
        target_channel = F.linear(target_f_a, target.state["f_b_proj.weight"].float()).view(
            *hidden.shape[:2],
            config.value_num_heads,
            config.qk_head_dim,
        )
        donor_channel = donor_raw.unsqueeze(-1).expand_as(target_channel)
        donor_dt = donor.state["dt_bias"].repeat_interleave(config.qk_head_dim)
        donor_decay = qwen_compat_decay(donor_channel, donor.state["A_log"], donor_dt)
        target_decay = qwen_compat_decay(target_channel, target.state["A_log"], target.state["dt_bias"])

    fp32_pairs = {
        "q_projection": (donor_q, target_q),
        "k_projection": (donor_k, target_k),
        "v_projection": (donor_v, target_v),
        "raw_a_logits": (donor_raw, target_f_a[..., : config.value_num_heads]),
        "channel_a_logits": (donor_channel, target_channel),
        "decay": (donor_decay, target_decay),
    }
    fp32_results = {
        name: compare(reference, actual, torch.float32)
        for name, (reference, actual) in fp32_pairs.items()
    }
    donor_channel_bf16 = donor_channel.to(torch.bfloat16)
    target_channel_bf16 = target_channel.to(torch.bfloat16)
    donor_decay_from_bf16 = qwen_compat_decay(
        donor_channel_bf16,
        donor.state["A_log"],
        donor_dt,
    )
    target_decay_from_bf16 = qwen_compat_decay(
        target_channel_bf16,
        target.state["A_log"],
        target.state["dt_bias"],
    )
    cast_results = {
        "channel_a_logits": compare(donor_channel_bf16, target_channel_bf16, torch.bfloat16),
        "decay": compare(donor_decay_from_bf16, target_decay_from_bf16, torch.float32),
    }
    native_to_forced = {
        "donor_raw_a": compare(donor_native.raw_a_logits.float(), donor_raw, torch.float32),
        "target_raw_a": compare(
            target_native.raw_a_logits.float(),
            target_f_a[..., : config.value_num_heads],
            torch.float32,
        ),
        "donor_decay": compare(donor_native.decay, donor_decay_from_bf16, torch.float32),
        "target_decay": compare(target_native.decay, target_decay_from_bf16, torch.float32),
    }
    return {
        "diagnostic_only": True,
        "formal_backend_modified": False,
        "input_storage_dtype": dtype_name(hidden.dtype),
        "weight_storage_dtype": dtype_name(target.state["f_a_proj.weight"].dtype),
        "gemm_accumulation_request": "IEEE float32 (TF32 disabled)",
        "fp32_donor_vs_target": fp32_results,
        "cast_back_to_bf16_donor_vs_target": cast_results,
        "native_bf16_vs_forced_fp32": native_to_forced,
        "all_fp32_pairs_exact": all(item["exact"] for item in fp32_results.values()),
        "decay_exact_after_bf16_cast": cast_results["decay"]["exact"],
        "decay_strict_parity_after_bf16_cast": cast_results["decay"]["passed"],
    }


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


def result_comparison(reference: MixerResult, actual: MixerResult, dtype: torch.dtype) -> dict[str, Any]:
    output = compare(reference.output, actual.output, dtype)
    state = compare(reference.final_state, actual.final_state, dtype)
    return {
        "output": output,
        "final_state": state,
        "passed": output["passed"] and state["passed"],
        "exact": output["exact"] and state["exact"],
    }


def injection_diagnostics(
    donor: QwenGDNReferenceMixer,
    target: DenseK3QwenCompatMixer,
    hidden: torch.Tensor,
    donor_trace: MixerTrace,
    target_trace: MixerTrace,
    initial_state: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Perform the four counterfactual recurrence injections on the exact failed input."""
    config = donor.config
    donor_qkv = _reshape_qkv(config, donor_trace.q_conv, donor_trace.k_conv, donor_trace.v_conv)
    target_qkv = _reshape_qkv(config, target_trace.q_conv, target_trace.k_conv, target_trace.v_conv)

    def run(
        qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        decay: torch.Tensor,
        beta: torch.Tensor,
    ) -> MixerResult:
        return target.forward_from_recurrence_inputs(
            hidden,
            *qkv,
            decay,
            beta,
            initial_state=initial_state,
        )

    actuals = {
        "case_0_native_transplant": target_trace.result(),
        "case_1_donor_decay": run(target_qkv, donor_trace.decay, target_trace.beta),
        "case_2_donor_postconv_qkv": run(donor_qkv, target_trace.decay, target_trace.beta),
        "case_3_both_injected": run(donor_qkv, donor_trace.decay, donor_trace.beta),
    }
    cases = {
        name: result_comparison(donor_trace.result(), actual, torch.bfloat16)
        for name, actual in actuals.items()
    }
    target_native_vs_decay_injected = result_comparison(
        target_trace.result(),
        actuals["case_1_donor_decay"],
        torch.bfloat16,
    )
    return {
        "diagnostic_only": True,
        "dtype": "bfloat16",
        "cases_vs_donor": cases,
        "target_native_vs_donor_decay_injected": target_native_vs_decay_injected,
        "both_injected_exact": cases["case_3_both_injected"]["exact"],
        "pass_definition": (
            "under identical donor recurrence inputs, the target recurrence and post-processing path "
            "exactly restores donor output and state"
        ),
    }


def summarize_case(
    donor: MixerTrace,
    target: MixerTrace,
    dtype: torch.dtype,
    *,
    include_distribution: bool,
) -> dict[str, Any]:
    comparisons = {
        "output": compare(donor.output, target.output, dtype),
        "final_state": compare(donor.final_state, target.final_state, dtype),
        "decay": compare(donor.decay, target.decay, dtype),
        "beta": compare(donor.beta, target.beta, dtype),
    }
    value = {
        "comparisons": comparisons,
        "standard_allclose": all(item["passed"] for item in comparisons.values()),
        "all_finite": all(
            _finite(tensor)
            for tensor in (
                donor.output,
                target.output,
                donor.final_state,
                target.final_state,
                donor.decay,
                target.decay,
                donor.beta,
                target.beta,
            )
        ),
    }
    if include_distribution:
        value["distributions"] = {
            "output": error_distribution(donor.output, target.output, dtype, top_k=4),
            "final_state": error_distribution(donor.final_state, target.final_state, dtype, top_k=4),
            "decay": error_distribution(
                donor.decay,
                target.decay,
                dtype,
                axis_names=("batch", "token", "value_head", "key_channel"),
                top_k=4,
            ),
        }
    return value


__all__ = [
    "P4_DIAGNOSTIC_LAYER",
    "P4_LOCAL_SEED",
    "P4_LOCAL_SEQUENCE",
    "compare",
    "decay_path_diagnostics",
    "error_distribution",
    "forced_fp32_diagnostics",
    "injection_diagnostics",
    "parameter_embedding_audit",
    "replay_p4_hidden",
    "seeded_hidden",
    "summarize_case",
    "trace_comparisons",
]
