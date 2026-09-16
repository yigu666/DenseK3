"""Bounded helpers for isolating P4.6 hybrid-cache semantics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from densek3_core.evaluation.p4_decay_diagnostics import compare, error_distribution
from densek3_core.kda.reference import kda_reference_from_raw
from densek3_core.modeling.hybrid_cache import GQACacheLayer
from densek3_core.modeling.modeling_densek3 import DenseK3KDA, _causal_depthwise_conv


def segment_ranges(sequence_length: int, prompt_length: int) -> list[tuple[int, int]]:
    """Return one prefill range followed by one-token decode ranges."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if prompt_length <= 0 or prompt_length > sequence_length:
        raise ValueError("prompt_length must be in [1, sequence_length]")
    return [(0, prompt_length), *[(index, index + 1) for index in range(prompt_length, sequence_length)]]


def tensor_comparison(
    reference: torch.Tensor,
    actual: torch.Tensor,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Record standard allclose and bounded distribution metrics together."""
    result = compare(reference, actual, dtype)
    result["distribution"] = error_distribution(reference, actual, dtype, top_k=0)
    return result


def partitioned_apply(
    function: Callable[[torch.Tensor], torch.Tensor],
    value: torch.Tensor,
    prompt_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one function to a whole sequence and to its production partition."""
    full = function(value)
    pieces = [function(value[:, start:end]) for start, end in segment_ranges(value.shape[1], prompt_length)]
    return full, torch.cat(pieces, dim=1)


def projection_partition_probe(
    layer: DenseK3KDA,
    hidden: torch.Tensor,
    prompt_length: int,
) -> dict[str, Any]:
    """Measure GEMM-shape differences before any cache operation."""
    functions: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        "q_projection": layer.q_proj,
        "k_projection": layer.k_proj,
        "v_projection": layer.v_proj,
        "channel_a_logits": lambda value: layer.f_b_proj(layer.f_a_proj(value)),
        "beta_logits": layer.b_proj,
        "output_gate": layer.g_proj,
    }
    comparisons = {}
    for name, function in functions.items():
        full, segmented = partitioned_apply(function, hidden, prompt_length)
        comparisons[name] = tensor_comparison(full, segmented, hidden.dtype)

    fp32_comparisons = {}
    fp32_functions: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        "q_projection": lambda value: torch.nn.functional.linear(
            value.float(), layer.q_proj.weight.float()
        ),
        "k_projection": lambda value: torch.nn.functional.linear(
            value.float(), layer.k_proj.weight.float()
        ),
        "v_projection": lambda value: torch.nn.functional.linear(
            value.float(), layer.v_proj.weight.float()
        ),
        "channel_a_logits": lambda value: torch.nn.functional.linear(
            torch.nn.functional.linear(value.float(), layer.f_a_proj.weight.float()),
            layer.f_b_proj.weight.float(),
        ),
        "beta_logits": lambda value: torch.nn.functional.linear(
            value.float(), layer.b_proj.weight.float()
        ),
        "output_gate": lambda value: torch.nn.functional.linear(
            value.float(), layer.g_proj.weight.float()
        ),
    }
    for name, function in fp32_functions.items():
        full, segmented = partitioned_apply(function, hidden, prompt_length)
        fp32_comparisons[name] = tensor_comparison(full, segmented, torch.float32)
    return {
        "sequence_length": hidden.shape[1],
        "prompt_length": prompt_length,
        "native_bf16": comparisons,
        "forced_fp32": fp32_comparisons,
        "all_finite": all(
            item["reference_nan_count"] == 0
            and item["actual_nan_count"] == 0
            and item["reference_inf_count"] == 0
            and item["actual_inf_count"] == 0
            for group in (comparisons, fp32_comparisons)
            for item in group.values()
        ),
    }


def convolution_partition_probe(
    projected: torch.Tensor,
    conv: torch.nn.Conv1d,
    prompt_length: int,
) -> dict[str, Any]:
    """Compare full convolution with prefill plus cached one-token decode."""
    full_output, full_state = _causal_depthwise_conv(projected, conv, None)
    state = None
    outputs = []
    for start, end in segment_ranges(projected.shape[1], prompt_length):
        output, state = _causal_depthwise_conv(projected[:, start:end], conv, state)
        outputs.append(output)
    if state is None:
        raise RuntimeError("Segmented convolution did not produce a cache state")
    segmented_output = torch.cat(outputs, dim=1)
    return {
        "output": tensor_comparison(full_output, segmented_output, projected.dtype),
        "final_state": tensor_comparison(full_state, state, projected.dtype),
        "final_state_exact": bool(torch.equal(full_state, state)),
    }


def _partitioned_recurrence(
    function: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None],
        tuple[torch.Tensor, torch.Tensor],
    ],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_decay: torch.Tensor,
    beta_logits: torch.Tensor,
    prompt_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    full_output, full_state = function(q, k, v, raw_decay, beta_logits, None)
    state = None
    outputs = []
    for start, end in segment_ranges(q.shape[1], prompt_length):
        output, state = function(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            raw_decay[:, start:end],
            beta_logits[:, start:end],
            state,
        )
        outputs.append(output)
    if state is None:
        raise RuntimeError("Segmented recurrence did not produce a final state")
    return full_output, full_state, torch.cat(outputs, dim=1), state


def recurrence_partition_probe(
    layer: DenseK3KDA,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_decay: torch.Tensor,
    beta_logits: torch.Tensor,
    prompt_length: int,
) -> dict[str, Any]:
    """Compare FLA and pure-reference recurrence under identical core inputs."""

    def production(
        q_value: torch.Tensor,
        k_value: torch.Tensor,
        v_value: torch.Tensor,
        decay_value: torch.Tensor,
        beta_value: torch.Tensor,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return layer._operator(q_value, k_value, v_value, decay_value, beta_value, state)

    def reference(
        q_value: torch.Tensor,
        k_value: torch.Tensor,
        v_value: torch.Tensor,
        decay_value: torch.Tensor,
        beta_value: torch.Tensor,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, final_state = kda_reference_from_raw(
            q_value,
            k_value,
            v_value,
            decay_value,
            beta_value,
            layer.A_log,
            layer.dt_bias,
            gate_mode="qwen_compat",
            initial_state=state,
            output_final_state=True,
        )
        if final_state is None:
            raise RuntimeError("Reference recurrence did not return its requested state")
        return output, final_state

    results = {}
    for name, function in (("production_fla", production), ("reference", reference)):
        full_output, full_state, segmented_output, segmented_state = _partitioned_recurrence(
            function,
            q,
            k,
            v,
            raw_decay,
            beta_logits,
            prompt_length,
        )
        results[name] = {
            "output": tensor_comparison(full_output, segmented_output, q.dtype),
            "final_state": tensor_comparison(full_state, segmented_state, q.dtype),
            "output_exact": bool(torch.equal(full_output, segmented_output)),
            "final_state_exact": bool(torch.equal(full_state, segmented_state)),
        }
    return results


def gqa_append_partition_probe(
    key: torch.Tensor,
    value: torch.Tensor,
    prompt_length: int,
) -> dict[str, Any]:
    """Verify that the GQA cache container concatenates canonical KV exactly."""
    full_cache = GQACacheLayer()
    full_key, full_value = full_cache.update(key, value)
    segmented_cache = GQACacheLayer()
    segmented_key = segmented_value = None
    for start, end in segment_ranges(key.shape[2], prompt_length):
        segmented_key, segmented_value = segmented_cache.update(
            key[:, :, start:end],
            value[:, :, start:end],
        )
    if segmented_key is None or segmented_value is None:
        raise RuntimeError("Segmented GQA append did not produce cache tensors")
    return {
        "key_exact": bool(torch.equal(full_key, segmented_key)),
        "value_exact": bool(torch.equal(full_value, segmented_value)),
        "sequence_length": segmented_cache.sequence_length,
        "expected_sequence_length": key.shape[2],
        "passed": (
            torch.equal(full_key, segmented_key)
            and torch.equal(full_value, segmented_value)
            and segmented_cache.sequence_length == key.shape[2]
        ),
    }


def hidden_error_curve(
    full_hidden: tuple[torch.Tensor, ...],
    segmented_hidden: tuple[torch.Tensor, ...],
) -> list[dict[str, Any]]:
    """Measure every model boundary without granting any model-level fallback."""
    if len(full_hidden) != len(segmented_hidden):
        raise ValueError("Full and segmented hidden-state tuples must have equal length")
    curve = []
    for boundary, (reference, actual) in enumerate(zip(full_hidden, segmented_hidden, strict=True)):
        item = tensor_comparison(reference, actual, torch.bfloat16)
        item.update(
            {
                "boundary_index": boundary,
                "boundary": (
                    "embedding"
                    if boundary == 0
                    else "final_norm"
                    if boundary == len(full_hidden) - 1
                    else f"after_layer_{boundary - 1}"
                ),
            }
        )
        curve.append(item)
    return curve
