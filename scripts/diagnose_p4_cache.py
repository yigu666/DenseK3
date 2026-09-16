"""Isolate P4.6 cache semantics from full-model BF16 error propagation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from densek3_core.evaluation.p4_cache_diagnostics import (
    convolution_partition_probe,
    gqa_append_partition_probe,
    hidden_error_curve,
    projection_partition_probe,
    recurrence_partition_probe,
    segment_ranges,
    tensor_comparison,
)
from densek3_core.evaluation.p4_decay_diagnostics import ieee_fp32_matmul
from densek3_core.evaluation.p4_runtime import load_densek3_model
from densek3_core.modeling.modeling_densek3 import _apply_partial_rope, _causal_depthwise_conv
from densek3_core.transplant.full_model import verify_p4_artifact

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = ROOT / "outputs/densek3-core/checkpoints/p4/densek3-4b-qwen-compat"
DEFAULT_MANIFEST_DIR = ROOT / "outputs/densek3-core/manifests/p4"
DEFAULT_OUTPUT = DEFAULT_MANIFEST_DIR / "p4-cache-causal-diagnostics.json"
FORMAL_FINAL_NAME = "p4-final-check.json"
FORMAL_CACHE_NAME = "p4-hybrid-cache.json"
P4_AMENDMENT = "P4-NUM-2026-08-11-01"
TOKEN_SEED = 20260811
CASES = ((1, 1), (7, 8), (64, 32))


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sanitize_json(item: Any) -> Any:
    if isinstance(item, float) and not math.isfinite(item):
        return None
    if isinstance(item, dict):
        return {key: _sanitize_json(value) for key, value in item.items()}
    if isinstance(item, (list, tuple)):
        return [_sanitize_json(value) for value in item]
    return item


def save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_sanitize_json(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_resume_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Cannot resume missing diagnostic report: {path}")
    previous_bytes = path.read_bytes()
    report = json.loads(previous_bytes)
    resume_checks = {
        "schema_version": report.get("schema_version") == 1,
        "stage": report.get("stage") == "P4.6-CACHE-DIAG",
        "token_seed": report.get("token_seed") == TOKEN_SEED,
        "partial_status": report.get("status") in {"EXCEPTION", "RUNNING"},
        "formal_status_unchanged": report.get("p4_6_formal_status") == "FAIL_UNCHANGED",
        "no_amendment": report.get("numerics_amendment_approved") is False,
    }
    if not all(resume_checks.values()):
        raise ValueError(f"Existing diagnostic report is not safe to resume: {resume_checks}")
    previous_sha256 = hashlib.sha256(previous_bytes).hexdigest()
    history = list(report.get("resume_history", []))
    history.append(
        {
            "resumed_at": now(),
            "previous_report_sha256": previous_sha256,
            "previous_status": report.get("status"),
            "previous_last_completed_phase": report.get("last_completed_phase"),
            "previous_exception_type": report.get("exception_type"),
            "previous_exception": report.get("exception"),
        }
    )
    report["resume_history"] = history
    report["resume"] = {
        "enabled": True,
        "safety_checks": resume_checks,
        "previous_report_sha256": previous_sha256,
    }
    for name in ("exception_type", "exception", "traceback", "completed_at"):
        report.pop(name, None)
    report.update(
        {
            "status": "RESUMING",
            "diagnostics_completed": False,
            "result_marker": "P4_06_CACHE_CAUSAL_DIAGNOSTICS=RUNNING",
        }
    )
    return report


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_formal_evidence(manifest_dir: Path) -> dict[str, Any]:
    final_path = manifest_dir / FORMAL_FINAL_NAME
    cache_path = manifest_dir / FORMAL_CACHE_NAME
    if not final_path.is_file() or not cache_path.is_file():
        raise FileNotFoundError("P4.6 diagnostic requires the formal final-check and hybrid-cache reports")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    checks = {
        "p4_4_passed": final.get("all_layer_local_parity") is True,
        "p4_5_completed": final.get("full_model_forward") is True,
        "first_failure_is_p4_6": final.get("first_failed_stage") == "P4.6",
        "p4_6_failed": final.get("hybrid_cache_generation") is False,
        "p4_amendment_recorded": P4_AMENDMENT in final.get("p4_numerics_amendments", []),
        "formal_cache_stage": cache.get("stage") == "P4.6",
        "formal_cache_failed": cache.get("status") == "FAIL",
        "three_cases_persisted": cache.get("cases_completed") == len(CASES),
    }
    return {
        "files": {
            "final_check": {"path": str(final_path), "sha256": file_sha256(final_path)},
            "hybrid_cache": {"path": str(cache_path), "sha256": file_sha256(cache_path)},
        },
        "checks": checks,
        "passed": all(checks.values()),
        "formal_summary": {
            "first_failed_stage": final.get("first_failed_stage"),
            "p4_4": final.get("all_layer_local_parity"),
            "p4_5": final.get("full_model_forward"),
            "p4_6": final.get("hybrid_cache_generation"),
            "p5_allowed": final.get("p5_allowed"),
        },
    }


def _case_tokens(config: Any) -> tuple[list[tuple[int, int, torch.LongTensor]], torch.LongTensor]:
    generator = torch.Generator(device="cpu").manual_seed(TOKEN_SEED)
    cases = []
    for prompt, decode in CASES:
        tokens = torch.randint(
            low=0,
            high=config.vocab_size,
            size=(1, prompt + decode),
            generator=generator,
        )
        cases.append((prompt, decode, tokens))
    batch = torch.randint(low=0, high=config.vocab_size, size=(2, 7), generator=generator)
    return cases, batch


def _canonical_gqa_kv(
    layer: Any,
    hidden: torch.Tensor,
    prompt_length: int,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    batch, sequence, _ = hidden.shape
    packed_q = layer.q_proj(hidden).view(batch, sequence, layer.num_heads, 2, layer.head_dim)
    q, _ = packed_q.unbind(dim=3)
    q = layer.q_norm(q).transpose(1, 2)
    k = layer.k_norm(
        layer.k_proj(hidden).view(batch, sequence, layer.num_key_value_heads, layer.head_dim)
    ).transpose(1, 2)
    v = layer.v_proj(hidden).view(
        batch, sequence, layer.num_key_value_heads, layer.head_dim
    ).transpose(1, 2)
    full_positions = torch.arange(sequence, device=hidden.device).unsqueeze(0)
    segmented_positions = torch.cat(
        [torch.arange(start, end, device=hidden.device) for start, end in segment_ranges(sequence, prompt_length)]
    ).unsqueeze(0)
    q, k = _apply_partial_rope(q, k, full_positions, layer.rotary_dim, layer.rope_theta)
    del q
    return k, v, bool(torch.equal(full_positions, segmented_positions))


def _direct_case(model: Any, tokens: torch.LongTensor, prompt: int, device: torch.device) -> dict[str, Any]:
    tokens = tokens.to(device)
    decoder = model.model.layers[0]
    kda = decoder.self_attn
    with torch.inference_mode():
        hidden = decoder.input_layernorm(model.model.embed_tokens(tokens))
        with ieee_fp32_matmul():
            projections = projection_partition_probe(kda, hidden, prompt)
        q_projected = kda.q_proj(hidden)
        k_projected = kda.k_proj(hidden)
        v_projected = kda.v_proj(hidden)
        convolution = {
            "q": convolution_partition_probe(q_projected, kda.q_conv1d, prompt),
            "k": convolution_partition_probe(k_projected, kda.k_conv1d, prompt),
            "v": convolution_partition_probe(v_projected, kda.v_conv1d, prompt),
        }
        q, _ = _causal_depthwise_conv(q_projected, kda.q_conv1d, None)
        k, _ = _causal_depthwise_conv(k_projected, kda.k_conv1d, None)
        v, _ = _causal_depthwise_conv(v_projected, kda.v_conv1d, None)
        batch, sequence, _ = q.shape
        q = q.view(batch, sequence, kda.qk_heads, kda.key_dim)
        k = k.view(batch, sequence, kda.qk_heads, kda.key_dim)
        v = v.view(batch, sequence, kda.value_heads, kda.value_dim)
        raw_decay = kda.f_b_proj(kda.f_a_proj(hidden)).view(
            batch,
            sequence,
            kda.value_heads,
            kda.key_dim,
        )
        beta = kda.b_proj(hidden)
        recurrence = recurrence_partition_probe(kda, q, k, v, raw_decay, beta, prompt)
        gqa = model.model.layers[3].self_attn
        gqa_key, gqa_value, positions_exact = _canonical_gqa_kv(gqa, hidden, prompt)
        gqa_append = gqa_append_partition_probe(gqa_key, gqa_value, prompt)
    case = {
        "prompt_length": prompt,
        "sequence_length": tokens.shape[1],
        "token_sha256": tensor_sha256(tokens),
        "projection_partition": projections,
        "convolution_partition": convolution,
        "recurrence_partition": recurrence,
        "gqa_append_partition": gqa_append,
        "position_sequence_exact": positions_exact,
    }
    case.update(_review_direct_case(case))
    return case


def _review_direct_case(case: dict[str, Any]) -> dict[str, Any]:
    projections = case["projection_partition"]
    convolution = case["convolution_partition"]
    recurrence = case["recurrence_partition"]
    gqa_append = case["gqa_append_partition"]
    cache_checks = {
        "convolution_outputs_pass": all(item["output"]["passed"] for item in convolution.values()),
        "convolution_states_exact": all(item["final_state_exact"] for item in convolution.values()),
        "fla_recurrence_output_pass": recurrence["production_fla"]["output"]["passed"],
        "fla_recurrence_state_pass": recurrence["production_fla"]["final_state"]["passed"],
        "reference_recurrence_output_pass": recurrence["reference"]["output"]["passed"],
        "reference_recurrence_state_pass": recurrence["reference"]["final_state"]["passed"],
        "gqa_append_exact": gqa_append["passed"],
        "position_sequence_exact": case["position_sequence_exact"],
        "all_values_finite": projections["all_finite"],
    }
    native_failures = [
        name for name, item in projections["native_bf16"].items() if not item["passed"]
    ]
    forced_fp32_tails = [
        {
            "tensor": name,
            "max_abs_error": item["max_abs_error"],
            "mean_abs_error": item["mean_abs_error"],
            "relative_l2_error": item["relative_l2_error"],
            "cosine_similarity": item["cosine_similarity"],
            "num_failed_allclose": item["distribution"]["num_failed_allclose"],
            "failed_allclose_fraction": item["distribution"]["failed_allclose_fraction"],
            "p99_9_abs_error": item["distribution"]["p99_9_abs_error"],
        }
        for name, item in projections["forced_fp32"].items()
        if not item["passed"]
    ]
    return {
        "cache_semantic_checks": cache_checks,
        "direct_cache_contracts_passed": all(cache_checks.values()),
        "direct_contracts_passed": all(cache_checks.values()),
        "direct_contract_definition": (
            "canonical-input convolution/KDA/GQA cache semantics; projection shape differences are causal inputs, "
            "not cache-contract failures"
        ),
        "projection_observations": {
            "native_bf16_standard_failures": native_failures,
            "forced_fp32_standard_tails": forced_fp32_tails,
            "forced_fp32_all_standard_pass": not forced_fp32_tails,
            "tails_retained_without_formal_acceptance": True,
        },
    }


def _validate_resumed_cases(
    saved_cases: list[dict[str, Any]],
    expected_cases: list[tuple[int, int, torch.LongTensor]],
    *,
    integrated: bool,
) -> None:
    if len(saved_cases) > len(expected_cases):
        raise ValueError("Resume report contains more cases than the frozen diagnostic matrix")
    for index, saved in enumerate(saved_cases):
        prompt, decode, tokens = expected_cases[index]
        checks = {
            "prompt_length": saved.get("prompt_length") == prompt,
            "sequence_length": saved.get("sequence_length") == prompt + decode,
            "token_sha256": saved.get("token_sha256") == tensor_sha256(tokens),
        }
        if integrated:
            checks["decode_length"] = saved.get("decode_length") == decode
        if not all(checks.values()):
            raise ValueError(f"Resume case {index} does not match frozen tokens/split: {checks}")


def _to_cpu_hidden(values: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(value.detach().cpu() for value in values)


def _integrated_case(
    model: Any,
    tokens: torch.LongTensor,
    prompt: int,
    decode: int,
    device: torch.device,
) -> dict[str, Any]:
    tokens = tokens.to(device)
    mask = torch.ones_like(tokens)
    with torch.inference_mode():
        full = model(
            tokens,
            attention_mask=mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
    full_hidden = _to_cpu_hidden(full.hidden_states)
    full_logits = full.logits.detach().cpu()
    del full
    segmented_hidden: list[list[torch.Tensor]] = [[] for _ in full_hidden]
    segmented_logits = []
    with torch.inference_mode():
        output = model(
            tokens[:, :prompt],
            attention_mask=mask[:, :prompt],
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        cache = output.past_key_values
        for index, value in enumerate(output.hidden_states):
            segmented_hidden[index].append(value.detach().cpu())
        segmented_logits.append(output.logits.detach().cpu())
        for position in range(prompt, tokens.shape[1]):
            output = model(
                tokens[:, position : position + 1],
                attention_mask=mask[:, : position + 1],
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            cache = output.past_key_values
            for index, value in enumerate(output.hidden_states):
                segmented_hidden[index].append(value.detach().cpu())
            segmented_logits.append(output.logits.detach().cpu())
    joined_hidden = tuple(torch.cat(values, dim=1) for values in segmented_hidden)
    joined_logits = torch.cat(segmented_logits, dim=1)
    curve = hidden_error_curve(full_hidden, joined_hidden)
    first_failed = next((item["boundary"] for item in curve if not item["passed"]), None)
    logits = tensor_comparison(full_logits, joined_logits, torch.bfloat16)
    return {
        "prompt_length": prompt,
        "decode_length": decode,
        "sequence_length": tokens.shape[1],
        "token_sha256": tensor_sha256(tokens),
        "seen_tokens": cache.seen_tokens,
        "seen_tokens_correct": cache.seen_tokens == tokens.shape[1],
        "first_failed_boundary": first_failed,
        "hidden_error_curve": curve,
        "logits": logits,
        "formal_failure_reproduced": not logits["passed"],
    }


def _batch_isolation(model: Any, tokens: torch.LongTensor, device: torch.device) -> dict[str, Any]:
    tokens = tokens.to(device)
    with torch.inference_mode():
        batched = model(tokens, attention_mask=torch.ones_like(tokens), use_cache=True, return_dict=True)
        singles = [
            model(
                tokens[index : index + 1],
                attention_mask=torch.ones_like(tokens[index : index + 1]),
                use_cache=True,
                return_dict=True,
            )
            for index in range(tokens.shape[0])
        ]
    joined_logits = torch.cat([item.logits for item in singles], dim=0)
    logits = tensor_comparison(joined_logits, batched.logits, torch.bfloat16)
    states = []
    for layer_index in model.config.kda_indices:
        expected = torch.cat(
            [item.past_key_values.kda_layer(layer_index).recurrent_state for item in singles],
            dim=0,
        )
        actual = batched.past_key_values.kda_layer(layer_index).recurrent_state
        states.append(
            {
                "layer_index": layer_index,
                "comparison": tensor_comparison(expected, actual, torch.bfloat16),
            }
        )
    return {
        "token_sha256": tensor_sha256(tokens),
        "logits": logits,
        "kda_recurrent_states": states,
        "states_passed": sum(item["comparison"]["passed"] for item in states),
        "states_total": len(states),
        "passed": logits["passed"] and all(item["comparison"]["passed"] for item in states),
    }


def run(args: argparse.Namespace, report: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("The authoritative P4.6 cache diagnostic requires CUDA")
    device = torch.device("cuda")
    formal = _load_formal_evidence(args.manifest_dir)
    if not formal["passed"]:
        raise ValueError("Formal P4.6 failure evidence did not match the frozen diagnostic scope")
    artifact = verify_p4_artifact(args.artifact_dir)
    report.update(
        {
            "status": "RUNNING",
            "last_completed_phase": "formal_evidence_and_artifact",
            "formal_evidence": formal,
            "artifact_verification": artifact,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device),
                "transformers": package_version("transformers"),
                "fla": package_version("flash-linear-attention"),
            },
        }
    )
    save_json(args.output, report)
    model = load_densek3_model(args.artifact_dir, device=device, dtype=torch.bfloat16)
    cases, batch_tokens = _case_tokens(model.config)

    direct_cases = list(report.get("direct_cases", []))
    _validate_resumed_cases(direct_cases, cases, integrated=False)
    for case in direct_cases:
        case.update(_review_direct_case(case))
    report.setdefault("resume", {})["direct_cases_reused"] = len(direct_cases)
    for prompt, _decode, tokens in cases[len(direct_cases) :]:
        direct_cases.append(_direct_case(model, tokens, prompt, device))
        report.update(
            {
                "last_completed_phase": "direct_cache_contracts",
                "direct_cases_completed": len(direct_cases),
                "direct_cases": direct_cases,
            }
        )
        save_json(args.output, report)

    integrated_cases = list(report.get("integrated_cases", []))
    _validate_resumed_cases(integrated_cases, cases, integrated=True)
    report.setdefault("resume", {})["integrated_cases_reused"] = len(integrated_cases)
    for prompt, decode, tokens in cases[len(integrated_cases) :]:
        integrated_cases.append(_integrated_case(model, tokens, prompt, decode, device))
        report.update(
            {
                "last_completed_phase": "integrated_error_propagation",
                "integrated_cases_completed": len(integrated_cases),
                "integrated_cases": integrated_cases,
            }
        )
        save_json(args.output, report)

    batch = _batch_isolation(model, batch_tokens, device)
    report.update(
        {
            "last_completed_phase": "batch_isolation",
            "batch_isolation": batch,
        }
    )
    save_json(args.output, report)

    direct_pass = all(case["direct_cache_contracts_passed"] for case in direct_cases)
    integrated_reproduced = all(case["formal_failure_reproduced"] for case in integrated_cases)
    seen_tokens_correct = all(case["seen_tokens_correct"] for case in integrated_cases)
    report.update(
        {
            "status": "PASS",
            "diagnostics_completed": True,
            "completed_at": now(),
            "last_completed_phase": "causal_classification",
            "causal_summary": {
                "direct_cache_contracts_passed": direct_pass,
                "integrated_failure_reproduced": integrated_reproduced,
                "seen_tokens_correct": seen_tokens_correct,
                "batch_isolation_persisted": True,
                "forced_fp32_projection_tails": [
                    {
                        "prompt_length": case["prompt_length"],
                        "sequence_length": case["sequence_length"],
                        "tails": case["projection_observations"]["forced_fp32_standard_tails"],
                    }
                    for case in direct_cases
                    if case["projection_observations"]["forced_fp32_standard_tails"]
                ],
                "classification": (
                    "SUPPORTED_NUMERICAL_PROPAGATION_NOT_CACHE_LAYOUT"
                    if direct_pass and integrated_reproduced and seen_tokens_correct
                    else "CACHE_SEMANTICS_OR_KERNEL_CONTROL_FAILED"
                ),
            },
            "formal_gate_modified": False,
            "p4_6_formal_status": "FAIL_UNCHANGED",
            "numerics_amendment_approved": False,
            "p5_allowed": False,
            "result_marker": "P4_06_CACHE_CAUSAL_DIAGNOSTICS=PASS",
        }
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse hash-verified completed cases from an existing partial diagnostic JSON",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.artifact_dir = args.artifact_dir.resolve()
    args.manifest_dir = args.manifest_dir.resolve()
    args.output = args.output.resolve()
    if args.resume:
        report = _load_resume_report(args.output)
    else:
        report = {
            "schema_version": 1,
            "stage": "P4.6-CACHE-DIAG",
            "status": "STARTING",
            "diagnostic_only": True,
            "diagnostics_completed": False,
            "formal_gate_modified": False,
            "p4_6_formal_status": "FAIL_UNCHANGED",
            "numerics_amendment_approved": False,
            "p5_allowed": False,
            "token_seed": TOKEN_SEED,
            "cases": [{"prompt": prompt, "decode": decode} for prompt, decode in CASES],
            "started_at": now(),
            "resume": {"enabled": False, "direct_cases_reused": 0, "integrated_cases_reused": 0},
            "result_marker": "P4_06_CACHE_CAUSAL_DIAGNOSTICS=RUNNING",
        }
    save_json(args.output, report)
    try:
        report = run(args, report)
    except Exception as exc:  # noqa: BLE001 - preserve every diagnostic failure envelope
        report.update(
            {
                "status": "EXCEPTION",
                "completed_at": now(),
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
                "result_marker": "P4_06_CACHE_CAUSAL_DIAGNOSTICS=EXCEPTION",
            }
        )
    save_json(args.output, report)
    print(report["result_marker"])
    print(f"P4_6_FORMAL_STATUS={report['p4_6_formal_status']}")
    print("P4_CACHE_NUMERICS_AMENDMENT_APPROVED=false")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
