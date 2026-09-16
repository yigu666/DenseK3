"""Run the Titan-only P7 true persistent latent-cache closeout."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from kda_sm75_backend import enable_sm75_fused_recurrent_fallback
from p7_latent_cache import (
    TitanP7HybridCache,
    disable_p7_latent_cache_runtime,
    enable_p7_latent_cache_runtime,
    reconstruct_kv_from_latent,
)

from densek3_core.evaluation.p4_runtime import load_densek3_model
from densek3_core.modeling.hybrid_cache import DenseK3HybridCache as ExpandedHybridCache
from densek3_core.recovery.p6_closeout import (
    P6_MLA_LAYERS,
    checkpoint_identity,
    sha256_file,
    verify_checkpoint_storage,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "titan/configs/p7-titan-latent-cache.yaml"
CACHE_CONTRACT = (
    ROOT / "titan/configs/p6-titan-fp16-cache-compat-2026-08-19-01.yaml"
)
CANDIDATE = (
    ROOT
    / "outputs/densek3-core/checkpoints/p6/"
    "densek3-4b-8mla-c3-qk-norm-calibrated-candidate"
)
P6_CLOSEOUT = (
    ROOT / "titan/manifests/reproduction/p6/p6-titan-compatibility-closeout.json"
)
PROMPTS = ROOT / "data/p4-smoke-prompts.jsonl"
REPORT_DIR = ROOT / "titan/manifests/reproduction/p7"
FINAL = REPORT_DIR / "p7-final-check.json"
RUNTIME_MANIFEST = REPORT_DIR / "p7-runtime-manifest.json"
MAX_START_TEMPERATURE_C = 75


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


def tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def distribution(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    decisions: bool = False,
) -> dict[str, Any]:
    if actual.shape != reference.shape:
        raise ValueError(
            f"Comparison shape mismatch: {tuple(actual.shape)} vs {tuple(reference.shape)}"
        )
    actual64 = actual.detach().double()
    reference64 = reference.detach().double()
    difference = actual64 - reference64
    absolute = difference.abs()
    failed = absolute > atol + rtol * reference64.abs()
    flat = absolute.reshape(-1).cpu()
    result: dict[str, Any] = {
        "numel": int(actual.numel()),
        "standard_allclose": bool(torch.allclose(actual, reference, rtol=rtol, atol=atol)),
        "exact": bool(torch.equal(actual, reference)),
        "finite": bool(torch.isfinite(actual).all() and torch.isfinite(reference).all()),
        "relative_l2": float(
            difference.norm().div(reference64.norm().clamp_min(1e-30)).item()
        ),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                actual64.reshape(1, -1),
                reference64.reshape(1, -1),
            ).item()
        ),
        "mean_abs": float(absolute.mean().item()),
        "p99_9_abs": float(torch.quantile(flat, 0.999).item()),
        "max_abs": float(absolute.max().item()),
        "failed_allclose_fraction": float(failed.double().mean().item()),
        "nan_count": int(torch.isnan(actual).sum().item() + torch.isnan(reference).sum().item()),
        "inf_count": int(torch.isinf(actual).sum().item() + torch.isinf(reference).sum().item()),
    }
    if decisions:
        reference_top1 = reference.argmax(dim=-1)
        actual_top1 = actual.argmax(dim=-1)
        reference_top10 = reference.topk(10, dim=-1).indices
        actual_top10 = actual.topk(10, dim=-1).indices
        overlaps = (
            reference_top10.unsqueeze(-1)
            .eq(actual_top10.unsqueeze(-2))
            .any(dim=-1)
            .sum(dim=-1)
        )
        result.update(
            {
                "argmax_agreement": float(
                    actual_top1.eq(reference_top1).double().mean().item()
                ),
                "top10_min_overlap": int(overlaps.min().item()),
                "top10_mean_overlap": float(overlaps.double().mean().item() / 10.0),
                "top10_histogram": {
                    str(key): value
                    for key, value in sorted(
                        Counter(int(item) for item in overlaps.reshape(-1).cpu()).items()
                    )
                },
            }
        )
    return result


def evaluate_compatibility(
    metrics: dict[str, Any],
    limits: dict[str, Any],
    *,
    decisions: bool,
) -> dict[str, Any]:
    checks = {
        "relative_l2": metrics["relative_l2"] <= limits["relative_l2_max"],
        "cosine": metrics["cosine"] >= limits["cosine_min"],
        "mean_abs": metrics["mean_abs"] <= limits["mean_abs_max"],
        "p99_9_abs": metrics["p99_9_abs"] <= limits["p99_9_abs_max"],
        "failed_allclose_fraction": metrics["failed_allclose_fraction"]
        <= limits["failed_allclose_fraction_max"],
        "max_abs": metrics["max_abs"] <= limits["max_abs_max"],
        "nan_count": metrics["nan_count"] <= limits["nan_count_max"],
        "inf_count": metrics["inf_count"] <= limits["inf_count_max"],
    }
    if decisions:
        checks.update(
            {
                "argmax": metrics["argmax_agreement"] == 1.0,
                "top10": metrics["top10_min_overlap"]
                >= limits["top10_min_overlap"],
            }
        )
    standard = bool(metrics["standard_allclose"])
    return {
        "standard_allclose": standard,
        "fallback_evaluated": not standard,
        "checks": checks,
        "passed": standard or all(checks.values()),
        "pass_mode": "STANDARD_ALLCLOSE" if standard else "EXISTING_TITAN_FP16_ENVELOPE",
    }


def gpu_start_state() -> dict[str, Any]:
    import subprocess

    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,temperature.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    records = [line.strip() for line in processes.stdout.splitlines() if line.strip()]
    fields = [item.strip() for item in gpu.stdout.strip().split(",")]
    visible = gpu.returncode == 0 and len(fields) == 4
    temperature = int(fields[1]) if visible else None
    return {
        "visible": visible,
        "description": gpu.stdout.strip(),
        "error": gpu.stderr.strip(),
        "compute_processes": records,
        "temperature_c": temperature,
        "idle": visible
        and not records
        and temperature is not None
        and temperature <= MAX_START_TEMPERATURE_C,
    }


@torch.inference_mode()
def incremental_logits(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    prefill: int,
    cache: Any,
) -> tuple[torch.Tensor, Any, dict[str, float]]:
    device = next(model.parameters()).device
    tokens = tokens.to(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    prefix = model(
        tokens[:, :prefill],
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )
    cache = prefix.past_key_values
    prefill_done = time.perf_counter()
    pieces = []
    for index in range(prefill, tokens.shape[1]):
        output = model(
            tokens[:, index : index + 1],
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = output.past_key_values
        pieces.append(output.logits.detach().cpu())
    torch.cuda.synchronize(device)
    finished = time.perf_counter()
    logits = torch.cat(pieces, dim=1)
    return logits, cache, {
        "prefill_seconds": prefill_done - started,
        "decode_seconds": finished - prefill_done,
        "decode_ms_per_token": (finished - prefill_done)
        * 1000
        / max(tokens.shape[1] - prefill, 1),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def direct_mixer_incremental(
    mixer: torch.nn.Module,
    hidden: torch.Tensor,
    prefill: int,
    cache: Any,
) -> tuple[torch.Tensor, Any]:
    batch, total, _ = hidden.shape
    device = hidden.device
    positions = torch.arange(total, device=device).unsqueeze(0).expand(batch, -1)
    mixer(
        hidden[:, :prefill],
        positions[:, :prefill],
        torch.ones((batch, prefill), device=device),
        cache,
        True,
    )
    cache.advance(prefill)
    pieces = []
    for index in range(prefill, total):
        pieces.append(
            mixer(
                hidden[:, index : index + 1],
                positions[:, index : index + 1],
                torch.ones((batch, index + 1), device=device),
                cache,
                True,
            ).detach()
        )
        cache.advance(1)
    return torch.cat(pieces, dim=1), cache


def capture_mla_inputs(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    layer_indices: tuple[int, ...],
) -> dict[int, torch.Tensor]:
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer_idx in layer_indices:
        mixer = model.model.layers[layer_idx].self_attn

        def hook(_module, args, index=layer_idx):
            captured[index] = args[0].detach()

        handles.append(mixer.register_forward_pre_hook(hook))
    try:
        model(tokens, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(layer_indices):
        raise RuntimeError("Failed to capture all frozen P6 MLA inputs")
    return captured


def single_layer_probe(
    model: torch.nn.Module,
    frozen: dict[str, Any],
    limits: dict[str, Any],
) -> dict[str, Any]:
    layers = tuple(int(index) for index in frozen["architecture"]["probe_layers"])
    cases = [tuple(int(value) for value in item) for item in frozen["execution"]["single_layer_cases"]]
    maximum_total = max(prefill + decode for prefill, decode in cases)
    generator = torch.Generator(device="cpu").manual_seed(20260819)
    tokens = torch.randint(1, model.config.vocab_size, (1, maximum_total), generator=generator).cuda()
    captured = capture_mla_inputs(model, tokens, layers)
    records = []
    for layer_idx in layers:
        mixer = model.model.layers[layer_idx].self_attn
        hidden_all = captured[layer_idx]
        for prefill, decode in cases:
            total = prefill + decode
            hidden = hidden_all[:, :total]
            positions = torch.arange(total, device=hidden.device).unsqueeze(0)
            full = mixer(
                hidden,
                positions,
                torch.ones((1, total), device=hidden.device),
                None,
                False,
            ).detach()
            expanded = ExpandedHybridCache(model.config)
            expanded_output, expanded = direct_mixer_incremental(
                mixer,
                hidden,
                prefill,
                expanded,
            )
            expanded_prefix_key = expanded.gqa_layer(layer_idx).key[:, :, :prefill].detach()
            expanded_prefix_value = expanded.gqa_layer(layer_idx).value[:, :, :prefill].detach()

            enable_p7_latent_cache_runtime((layer_idx,))
            try:
                latent = TitanP7HybridCache(model.config)
                latent_output, latent = direct_mixer_incremental(
                    mixer,
                    hidden,
                    prefill,
                    latent,
                )
                latent_prefix = latent.latent_layer(layer_idx).latent[:, :prefill]
                reconstructed_key, reconstructed_value = reconstruct_kv_from_latent(
                    mixer,
                    latent_prefix,
                )
                branch_source = TitanP7HybridCache(model.config)
                mixer(
                    hidden[:, :prefill],
                    positions[:, :prefill],
                    torch.ones((1, prefill), device=hidden.device),
                    branch_source,
                    True,
                )
                branch_source.advance(prefill)
                branch_a = branch_source.clone()
                branch_b = branch_source.clone()
                branch_input = hidden[:, prefill : prefill + 1]
                branch_position = positions[:, prefill : prefill + 1]
                branch_mask = torch.ones((1, prefill + 1), device=hidden.device)
                branch_a_output = mixer(
                    branch_input,
                    branch_position,
                    branch_mask,
                    branch_a,
                    True,
                )
                branch_b_output = mixer(
                    branch_input,
                    branch_position,
                    branch_mask,
                    branch_b,
                    True,
                )
                branch_a.advance(1)
                branch_b.advance(1)
            finally:
                disable_p7_latent_cache_runtime()

            output_metrics = distribution(
                latent_output,
                expanded_output,
                rtol=0.006,
                atol=0.006,
            )
            output_gate = evaluate_compatibility(
                output_metrics,
                limits,
                decisions=False,
            )
            key_metrics = distribution(
                reconstructed_key,
                expanded_prefix_key,
                rtol=0.006,
                atol=0.006,
            )
            value_metrics = distribution(
                reconstructed_value,
                expanded_prefix_value,
                rtol=0.006,
                atol=0.006,
            )
            no_cache_metrics = distribution(
                latent_output,
                full[:, prefill:],
                rtol=0.006,
                atol=0.006,
            )
            branch_exact = bool(torch.equal(branch_a_output, branch_b_output))
            source_unchanged = branch_source.seen_tokens == prefill and branch_source.latent_layer(
                layer_idx
            ).sequence_length == prefill
            cache_contract = latent.latent_contract()
            checks = {
                "expanded_vs_latent": output_gate["passed"],
                "reconstructed_key": key_metrics["standard_allclose"],
                "reconstructed_value": value_metrics["standard_allclose"],
                "branch_exact": branch_exact,
                "branch_source_unchanged": source_unchanged,
                "seen_tokens": latent.seen_tokens == total,
                "latent_length": latent.latent_layer(layer_idx).sequence_length == total,
                "persistent_expanded_k_zero": cache_contract["bytes"][
                    "persistent_expanded_k_numel"
                ]
                == 0,
                "persistent_expanded_v_zero": cache_contract["bytes"][
                    "persistent_expanded_v_numel"
                ]
                == 0,
                "latent_finite": all(cache_contract["latent_finite"].values()),
                "finite": output_metrics["finite"],
            }
            records.append(
                {
                    "layer": layer_idx,
                    "prefill": prefill,
                    "decode": decode,
                    "status": "PASS" if all(checks.values()) else "FAIL",
                    "checks": checks,
                    "expanded_vs_latent": output_metrics,
                    "compatibility_gate": output_gate,
                    "latent_vs_no_cache": no_cache_metrics,
                    "reconstructed_key": key_metrics,
                    "reconstructed_value": value_metrics,
                    "branch": {
                        "output_exact": branch_exact,
                        "source_unchanged": source_unchanged,
                        "branch_seen_tokens": [branch_a.seen_tokens, branch_b.seen_tokens],
                    },
                    "cache_contract": cache_contract,
                }
            )
    report = {
        "stage": "TITAN_P7_2_SINGLE_LAYER_SEMANTIC_PROBE",
        "status": "PASS" if all(item["status"] == "PASS" for item in records) else "FAIL",
        "cases": records,
        "heldout_accessed": False,
        "training_performed": False,
    }
    atomic_json(REPORT_DIR / "p7-2-single-layer-probe.json", report)
    return report


def scaleout_cases(
    model: torch.nn.Module,
    frozen: dict[str, Any],
    limits: dict[str, Any],
) -> dict[str, Any]:
    enable_p7_latent_cache_runtime(P6_MLA_LAYERS)
    records = []
    try:
        for item in frozen["execution"]["full_model_cases"]:
            batch = int(item["batch"])
            prefill = int(item["prefill"])
            decode = int(item["decode"])
            generator = torch.Generator(device="cpu").manual_seed(
                20260819 + batch * 100_000 + prefill * 100 + decode
            )
            tokens = torch.randint(
                1,
                model.config.vocab_size,
                (batch, prefill + decode),
                generator=generator,
            )
            expanded_logits, expanded_cache, expanded_performance = incremental_logits(
                model,
                tokens,
                prefill,
                ExpandedHybridCache(model.config),
            )
            expanded_cache_bytes = expanded_cache.cache_bytes()
            del expanded_cache
            gc.collect()
            torch.cuda.empty_cache()
            latent_logits, latent_cache, latent_performance = incremental_logits(
                model,
                tokens,
                prefill,
                TitanP7HybridCache(model.config),
            )
            metrics = distribution(
                latent_logits,
                expanded_logits,
                rtol=0.006,
                atol=0.006,
                decisions=True,
            )
            gate = evaluate_compatibility(metrics, limits, decisions=True)
            contract = latent_cache.latent_contract()
            checks = {
                "compatibility": gate["passed"],
                "seen_tokens": latent_cache.seen_tokens == prefill + decode,
                "latent_layers": contract["latent_cache_layers"] == list(P6_MLA_LAYERS),
                "latent_dims": set(contract["latent_dims"].values()) == {512},
                "latent_lengths": all(
                    shape == [batch, prefill + decode, 512]
                    for shape in contract["latent_shapes"].values()
                ),
                "persistent_expanded_k_zero": contract["bytes"][
                    "persistent_expanded_k_numel"
                ]
                == 0,
                "persistent_expanded_v_zero": contract["bytes"][
                    "persistent_expanded_v_numel"
                ]
                == 0,
                "latent_finite": all(contract["latent_finite"].values()),
                "finite": metrics["finite"],
            }
            records.append(
                {
                    "batch": batch,
                    "prefill": prefill,
                    "decode": decode,
                    "status": "PASS" if all(checks.values()) else "FAIL",
                    "checks": checks,
                    "metrics": metrics,
                    "compatibility_gate": gate,
                    "expanded_performance": expanded_performance,
                    "latent_performance": latent_performance,
                    "expanded_cache_bytes": expanded_cache_bytes,
                    "latent_cache_contract": contract,
                }
            )
            del latent_cache
            gc.collect()
            torch.cuda.empty_cache()
            print(
                f"TITAN_P7_SCALEOUT B={batch} P={prefill} D={decode} "
                f"STATUS={records[-1]['status']}",
                flush=True,
            )
    finally:
        disable_p7_latent_cache_runtime()
    report = {
        "stage": "TITAN_P7_3_EIGHT_LAYER_SCALEOUT",
        "status": "PASS" if all(item["status"] == "PASS" for item in records) else "FAIL",
        "cases": records,
        "heldout_accessed": False,
        "training_performed": False,
    }
    atomic_json(REPORT_DIR / "p7-3-eight-layer-scaleout.json", report)
    return report


@torch.inference_mode()
def generation_replay(
    model: torch.nn.Module,
    tokenizer: Any,
    limits: dict[str, Any],
) -> dict[str, Any]:
    prompts = [json.loads(line) for line in PROMPTS.read_text(encoding="utf-8").splitlines() if line]
    enable_p7_latent_cache_runtime(P6_MLA_LAYERS)
    records = []
    try:
        for item in prompts:
            input_ids = tokenizer(item["prompt"], return_tensors="pt")["input_ids"].cuda()

            def greedy(
                cache: Any,
                current_input_ids: torch.Tensor = input_ids,
            ) -> tuple[list[int], torch.Tensor]:
                output = model(
                    current_input_ids,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
                current_cache = output.past_key_values
                logits = [output.logits[:, -1:].detach().cpu()]
                generated = []
                token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
                for _ in range(8):
                    generated.append(int(token.item()))
                    output = model(
                        token,
                        past_key_values=current_cache,
                        use_cache=True,
                        return_dict=True,
                    )
                    current_cache = output.past_key_values
                    logits.append(output.logits[:, -1:].detach().cpu())
                    token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
                return generated, torch.cat(logits, dim=1)

            expanded_tokens, expanded_logits = greedy(ExpandedHybridCache(model.config))
            latent_tokens, latent_logits = greedy(TitanP7HybridCache(model.config))
            metrics = distribution(
                latent_logits,
                expanded_logits,
                rtol=0.006,
                atol=0.006,
                decisions=True,
            )
            gate = evaluate_compatibility(metrics, limits, decisions=True)
            records.append(
                {
                    "category": item.get("category", "unknown"),
                    "expanded_tokens": expanded_tokens,
                    "latent_tokens": latent_tokens,
                    "tokens_exact": expanded_tokens == latent_tokens,
                    "metrics": metrics,
                    "compatibility_gate": gate,
                    "passed": expanded_tokens == latent_tokens and gate["passed"],
                }
            )
    finally:
        disable_p7_latent_cache_runtime()
    return {
        "status": "PASS" if all(item["passed"] for item in records) else "FAIL",
        "categories": records,
    }


@torch.inference_mode()
def memory_gate(model: torch.nn.Module, frozen: dict[str, Any]) -> dict[str, Any]:
    contexts = [int(value) for value in frozen["execution"]["memory_contexts"]]
    chunk_size = int(frozen["execution"]["memory_prefill_chunk_size"])
    if chunk_size <= 0:
        raise ValueError("P7 memory prefill chunk size must be positive")
    expected_expanded = int(
        frozen["memory"]["expected_expanded_mla_bytes_per_token"]
    )
    expected_latent = int(frozen["memory"]["expected_latent_mla_bytes_per_token"])
    enable_p7_latent_cache_runtime(P6_MLA_LAYERS)
    records = []
    try:
        for context in contexts:
            generator = torch.Generator(device="cpu").manual_seed(20260819 + context)
            tokens = torch.randint(
                1,
                model.config.vocab_size,
                (1, context),
                generator=generator,
            ).cuda()

            def prefill(
                cache: Any,
                current_tokens: torch.Tensor = tokens,
            ) -> tuple[Any, dict[str, float]]:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                chunks = 0
                for start in range(0, current_tokens.shape[1], chunk_size):
                    output = model(
                        current_tokens[:, start : start + chunk_size],
                        past_key_values=cache,
                        use_cache=True,
                        return_dict=True,
                    )
                    cache = output.past_key_values
                    del output
                    chunks += 1
                torch.cuda.synchronize()
                return cache, {
                    "seconds": time.perf_counter() - started,
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "prefill_chunk_size": chunk_size,
                    "prefill_chunks": chunks,
                }

            expanded_cache, expanded_performance = prefill(
                ExpandedHybridCache(model.config)
            )
            expanded_bytes = expanded_cache.cache_bytes()["gqa"]
            del expanded_cache
            gc.collect()
            torch.cuda.empty_cache()
            latent_cache, latent_performance = prefill(TitanP7HybridCache(model.config))
            latent_contract = latent_cache.latent_contract()
            latent_bytes = latent_contract["bytes"]["latent"]
            reduction = 1.0 - latent_bytes / expanded_bytes
            checks = {
                "expanded_linear_exact": expanded_bytes == expected_expanded * context,
                "latent_linear_exact": latent_bytes == expected_latent * context,
                "reduction_exact": math.isclose(reduction, 0.9375, abs_tol=1e-12),
                "persistent_expanded_k_zero": latent_contract["bytes"][
                    "persistent_expanded_k_numel"
                ]
                == 0,
                "persistent_expanded_v_zero": latent_contract["bytes"][
                    "persistent_expanded_v_numel"
                ]
                == 0,
                "latent_finite": all(latent_contract["latent_finite"].values()),
                "seen_tokens": latent_cache.seen_tokens == context,
            }
            records.append(
                {
                    "context": context,
                    "status": "PASS" if all(checks.values()) else "FAIL",
                    "checks": checks,
                    "expanded_mla_bytes": expanded_bytes,
                    "latent_mla_bytes": latent_bytes,
                    "reduction": reduction,
                    "expanded_performance": expanded_performance,
                    "latent_performance": latent_performance,
                }
            )
            del latent_cache
            gc.collect()
            torch.cuda.empty_cache()
            print(f"TITAN_P7_MEMORY T={context} STATUS={records[-1]['status']}", flush=True)
    finally:
        disable_p7_latent_cache_runtime()
    report = {
        "stage": "TITAN_P7_4_CACHE_MEMORY",
        "status": "PASS" if all(item["status"] == "PASS" for item in records) else "FAIL",
        "contexts": records,
        "theory": {
            "p6_expanded_bytes_per_token": expected_expanded,
            "p7_latent_bytes_per_token": expected_latent,
            "p6_to_p7_reduction": 0.9375,
            "p5_gqa_bytes_per_token": frozen["memory"][
                "expected_p5_gqa_bytes_per_token"
            ],
            "p5_to_p7_reduction": frozen["memory"][
                "expected_p5_gqa_to_latent_reduction"
            ],
        },
        "optional_8192": {
            "status": "SKIPPED_OPTIONAL",
            "reason": "reference SDPA and temporary K/V remain quadratic; 2048 proves linear persistent-cache accounting",
        },
    }
    atomic_json(REPORT_DIR / "p7-4-cache-memory.json", report)
    return report


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    state = gpu_start_state()
    envelope: dict[str, Any] = {
        "stage": "TITAN_P7_TRUE_LATENT_CACHE",
        "status": "RUNNING",
        "started_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "gpu": state,
        "p7_formal_allowed": False,
        "p7_canonical_frozen": False,
        "p8_titan_migration_allowed": False,
        "p8_canonical_allowed": False,
        "training_performed": False,
        "heldout_accessed": False,
        "weights_modified": False,
        "canonical_source_modified": False,
    }
    atomic_json(FINAL, envelope)
    try:
        if not state["idle"]:
            envelope["status"] = "DEFERRED_GPU_NOT_IDLE"
            atomic_json(FINAL, envelope)
            print("TITAN_P7_TRUE_LATENT_CACHE=DEFERRED_GPU_NOT_IDLE")
            return 3
        frozen = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        cache_contract = yaml.safe_load(CACHE_CONTRACT.read_text(encoding="utf-8"))
        limits = cache_contract["evaluation"]["fallback_requires_all"]
        p6_closeout = load_json(P6_CLOSEOUT)
        candidate_manifest = load_json(CANDIDATE / "p6-candidate-manifest.json")
        candidate_config = load_json(CANDIDATE / "config.json")
        identity_before = checkpoint_identity(CANDIDATE)
        storage_before = verify_checkpoint_storage(CANDIDATE, candidate_manifest)
        baseline_checks = {
            "p6_titan_compatibility": p6_closeout["status"] == "PASS",
            "p6_formal_pending": p6_closeout["formal_p6_status"]
            == "AWAITING_NATIVE_BF16_REPLAY",
            "storage": storage_before["passed"],
            "tensor_count": storage_before["tensor_count"] == 562,
            "parameter_count": storage_before["parameter_count"] == 4_226_431_232,
            "architecture": candidate_manifest["architecture"]
            == "24_KDA_PLUS_8_STRICT_NOPE_MLA_QK_NORM_PRESERVED",
            "mla_layers": tuple(
                index
                for index, mixer_type in enumerate(candidate_config["mixer_types"])
                if mixer_type == "mla"
            )
            == P6_MLA_LAYERS,
            "rank": candidate_config["mla_kv_lora_rank"] == 512,
            "qk_norm": candidate_config["mla_qk_norm"] is True,
        }
        if not all(baseline_checks.values()):
            raise ValueError(f"P7.0 frozen P6 baseline failed: {baseline_checks}")
        kda_override = enable_sm75_fused_recurrent_fallback()
        from transformers import AutoTokenizer

        tokenizer_source = (
            ROOT / "outputs/densek3-core/checkpoints/p5/densek3-4b-k3-decay-silu-i1"
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=False)
        model = load_densek3_model(CANDIDATE, device="cuda", dtype=torch.float16)
        model.eval()
        generator = torch.Generator(device="cpu").manual_seed(20260819)
        baseline_tokens = torch.randint(
            1,
            model.config.vocab_size,
            (1, 15),
            generator=generator,
        )
        full = model(baseline_tokens.cuda(), use_cache=False, return_dict=True).logits[:, 7:].cpu()
        expanded_logits, expanded_cache, _ = incremental_logits(
            model,
            baseline_tokens,
            7,
            ExpandedHybridCache(model.config),
        )
        baseline_metrics = distribution(
            expanded_logits,
            full,
            rtol=0.006,
            atol=0.006,
            decisions=True,
        )
        baseline_gate = evaluate_compatibility(
            baseline_metrics,
            limits,
            decisions=True,
        )
        baseline_checks.update(
            {
                "expanded_cache": expanded_cache.seen_tokens == 15,
                "expanded_compatibility": baseline_gate["passed"],
                "finite": baseline_metrics["finite"],
            }
        )
        baseline = {
            "stage": "TITAN_P7_0_FROZEN_P6_BASELINE",
            "status": "PASS" if all(baseline_checks.values()) else "FAIL",
            "checks": baseline_checks,
            "checkpoint_identity": identity_before,
            "storage": storage_before,
            "expanded_cache_metrics": baseline_metrics,
            "expanded_cache_gate": baseline_gate,
            "kda_backend_override": kda_override,
        }
        atomic_json(REPORT_DIR / "p7-0-baseline.json", baseline)
        if baseline["status"] != "PASS":
            raise ValueError("P7.0 baseline did not pass")
        print("TITAN_P7_0_BASELINE=PASS", flush=True)

        single = single_layer_probe(model, frozen, limits)
        print(f"TITAN_P7_2_SINGLE_LAYER_PROBE={single['status']}", flush=True)
        if single["status"] != "PASS":
            raise ValueError("P7.2 single-layer semantic probe failed")

        scaleout = scaleout_cases(model, frozen, limits)
        print(f"TITAN_P7_3_EIGHT_LAYER_SCALEOUT={scaleout['status']}", flush=True)
        if scaleout["status"] != "PASS":
            raise ValueError("P7.3 full-model scaleout failed")

        generation = generation_replay(model, tokenizer, limits)
        atomic_json(REPORT_DIR / "p7-3-generation.json", generation)
        print(f"TITAN_P7_GENERATION={generation['status']}", flush=True)
        if generation["status"] != "PASS":
            raise ValueError("P7 generation replay failed")

        memory = memory_gate(model, frozen)
        print(f"TITAN_P7_4_CACHE_MEMORY={memory['status']}", flush=True)
        if memory["status"] != "PASS":
            raise ValueError("P7 memory Gate failed")

        enable_p7_latent_cache_runtime(P6_MLA_LAYERS)
        reload_tokens = baseline_tokens[:, :8].cuda()
        initial_reload = model(
            reload_tokens,
            past_key_values=TitanP7HybridCache(model.config),
            use_cache=True,
            return_dict=True,
        )
        initial_observation = {
            "logits_sha256": tensor_sha256(initial_reload.logits[:, -1]),
            "seen_tokens": initial_reload.past_key_values.seen_tokens,
            "latent_contract": initial_reload.past_key_values.latent_contract(),
        }
        disable_p7_latent_cache_runtime()
        del initial_reload, model
        gc.collect()
        torch.cuda.empty_cache()
        model = load_densek3_model(CANDIDATE, device="cuda", dtype=torch.float16)
        model.eval()
        enable_p7_latent_cache_runtime(P6_MLA_LAYERS)
        reloaded = model(
            reload_tokens,
            past_key_values=TitanP7HybridCache(model.config),
            use_cache=True,
            return_dict=True,
        )
        reload_observation = {
            "logits_sha256": tensor_sha256(reloaded.logits[:, -1]),
            "seen_tokens": reloaded.past_key_values.seen_tokens,
            "latent_contract": reloaded.past_key_values.latent_contract(),
        }
        disable_p7_latent_cache_runtime()
        identity_after = checkpoint_identity(CANDIDATE)
        storage_after = verify_checkpoint_storage(CANDIDATE, candidate_manifest)
        provenance_checks = {
            "checkpoint_identity_exact": identity_before == identity_after,
            "all_tensor_hashes_exact": storage_after["passed"],
            "tensor_count": storage_after["tensor_count"] == 562,
            "parameter_count": storage_after["parameter_count"] == 4_226_431_232,
            "reload_exact": initial_observation == reload_observation,
            "config_still_expanded_reference": model.config.mla_cache_mode
            == "expanded_reference",
            "runtime_manifest_only": True,
        }
        runtime_manifest = {
            "stage": "TITAN_P7_6_RUNTIME_PROVENANCE",
            "status": "PASS" if all(provenance_checks.values()) else "FAIL",
            "source_checkpoint": str(CANDIDATE),
            "source_checkpoint_identity": identity_before,
            "source_candidate_manifest_sha256": sha256_file(
                CANDIDATE / "p6-candidate-manifest.json"
            ),
            "p7_config_sha256": sha256_file(CONFIG),
            "cache_implementation_sha256": sha256_file(
                ROOT / "titan/scripts/p7_latent_cache.py"
            ),
            "runner_sha256": sha256_file(Path(__file__)),
            "checks": provenance_checks,
            "initial_observation": initial_observation,
            "reload_observation": reload_observation,
            "weights_copied": False,
            "weights_modified": False,
            "canonical_source_modified": False,
            "training_performed": False,
            "formal_bf16_claim": False,
        }
        atomic_json(RUNTIME_MANIFEST, runtime_manifest)
        if runtime_manifest["status"] != "PASS":
            raise ValueError("P7 runtime provenance failed")

        final_checks = {
            "p7_0": baseline["status"] == "PASS",
            "p7_2": single["status"] == "PASS",
            "p7_3": scaleout["status"] == "PASS",
            "generation": generation["status"] == "PASS",
            "p7_4": memory["status"] == "PASS",
            "p7_6": runtime_manifest["status"] == "PASS",
            "weights_unchanged": provenance_checks["checkpoint_identity_exact"]
            and provenance_checks["all_tensor_hashes_exact"],
            "persistent_expanded_kv_zero": all(
                case["latent_cache_contract"]["bytes"][
                    "persistent_expanded_k_numel"
                ]
                == 0
                and case["latent_cache_contract"]["bytes"][
                    "persistent_expanded_v_numel"
                ]
                == 0
                for case in scaleout["cases"]
            ),
            "formal_bf16_not_claimed": True,
        }
        envelope.update(
            {
                "status": "PASS" if all(final_checks.values()) else "FAIL",
                "checks": final_checks,
                "compatibility_pass_mode": "EXISTING_TITAN_SM75_FP16_SEMANTICS",
                "p7_formal_status": "PENDING_NATIVE_BF16_REPLAY",
                "p7_formal_allowed": False,
                "p7_canonical_frozen": False,
                "p8_titan_migration_allowed": all(final_checks.values()),
                "p8_canonical_allowed": False,
                "completed_at": datetime.now(timezone.utc).astimezone().isoformat(),
                "reports": {
                    "baseline": str(REPORT_DIR / "p7-0-baseline.json"),
                    "single_layer": str(REPORT_DIR / "p7-2-single-layer-probe.json"),
                    "scaleout": str(REPORT_DIR / "p7-3-eight-layer-scaleout.json"),
                    "generation": str(REPORT_DIR / "p7-3-generation.json"),
                    "memory": str(REPORT_DIR / "p7-4-cache-memory.json"),
                    "provenance": str(RUNTIME_MANIFEST),
                },
            }
        )
        atomic_json(FINAL, envelope)
        print(f"TITAN_P7_TRUE_LATENT_CACHE={envelope['status']}")
        print(f"TITAN_P7_FP16_COMPATIBILITY={envelope['status']}")
        print("P7_CANONICAL_BF16_REPLAY=PENDING")
        print("P7_CANONICAL_FROZEN=false")
        print(
            "P8_TITAN_MIGRATION_ALLOWED="
            f"{str(envelope['p8_titan_migration_allowed']).lower()}"
        )
        return 0 if envelope["status"] == "PASS" else 1
    except Exception as error:  # noqa: BLE001
        disable_p7_latent_cache_runtime()
        envelope.update(
            {
                "status": "ERROR",
                "exception_type": type(error).__name__,
                "exception": str(error),
                "completed_at": datetime.now(timezone.utc).astimezone().isoformat(),
                "p8_titan_migration_allowed": False,
            }
        )
        atomic_json(FINAL, envelope)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
