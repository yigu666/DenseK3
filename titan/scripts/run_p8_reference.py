"""Run Titan P8.0-P8.2 Block AttnRes reference and zero-shock gates."""

from __future__ import annotations

import gc
import json
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
TITAN = ROOT / "titan"
sys.path.insert(0, str(TITAN))

from attnres import (  # noqa: E402
    BLOCK_BOUNDARY_LAYERS,
    BlockAttnResReference,
    disable_block_attnres_runtime,
    enable_block_attnres_runtime,
)
from attnres.contracts import (  # noqa: E402
    SOURCE_PARAMETER_COUNT,
    SOURCE_TENSOR_COUNT,
)
from attnres.metrics import tensor_metrics  # noqa: E402
from kda_sm75_backend import enable_sm75_fused_recurrent_fallback  # noqa: E402
from p7_latent_cache import (  # noqa: E402
    TitanP7HybridCache,
    disable_p7_latent_cache_runtime,
    enable_p7_latent_cache_runtime,
)
from run_p7 import (  # noqa: E402
    distribution,
    evaluate_compatibility,
    incremental_logits,
)

from densek3_core.evaluation.p4_runtime import load_densek3_model  # noqa: E402
from densek3_core.modeling.hybrid_cache import (  # noqa: E402
    DenseK3HybridCache as ExpandedHybridCache,
)
from densek3_core.recovery.p5_corpus import (  # noqa: E402
    PackedTokenDataset,
    verify_frozen_corpus,
)
from densek3_core.recovery.p6_closeout import (  # noqa: E402
    checkpoint_identity,
    sha256_file,
    verify_checkpoint_storage,
)

CONFIG = TITAN / "configs/p8-titan-block-attnres.yaml"
CACHE_CONTRACT = TITAN / "configs/p6-titan-fp16-cache-compat-2026-08-19-01.yaml"
CANDIDATE = (
    ROOT
    / "outputs/densek3-core/checkpoints/p6/"
    "densek3-4b-8mla-c3-qk-norm-calibrated-candidate"
)
P7_FINAL = TITAN / "manifests/reproduction/p7/p7-final-check.json"
P7_RUNTIME = TITAN / "manifests/reproduction/p7/p7-runtime-manifest.json"
DATA = ROOT / "data/p5-probe"
PROMPTS = ROOT / "data/p4-smoke-prompts.jsonl"
REPORT_DIR = TITAN / "manifests/reproduction/p8"
FINAL = REPORT_DIR / "p8-reference-final-check.json"
MAX_START_TEMPERATURE_C = 75


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
    fields = [item.strip() for item in gpu.stdout.strip().split(",")]
    records = [line.strip() for line in processes.stdout.splitlines() if line.strip()]
    visible = gpu.returncode == 0 and len(fields) == 4
    temperature = int(fields[1]) if visible else None
    return {
        "visible": visible,
        "description": gpu.stdout.strip(),
        "compute_processes": records,
        "temperature_c": temperature,
        "idle": visible
        and not records
        and temperature is not None
        and temperature <= MAX_START_TEMPERATURE_C,
    }


@torch.inference_mode()
def corpus_ce(
    model: torch.nn.Module,
    dataset: PackedTokenDataset,
    *,
    label: str,
) -> float:
    if dataset.split != "dev" or len(dataset) != 97:
        raise ValueError("P8 requires the frozen 97-sequence dev split")
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0
    for index in range(len(dataset)):
        tokens = torch.from_numpy(dataset[index].copy()).unsqueeze(0).to(device)
        hidden = model.model(
            input_ids=tokens,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state[:, :-1]
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
        if (index + 1) % 16 == 0 or index + 1 == len(dataset):
            print(f"TITAN_P8_{label}_DEV_PROGRESS={index + 1}/97", flush=True)
    return total_loss / total_tokens


def capture_standard_trace(
    model: torch.nn.Module,
    tokens: torch.Tensor,
) -> tuple[torch.Tensor, dict[int, dict[str, torch.Tensor]]]:
    records: dict[int, dict[str, torch.Tensor]] = {
        index: {} for index in range(len(model.model.layers))
    }
    handles = []
    for index, layer in enumerate(model.model.layers):
        handles.append(
            layer.input_layernorm.register_forward_pre_hook(
                lambda _module, args, layer_index=index: records[layer_index].__setitem__(
                    "pre_mixer", args[0].detach()
                )
            )
        )
        handles.append(
            layer.self_attn.register_forward_hook(
                lambda _module, _args, output, layer_index=index: records[layer_index].__setitem__(
                    "mixer_output", output.detach()
                )
            )
        )
        handles.append(
            layer.post_attention_layernorm.register_forward_pre_hook(
                lambda _module, args, layer_index=index: records[layer_index].__setitem__(
                    "pre_mlp", args[0].detach()
                )
            )
        )
        handles.append(
            layer.mlp.register_forward_hook(
                lambda _module, _args, output, layer_index=index: records[layer_index].__setitem__(
                    "mlp_output", output.detach()
                )
            )
        )
        handles.append(
            layer.register_forward_hook(
                lambda _module, _args, output, layer_index=index: records[layer_index].__setitem__(
                    "post_mlp", output.detach()
                )
            )
        )
    try:
        output = model(tokens, use_cache=False, return_dict=True).logits.detach()
    finally:
        for handle in handles:
            handle.remove()
    return output, records


@torch.inference_mode()
def probe_runtime(
    model: torch.nn.Module,
    runtime: BlockAttnResReference,
    lengths: list[int],
    *,
    seed: int,
    label: str,
) -> dict[str, Any]:
    records = []
    device = next(model.parameters()).device
    for length in lengths:
        generator = torch.Generator(device="cpu").manual_seed(seed + length)
        tokens = torch.randint(
            1,
            model.config.vocab_size,
            (1, length),
            generator=generator,
        ).to(device)
        standard_logits, standard_trace = capture_standard_trace(model, tokens)
        runtime.clear_diagnostics()
        runtime.capture_diagnostics = True
        runtime.capture_tensor_metrics = True
        runtime.capture_activations = True
        enable_block_attnres_runtime(model, runtime)
        try:
            actual_logits = model(tokens, use_cache=False, return_dict=True).logits.detach()
        finally:
            disable_block_attnres_runtime(model)
            runtime.capture_diagnostics = False
            runtime.capture_tensor_metrics = False
            runtime.capture_activations = False
        activation = {item["site"]: item for item in runtime.activation_records}
        layer_metrics = []
        for layer_index in range(len(model.model.layers)):
            mixer = activation[layer_index * 2]
            mlp = activation[layer_index * 2 + 1]
            layer_metrics.append(
                {
                    "layer": layer_index,
                    "pre_mixer": tensor_metrics(
                        mixer["read"], standard_trace[layer_index]["pre_mixer"]
                    ),
                    "mixer_output": tensor_metrics(
                        mixer["branch_output"],
                        standard_trace[layer_index]["mixer_output"],
                    ),
                    "post_mixer": tensor_metrics(
                        mixer["post_residual"], standard_trace[layer_index]["pre_mlp"]
                    ),
                    "pre_mlp": tensor_metrics(
                        mlp["read"], standard_trace[layer_index]["pre_mlp"]
                    ),
                    "mlp_output": tensor_metrics(
                        mlp["branch_output"], standard_trace[layer_index]["mlp_output"]
                    ),
                    "post_mlp": tensor_metrics(
                        mlp["post_residual"], standard_trace[layer_index]["post_mlp"]
                    ),
                }
            )
        source_partition = [item["source_partition"] for item in runtime.tensor_records]
        routing = list(runtime.routing_records)
        checks = {
            "routing_site_count": len(routing) == 65,
            "activation_site_count": len(activation) == 64,
            "zero_scores": all(item["scores_zero"] for item in routing),
            "uniform_exact": all(item["alpha_uniform_exact"] for item in routing),
            "source_partition_finite": all(item["finite"] for item in source_partition),
            "source_partition_relative_l2": max(
                item["relative_l2"] for item in source_partition
            )
            <= 0.002,
            "finite": bool(torch.isfinite(actual_logits).all()),
        }
        records.append(
            {
                "length": length,
                "status": "PASS" if all(checks.values()) else "FAIL",
                "checks": checks,
                "logits": tensor_metrics(actual_logits, standard_logits),
                "layers": layer_metrics,
                "routing": routing,
                "source_partition_max_relative_l2": max(
                    item["relative_l2"] for item in source_partition
                ),
            }
        )
        del tokens, standard_logits, actual_logits, standard_trace
        runtime.clear_diagnostics()
        gc.collect()
        torch.cuda.empty_cache()
        print(f"TITAN_P8_{label} T={length} STATUS={records[-1]['status']}", flush=True)
    return {
        "status": "PASS" if all(item["status"] == "PASS" for item in records) else "FAIL",
        "cases": records,
    }


@torch.inference_mode()
def generation(
    model: torch.nn.Module,
    tokenizer: Any,
    runtime: BlockAttnResReference | None,
) -> list[dict[str, Any]]:
    prompts = [
        json.loads(line)
        for line in PROMPTS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = []
    if runtime is not None:
        enable_block_attnres_runtime(model, runtime)
    try:
        for item in prompts:
            cache = TitanP7HybridCache(model.config)
            tokens = tokenizer(item["prompt"], return_tensors="pt")["input_ids"].cuda()
            output = model(tokens, past_key_values=cache, use_cache=True, return_dict=True)
            generated = []
            token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            for _ in range(8):
                generated.append(int(token.item()))
                output = model(
                    token,
                    past_key_values=output.past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            records.append(
                {
                    "category": item["id"],
                    "tokens": generated,
                    "finite": bool(torch.isfinite(output.logits).all()),
                    "cache_contract": output.past_key_values.latent_contract(),
                }
            )
    finally:
        if runtime is not None:
            disable_block_attnres_runtime(model)
    return records


@torch.inference_mode()
def cache_equivalence(
    model: torch.nn.Module,
    runtime: BlockAttnResReference | None,
    limits: dict[str, Any],
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(20260820)
    tokens = torch.randint(1, model.config.vocab_size, (1, 31), generator=generator)
    if runtime is not None:
        enable_block_attnres_runtime(model, runtime)
    try:
        expanded, _, _ = incremental_logits(
            model, tokens, 7, ExpandedHybridCache(model.config)
        )
        latent, cache, _ = incremental_logits(
            model, tokens, 7, TitanP7HybridCache(model.config)
        )
    finally:
        if runtime is not None:
            disable_block_attnres_runtime(model)
    metrics = distribution(latent, expanded, rtol=0.006, atol=0.006, decisions=True)
    gate = evaluate_compatibility(metrics, limits, decisions=True)
    contract = cache.latent_contract()
    checks = {
        "compatibility": gate["passed"],
        "seen_tokens": contract["seen_tokens"] == 31,
        "latent_layers": contract["latent_layer_count"] == 8,
        "persistent_expanded_k_zero": contract["bytes"]["persistent_expanded_k_numel"] == 0,
        "persistent_expanded_v_zero": contract["bytes"]["persistent_expanded_v_numel"] == 0,
        "finite": metrics["finite"],
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "metrics": metrics,
        "gate": gate,
        "cache_contract": contract,
    }


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    state = gpu_start_state()
    envelope: dict[str, Any] = {
        "stage": "TITAN_P8_REFERENCE",
        "status": "RUNNING",
        "started_at": now(),
        "gpu": state,
        "training_performed": False,
        "heldout_accessed": False,
        "canonical_source_modified": False,
        "weights_modified": False,
        "p8_3_training_allowed": False,
        "p9_titan_migration_allowed": False,
    }
    atomic_json(FINAL, envelope)
    try:
        if not state["idle"]:
            envelope["status"] = "DEFERRED_GPU_NOT_IDLE"
            atomic_json(FINAL, envelope)
            print("TITAN_P8_REFERENCE=DEFERRED_GPU_NOT_IDLE")
            return 3
        config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        cache_limits = yaml.safe_load(CACHE_CONTRACT.read_text(encoding="utf-8"))[
            "evaluation"
        ]["fallback_requires_all"]
        p7_final = load_json(P7_FINAL)
        p7_runtime = load_json(P7_RUNTIME)
        corpus = verify_frozen_corpus(DATA)
        candidate_manifest = load_json(CANDIDATE / "p6-candidate-manifest.json")
        identity_before = checkpoint_identity(CANDIDATE)
        storage = verify_checkpoint_storage(CANDIDATE, candidate_manifest)
        baseline_checks = {
            "p7_pass": p7_final["status"] == "PASS",
            "p7_titan_compatibility": p7_final["p8_titan_migration_allowed"] is True,
            "p7_canonical_pending": p7_final["p7_formal_status"]
            == "PENDING_NATIVE_BF16_REPLAY",
            "p7_runtime": p7_runtime["status"] == "PASS",
            "source_storage": storage["passed"],
            "source_tensor_count": storage["tensor_count"] == SOURCE_TENSOR_COUNT,
            "source_parameter_count": storage["parameter_count"]
            == SOURCE_PARAMETER_COUNT,
            "corpus_frozen": corpus["p5_probe_corpus_frozen"] is True,
            "block_boundaries": tuple(config["architecture"]["block_boundary_layers"])
            == BLOCK_BOUNDARY_LAYERS,
        }
        if not all(baseline_checks.values()):
            raise ValueError(f"P8.0 prerequisites failed: {baseline_checks}")
        enable_sm75_fused_recurrent_fallback()
        enable_p7_latent_cache_runtime()
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            ROOT / "outputs/densek3-core/checkpoints/p5/densek3-4b-k3-decay-silu-i1",
            trust_remote_code=False,
        )
        model = load_densek3_model(CANDIDATE, device="cuda", dtype=torch.float16)
        model.eval()
        dev = PackedTokenDataset(DATA / "packed", "dev")
        baseline_ce = corpus_ce(model, dev, label="P7_BASELINE")
        baseline_generation = generation(model, tokenizer, None)
        baseline_cache = cache_equivalence(model, None, cache_limits)
        baseline_checks.update(
            {
                "full_dev_finite": math.isfinite(baseline_ce),
                "generation_finite": all(item["finite"] for item in baseline_generation),
                "cache": baseline_cache["status"] == "PASS",
            }
        )
        baseline = {
            "stage": "TITAN_P8_0_FROZEN_P7_BASELINE",
            "status": "PASS" if all(baseline_checks.values()) else "FAIL",
            "checks": baseline_checks,
            "source_checkpoint_identity": identity_before,
            "source_weight_hash": identity_before["fingerprint"],
            "source_storage": storage,
            "p7_cache_contract_sha256": sha256_file(P7_RUNTIME),
            "p7_cache_implementation_sha256": p7_runtime[
                "cache_implementation_sha256"
            ],
            "full_dev_ce": baseline_ce,
            "generation": baseline_generation,
            "cache": baseline_cache,
        }
        atomic_json(REPORT_DIR / "p8-0-baseline.json", baseline)
        if baseline["status"] != "PASS":
            raise ValueError("P8.0 baseline failed")
        print("TITAN_P8_0_BASELINE=PASS", flush=True)

        n1 = BlockAttnResReference(
            model.config.hidden_size,
            model.config.rms_norm_eps,
            block_size_residuals=int(
                config["reference_semantics"]["n1_block_size_residuals"]
            ),
        ).cuda()
        n1_report = probe_runtime(
            model,
            n1,
            [int(value) for value in config["reference_semantics"]["n1_probe_lengths"]],
            seed=20260820,
            label="P8_1_N1",
        )
        n1_report.update(
            {
                "stage": "TITAN_P8_1_REFERENCE_SEMANTICS",
                "parameter_contract": n1.parameter_contract(),
                "training_performed": False,
                "heldout_accessed": False,
            }
        )
        atomic_json(REPORT_DIR / "p8-1-reference-semantics.json", n1_report)
        if n1_report["status"] != "PASS":
            raise ValueError("P8.1 reference semantics failed")
        print("TITAN_P8_1_REFERENCE_SEMANTICS=PASS", flush=True)
        del n1
        gc.collect()
        torch.cuda.empty_cache()

        runtime = BlockAttnResReference(
            model.config.hidden_size,
            model.config.rms_norm_eps,
            block_size_residuals=int(config["architecture"]["block_size_residuals"]),
        ).cuda()
        shock_probe = probe_runtime(
            model,
            runtime,
            [int(value) for value in config["zero_init_shock"]["probe_lengths"]],
            seed=20260821,
            label="P8_2_ZERO",
        )
        if shock_probe["status"] != "PASS":
            raise ValueError("P8.2 small zero-init probe failed")
        enable_block_attnres_runtime(model, runtime)
        try:
            zero_ce = corpus_ce(model, dev, label="ZERO_INIT")
        finally:
            disable_block_attnres_runtime(model)
        zero_generation = generation(model, tokenizer, runtime)
        zero_cache = cache_equivalence(model, runtime, cache_limits)
        delta = zero_ce - baseline_ce
        if delta <= float(config["zero_init_shock"]["strong_go_delta_ce_max"]):
            decision = "STRONG_GO"
        elif delta <= float(
            config["zero_init_shock"]["trainable_recovery_delta_ce_max"]
        ):
            decision = "MODIFY_TRAINABLE_RECOVERY"
        else:
            decision = "REDESIGN"
        identity_after = checkpoint_identity(CANDIDATE)
        checks = {
            "small_probe": shock_probe["status"] == "PASS",
            "full_dev_finite": math.isfinite(zero_ce),
            "generation_finite": all(item["finite"] for item in zero_generation),
            "cache_preserved": zero_cache["status"] == "PASS",
            "weights_unchanged": identity_after == identity_before,
            "queries_zero": runtime.parameter_contract()["all_queries_zero"],
            "norms_one": runtime.parameter_contract()["all_norms_one"],
        }
        shock = {
            "stage": "TITAN_P8_2_ZERO_INIT_SHOCK",
            "status": "PASS" if all(checks.values()) else "FAIL",
            "decision": decision,
            "checks": checks,
            "small_probe": shock_probe,
            "p7_full_dev_ce": baseline_ce,
            "p8_zero_full_dev_ce": zero_ce,
            "delta_ce": delta,
            "generation": {
                "baseline": baseline_generation,
                "zero": zero_generation,
                "tokens_exact": [
                    left["tokens"] == right["tokens"]
                    for left, right in zip(
                        baseline_generation,
                        zero_generation,
                        strict=True,
                    )
                ],
            },
            "cache": zero_cache,
            "parameter_contract": runtime.parameter_contract(),
            "source_checkpoint_identity_after": identity_after,
            "training_performed": False,
            "heldout_accessed": False,
        }
        atomic_json(REPORT_DIR / "p8-2-zero-init-shock.json", shock)
        if shock["status"] != "PASS":
            raise ValueError("P8.2 zero-init audit failed")
        training_allowed = decision in {"STRONG_GO", "MODIFY_TRAINABLE_RECOVERY"}
        envelope.update(
            {
                "status": "PASS",
                "completed_at": now(),
                "p8_0": True,
                "p8_1": True,
                "p8_2": True,
                "p8_2_decision": decision,
                "p8_3_training_allowed": training_allowed,
                "p8_formal_status": "TITAN_REFERENCE_ONLY_PENDING_NATIVE_BF16",
                "p8_canonical_frozen": False,
            }
        )
        atomic_json(FINAL, envelope)
        print("TITAN_P8_2_ZERO_INIT_SHOCK=PASS", flush=True)
        print(f"TITAN_P8_2_DECISION={decision}", flush=True)
        print(f"P8_3_TRAINING_ALLOWED={str(training_allowed).lower()}", flush=True)
        print("P9_TITAN_MIGRATION_ALLOWED=false", flush=True)
        return 0
    except Exception as exc:
        envelope.update(
            {
                "status": "ERROR",
                "completed_at": now(),
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        atomic_json(FINAL, envelope)
        print(f"TITAN_P8_REFERENCE_EXCEPTION={type(exc).__name__}: {exc}", flush=True)
        return 1
    finally:
        disable_p7_latent_cache_runtime()


if __name__ == "__main__":
    raise SystemExit(main())
