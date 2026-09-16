"""Run the immutable P6 candidate closeout in Titan FP16 compatibility mode.

This executes all portable P6.3/P6.4 capability, storage, reload, generation,
and expanded-cache checks.  SM75 FP16 evidence never writes the canonical P6
final manifest and never enables P7; the frozen contract still requires a
native BF16 replay for that decision.
"""

from __future__ import annotations

import gc
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from kda_sm75_backend import enable_sm75_fused_recurrent_fallback

from densek3_core.evaluation.p4_runtime import (
    load_densek3_model,
    load_densek3_model_from_pretrained,
)
from densek3_core.modeling.hybrid_cache import GQACacheLayer, KDACacheLayer
from densek3_core.recovery.p5_corpus import PackedTokenDataset, verify_frozen_corpus
from densek3_core.recovery.p6_closeout import (
    P6_MLA_LAYERS,
    P6_SELECTED_STEPS,
    capability_gate,
    checkpoint_identity,
    sha256_file,
    verify_checkpoint_storage,
    verify_local_artifact_binding,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/architecture/p6-closeout.yaml"
TITAN_CACHE_CONFIG = (
    ROOT / "titan/configs/p6-titan-fp16-cache-compat-2026-08-19-01.yaml"
)
CANDIDATE = (
    ROOT
    / "outputs/densek3-core/checkpoints/p6/"
    "densek3-4b-8mla-c3-qk-norm-calibrated-candidate"
)
P5_TOKENIZER = (
    ROOT / "outputs/densek3-core/checkpoints/p5/densek3-4b-k3-decay-silu-i1"
)
P6_2D = ROOT / "outputs/densek3-core/manifests/p6/p6-2d/p6-2d-scaleout.json"
LOCAL_ARTIFACT = (
    ROOT
    / "outputs/densek3-core/artifacts/p6/p6-2d-local/"
    "p6-2d-eight-layer-best-trainable.safetensors"
)
DATA = ROOT / "data/p5-probe"
PROMPTS = ROOT / "data/p4-smoke-prompts.jsonl"
P0_P5 = ROOT / "titan/manifests/reproduction/p0-p5/titan-p0-p5-final-check.json"
P6_ASSETS = ROOT / "titan/manifests/reproduction/p6/p6-candidate-asset-reconstruction.json"
P6_CPU_CONTRACTS = ROOT / "titan/manifests/reproduction/cpu-contracts/p6-cpu-contracts.log"
REPORT = ROOT / "titan/manifests/reproduction/p6/p6-titan-compatibility-closeout.json"
MAX_START_TEMPERATURE_C = 75


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def gpu_state() -> dict[str, Any]:
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,temperature.gpu",
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
    fields = [field.strip() for field in gpu.stdout.strip().split(",")]
    visible = gpu.returncode == 0 and len(fields) == 2
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_prompts() -> list[dict[str, str]]:
    prompts = [
        json.loads(line)
        for line in PROMPTS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if {item["id"] for item in prompts} != {"english", "chinese", "code", "factual", "longer"}:
        raise ValueError("P6 closeout requires the frozen five-category prompts")
    return prompts


@torch.inference_mode()
def corpus_ce(model: torch.nn.Module, dataset: PackedTokenDataset) -> float:
    if dataset.split not in {"dev", "heldout"} or len(dataset) != 97:
        raise ValueError("P6 closeout requires exactly 97 dev/heldout sequences")
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
            print(f"TITAN_P6_{dataset.split.upper()}_PROGRESS={index + 1}/97", flush=True)
    return total_loss / total_tokens


@torch.inference_mode()
def short_observation(model: torch.nn.Module) -> dict[str, Any]:
    import hashlib

    device = next(model.parameters()).device
    tokens = torch.tensor([[1, 17, 29, 31, 43, 47, 53]], device=device)
    output = model(tokens, use_cache=False, return_dict=True).logits
    raw = output.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return {
        "logits_sha256": hashlib.sha256(raw).hexdigest(),
        "finite": bool(torch.isfinite(output).all()),
        "argmax": output.argmax(dim=-1).detach().cpu().tolist(),
    }


@torch.inference_mode()
def generation_report(model: torch.nn.Module, tokenizer: Any) -> dict[str, Any]:
    device = next(model.parameters()).device
    records = []
    for item in load_prompts():
        encoded = tokenizer(item["prompt"], return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        generated = model.greedy_generate(
            input_ids,
            attention_mask=encoded["attention_mask"].to(device),
            max_new_tokens=8,
        )
        tokens = generated[:, input_ids.shape[1] :].detach().cpu().flatten().tolist()
        counts = Counter(tokens)
        not_collapsed = (
            bool(tokens)
            and max(counts.values()) / len(tokens) <= 0.75
            and len(counts) / len(tokens) >= 0.25
        )
        records.append({"id": item["id"], "new_tokens": tokens, "not_collapsed": not_collapsed})
    return {"records": records, "not_collapsed": all(item["not_collapsed"] for item in records)}


def cache_structure(cache: Any, expected_tokens: int) -> dict[str, Any]:
    kda = 0
    mla = 0
    checks = []
    for index, layer in enumerate(cache.layers):
        if isinstance(layer, KDACacheLayer):
            kda += 1
            checks.append(
                layer.recurrent_state is not None
                and layer.recurrent_state.dtype == torch.float32
                and layer.recurrent_state_layout == "v_first"
            )
        elif isinstance(layer, GQACacheLayer):
            mla += 1
            checks.append(
                layer.key is not None
                and layer.value is not None
                and layer.sequence_length == expected_tokens
                and index in P6_MLA_LAYERS
            )
        else:
            checks.append(False)
    return {
        "seen_tokens": cache.seen_tokens,
        "kda_layers": kda,
        "mla_layers": mla,
        "cache_bytes": cache.cache_bytes(),
        "passed": cache.seen_tokens == expected_tokens
        and kda == 24
        and mla == 8
        and all(checks),
    }


@torch.inference_mode()
def cache_case(
    model: torch.nn.Module,
    prefill: int,
    decode: int,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    total = prefill + decode
    generator = torch.Generator(device="cpu").manual_seed(20260814 + prefill * 100 + decode)
    tokens = torch.randint(1, model.config.vocab_size, (1, total), generator=generator).to(device)
    full = model(tokens, use_cache=False, return_dict=True).logits[:, prefill:]
    prefix = model(tokens[:, :prefill], use_cache=True, return_dict=True)
    cache = prefix.past_key_values
    pieces = []
    for offset in range(decode):
        step = model(
            tokens[:, prefill + offset : prefill + offset + 1],
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = step.past_key_values
        pieces.append(step.logits)
    incremental = torch.cat(pieces, dim=1)
    difference = incremental.double() - full.double()
    absolute = difference.abs().reshape(-1).cpu()
    tolerance = atol + rtol * full.double().abs()
    failed = difference.abs() > tolerance
    full_top10 = full.topk(10, dim=-1).indices
    incremental_top10 = incremental.topk(10, dim=-1).indices
    top10_overlap = (
        full_top10.unsqueeze(-1)
        .eq(incremental_top10.unsqueeze(-2))
        .any(dim=-1)
        .sum(dim=-1)
    )
    structure = cache_structure(cache, total)
    return {
        "prefill": prefill,
        "decode": decode,
        "finite": bool(torch.isfinite(full).all() and torch.isfinite(incremental).all()),
        "standard_allclose": bool(torch.allclose(incremental, full, rtol=rtol, atol=atol)),
        "argmax_equal": bool(
            torch.equal(incremental.argmax(dim=-1), full.argmax(dim=-1))
        ),
        "max_abs": float(difference.abs().max().item()),
        "mean_abs": float(difference.abs().mean().item()),
        "p99_9_abs": float(torch.quantile(absolute, 0.999).item()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                incremental.double().reshape(1, -1),
                full.double().reshape(1, -1),
            ).item()
        ),
        "failed_allclose_fraction": float(failed.double().mean().item()),
        "top10_min_overlap": int(top10_overlap.min().item()),
        "top10_mean_overlap": float(top10_overlap.double().mean().item() / 10.0),
        "nan_count": int(torch.isnan(full).sum().item() + torch.isnan(incremental).sum().item()),
        "inf_count": int(torch.isinf(full).sum().item() + torch.isinf(incremental).sum().item()),
        "relative_l2": float(difference.norm().div(full.double().norm().clamp_min(1e-30)).item()),
        "structure": structure,
    }


def evaluate_titan_cache_case(
    case: dict[str, Any],
    limits: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate the Titan-only FP16 cache envelope after standard allclose."""
    checks = {
        "relative_l2": case["relative_l2"] <= limits["relative_l2_max"],
        "cosine": case["cosine"] >= limits["cosine_min"],
        "mean_abs": case["mean_abs"] <= limits["mean_abs_max"],
        "p99_9_abs": case["p99_9_abs"] <= limits["p99_9_abs_max"],
        "failed_allclose_fraction": case["failed_allclose_fraction"]
        <= limits["failed_allclose_fraction_max"],
        "max_abs": case["max_abs"] <= limits["max_abs_max"],
        "nan_count": case["nan_count"] <= limits["nan_count_max"],
        "inf_count": case["inf_count"] <= limits["inf_count_max"],
        "argmax_equal": case["argmax_equal"] is limits["argmax_equal"],
        "top10_min_overlap": case["top10_min_overlap"]
        >= limits["top10_min_overlap"],
        "cache_structure": case["structure"]["passed"]
        is limits["cache_structure_passed"],
    }
    standard = bool(case["standard_allclose"])
    return {
        "standard_allclose": standard,
        "fallback_evaluated": not standard,
        "fallback_checks": checks,
        "passed": standard or all(checks.values()),
        "pass_mode": (
            "STANDARD_ALLCLOSE"
            if standard
            else "TITAN_SM75_FP16_COMPOSITE_CACHE_ENVELOPE"
        ),
    }


def main() -> int:
    state = gpu_state()
    if not state["idle"]:
        atomic_json(
            REPORT,
            {
                "schema_version": 1,
                "stage": "P6_TITAN_COMPATIBILITY_CLOSEOUT",
                "status": "DEFERRED_GPU_NOT_IDLE",
                "completed_at": datetime.now(timezone.utc).astimezone().isoformat(),
                "gpu": state,
                "p7_allowed": False,
            },
        )
        print("P6_TITAN_COMPATIBILITY_CLOSEOUT=DEFERRED_GPU_NOT_IDLE")
        return 3

    report: dict[str, Any] = {
        "schema_version": 1,
        "stage": "P6_TITAN_COMPATIBILITY_CLOSEOUT",
        "status": "RUNNING",
        "started_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "gpu": state,
        "dtype": "float16",
        "canonical_dtype": "bfloat16",
        "formal_p6_final_check_written": False,
        "formal_bf16_gate_reclassified": False,
        "p7_allowed": False,
        "training_performed": False,
        "candidate_modified": False,
    }
    atomic_json(REPORT, report)
    try:
        report["kda_backend_override"] = enable_sm75_fused_recurrent_fallback()
        atomic_json(REPORT, report)
        from transformers import AutoTokenizer

        frozen = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        titan_cache_contract = yaml.safe_load(
            TITAN_CACHE_CONFIG.read_text(encoding="utf-8")
        )
        p0_p5 = load_json(P0_P5)
        p6_assets = load_json(P6_ASSETS)
        candidate = load_json(CANDIDATE / "p6-candidate-manifest.json")
        p6_2d = load_json(P6_2D)
        tokenizer_files = (
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
        )
        tokenizer_identity = {
            name: sha256_file(P5_TOKENIZER / name) for name in tokenizer_files
        }
        source_checks = {
            "p0_p5": p0_p5["status"] == "PASS",
            "p6_assets": p6_assets["status"] == "PASS",
            "candidate_manifest": sha256_file(CANDIDATE / "p6-candidate-manifest.json")
            == frozen["source"]["candidate_manifest_sha256"],
            "candidate_config": sha256_file(CANDIDATE / "config.json")
            == frozen["source"]["candidate_config_sha256"],
            "candidate_index": sha256_file(CANDIDATE / "model.safetensors.index.json")
            == frozen["source"]["candidate_index_sha256"],
            "p6_2d": sha256_file(P6_2D) == frozen["source"]["p6_2d_report_sha256"],
            "local_artifact": sha256_file(LOCAL_ARTIFACT)
            == frozen["source"]["local_artifact_sha256"],
            "p6_2d_pass": p6_2d["status"] == "PASS",
            "p6_cpu_contracts": "41 passed" in P6_CPU_CONTRACTS.read_text(encoding="utf-8"),
            "p5_tokenizer_assets": all(
                (P5_TOKENIZER / name).is_file() for name in tokenizer_files
            ),
            "selected_steps": {
                int(key): int(value)
                for key, value in candidate["provenance"]["per_layer_best_steps"].items()
            }
            == P6_SELECTED_STEPS,
        }
        if not all(source_checks.values()):
            raise ValueError(f"P6 closeout source identity failed: {source_checks}")
        identity = checkpoint_identity(CANDIDATE)
        storage = verify_checkpoint_storage(CANDIDATE, candidate)
        binding = verify_local_artifact_binding(LOCAL_ARTIFACT, storage["mla_tensor_hashes"])
        corpus = verify_frozen_corpus(DATA)
        if not storage["passed"] or not binding["passed"] or corpus["status"] != "PASS":
            raise ValueError("P6 storage, local artifact, or corpus verification failed")

        tokenizer = AutoTokenizer.from_pretrained(
            P5_TOKENIZER,
            trust_remote_code=False,
        )
        model = load_densek3_model(CANDIDATE, device="cuda", dtype=torch.float16)
        indexed_observation = short_observation(model)
        dev_ce = corpus_ce(model, PackedTokenDataset(DATA / "packed", "dev"))
        heldout_ce = corpus_ce(model, PackedTokenDataset(DATA / "packed", "heldout"))
        generation = generation_report(model, tokenizer)
        rtol = float(frozen["engineering"]["cache_standard_bf16_rtol"])
        atol = float(frozen["engineering"]["cache_standard_bf16_atol"])
        cache_cases = [
            cache_case(model, int(prefill), int(decode), rtol=rtol, atol=atol)
            for prefill, decode in frozen["engineering"]["cache_cases"]
        ]
        cache_evaluations = [
            evaluate_titan_cache_case(
                item,
                titan_cache_contract["evaluation"]["fallback_requires_all"],
            )
            for item in cache_cases
        ]
        topology = {
            "kda_24": len(model.config.kda_indices) == 24,
            "mla_8": tuple(model.config.mla_indices) == P6_MLA_LAYERS,
            "gqa_0": not model.config.gqa_indices,
            "rank_512": model.config.mla_kv_lora_rank == 512,
            "nope": model.config.mla_position_encoding == "none",
            "qk_norm": model.config.mla_qk_norm is True,
            "expanded_reference": model.config.mla_cache_mode == "expanded_reference",
        }
        del model
        gc.collect()
        torch.cuda.empty_cache()

        public_model = load_densek3_model_from_pretrained(
            CANDIDATE,
            device="cuda",
            dtype=torch.float16,
        )
        public_observation = short_observation(public_model)
        del public_model
        gc.collect()
        torch.cuda.empty_cache()

        dev_gate = capability_gate(
            p5_ce=float(frozen["capability"]["full_dev"]["p5_ce"]),
            p6_ce=dev_ce,
            maximum_delta=float(frozen["capability"]["full_dev"]["maximum_delta"]),
        )
        heldout_gate = capability_gate(
            p5_ce=float(frozen["capability"]["heldout"]["p5_ce"]),
            p6_ce=heldout_ce,
            maximum_delta=float(frozen["capability"]["heldout"]["maximum_delta"]),
        )
        checks = {
            "source_identity": all(source_checks.values()),
            "storage": storage["passed"],
            "local_artifact_binding": binding["passed"],
            "corpus": corpus["status"] == "PASS",
            "topology": all(topology.values()),
            "dev_capability": dev_gate["passed"],
            "heldout_capability": heldout_gate["passed"],
            "generation": generation["not_collapsed"],
            "cache": all(
                item["finite"] and evaluation["passed"]
                for item, evaluation in zip(
                    cache_cases,
                    cache_evaluations,
                    strict=True,
                )
            ),
            "indexed_public_reload_exact": indexed_observation == public_observation,
        }
        report.update(
            {
                "status": "PASS" if all(checks.values()) else "FAIL",
                "source_checks": source_checks,
                "tokenizer_source": {
                    "checkpoint": str(P5_TOKENIZER),
                    "reason": "P6 preserves the frozen P5 vocabulary and tokenizer identity",
                    "file_sha256": tokenizer_identity,
                    "candidate_checkpoint_modified": False,
                },
                "checkpoint_identity": identity,
                "storage": storage,
                "local_artifact_binding": binding,
                "corpus": corpus,
                "topology": topology,
                "capability": {"dev": dev_gate, "heldout": heldout_gate},
                "frozen_bf16_expected_dev_ce": frozen["capability"]["full_dev"][
                    "expected_p6_ce"
                ],
                "bf16_exact_replay_not_applicable_on_sm75": True,
                "generation": generation,
                "cache": {
                    "cases": cache_cases,
                    "evaluations": cache_evaluations,
                    "rtol": rtol,
                    "atol": atol,
                    "compatibility_contract": titan_cache_contract,
                    "compatibility_contract_sha256": sha256_file(TITAN_CACHE_CONFIG),
                    "diagnostic_evidence": str(
                        ROOT
                        / "titan/manifests/reproduction/p6/"
                        "p6-cache-fp16-diagnostic.json"
                    ),
                    "fallback_used": any(
                        item["fallback_evaluated"] for item in cache_evaluations
                    ),
                    "true_latent_cache": False,
                },
                "reload": {
                    "indexed": indexed_observation,
                    "public": public_observation,
                    "exact": indexed_observation == public_observation,
                },
                "checks": checks,
                "compatibility_passed": all(checks.values()),
                "compatibility_pass_mode": (
                    titan_cache_contract["pass_mode"]
                    if all(checks.values())
                    and any(
                        item["fallback_evaluated"] for item in cache_evaluations
                    )
                    else "STANDARD_ALL_CHECKS"
                ),
                "formal_p6_status": "AWAITING_NATIVE_BF16_REPLAY",
                "p7_allowed": False,
                "completed_at": datetime.now(timezone.utc).astimezone().isoformat(),
            }
        )
        atomic_json(REPORT, report)
        print(f"P6_TITAN_COMPATIBILITY_CLOSEOUT={report['status']}")
        print("P6_FORMAL_STATUS=AWAITING_NATIVE_BF16_REPLAY")
        print("P7_ALLOWED=false")
        return 0 if report["status"] == "PASS" else 1
    except Exception as error:  # noqa: BLE001
        report.update(
            {
                "status": "ERROR",
                "exception_type": type(error).__name__,
                "exception": str(error),
                "completed_at": datetime.now(timezone.utc).astimezone().isoformat(),
            }
        )
        atomic_json(REPORT, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
