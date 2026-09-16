"""Run the bounded Titan-only P11.5a dual-teacher 256K probe.

This is a Titan overlay. It never modifies canonical model/config/test sources.
Every expensive phase is resumable and API attempts are write-ahead logged.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file, save_file
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[2]
TITAN = ROOT / "titan"
SCRIPTS = TITAN / "scripts"
sys.path[:0] = [str(TITAN), str(SCRIPTS)]

from p7_latent_cache import (  # noqa: E402
    TitanP7HybridCache,
    enable_p7_latent_cache_runtime,
)
from p10 import audit_gradient_coverage, build_p10_titan_reduced_scope  # noqa: E402
from p10.training import cosine_learning_rate  # noqa: E402
from p11 import (  # noqa: E402
    KimiAPIConfig,
    KimiAPICostLedger,
    KimiAPITeacher,
    accounted_ledger_cost,
    api_call_allowed,
    contract_manifest,
    exact_token_advantage,
    select_high_information,
)
from p11.cost_calibration import DOMAINS  # noqa: E402
from p11.probe import (  # noqa: E402
    KIMI_MAX_REVIEW_ATTEMPTS,
    KIMI_REVIEWS_PER_DOMAIN_PER_WAVE,
    KIMI_REVIEWS_PER_WAVE,
    P11_5A_AMENDMENT,
    P11_API_TOTAL_HARD_CAP_CNY,
    P11_FP16_INITIAL_LOSS_SCALE,
    P11_FP16_LOSS_SCALE_GROWTH_INTERVAL,
    P11_GPU_HISTORICAL_PEAK_MEMORY_MIB,
    P11_GPU_MAX_START_TEMPERATURE_C,
    P11_GPU_MEMORY_GATE_AMENDMENT,
    P11_GPU_SAFETY_MARGIN_MIB,
    QWEN_CANDIDATE_ROLLOUT_TOKENS,
    QWEN_CANDIDATES_PER_DOMAIN,
    QWEN_LONG_ROLLOUT_TOKENS,
    STREAM_LOSS_WEIGHTS,
    UPDATE_BUCKETS_PER_WAVE,
    WAVE_COUNT,
    assign_update_buckets,
    cumulative_main_quotas,
    gpu_memory_envelope,
    raw_lm_target_lengths,
    wave_main_quotas,
)
from p11.runtime import (  # noqa: E402
    QWEN,
    TOKENIZER,
    force_qwen_torch_gdn_backend,
    load_p10_t_student,
    qwen_token_logprobs,
    token_hash,
)
from run_p10 import (  # noqa: E402
    build_optimizer,
    cache_equivalence,
    compensation_warning,
    corpus_ce,
    frozen_scope_audit,
    generation,
    gpu_start_state,
    load_trainable_state,
    optimizer_gradient_norm_by_family,
    optimizer_parameter_names,
    parameter_update_by_family,
    routing_diagnostic,
    trainable_state,
    unexpected_gradient_names,
)

from densek3_core.recovery.p5_corpus import PackedTokenDataset  # noqa: E402
from densek3_core.recovery.p5_probe import chunked_causal_lm_loss  # noqa: E402
from densek3_core.recovery.p6_closeout import P6_MLA_LAYERS, sha256_file  # noqa: E402

CONFIG = TITAN / "configs/p11-5a-probe-contract-2026-08-21.yaml"
API_CONFIG = TITAN / "configs/p11-kimi-api.json"
REPORT_ROOT = TITAN / "manifests/reproduction/p11-t"
REPORT_DIR = REPORT_ROOT / "p11-5a"
PREFLIGHT_REPORT = REPORT_DIR / "p11-5a-preflight.json"
BASELINE_REPORT = REPORT_DIR / "p11-5a-p10t-baseline.json"
FINAL_REPORT = REPORT_DIR / "p11-5a-final-check.json"
PROMPTS = ROOT / "data/p11-general/prompts/general-train-prompts-v1.jsonl"
GENERAL_DEV = ROOT / "data/p11-general/eval/general-dev-micro-v1.jsonl"
RAW_DATA = ROOT / "data/p5-probe/packed"
QWEN_CACHE_ROOT = ROOT / "data/p11-general/teacher-cache/p11-5a/qwen-opd"
KIMI_CACHE_ROOT = ROOT / "data/p11-general/teacher-cache/p11-5a/kimi"
CHECKPOINT_ROOT = ROOT / "outputs/densek3-core/checkpoints/p11-t/probe-256k"
PROGRESS_ROOT = CHECKPOINT_ROOT / "progress"
P10_FINAL = TITAN / "manifests/reproduction/p10-t/p10-t-final-check.json"
P10_SOURCE = ROOT / "outputs/densek3-core/checkpoints/p10-t/densek3-4b-core-k3-joint-reduced-titan"
P11_4 = REPORT_ROOT / "p11-4-dual-teacher-joint-preflight.json"
P11_4B = REPORT_ROOT / "p11-4b-kimi-cost-calibration.json"
P11_4B_SELECTION = REPORT_ROOT / "p11-4b-high-information-selection.json"
MAX_PROMPT_TOKENS = 256
MAX_KIMI_COMPLETION_TOKENS = 768
RATE_LIMIT_SECONDS = 21.0
SEED = 20260821


def now() -> str:
    return datetime.now().astimezone().isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def p11_gpu_start_state(*, stage: str, wave_index: int | None = None) -> dict[str, Any]:
    """Audit the Titan memory envelope while preserving the actual idle state."""
    state = gpu_start_state()
    fields = [item.strip() for item in str(state.get("description", "")).split(",")]
    temperature_c: int | None = None
    memory_used_mib: int | None = None
    memory_total_mib: int | None = None
    if bool(state.get("visible")) and len(fields) == 4:
        try:
            temperature_c = int(fields[1])
            memory_used_mib = int(fields[2])
            memory_total_mib = int(fields[3])
        except ValueError:
            pass
    envelope = gpu_memory_envelope(
        visible=bool(state.get("visible")),
        temperature_c=temperature_c,
        memory_used_mib=memory_used_mib,
        memory_total_mib=memory_total_mib,
    )
    audited = {
        **state,
        "actual_idle": bool(state.get("idle")),
        "allowed": bool(envelope["allowed"]),
        "memory_envelope": envelope,
        "formal_gate_amendment": P11_GPU_MEMORY_GATE_AMENDMENT,
        "formal_gate_requires_idle": False,
        "completed_at": now(),
        "stage": stage,
        "wave_index": wave_index,
    }
    directory = REPORT_DIR / f"wave-{wave_index}" if wave_index is not None else REPORT_DIR
    atomic_json(directory / f"{stage}-gpu-gate.json", audited)
    return audited


def atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(encoded)


def release(*values: object) -> None:
    del values
    gc.collect()
    torch.cuda.empty_cache()


def load_tokenizer() -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=False)


def load_prompts() -> list[dict[str, Any]]:
    return [json.loads(line) for line in PROMPTS.read_text(encoding="utf-8").splitlines() if line.strip()]


def stable_rank(namespace: str, record_id: str) -> str:
    return sha256_text(f"{P11_5A_AMENDMENT}\0{namespace}\0{record_id}")


def stable_seed(namespace: str, record_id: str) -> int:
    return int(stable_rank(namespace, record_id)[:16], 16) % (2**63 - 1)


def source_checkpoint(wave_index: int) -> Path | None:
    if wave_index == 0:
        return None
    path = segment_checkpoint(wave_index - 1)
    if not (path / "checkpoint-manifest.json").is_file():
        raise FileNotFoundError(f"P11.5a prior segment checkpoint is absent: {path}")
    return path


def segment_checkpoint(wave_index: int) -> Path:
    tokens = cumulative_main_quotas(wave_index + 1)["main"]
    return CHECKPOINT_ROOT / f"main-tokens-{tokens:06d}"


def checkpoint_fingerprint(path: Path | None) -> str:
    if path is None:
        manifest = P10_SOURCE / "p10-t-candidate-manifest.json"
        return sha256_file(manifest)
    return sha256_file(path / "trainable.safetensors")


def load_probe_student(checkpoint_dir: Path | None) -> tuple[Any, Any, Any, dict[str, Any]]:
    model, attnres, load = load_p10_t_student()
    scope = build_p10_titan_reduced_scope(model, attnres, rank=8, alpha=16.0)
    if checkpoint_dir is not None:
        manifest = load_json(checkpoint_dir / "checkpoint-manifest.json")
        weights = checkpoint_dir / "trainable.safetensors"
        if sha256_file(weights) != manifest["trainable_sha256"]:
            raise ValueError("P11.5a trainable checkpoint hash mismatch")
        load_trainable_state(scope, weights)
    enable_p7_latent_cache_runtime(P6_MLA_LAYERS)
    return model, attnres, scope, load


def run_preflight() -> dict[str, Any]:
    contract = contract_manifest()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    p10 = load_json(P10_FINAL)
    p11_4 = load_json(P11_4)
    p11_4b = load_json(P11_4B)
    ledger = accounted_ledger_cost(REPORT_ROOT)
    api = KimiAPIConfig.from_json(API_CONFIG)
    checks = {
        "contract_exact": contract["passed"],
        "config_amendment_exact": config["amendment"] == P11_5A_AMENDMENT,
        "p10_t_pass": p10.get("status") == "PASS",
        "p10_t_source_exact": Path(p10.get("p11_titan_source", "")).resolve() == P10_SOURCE.resolve(),
        "p11_4_pass": p11_4.get("status") == "PASS",
        "p11_4b_pass": p11_4b.get("status") == "PASS",
        "source_present": P10_SOURCE.is_dir(),
        "prompts_present": PROMPTS.is_file(),
        "general_dev_present": GENERAL_DEV.is_file(),
        "raw_train_present": (RAW_DATA / "train.bin").is_file(),
        "raw_dev_present": (RAW_DATA / "dev.bin").is_file(),
        "kimi_api_model": api.model == "kimi-k3",
        "kimi_low_reasoning": api.reasoning_effort == "low",
        "kimi_auto_retry_zero": api.max_retries == 0,
        "daily_budget_20": api.daily_budget_cny == 20.0,
        "p11_cumulative_budget_below_5": ledger["accounted_cost_cny"] < P11_API_TOTAL_HARD_CAP_CNY,
        "heldout_forbidden": config["forbidden"]["heldout_access"] is True,
        "expected_scope_563": config["optimization"]["expected_gradient_tensors"] == 563,
        "p11_fp16_initial_loss_scale": (
            float(config["optimization"]["fp16_initial_loss_scale"])
            == P11_FP16_INITIAL_LOSS_SCALE
        ),
        "p11_fp16_loss_scale_growth_interval": (
            int(config["optimization"]["fp16_loss_scale_growth_interval"])
            == P11_FP16_LOSS_SCALE_GROWTH_INTERVAL
        ),
        "p11_gpu_memory_gate_amendment": (
            config["gpu_start_gate"]["amendment"]
            == P11_GPU_MEMORY_GATE_AMENDMENT
        ),
        "p11_gpu_gate_does_not_require_idle": (
            config["gpu_start_gate"]["requires_gpu_idle"] is False
        ),
        "p11_gpu_historical_peak_memory": (
            int(config["gpu_start_gate"]["historical_peak_memory_mib"])
            == P11_GPU_HISTORICAL_PEAK_MEMORY_MIB
        ),
        "p11_gpu_safety_margin": (
            int(config["gpu_start_gate"]["safety_margin_mib"])
            == P11_GPU_SAFETY_MARGIN_MIB
        ),
        "p11_gpu_required_free_memory": (
            int(config["gpu_start_gate"]["required_free_memory_mib"])
            == P11_GPU_HISTORICAL_PEAK_MEMORY_MIB
            + P11_GPU_SAFETY_MARGIN_MIB
        ),
        "p11_gpu_maximum_start_temperature": (
            int(config["gpu_start_gate"]["maximum_start_temperature_c"])
            == P11_GPU_MAX_START_TEMPERATURE_C
        ),
    }
    passed = all(checks.values())
    report = {
        "stage": "P11.5a-PREFLIGHT",
        "status": "PASS" if passed else "FAIL",
        "completed_at": now(),
        "amendment": P11_5A_AMENDMENT,
        "contract": contract,
        "p11_cumulative_api_ledger": ledger,
        "checks": checks,
        "p11_4b_cost_calibration": "PASS" if p11_4b.get("status") == "PASS" else "FAIL",
        "p11_5a_probe_allowed": passed,
        "p11_5a_formal_training_started": False,
        "p11_6_phase1_allowed": False,
        "p12_titan_migration_allowed": False,
        "heldout_accessed": False,
        "result_marker": "P11_5A_PREFLIGHT=PASS" if passed else "P11_5A_PREFLIGHT=FAIL",
    }
    atomic_json(PREFLIGHT_REPORT, report)
    print(report["result_marker"], flush=True)
    print(f"P11_5A_PROBE_ALLOWED={str(passed).lower()}", flush=True)
    print("P11_5A_FORMAL_TRAINING_STARTED=false", flush=True)
    return report


@torch.inference_mode()
def rollout_one(
    model: torch.nn.Module,
    tokenizer: Any,
    record: dict[str, Any],
    *,
    maximum_tokens: int,
    namespace: str,
) -> dict[str, Any]:
    prompt_ids = tokenizer(
        record["prompt"], return_tensors="pt", truncation=True, max_length=MAX_PROMPT_TOKENS
    ).input_ids.cuda()
    cache = TitanP7HybridCache(model.config)
    output = model(prompt_ids, past_key_values=cache, use_cache=True, return_dict=True)
    generator = torch.Generator(device="cuda").manual_seed(stable_seed(namespace, record["id"]))
    generated: list[int] = []
    old_logprobs: list[float] = []
    entropies: list[float] = []
    for _ in range(maximum_tokens):
        logits = output.logits[:, -1].float()
        logprob = torch.log_softmax(logits, dim=-1)
        probabilities = torch.exp(logprob)
        token = torch.multinomial(probabilities, 1, generator=generator)
        generated.append(int(token.item()))
        old_logprobs.append(float(logprob.gather(-1, token).item()))
        entropies.append(float(-(probabilities * logprob).sum()))
        if generated[-1] == tokenizer.eos_token_id:
            break
        output = model(
            token, past_key_values=output.past_key_values, use_cache=True, return_dict=True
        )
    if not generated:
        raise ValueError("P11.5a rollout is empty")
    generated_tensor = torch.tensor([generated], dtype=torch.long)
    full = torch.cat((prompt_ids.cpu(), generated_tensor), dim=1)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return {
        "id": record["id"],
        "domain": record["domain"],
        "language": record["language"],
        "prompt": record["prompt"],
        "prompt_sha256": sha256_text(record["prompt"]),
        "prompt_tokens": int(prompt_ids.shape[1]),
        "rollout_tokens": len(generated),
        "rollout_token_ids": generated,
        "full_token_ids": full[0].tolist(),
        "rollout_token_hash": token_hash(generated_tensor),
        "student_text": text,
        "student_text_sha256": sha256_text(text),
        "old_student_logprobs": old_logprobs,
        "student_mean_token_entropy": sum(entropies) / len(entropies),
    }


def calibration_exclusions() -> set[str]:
    report = load_json(P11_4B_SELECTION)
    return {str(item["id"]) for item in report.get("selected", [])}


def previous_wave_prompt_ids(wave_index: int) -> set[str]:
    used = calibration_exclusions()
    for index in range(wave_index):
        manifest = REPORT_DIR / f"wave-{index}/qwen-opd-manifest.json"
        if manifest.is_file():
            used.update(str(item["prompt_id"]) for item in load_json(manifest).get("trajectories", []))
    return used


def candidate_records(prompts: list[dict[str, Any]], wave_index: int, excluded: set[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in prompts:
        if record["id"] not in excluded and record["domain"] in DOMAINS:
            grouped[record["domain"]].append(record)
    selected: list[dict[str, Any]] = []
    for domain in DOMAINS:
        values = sorted(
            grouped[domain], key=lambda item: stable_rank(f"wave-{wave_index}-candidate-{domain}", item["id"])
        )
        if len(values) < QWEN_CANDIDATES_PER_DOMAIN:
            raise ValueError(f"P11.5a has too few candidate prompts for {domain}")
        selected.extend(values[:QWEN_CANDIDATES_PER_DOMAIN])
    return selected


def long_prompt_records(prompts: list[dict[str, Any]], wave_index: int, excluded: set[str]) -> list[dict[str, Any]]:
    values = [record for record in prompts if record["id"] not in excluded]
    return sorted(values, key=lambda item: stable_rank(f"wave-{wave_index}-long", item["id"]))


def qwen_cache_path(wave_index: int, trajectory_index: int) -> Path:
    return QWEN_CACHE_ROOT / f"wave-{wave_index}" / f"trajectory-{trajectory_index:04d}.json"


def build_wave_qwen_cache(wave_index: int) -> dict[str, Any]:
    state = p11_gpu_start_state(stage="qwen-cache", wave_index=wave_index)
    if not state["allowed"]:
        return {
            "stage": "P11.5a-QWEN-CACHE",
            "status": "DEFERRED_GPU_MEMORY_UNSAFE",
            "gpu": state,
        }
    quota = wave_main_quotas(wave_index)["qwen"]
    checkpoint_dir = source_checkpoint(wave_index)
    source_hash = checkpoint_fingerprint(checkpoint_dir)
    prompts = load_prompts()
    excluded = previous_wave_prompt_ids(wave_index)
    candidates = candidate_records(prompts, wave_index, excluded)
    excluded.update(item["id"] for item in candidates)
    long_records = long_prompt_records(prompts, wave_index, excluded)
    tokenizer = load_tokenizer()
    wave_dir = QWEN_CACHE_ROOT / f"wave-{wave_index}"
    wave_dir.mkdir(parents=True, exist_ok=True)
    cached = sorted(wave_dir.glob("trajectory-*.json"))
    records: list[dict[str, Any]] = []
    for path in cached:
        item = load_json(path)
        if item["source_checkpoint_fingerprint"] != source_hash:
            raise ValueError("P11.5a cached rollout source changed")
        records.append(item)
    generated_tokens = sum(int(item["rollout_tokens"]) for item in records)
    if generated_tokens > quota:
        raise ValueError("P11.5a cached Qwen tokens exceed the wave quota")

    if generated_tokens < quota:
        model, attnres, scope, student_load = load_probe_student(checkpoint_dir)
        model.eval()
        attnres.eval()
        index = len(records)
        long_index = max(0, index - len(candidates))
        while generated_tokens < quota:
            if index < len(candidates):
                record = candidates[index]
                role = "kimi_candidate"
                maximum = min(QWEN_CANDIDATE_ROLLOUT_TOKENS, quota - generated_tokens)
            else:
                record = long_records[long_index % len(long_records)]
                long_index += 1
                role = "main_opd"
                maximum = min(QWEN_LONG_ROLLOUT_TOKENS, quota - generated_tokens)
            item = rollout_one(
                model,
                tokenizer,
                record,
                maximum_tokens=maximum,
                namespace=f"wave-{wave_index}-trajectory-{index}",
            )
            item.update(
                {
                    "schema_version": 1,
                    "stage": "P11.5a-QWEN-EXACT-OPD-CACHE",
                    "wave_index": wave_index,
                    "trajectory_index": index,
                    "role": role,
                    "source_checkpoint": str(checkpoint_dir or P10_SOURCE),
                    "source_checkpoint_fingerprint": source_hash,
                    "qwen_scored": False,
                }
            )
            atomic_json(qwen_cache_path(wave_index, index), item)
            records.append(item)
            generated_tokens += int(item["rollout_tokens"])
            index += 1
            if index % 8 == 0 or generated_tokens == quota:
                print(
                    f"P11_5A_WAVE_{wave_index}_ROLLOUT_TOKENS={generated_tokens}/{quota}",
                    flush=True,
                )
        del model, attnres, scope
        release()
    else:
        student_load = {"status": "RESUMED_CACHED_ROLLOUTS"}

    from transformers import AutoModelForCausalLM

    pending = [item for item in records if not item.get("qwen_scored")]
    qwen_backend: dict[str, Any] | None = None
    if pending:
        qwen = AutoModelForCausalLM.from_pretrained(
            QWEN, dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=False
        )
        qwen_backend = force_qwen_torch_gdn_backend(qwen)
        qwen = qwen.cuda()
        qwen.eval()
        for number, item in enumerate(pending, start=1):
            tokens = torch.tensor([item["full_token_ids"]], dtype=torch.long, device="cuda")
            values = qwen_token_logprobs(
                qwen, tokens, target_start=int(item["prompt_tokens"])
            ).detach().float().cpu()[0]
            old = torch.tensor(item["old_student_logprobs"], dtype=torch.float32)
            if values.shape != old.shape:
                raise ValueError("P11.5a Qwen/student score shape mismatch")
            difference = values - old
            item.update(
                {
                    "qwen_teacher_logprobs": values.tolist(),
                    "qwen_student_advantage_mean": float(difference.mean()),
                    "qwen_student_abs_logprob_gap_mean": float(difference.abs().mean()),
                    "disagreement_score": float(difference.abs().mean() + difference.mean().clamp_min(0)),
                    "qwen_scored": True,
                    "qwen_scored_at": now(),
                }
            )
            atomic_json(qwen_cache_path(wave_index, int(item["trajectory_index"])), item)
            if number % 8 == 0 or number == len(pending):
                print(f"P11_5A_WAVE_{wave_index}_QWEN_SCORE={number}/{len(pending)}", flush=True)
        del qwen
        release()
    records = [load_json(path) for path in sorted(wave_dir.glob("trajectory-*.json"))]
    checks = {
        "qwen_tokens_exact": sum(int(item["rollout_tokens"]) for item in records) == quota,
        "candidate_pool_48": sum(item["role"] == "kimi_candidate" for item in records) == 48,
        "all_qwen_scored": all(item.get("qwen_scored") for item in records),
        "all_scores_finite": all(math.isfinite(float(item["disagreement_score"])) for item in records),
        "source_checkpoint_exact": all(item["source_checkpoint_fingerprint"] == source_hash for item in records),
        "heldout_not_accessed": True,
    }
    trajectories = [
        {
            "trajectory_index": int(item["trajectory_index"]),
            "prompt_id": item["id"],
            "domain": item["domain"],
            "role": item["role"],
            "rollout_tokens": int(item["rollout_tokens"]),
            "cache_path": str(qwen_cache_path(wave_index, int(item["trajectory_index"])).relative_to(ROOT)),
            "cache_sha256": sha256_file(qwen_cache_path(wave_index, int(item["trajectory_index"]))),
        }
        for item in records
    ]
    report = {
        "stage": "P11.5a-QWEN-EXACT-OPD-CACHE",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "completed_at": now(),
        "wave_index": wave_index,
        "source_checkpoint": str(checkpoint_dir or P10_SOURCE),
        "source_checkpoint_fingerprint": source_hash,
        "qwen_token_quota": quota,
        "student_load": student_load,
        "qwen_backend": qwen_backend or "RESUMED_CACHED_QWEN_SCORES",
        "trajectories": trajectories,
        "checks": checks,
        "heldout_accessed": False,
        "result_marker": f"P11_5A_WAVE_{wave_index}_QWEN_CACHE={'PASS' if all(checks.values()) else 'FAIL'}",
    }
    path = REPORT_DIR / f"wave-{wave_index}/qwen-opd-manifest.json"
    atomic_json(path, report)
    print(report["result_marker"], flush=True)
    return report


def request_upper_bound(config: KimiAPIConfig, messages: list[dict[str, str]], response_format: dict[str, Any]) -> float:
    payload = {
        "model": config.model,
        "messages": messages,
        "max_completion_tokens": MAX_KIMI_COMPLETION_TOKENS,
        "reasoning_effort": config.reasoning_effort,
        "stream": False,
        "response_format": response_format,
    }
    input_upper = len(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode())
    return (
        input_upper * config.input_cny_per_million
        + MAX_KIMI_COMPLETION_TOKENS * config.output_cny_per_million
    ) / 1_000_000


def all_probe_attempts() -> list[dict[str, Any]]:
    return [load_json(path) for path in sorted(KIMI_CACHE_ROOT.glob("wave-*/attempt-*.json"))]


def wave_selection(wave_index: int) -> dict[str, Any]:
    output = REPORT_DIR / f"wave-{wave_index}/kimi-selection.json"
    if output.is_file():
        return load_json(output)
    manifest = load_json(REPORT_DIR / f"wave-{wave_index}/qwen-opd-manifest.json")
    if manifest.get("status") != "PASS":
        raise ValueError("P11.5a Kimi selection requires a PASS Qwen wave cache")
    candidates = []
    for record in manifest["trajectories"]:
        if record["role"] == "kimi_candidate":
            candidates.append(load_json(ROOT / record["cache_path"]))
    selected = select_high_information(candidates, per_domain=KIMI_REVIEWS_PER_DOMAIN_PER_WAVE)
    checks = {
        "selected_16": len(selected) == KIMI_REVIEWS_PER_WAVE,
        "balanced_four_per_domain": all(
            sum(item["domain"] == domain for item in selected) == KIMI_REVIEWS_PER_DOMAIN_PER_WAVE
            for domain in DOMAINS
        ),
        "student_is_current_wave_policy": all(
            item["source_checkpoint_fingerprint"] == manifest["source_checkpoint_fingerprint"]
            for item in selected
        ),
        "heldout_not_accessed": True,
    }
    report = {
        "stage": "P11.5a-KIMI-HIGH-INFORMATION-SELECTION",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "wave_index": wave_index,
        "completed_at": now(),
        "selection_method": "TOP4_PER_DOMAIN_BY_QWEN_STUDENT_DISAGREEMENT",
        "selected": selected,
        "checks": checks,
        "api_calls": 0,
        "heldout_accessed": False,
    }
    atomic_json(output, report)
    return report


def run_wave_reviews(wave_index: int) -> dict[str, Any]:
    selection = wave_selection(wave_index)
    if selection.get("status") != "PASS":
        raise ValueError("P11.5a Kimi wave selection failed")
    config = KimiAPIConfig.from_json(API_CONFIG)
    if config.reasoning_effort != "low" or config.max_retries != 0:
        raise ValueError("P11.5a requires low reasoning and zero automatic retry")
    ledger = KimiAPICostLedger(REPORT_ROOT, config)
    teacher = KimiAPITeacher(config, ledger=ledger)
    tokenizer = load_tokenizer()
    wave_cache = KIMI_CACHE_ROOT / f"wave-{wave_index}"
    wave_cache.mkdir(parents=True, exist_ok=True)
    last_started: float | None = None
    stop_reason: str | None = None
    for slot, selected in enumerate(selection["selected"]):
        path = wave_cache / f"attempt-{slot:02d}.json"
        if path.is_file():
            existing = load_json(path)
            if existing.get("status") == "REQUEST_PENDING":
                existing.update(
                    {
                        "status": "UNCERTAIN_CHARGE_NO_RETRY",
                        "completed_at": now(),
                        "slot_consumed": True,
                        "reasoning_content_persisted": False,
                    }
                )
                atomic_json(path, existing)
            continue
        attempts = len(all_probe_attempts())
        messages, response_format = teacher.structured_trajectory_contract(
            selected["prompt"], selected["student_text"]
        )
        upper = request_upper_bound(config, messages, response_format)
        cumulative = accounted_ledger_cost(REPORT_ROOT)
        gate = api_call_allowed(
            cumulative_cost_cny=float(cumulative["accounted_cost_cny"]),
            next_upper_bound_cny=upper,
            attempts=attempts,
        )
        if not gate["allowed"]:
            stop_reason = str(gate["stop_reason"])
            break
        if last_started is not None:
            remaining = RATE_LIMIT_SECONDS - (time.monotonic() - last_started)
            if remaining > 0:
                print(f"P11_5A_KIMI_RATE_LIMIT_SECONDS={remaining:.3f}", flush=True)
                time.sleep(remaining)
        fingerprint = sha256_json(
            {
                "amendment": P11_5A_AMENDMENT,
                "wave": wave_index,
                "slot": slot,
                "selected_id": selected["id"],
                "student_text_sha256": selected["student_text_sha256"],
                "model": config.model,
                "reasoning_effort": config.reasoning_effort,
            }
        )
        pending = {
            "schema_version": 1,
            "stage": "P11.5a-KIMI-REVIEW",
            "status": "REQUEST_PENDING",
            "wave_index": wave_index,
            "slot": slot,
            "slot_consumed": True,
            "selected_id": selected["id"],
            "domain": selected["domain"],
            "prompt": selected["prompt"],
            "prompt_sha256": selected["prompt_sha256"],
            "student_text": selected["student_text"],
            "student_text_sha256": selected["student_text_sha256"],
            "request_fingerprint": fingerprint,
            "request_upper_bound_cny": upper,
            "p11_cumulative_cost_before_request_cny": cumulative["accounted_cost_cny"],
            "reasoning_content_persisted": False,
            "reasoning_content_trained": False,
            "api_secret_persisted": False,
            "started_at": now(),
        }
        atomic_json(path, pending)
        last_started = time.monotonic()
        try:
            trajectory, result = teacher.structured_trajectory(
                selected["prompt"],
                selected["student_text"],
                max_completion_tokens=MAX_KIMI_COMPLETION_TOKENS,
            )
            corrected = trajectory.corrected_text or ""
            corrected_tokens = (
                int(tokenizer(corrected, add_special_tokens=False, return_tensors="pt").input_ids.numel())
                if corrected
                else 0
            )
            pending.update(
                {
                    "status": "SUCCEEDED",
                    "decision": "keep" if trajectory.mode == "KEEP" else "correct",
                    "corrected_answer": corrected,
                    "corrected_target_tokens": corrected_tokens,
                    "teacher_model": result.model,
                    "finish_reason": result.finish_reason,
                    "usage": result.usage,
                    "cost_cny": result.cost_cny,
                    "teacher_response_sha256": sha256_text(result.text),
                    "structured_output_valid": True,
                    "reasoning_content_present": result.reasoning_content_present,
                    "reasoning_content_characters": result.reasoning_content_characters,
                    "completed_at": now(),
                }
            )
            atomic_json(path, pending)
            print(
                f"P11_5A_WAVE_{wave_index}_KIMI_SLOT={slot + 1}/16 "
                f"DECISION={pending['decision'].upper()} COST_CNY={result.cost_cny:.6f}",
                flush=True,
            )
        except Exception as error:
            pending.update(
                {
                    "status": "FAILED_NO_RETRY",
                    "exception_type": type(error).__name__,
                    "exception": str(error),
                    "completed_at": now(),
                }
            )
            atomic_json(path, pending)
            stop_reason = "PAUSED_AFTER_SINGLE_API_FAILURE"
            break
    attempts = [load_json(path) for path in sorted(wave_cache.glob("attempt-*.json"))]
    decisions = Counter(item.get("decision", "failed") for item in attempts)
    complete = len(attempts) == KIMI_REVIEWS_PER_WAVE or stop_reason == "P11_CUMULATIVE_API_HARD_CAP"
    ledger_summary = accounted_ledger_cost(REPORT_ROOT)
    checks = {
        "attempts_not_above_wave_cap": len(attempts) <= KIMI_REVIEWS_PER_WAVE,
        "attempts_not_above_global_cap": len(all_probe_attempts()) <= KIMI_MAX_REVIEW_ATTEMPTS,
        "p11_cumulative_cost_not_above_5": ledger_summary["accounted_cost_cny"] <= P11_API_TOTAL_HARD_CAP_CNY,
        "reasoning_content_not_persisted": all(item.get("reasoning_content_persisted") is False for item in attempts),
        "no_automatic_retry": True,
        "heldout_not_accessed": True,
    }
    report = {
        "stage": "P11.5a-KIMI-REVIEW-WAVE",
        "status": "PASS" if complete and all(checks.values()) else "PAUSED",
        "completed_at": now(),
        "wave_index": wave_index,
        "attempts": len(attempts),
        "successful": sum(item.get("status") == "SUCCEEDED" for item in attempts),
        "keep": decisions["keep"],
        "correct": decisions["correct"],
        "correction_rate": (
            decisions["correct"] / (decisions["keep"] + decisions["correct"])
            if decisions["keep"] + decisions["correct"]
            else None
        ),
        "stop_reason": stop_reason,
        "p11_cumulative_api_ledger": ledger_summary,
        "attempt_paths": [str((wave_cache / f"attempt-{index:02d}.json").relative_to(ROOT)) for index in range(len(attempts))],
        "checks": checks,
        "heldout_accessed": False,
        "result_marker": (
            f"P11_5A_WAVE_{wave_index}_KIMI=PASS"
            if complete and all(checks.values())
            else f"P11_5A_WAVE_{wave_index}_KIMI=PAUSED"
        ),
    }
    atomic_json(REPORT_DIR / f"wave-{wave_index}/kimi-wave.json", report)
    print(report["result_marker"], flush=True)
    return report


def run_wave(wave_index: int) -> dict[str, Any]:
    qwen = build_wave_qwen_cache(wave_index)
    if qwen.get("status") != "PASS":
        return qwen
    selection = wave_selection(wave_index)
    if selection.get("status") != "PASS":
        return selection
    return run_wave_reviews(wave_index)


def chunked_target_ce(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    *,
    target_start: int,
    lm_head_weight: torch.Tensor,
    chunk_size: int = 16,
) -> torch.Tensor:
    predictors = hidden[:, target_start - 1 : -1]
    targets = labels[:, target_start:]
    total = targets.numel()
    if total <= 0 or predictors.shape[:2] != targets.shape:
        raise ValueError("Invalid target-only CE tensors")
    values = []
    for start in range(0, targets.shape[1], chunk_size):
        end = min(start + chunk_size, targets.shape[1])

        def loss_chunk(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            logits = F.linear(value, lm_head_weight).float()
            return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="sum")

        if torch.is_grad_enabled():
            values.append(
                checkpoint(
                    loss_chunk,
                    predictors[:, start:end],
                    targets[:, start:end],
                    use_reentrant=False,
                )
            )
        else:
            values.append(loss_chunk(predictors[:, start:end], targets[:, start:end]))
    return torch.stack(values).sum() / total


def chunked_opd_loss(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    *,
    target_start: int,
    advantage: torch.Tensor,
    lm_head_weight: torch.Tensor,
    chunk_size: int = 16,
) -> torch.Tensor:
    predictors = hidden[:, target_start - 1 : -1]
    targets = labels[:, target_start:]
    if advantage.shape != targets.shape or targets.numel() == 0:
        raise ValueError("Invalid P11.5a OPD tensors")
    values = []
    for start in range(0, targets.shape[1], chunk_size):
        end = min(start + chunk_size, targets.shape[1])

        def loss_chunk(
            value: torch.Tensor,
            target: torch.Tensor,
            token_advantage: torch.Tensor,
        ) -> torch.Tensor:
            logits = F.linear(value, lm_head_weight).float()
            logprob = torch.log_softmax(logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
            return -(token_advantage * logprob).sum()

        values.append(
            checkpoint(
                loss_chunk,
                predictors[:, start:end],
                targets[:, start:end],
                advantage[:, start:end],
                use_reentrant=False,
            )
        )
    return torch.stack(values).sum() / targets.numel()


@torch.inference_mode()
def correction_metrics(model: torch.nn.Module, tokenizer: Any, attempts: list[dict[str, Any]]) -> dict[str, Any]:
    values = []
    for item in attempts:
        if item.get("status") != "SUCCEEDED" or item.get("decision") != "correct":
            continue
        prompt = tokenizer(
            item["prompt"], return_tensors="pt", truncation=True, max_length=MAX_PROMPT_TOKENS
        ).input_ids
        corrected = tokenizer(
            item["corrected_answer"], add_special_tokens=False, return_tensors="pt", truncation=True, max_length=768
        ).input_ids
        student = tokenizer(
            item["student_text"], add_special_tokens=False, return_tensors="pt", truncation=True, max_length=768
        ).input_ids
        losses = {}
        for name, target in (("corrected", corrected), ("student", student)):
            if target.numel() == 0:
                losses[name] = None
                continue
            sequence = torch.cat((prompt, target), dim=1).cuda()
            hidden = model.model(sequence, use_cache=False, return_dict=True).last_hidden_state
            loss = chunked_target_ce(
                hidden,
                sequence,
                target_start=int(prompt.shape[1]),
                lm_head_weight=model.lm_head.weight,
            )
            losses[name] = float(loss)
        if losses["corrected"] is not None and losses["student"] is not None:
            values.append(
                {
                    "slot": item["slot"],
                    "domain": item["domain"],
                    "corrected_ce": losses["corrected"],
                    "student_trajectory_ce": losses["student"],
                    "corrected_preference": losses["student"] - losses["corrected"],
                }
            )
    return {
        "corrected_trajectories": len(values),
        "mean_corrected_ce": sum(item["corrected_ce"] for item in values) / len(values) if values else None,
        "mean_corrected_preference": sum(item["corrected_preference"] for item in values) / len(values) if values else None,
        "trajectories": values,
    }


def progress_paths() -> tuple[Path, Path, Path]:
    return (
        PROGRESS_ROOT / "trainable.safetensors",
        PROGRESS_ROOT / "training-state.pt",
        PROGRESS_ROOT / "checkpoint-manifest.json",
    )


def save_training_snapshot(
    directory: Path,
    *,
    scope: Any,
    optimizer: Any,
    scaler: Any,
    wave_index: int,
    completed_bucket: int,
    global_optimizer_step: int,
    evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    weights = directory / "trainable.safetensors"
    temporary_weights = weights.with_suffix(".safetensors.tmp")
    save_file(trainable_state(scope), str(temporary_weights))
    temporary_weights.replace(weights)
    training_state = directory / "training-state.pt"
    atomic_torch_save(
        training_state,
        {
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "wave_index": wave_index,
            "completed_bucket": completed_bucket,
            "global_optimizer_step": global_optimizer_step,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state": torch.cuda.get_rng_state_all(),
        },
    )
    manifest = {
        "stage": "P11.5a-TRAINING-CHECKPOINT",
        "status": "PASS",
        "completed_at": now(),
        "wave_index": wave_index,
        "completed_bucket": completed_bucket,
        "global_optimizer_step": global_optimizer_step,
        "completed_main_tokens": (
            cumulative_main_quotas(wave_index + 1)["main"]
            if completed_bucket == UPDATE_BUCKETS_PER_WAVE
            else cumulative_main_quotas(wave_index)["main"]
        ),
        "trainable_sha256": sha256_file(weights),
        "training_state_sha256": sha256_file(training_state),
        "evaluation": evaluation,
        "heldout_accessed": False,
    }
    atomic_json(directory / "checkpoint-manifest.json", manifest)
    return manifest


def restore_training_snapshot(
    directory: Path,
    *,
    scope: Any,
    optimizer: Any,
    scaler: Any,
) -> dict[str, Any]:
    manifest = load_json(directory / "checkpoint-manifest.json")
    weights = directory / "trainable.safetensors"
    training_state = directory / "training-state.pt"
    if (
        sha256_file(weights) != manifest["trainable_sha256"]
        or sha256_file(training_state) != manifest["training_state_sha256"]
    ):
        raise ValueError("P11.5a training snapshot hash mismatch")
    load_trainable_state(scope, weights)
    state = torch.load(training_state, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_random_state"])
    torch.cuda.set_rng_state_all(state["cuda_random_state"])
    return state


def run_segment(wave_index: int) -> dict[str, Any]:
    segment_report_path = REPORT_DIR / f"wave-{wave_index}/training-segment.json"
    if segment_report_path.is_file():
        existing = load_json(segment_report_path)
        checkpoint_dir = segment_checkpoint(wave_index)
        manifest_path = checkpoint_dir / "checkpoint-manifest.json"
        if existing.get("status") == "PASS" and manifest_path.is_file():
            manifest = load_json(manifest_path)
            if sha256_file(checkpoint_dir / "trainable.safetensors") == manifest["trainable_sha256"]:
                print(f"P11_5A_SEGMENT_{wave_index}=RESUMED", flush=True)
                return existing
    wave_report = load_json(REPORT_DIR / f"wave-{wave_index}/kimi-wave.json")
    if wave_report.get("status") != "PASS":
        raise ValueError("P11.5a segment requires completed Kimi wave or monetary hard-cap stop")
    state = p11_gpu_start_state(stage="training-segment", wave_index=wave_index)
    if not state["allowed"]:
        return {
            "stage": "P11.5a-SEGMENT",
            "status": "DEFERRED_GPU_MEMORY_UNSAFE",
            "gpu": state,
        }
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    print("P11_5A_FORMAL_TRAINING_STARTED=true", flush=True)
    quota = wave_main_quotas(wave_index)
    qwen_manifest = load_json(REPORT_DIR / f"wave-{wave_index}/qwen-opd-manifest.json")
    qwen_records = [load_json(ROOT / item["cache_path"]) for item in qwen_manifest["trajectories"]]
    qwen_lengths = [int(item["rollout_tokens"]) for item in qwen_records]
    qwen_buckets = assign_update_buckets(qwen_lengths)
    raw_lengths = raw_lm_target_lengths(quota["raw_lm"])
    raw_buckets = assign_update_buckets(raw_lengths)
    attempts = [load_json(ROOT / path) for path in wave_report["attempt_paths"]]
    corrections = [item for item in attempts if item.get("status") == "SUCCEEDED" and item.get("decision") == "correct"]
    checkpoint_dir = source_checkpoint(wave_index)
    model, attnres, scope, student_load = load_probe_student(checkpoint_dir)
    frozen_before = frozen_scope_audit(model, scope)
    if wave_index == 0:
        initial_values = trainable_state(scope)
        CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
        save_file(initial_values, str(CHECKPOINT_ROOT / "step-zero-trainable.safetensors"))
    else:
        initial_values = load_file(str(CHECKPOINT_ROOT / "step-zero-trainable.safetensors"))
    masters, optimizer, _p10_scaler = build_optimizer(scope, config)
    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=float(config["optimization"]["fp16_initial_loss_scale"]),
        growth_interval=int(
            config["optimization"]["fp16_loss_scale_growth_interval"]
        ),
    )
    start_bucket = 0
    global_step = wave_index * UPDATE_BUCKETS_PER_WAVE
    _progress_weights, _progress_state, progress_manifest = progress_paths()
    restored_progress = False
    if progress_manifest.is_file():
        progress = load_json(progress_manifest)
        if int(progress["wave_index"]) == wave_index and int(progress["completed_bucket"]) < UPDATE_BUCKETS_PER_WAVE:
            restore_training_snapshot(PROGRESS_ROOT, scope=scope, optimizer=optimizer, scaler=scaler)
            start_bucket = int(progress["completed_bucket"])
            global_step = int(progress["global_optimizer_step"])
            restored_progress = True
    if not restored_progress and wave_index > 0:
        prior = segment_checkpoint(wave_index - 1)
        restore_training_snapshot(prior, scope=scope, optimizer=optimizer, scaler=scaler)
        start_bucket = 0
        global_step = wave_index * UPDATE_BUCKETS_PER_WAVE
    tokenizer = load_tokenizer()
    model.eval()
    attnres.eval()
    correction_before = correction_metrics(model, tokenizer, corrections)
    model.train()
    attnres.train()
    model.model.gradient_checkpointing = True
    raw_dataset = PackedTokenDataset(RAW_DATA, "train")
    recipe = config["optimization"]
    training_curve = []
    total_optimizer_steps = WAVE_COUNT * UPDATE_BUCKETS_PER_WAVE
    for bucket in range(start_bucket, UPDATE_BUCKETS_PER_WAVE):
        lora_lr = cosine_learning_rate(
            global_step,
            total_optimizer_steps,
            float(recipe["lora_learning_rate"]),
            warmup_ratio=float(recipe["warmup_ratio"]),
            minimum_ratio=float(recipe["minimum_learning_rate_ratio"]),
        )
        direct_lr = cosine_learning_rate(
            global_step,
            total_optimizer_steps,
            float(recipe["direct_learning_rate"]),
            warmup_ratio=float(recipe["warmup_ratio"]),
            minimum_ratio=float(recipe["minimum_learning_rate_ratio"]),
        )
        optimizer.param_groups[0]["lr"] = lora_lr
        optimizer.param_groups[1]["lr"] = direct_lr
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        attnres.zero_grad(set_to_none=True)
        loss_log: dict[str, list[float]] = {"qwen": [], "raw_lm": [], "kimi": []}
        for index in qwen_buckets[bucket]:
            item = qwen_records[index]
            tokens = torch.tensor([item["full_token_ids"]], dtype=torch.long, device="cuda")
            hidden = model.model(tokens, use_cache=False, return_dict=True).last_hidden_state
            old = torch.tensor([item["old_student_logprobs"]], dtype=torch.float32, device="cuda")
            teacher = torch.tensor([item["qwen_teacher_logprobs"]], dtype=torch.float32, device="cuda")
            advantage = exact_token_advantage(teacher, old)
            loss = chunked_opd_loss(
                hidden,
                tokens,
                target_start=int(item["prompt_tokens"]),
                advantage=advantage,
                lm_head_weight=model.lm_head.weight,
            )
            weight = STREAM_LOSS_WEIGHTS["qwen"] * UPDATE_BUCKETS_PER_WAVE * int(item["rollout_tokens"]) / quota["qwen"]
            scaler.scale(weight * loss).backward()
            loss_log["qwen"].append(float(loss.detach()))
            del tokens, hidden, old, teacher, advantage, loss
        for local_index in raw_buckets[bucket]:
            target_tokens = raw_lengths[local_index]
            dataset_index = wave_index * len(raw_lengths) + local_index
            tokens = torch.from_numpy(raw_dataset[dataset_index][: target_tokens + 1].copy()).unsqueeze(0).cuda()
            hidden = model.model(tokens, use_cache=False, return_dict=True).last_hidden_state
            loss = chunked_causal_lm_loss(
                hidden, tokens, model.lm_head.weight, chunk_size=16, checkpoint_chunks=True
            )
            weight = STREAM_LOSS_WEIGHTS["raw_lm"] * UPDATE_BUCKETS_PER_WAVE * target_tokens / quota["raw_lm"]
            scaler.scale(weight * loss).backward()
            loss_log["raw_lm"].append(float(loss.detach()))
            del tokens, hidden, loss
        if bucket < len(corrections):
            item = corrections[bucket]
            prompt = tokenizer(
                item["prompt"], return_tensors="pt", truncation=True, max_length=MAX_PROMPT_TOKENS
            ).input_ids
            target = tokenizer(
                item["corrected_answer"], add_special_tokens=False, return_tensors="pt", truncation=True, max_length=768
            ).input_ids
            tokens = torch.cat((prompt, target), dim=1).cuda()
            hidden = model.model(tokens, use_cache=False, return_dict=True).last_hidden_state
            loss = chunked_target_ce(
                hidden,
                tokens,
                target_start=int(prompt.shape[1]),
                lm_head_weight=model.lm_head.weight,
            )
            scaler.scale(STREAM_LOSS_WEIGHTS["kimi"] * loss).backward()
            loss_log["kimi"].append(float(loss.detach()))
            del prompt, target, tokens, hidden, loss
        masters.copy_scaled_grads_from_model()
        scaler.unscale_(optimizer)
        parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        parameter_names = optimizer_parameter_names(scope, masters)
        coverage = audit_gradient_coverage(parameters, parameter_names)
        unexpected = unexpected_gradient_names(model, attnres, scope)
        gradients = [parameter for parameter in parameters if parameter.grad is not None]
        nonfinite_gradient_names = [
            parameter_names[id(parameter)]
            for parameter in gradients
            if not bool(torch.isfinite(parameter.grad).all())
        ]
        finite = not nonfinite_gradient_names
        gradient_norm_by_family = optimizer_gradient_norm_by_family(scope, masters)
        gradient_norm = torch.nn.utils.clip_grad_norm_(gradients, float(recipe["gradient_clip_norm"]))
        if not coverage["complete"] or coverage["optimizer_tensors"] != 563 or unexpected:
            raise RuntimeError(f"P11.5a gradient contract failed: coverage={coverage} unexpected={unexpected[:8]}")
        if not finite or not math.isfinite(float(gradient_norm)):
            atomic_json(
                REPORT_DIR / f"wave-{wave_index}/gradient-failure.json",
                {
                    "stage": "P11.5a-GRADIENT-DIAGNOSTIC",
                    "status": "FAIL_NONFINITE_GRADIENTS",
                    "completed_at": now(),
                    "wave_index": wave_index,
                    "bucket": bucket + 1,
                    "global_optimizer_step": global_step,
                    "loss_scale": scaler.get_scale(),
                    "nonfinite_gradient_count": len(nonfinite_gradient_names),
                    "nonfinite_gradient_names": nonfinite_gradient_names,
                    "gradient_tensors": coverage["gradient_tensors"],
                    "optimizer_tensors": coverage["optimizer_tensors"],
                    "losses": loss_log,
                    "optimizer_step_performed": False,
                    "heldout_accessed": False,
                },
            )
            raise FloatingPointError("P11.5a gradients are not finite")
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        masters.copy_to_model()
        global_step += 1
        training_curve.append(
            {
                "bucket": bucket + 1,
                "global_optimizer_step": global_step,
                "lora_learning_rate": lora_lr,
                "direct_learning_rate": direct_lr,
                "gradient_tensors": coverage["gradient_tensors"],
                "optimizer_tensors": coverage["optimizer_tensors"],
                "missing_gradient_count": coverage["missing_gradient_count"],
                "unexpected_gradient_count": len(unexpected),
                "gradient_norm_before_clip": float(gradient_norm),
                "gradient_norm_by_family_before_clip": gradient_norm_by_family,
                "grad_scaler_before": scale_before,
                "grad_scaler_after": scaler.get_scale(),
                "losses": loss_log,
            }
        )
        save_training_snapshot(
            PROGRESS_ROOT,
            scope=scope,
            optimizer=optimizer,
            scaler=scaler,
            wave_index=wave_index,
            completed_bucket=bucket + 1,
            global_optimizer_step=global_step,
        )
        print(
            f"P11_5A_SEGMENT={wave_index} BUCKET={bucket + 1}/16 "
            f"GRADIENTS={coverage['gradient_tensors']}/{coverage['optimizer_tensors']}",
            flush=True,
        )
    model.eval()
    attnres.eval()
    correction_after = correction_metrics(model, tokenizer, corrections)
    diagnostic_tokens = torch.from_numpy(PackedTokenDataset(RAW_DATA, "dev")[0][:128].copy()).unsqueeze(0).cuda()
    routing = routing_diagnostic(model, attnres, diagnostic_tokens)
    updates = parameter_update_by_family(scope, initial_values)
    compensation = compensation_warning(updates, routing)
    frozen_after = frozen_scope_audit(model, scope)
    qwen_validation = qwen_records[: min(8, len(qwen_records))]
    qwen_gaps = []
    with torch.inference_mode():
        for item in qwen_validation:
            tokens = torch.tensor([item["full_token_ids"]], dtype=torch.long, device="cuda")
            hidden = model.model(tokens, use_cache=False, return_dict=True).last_hidden_state
            predictors = hidden[:, int(item["prompt_tokens"]) - 1 : -1]
            targets = tokens[:, int(item["prompt_tokens"]):]
            current_parts = []
            for start in range(0, targets.shape[1], 16):
                end = min(start + 16, targets.shape[1])
                logits = F.linear(predictors[:, start:end], model.lm_head.weight).float()
                current_parts.append(
                    torch.log_softmax(logits, dim=-1)
                    .gather(-1, targets[:, start:end].unsqueeze(-1))
                    .squeeze(-1)
                    .cpu()
                )
            current = torch.cat(current_parts, dim=1)[0]
            teacher = torch.tensor(item["qwen_teacher_logprobs"])
            old = torch.tensor(item["old_student_logprobs"])
            qwen_gaps.append(
                {
                    "trajectory_index": item["trajectory_index"],
                    "old_abs_gap": float((teacher - old).abs().mean()),
                    "current_abs_gap": float((teacher - current).abs().mean()),
                }
            )
    evaluation = {
        "correction_before": correction_before,
        "correction_after": correction_after,
        "qwen_validation": qwen_gaps,
        "routing": routing,
        "parameter_update_by_family": updates,
        "kda_freeze_compensation": compensation,
        "gradient_contract": {
            "expected": 563,
            "observed_last": training_curve[-1]["gradient_tensors"],
            "missing_last": training_curve[-1]["missing_gradient_count"],
            "unexpected_last": training_curve[-1]["unexpected_gradient_count"],
        },
    }
    final_checkpoint = segment_checkpoint(wave_index)
    checkpoint_manifest = save_training_snapshot(
        final_checkpoint,
        scope=scope,
        optimizer=optimizer,
        scaler=scaler,
        wave_index=wave_index,
        completed_bucket=UPDATE_BUCKETS_PER_WAVE,
        global_optimizer_step=global_step,
        evaluation=evaluation,
    )
    checks = {
        "qwen_tokens_exact": sum(qwen_lengths) == quota["qwen"],
        "raw_lm_tokens_exact": sum(raw_lengths) == quota["raw_lm"],
        "gradient_coverage_563": all(item["gradient_tensors"] == 563 for item in training_curve),
        "missing_zero": all(item["missing_gradient_count"] == 0 for item in training_curve),
        "unexpected_zero": all(item["unexpected_gradient_count"] == 0 for item in training_curve),
        "frozen_kda_exact": frozen_before["tensor_hash"] == frozen_after["tensor_hash"],
        "attnres_active": routing["active"],
        "compensation_not_high": compensation["status"] != "HIGH",
        "heldout_not_accessed": True,
    }
    report = {
        "stage": "P11.5a-TRAINING-SEGMENT",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "completed_at": now(),
        "wave_index": wave_index,
        "main_tokens_before": cumulative_main_quotas(wave_index)["main"],
        "main_tokens_after": cumulative_main_quotas(wave_index + 1)["main"],
        "qwen_supervised_tokens": sum(qwen_lengths),
        "raw_lm_supervised_tokens": sum(raw_lengths),
        "kimi_corrected_trajectories": len(corrections),
        "kimi_tokens_in_main_denominator": False,
        "kimi_loss_reduction": "PER_TRAJECTORY_MEAN_TARGET_CE",
        "p11_5a_formal_training_started": True,
        "training_curve": training_curve,
        "evaluation": evaluation,
        "source_checkpoint": str(checkpoint_dir or P10_SOURCE),
        "student_load": student_load,
        "checkpoint": checkpoint_manifest,
        "checks": checks,
        "heldout_accessed": False,
        "result_marker": f"P11_5A_SEGMENT_{wave_index}={'PASS' if all(checks.values()) else 'FAIL'}",
    }
    atomic_json(segment_report_path, report)
    print(report["result_marker"], flush=True)
    return report


def strict_instruction_pass(text: str, scoring: dict[str, Any]) -> bool:
    checks = []
    for instruction, kwargs in zip(scoring["instruction_id_list"], scoring["kwargs"], strict=True):
        if instruction == "change_case:english_lowercase":
            checks.append(not any(character.isascii() and character.isupper() for character in text))
        elif instruction == "change_case:english_capital":
            checks.append(not any(character.isascii() and character.islower() for character in text))
        elif instruction == "punctuation:no_comma":
            checks.append("," not in text)
        elif instruction == "keywords:forbidden_words":
            checks.append(all(word.lower() not in text.lower() for word in kwargs["forbidden_words"]))
        elif instruction == "detectable_format:json_format":
            try:
                json.loads(text)
                checks.append(True)
            except json.JSONDecodeError:
                checks.append(False)
        elif instruction == "startend:quotation":
            stripped = text.strip()
            checks.append(len(stripped) >= 2 and stripped[0] == '"' and stripped[-1] == '"')
        else:
            raise ValueError(f"Unsupported frozen P11 general-dev instruction: {instruction}")
    return all(checks)


@torch.inference_mode()
def general_capability(model: torch.nn.Module, tokenizer: Any, *, label: str) -> dict[str, Any]:
    records = [json.loads(line) for line in GENERAL_DEV.read_text(encoding="utf-8").splitlines() if line.strip()]
    results = []
    for index, record in enumerate(records, start=1):
        scoring = record["scoring"]
        prompt = tokenizer(record["prompt"], return_tensors="pt", truncation=True, max_length=1024).input_ids.cuda()
        if scoring["type"] == "MULTIPLE_CHOICE":
            scores = {}
            for letter in "ABCD":
                target = tokenizer(f"ANSWER: {letter}", add_special_tokens=False, return_tensors="pt").input_ids.cuda()
                sequence = torch.cat((prompt, target), dim=1)
                hidden = model.model(sequence, use_cache=False, return_dict=True).last_hidden_state
                predictors = hidden[:, prompt.shape[1] - 1 : -1]
                logits = F.linear(predictors.float(), model.lm_head.weight.float())
                logprob = torch.log_softmax(logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
                scores[letter] = float(logprob.mean())
            prediction = max(scores, key=scores.get)
            passed = prediction == scoring["answer"]
            response_hash = None
        else:
            cache = TitanP7HybridCache(model.config)
            output = model(prompt, past_key_values=cache, use_cache=True, return_dict=True)
            generated = []
            for _ in range(256):
                token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
                generated.append(int(token.item()))
                if generated[-1] == tokenizer.eos_token_id:
                    break
                output = model(token, past_key_values=output.past_key_values, use_cache=True, return_dict=True)
            response = tokenizer.decode(generated, skip_special_tokens=True)
            passed = strict_instruction_pass(response, scoring)
            prediction = None
            response_hash = sha256_text(response)
        results.append(
            {
                "id": record["id"],
                "domain": record["domain"],
                "passed": passed,
                "prediction": prediction,
                "response_sha256": response_hash,
            }
        )
        if index % 8 == 0 or index == len(records):
            print(f"P11_5A_GENERAL_{label}_PROGRESS={index}/{len(records)}", flush=True)
    domains = {}
    for domain in DOMAINS:
        values = [item for item in results if item["domain"] == domain]
        domains[domain] = sum(item["passed"] for item in values) / len(values)
    return {
        "records": len(results),
        "accuracy": sum(item["passed"] for item in results) / len(results),
        "domain_accuracy": domains,
        "general_composite": sum(domains.values()) / len(domains),
        "results": results,
    }


def run_baseline() -> dict[str, Any]:
    if BASELINE_REPORT.is_file() and load_json(BASELINE_REPORT).get("status") == "PASS":
        report = load_json(BASELINE_REPORT)
        print("P11_5A_P10T_BASELINE=RESUMED", flush=True)
        return report
    state = p11_gpu_start_state(stage="baseline")
    if not state["allowed"]:
        return {
            "stage": "P11.5a-BASELINE",
            "status": "DEFERRED_GPU_MEMORY_UNSAFE",
            "gpu": state,
        }
    model, attnres, scope, load = load_probe_student(None)
    model.eval()
    attnres.eval()
    tokenizer = load_tokenizer()
    capability = general_capability(model, tokenizer, label="P10T")
    report = {
        "stage": "P11.5a-P10T-BASELINE",
        "status": "PASS",
        "completed_at": now(),
        "general_capability": capability,
        "p10_t_full_dev_ce": 2.38968004745761,
        "source": str(P10_SOURCE),
        "source_fingerprint": checkpoint_fingerprint(None),
        "student_load": load,
        "heldout_accessed": False,
        "training_performed": False,
        "result_marker": "P11_5A_P10T_BASELINE=PASS",
    }
    atomic_json(BASELINE_REPORT, report)
    print(report["result_marker"], flush=True)
    return report


def run_closeout() -> dict[str, Any]:
    state = p11_gpu_start_state(stage="closeout")
    envelope = {
        "stage": "P11.5a-CLOSEOUT",
        "status": "RUNNING",
        "started_at": now(),
        "gpu": state,
        "heldout_accessed": False,
        "p11_6_phase1_allowed": False,
        "p12_titan_migration_allowed": False,
    }
    atomic_json(FINAL_REPORT, envelope)
    if not state["allowed"]:
        envelope.update(
            {
                "status": "DEFERRED_GPU_MEMORY_UNSAFE",
                "completed_at": now(),
                "gpu": state,
            }
        )
        atomic_json(FINAL_REPORT, envelope)
        return envelope
    final_checkpoint = segment_checkpoint(WAVE_COUNT - 1)
    baseline = load_json(BASELINE_REPORT)
    model, attnres, scope, load = load_probe_student(final_checkpoint)
    model.eval()
    attnres.eval()
    tokenizer = load_tokenizer()
    dev = PackedTokenDataset(RAW_DATA, "dev")
    full_dev_ce = corpus_ce(model, dev, label="P11_5A_CLOSEOUT")
    capability = general_capability(model, tokenizer, label="FINAL")
    diagnostic_tokens = torch.from_numpy(dev[0][:128].copy()).unsqueeze(0).cuda()
    routing = routing_diagnostic(model, attnres, diagnostic_tokens)
    initial = load_file(str(CHECKPOINT_ROOT / "step-zero-trainable.safetensors"))
    updates = parameter_update_by_family(scope, initial)
    compensation = compensation_warning(updates, routing)
    frozen = frozen_scope_audit(model, scope)
    from attnres import disable_block_attnres_runtime, enable_block_attnres_runtime

    disable_block_attnres_runtime(model)
    generated = generation(model, tokenizer, attnres)
    cache_limits = yaml.safe_load(
        (TITAN / "configs/p6-titan-fp16-cache-compat-2026-08-19-01.yaml").read_text(encoding="utf-8")
    )["evaluation"]["fallback_requires_all"]
    cache = cache_equivalence(model, attnres, cache_limits)
    enable_block_attnres_runtime(model, attnres)
    segments = [load_json(REPORT_DIR / f"wave-{index}/training-segment.json") for index in range(WAVE_COUNT)]
    correction_before = [
        item["evaluation"]["correction_before"]["mean_corrected_ce"]
        for item in segments
        if item["evaluation"]["correction_before"]["mean_corrected_ce"] is not None
    ]
    correction_after = [
        item["evaluation"]["correction_after"]["mean_corrected_ce"]
        for item in segments
        if item["evaluation"]["correction_after"]["mean_corrected_ce"] is not None
    ]
    kimi_learned = bool(correction_before) and (
        sum(correction_after) / len(correction_after)
        < sum(correction_before) / len(correction_before)
    )
    general_delta = capability["general_composite"] - baseline["general_capability"]["general_composite"]
    checks = {
        "all_segments_pass": all(item["status"] == "PASS" for item in segments),
        "training_finite": all(
            math.isfinite(float(point["gradient_norm_before_clip"]))
            for item in segments
            for point in item["training_curve"]
        ),
        "gradient_coverage_563": all(
            point["gradient_tensors"] == 563 and point["optimizer_tensors"] == 563
            for item in segments
            for point in item["training_curve"]
        ),
        "missing_zero": all(
            point["missing_gradient_count"] == 0
            for item in segments
            for point in item["training_curve"]
        ),
        "unexpected_zero": all(
            point["unexpected_gradient_count"] == 0
            for item in segments
            for point in item["training_curve"]
        ),
        "frozen_kda_hash_unchanged": (
            all(item["checks"]["frozen_kda_exact"] for item in segments)
            and not frozen["missing"]
        ),
        "general_capability_no_material_regression": general_delta >= -0.05,
        "qwen_retention_healthy": full_dev_ce <= 2.41968004745761,
        "kimi_correction_signal_learned": kimi_learned,
        "raw_lm_full_dev_ce": full_dev_ce <= 2.41968004745761,
        "attnres_active": routing["active"],
        "kda_compensation_not_high": compensation["status"] != "HIGH",
        "latent_cache_pass": cache["status"] == "PASS",
        "generation_pass": all(item["finite"] for item in generated),
        "heldout_not_accessed": True,
    }
    passed = all(checks.values())
    ledger = accounted_ledger_cost(REPORT_ROOT)
    report = {
        **envelope,
        "status": "PASS" if passed else "FAIL",
        "completed_at": now(),
        "amendment": P11_5A_AMENDMENT,
        "source": str(P10_SOURCE),
        "checkpoint": str(final_checkpoint),
        "main_training_tokens": cumulative_main_quotas(WAVE_COUNT),
        "full_dev_ce": full_dev_ce,
        "p10_t_full_dev_ce": 2.38968004745761,
        "full_dev_ce_delta": full_dev_ce - 2.38968004745761,
        "general_capability": capability,
        "p10_t_general_capability": baseline["general_capability"],
        "general_composite_delta": general_delta,
        "kimi_correction_ce_before_mean": sum(correction_before) / len(correction_before) if correction_before else None,
        "kimi_correction_ce_after_mean": sum(correction_after) / len(correction_after) if correction_after else None,
        "p11_cumulative_api_ledger": ledger,
        "routing": routing,
        "parameter_update_by_family": updates,
        "kda_freeze_compensation": compensation,
        "cache": cache,
        "generation": generated,
        "student_load": load,
        "checks": checks,
        "p11_5a_probe": "PASS" if passed else "FAIL",
        "p11_6_phase1_allowed": passed,
        "p12_titan_migration_allowed": False,
        "heldout_accessed": False,
        "canonical_p11_status": "PENDING_NATIVE_BF16_FULL_SCOPE_RETRAIN",
        "result_marker": "P11_5A_PROBE=PASS" if passed else "P11_5A_PROBE=FAIL",
    }
    atomic_json(FINAL_REPORT, report)
    print(report["result_marker"], flush=True)
    print(f"P11_6_PHASE1_ALLOWED={str(passed).lower()}", flush=True)
    print("P12_TITAN_MIGRATION_ALLOWED=false", flush=True)
    return report


def run_all() -> dict[str, Any]:
    preflight = run_preflight()
    if preflight["status"] != "PASS":
        return preflight
    baseline = run_baseline()
    if baseline["status"] != "PASS":
        return baseline
    for wave_index in range(WAVE_COUNT):
        wave = run_wave(wave_index)
        if wave.get("status") != "PASS":
            return wave
        segment = run_segment(wave_index)
        if segment.get("status") != "PASS":
            return segment
    return run_closeout()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("preflight", "baseline", "wave", "segment", "closeout", "all"),
        required=True,
    )
    parser.add_argument("--wave-index", type=int)
    args = parser.parse_args()
    envelope = {
        "stage": "P11.5a",
        "status": "RUNNING",
        "started_at": now(),
        "phase": args.phase,
        "wave_index": args.wave_index,
        "heldout_accessed": False,
        "p11_6_phase1_allowed": False,
        "p12_titan_migration_allowed": False,
    }
    try:
        if args.phase in {"wave", "segment"} and (args.wave_index is None or not 0 <= args.wave_index < WAVE_COUNT):
            raise ValueError("--wave-index 0..3 is required")
        if args.phase == "preflight":
            result = run_preflight()
        elif args.phase == "baseline":
            result = run_baseline()
        elif args.phase == "wave":
            result = run_wave(int(args.wave_index))
        elif args.phase == "segment":
            result = run_segment(int(args.wave_index))
        elif args.phase == "closeout":
            result = run_closeout()
        else:
            result = run_all()
    except Exception as error:
        result = {
            **envelope,
            "status": "ERROR",
            "completed_at": now(),
            "exception_type": type(error).__name__,
            "exception": str(error),
            "result_marker": "P11_5A=ERROR",
        }
        atomic_json(REPORT_DIR / "p11-5a-last-error.json", result)
        print(f"P11_5A_EXCEPTION={type(error).__name__}: {error}", flush=True)
    return 0 if result.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
