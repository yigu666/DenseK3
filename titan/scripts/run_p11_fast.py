"""Run Titan-only P11.6 FAST General Dual-Teacher OPD.

This is an additive Titan overlay.  It resumes the frozen P11.5a optimizer,
scaler, token counter, and RNG state exactly; canonical model/config/test files
are never modified.  All expensive phases are idempotent and persist progress
before they can be repeated.
"""

from __future__ import annotations

import argparse
import difflib
import gc
import hashlib
import json
import math
import random
import shutil
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

ROOT = Path(__file__).resolve().parents[2]
TITAN = ROOT / "titan"
SCRIPTS = TITAN / "scripts"
sys.path[:0] = [str(TITAN), str(SCRIPTS)]

import run_p11_probe as probe  # noqa: E402
from attnres import disable_block_attnres_runtime, enable_block_attnres_runtime  # noqa: E402
from p7_latent_cache import enable_p7_latent_cache_runtime  # noqa: E402
from p10 import merge_all_lora  # noqa: E402
from p11 import (  # noqa: E402
    KimiAPIConfig,
    KimiAPICostLedger,
    KimiAPITeacher,
    accounted_ledger_cost,
    exact_token_advantage,
    select_high_information,
)
from p11.cost_calibration import DOMAINS  # noqa: E402
from p11.fast import (  # noqa: E402
    CHUNK_COUNT,
    FAST_GPU_MEMORY_GATE_AMENDMENT,
    FAST_GPU_SAFETY_MARGIN_MIB,
    FROZEN_KDA_AUDIT_MAIN_TOKENS,
    FULL_DEV_CE_MAX,
    KIMI_ATTEMPTS_PER_DOMAIN,
    KIMI_MAX_ATTEMPTS_PER_REFRESH,
    KIMI_MAX_NEW_ATTEMPTS,
    NEW_OPTIMIZER_STEPS,
    P10_FULL_DEV_CE,
    P11_6_FAST_CONTRACT,
    P11_CUMULATIVE_API_HARD_CAP_CNY,
    SOURCE_MAIN_TOKENS,
    SOURCE_OPTIMIZER_STEPS,
    TARGET_MAIN_TOKENS,
    TOKENS_PER_CHUNK,
    UPDATE_BUCKETS_PER_CHUNK,
    chunk_quotas,
    continuation_cosine_learning_rate,
    contract_manifest,
    cumulative_main_quotas,
    fast_api_call_allowed,
    final_decision,
    is_recovery_marker,
    kimi_refresh_early_stop,
    refresh_index_for_chunk,
    refresh_required_for_chunk,
)
from p11.probe import (  # noqa: E402
    P11_GPU_MEMORY_GATE_AMENDMENT,
    STREAM_LOSS_WEIGHTS,
    assign_update_buckets,
    gpu_memory_envelope,
    raw_lm_target_lengths,
)
from p11.runtime import (  # noqa: E402
    QWEN,
    load_p10_t_student,
    qwen_token_logprobs,
)
from run_p10 import (  # noqa: E402
    apply_dense_overrides,
    candidate_override_state,
    frozen_parameter_audit,
)

from densek3_core.recovery.p5_corpus import PackedTokenDataset  # noqa: E402
from densek3_core.recovery.p5_probe import chunked_causal_lm_loss  # noqa: E402
from densek3_core.recovery.p6_closeout import P6_MLA_LAYERS, sha256_file  # noqa: E402

CONFIG = TITAN / "configs/p11-6-fast-contract-2026-08-24.yaml"
P11_5_CONFIG = TITAN / "configs/p11-5a-probe-contract-2026-08-21.yaml"
API_CONFIG = TITAN / "configs/p11-kimi-api.json"
REPORT_ROOT = TITAN / "manifests/reproduction/p11-t"
REPORT_DIR = REPORT_ROOT / "p11-6-fast"
PREFLIGHT_REPORT = REPORT_DIR / "p11-6-fast-preflight.json"
SOURCE_AUDIT_REPORT = REPORT_DIR / "p11-6-fast-source-runtime-audit.json"
EVALUATION_REPORT = REPORT_DIR / "p11-6-fast-final-evaluation.json"
FINAL_REPORT = REPORT_DIR / "p11-6-fast-final-check.json"
LAST_ERROR_REPORT = REPORT_DIR / "p11-6-fast-last-error.json"

P11_5_REPORT = REPORT_ROOT / "p11-5a/p11-5a-final-check.json"
P11_5_BASELINE = REPORT_ROOT / "p11-5a/p11-5a-p10t-baseline.json"
P11_5_SOURCE = (
    ROOT
    / "outputs/densek3-core/checkpoints/p11-t/probe-256k/main-tokens-262144"
)
P11_5_STEP_ZERO = (
    ROOT / "outputs/densek3-core/checkpoints/p11-t/probe-256k/step-zero-trainable.safetensors"
)
CHECKPOINT_ROOT = ROOT / "outputs/densek3-core/checkpoints/p11-t/fast-1m"
PROGRESS_ROOT = CHECKPOINT_ROOT / "progress"
RECOVERY_ROOT = CHECKPOINT_ROOT / "recovery"
CANDIDATE = CHECKPOINT_ROOT / "densek3-4b-general-dual-teacher-opd-fast-titan"

PROMPTS = ROOT / "data/p11-general/prompts/general-train-prompts-v1.jsonl"
RAW_DATA = ROOT / "data/p5-probe/packed"
QWEN_CACHE_ROOT = ROOT / "data/p11-general/teacher-cache/p11-6-fast/qwen-opd"
KIMI_CACHE_ROOT = ROOT / "data/p11-general/teacher-cache/p11-6-fast/kimi"

MAX_PROMPT_TOKENS = 256
MAX_KIMI_COMPLETION_TOKENS = 768
RATE_LIMIT_SECONDS = 21.0
QWEN_LONG_ROLLOUT_TOKENS = 512
QWEN_CANDIDATE_ROLLOUT_TOKENS = 16
QWEN_CANDIDATE_POOL_PER_DOMAIN = 48
SEED = 20260824


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


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def release() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def gpu_start_state(*, stage: str, chunk_index: int | None = None) -> dict[str, Any]:
    raw = probe.gpu_start_state()
    fields = [item.strip() for item in str(raw.get("description", "")).split(",")]
    temperature_c: int | None = None
    memory_used_mib: int | None = None
    memory_total_mib: int | None = None
    if bool(raw.get("visible")) and len(fields) == 4:
        try:
            temperature_c = int(fields[1])
            memory_used_mib = int(fields[2])
            memory_total_mib = int(fields[3])
        except ValueError:
            pass
    envelope = gpu_memory_envelope(
        visible=bool(raw.get("visible")),
        temperature_c=temperature_c,
        memory_used_mib=memory_used_mib,
        memory_total_mib=memory_total_mib,
    )
    # P11.6 FAST is explicitly authorized to run alongside another workload.
    # Admission is therefore memory-only: GPU visibility, valid memory fields,
    # and the historical P11.5a peak plus a positive safety margin remain hard
    # checks. Temperature is retained as a diagnostic but does not block start.
    fast_checks = dict(envelope["checks"])
    temperature_diagnostic_safe = fast_checks.pop("temperature_safe", None)
    fast_checks["historical_peak_plus_margin_fits"] = (
        envelope["memory_free_mib"] is not None
        and envelope["memory_free_mib"]
        >= envelope["historical_peak_memory_mib"] + FAST_GPU_SAFETY_MARGIN_MIB
    )
    envelope.update(
        {
            "amendment_id": FAST_GPU_MEMORY_GATE_AMENDMENT,
            "mode": "HISTORICAL_PEAK_PLUS_USER_APPROVED_MEMORY_ONLY_MARGIN",
            "safety_margin_mib": FAST_GPU_SAFETY_MARGIN_MIB,
            "required_free_mib": envelope["historical_peak_memory_mib"]
            + FAST_GPU_SAFETY_MARGIN_MIB,
            "checks": fast_checks,
            "allowed": all(fast_checks.values()),
            "temperature_diagnostic_safe": temperature_diagnostic_safe,
            "temperature_blocks_start": False,
            "concurrent_gpu_compute_allowed": True,
            "user_approved": True,
        }
    )
    report = {
        **raw,
        "actual_idle": bool(raw.get("idle")),
        "allowed": bool(envelope["allowed"]),
        "memory_envelope": envelope,
        "formal_gate_amendment": P11_GPU_MEMORY_GATE_AMENDMENT,
        "formal_gate_requires_idle": False,
        "stage": stage,
        "chunk_index": chunk_index,
        "completed_at": now(),
    }
    directory = REPORT_DIR / f"chunk-{chunk_index:02d}" if chunk_index is not None else REPORT_DIR
    atomic_json(directory / f"{stage}-gpu-gate.json", report)
    return report


def stable_rank(namespace: str, record_id: str) -> str:
    return sha256_text(f"{P11_6_FAST_CONTRACT}\0{namespace}\0{record_id}")


def load_prompts() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in PROMPTS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_student(weights: Path | None = None) -> tuple[Any, Any, Any, dict[str, Any]]:
    model, attnres, scope, report = probe.load_probe_student(P11_5_SOURCE)
    if weights is not None and weights.resolve() != (P11_5_SOURCE / "trainable.safetensors").resolve():
        probe.load_trainable_state(scope, weights)
    enable_p7_latent_cache_runtime(P6_MLA_LAYERS)
    return model, attnres, scope, report


def progress_manifest() -> dict[str, Any] | None:
    path = PROGRESS_ROOT / "checkpoint-manifest.json"
    return load_json(path) if path.is_file() else None


def current_weights_for_chunk(chunk_index: int) -> Path:
    progress = progress_manifest()
    if chunk_index == 0 and progress is None:
        return P11_5_SOURCE / "trainable.safetensors"
    if progress is None:
        raise FileNotFoundError("P11.6 progress is absent after chunk zero")
    expected_before = cumulative_main_quotas(chunk_index)["main"]
    completed = int(progress["completed_main_tokens"])
    current_chunk = int(progress["chunk_index"])
    completed_bucket = int(progress["completed_bucket"])
    valid = (
        completed == expected_before
        or (
            current_chunk == chunk_index
            and 0 < completed_bucket <= UPDATE_BUCKETS_PER_CHUNK
            and completed == expected_before + completed_bucket * (TOKENS_PER_CHUNK // UPDATE_BUCKETS_PER_CHUNK)
        )
    )
    if not valid:
        raise ValueError(
            f"P11.6 progress/token mismatch for chunk {chunk_index}: "
            f"completed={completed} current_chunk={current_chunk} bucket={completed_bucket}"
        )
    return PROGRESS_ROOT / "trainable.safetensors"


def checkpoint_fingerprint(weights: Path) -> str:
    return sha256_file(weights)


def preflight() -> dict[str, Any]:
    if PREFLIGHT_REPORT.is_file() and load_json(PREFLIGHT_REPORT).get("status") == "PASS":
        report = load_json(PREFLIGHT_REPORT)
        print("P11_6_FAST_PREFLIGHT=RESUMED", flush=True)
        return report
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    source_final = load_json(P11_5_REPORT)
    source_manifest = load_json(P11_5_SOURCE / "checkpoint-manifest.json")
    state = torch.load(P11_5_SOURCE / "training-state.pt", map_location="cpu", weights_only=False)
    optimizer = state["optimizer"]
    group_summary = [
        {
            "group_name": group.get("group_name"),
            "learning_rate": float(group["lr"]),
            "weight_decay": float(group["weight_decay"]),
            "parameter_tensors": len(group["params"]),
        }
        for group in optimizer["param_groups"]
    ]
    ledger = accounted_ledger_cost(REPORT_ROOT)
    checks = {
        "pure_contract_pass": contract_manifest()["passed"],
        "yaml_contract_exact": config["contract"] == P11_6_FAST_CONTRACT,
        "p11_5a_pass": source_final.get("status") == "PASS",
        "p11_6_phase1_allowed": source_final.get("p11_6_phase1_allowed") is True,
        "source_tokens_262144": source_manifest.get("completed_main_tokens") == SOURCE_MAIN_TOKENS,
        "source_step_64": state.get("global_optimizer_step") == SOURCE_OPTIMIZER_STEPS,
        "source_bucket_complete": state.get("completed_bucket") == 16,
        "source_trainable_hash": sha256_file(P11_5_SOURCE / "trainable.safetensors")
        == source_manifest["trainable_sha256"],
        "source_training_state_hash": sha256_file(P11_5_SOURCE / "training-state.pt")
        == source_manifest["training_state_sha256"],
        "optimizer_state_563": len(optimizer["state"]) == 563,
        "optimizer_group_tensors_563": sum(item["parameter_tensors"] for item in group_summary) == 563,
        "optimizer_groups_exact": [item["group_name"] for item in group_summary]
        == ["mergeable_lora", "direct_fp32_master"],
        "api_spend_below_15": float(ledger["accounted_cost_cny"])
        <= P11_CUMULATIVE_API_HARD_CAP_CNY,
        "heldout_not_accessed": source_final.get("heldout_accessed") is False,
        "p11_7_forbidden": config["next_stage"]["p11_7_allowed"] is False,
        "automatic_2m_forbidden": config["next_stage"]["p11_6_2m_automatically_allowed"] is False,
    }
    report = {
        "stage": "P11.6-FAST-PREFLIGHT",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "completed_at": now(),
        "contract": contract_manifest(),
        "source_checkpoint": str(P11_5_SOURCE),
        "source_checkpoint_manifest": source_manifest,
        "source_optimizer_groups": group_summary,
        "source_scaler": state["scaler"],
        "source_rng_keys": sorted(key for key in state if "random" in key),
        "p11_cumulative_api_ledger": ledger,
        "checks": checks,
        "heldout_accessed": False,
        "p11_6_training_started": False,
        "p11_6_2m_automatically_allowed": False,
        "p11_7_allowed": False,
        "p12_long_context_allowed": False,
        "result_marker": "P11_6_FAST_PREFLIGHT=PASS" if all(checks.values()) else "P11_6_FAST_PREFLIGHT=FAIL",
    }
    atomic_json(PREFLIGHT_REPORT, report)
    print(report["result_marker"], flush=True)
    return report


def source_runtime_audit() -> dict[str, Any]:
    if SOURCE_AUDIT_REPORT.is_file() and load_json(SOURCE_AUDIT_REPORT).get("status") == "PASS":
        report = load_json(SOURCE_AUDIT_REPORT)
        print("P11_6_FAST_SOURCE_AUDIT=RESUMED", flush=True)
        return report
    gpu = gpu_start_state(stage="source-runtime-audit")
    if not gpu["allowed"]:
        return {"stage": "P11.6-FAST-SOURCE-AUDIT", "status": "DEFERRED_GPU_MEMORY_UNSAFE", "gpu": gpu}
    model, attnres, scope, load = load_student()
    frozen = probe.frozen_scope_audit(model, scope)
    tokens = torch.from_numpy(PackedTokenDataset(RAW_DATA, "dev")[0][:128].copy()).unsqueeze(0).cuda()
    routing = probe.routing_diagnostic(model, attnres, tokens)
    checks = {
        "frozen_kda_names_complete": not frozen["missing"],
        "frozen_kda_requires_grad_false": frozen["all_requires_grad_false"],
        "attnres_active": routing["active"],
        "heldout_not_accessed": True,
    }
    report = {
        "stage": "P11.6-FAST-SOURCE-AUDIT",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "completed_at": now(),
        "source_checkpoint": str(P11_5_SOURCE),
        "source_frozen_kda": frozen,
        "routing": routing,
        "student_load": load,
        "checks": checks,
        "heldout_accessed": False,
        "result_marker": "P11_6_FAST_SOURCE_AUDIT=PASS" if all(checks.values()) else "P11_6_FAST_SOURCE_AUDIT=FAIL",
    }
    atomic_json(SOURCE_AUDIT_REPORT, report)
    print(report["result_marker"], flush=True)
    del model, attnres, scope, tokens
    release()
    return report


def candidate_records(prompts: list[dict[str, Any]], chunk_index: int) -> list[dict[str, Any]]:
    if not refresh_required_for_chunk(chunk_index):
        return []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in prompts:
        if record["domain"] in DOMAINS:
            grouped[record["domain"]].append(record)
    selected: list[dict[str, Any]] = []
    for domain in DOMAINS:
        values = sorted(
            grouped[domain],
            key=lambda item: stable_rank(f"chunk-{chunk_index}-candidate-{domain}", item["id"]),
        )
        if len(values) < QWEN_CANDIDATE_POOL_PER_DOMAIN:
            raise ValueError(f"P11.6 has too few candidate prompts for {domain}")
        selected.extend(values[:QWEN_CANDIDATE_POOL_PER_DOMAIN])
    return selected


def long_prompt_records(
    prompts: list[dict[str, Any]], chunk_index: int, excluded: set[str]
) -> list[dict[str, Any]]:
    values = [record for record in prompts if record["id"] not in excluded]
    return sorted(values, key=lambda item: stable_rank(f"chunk-{chunk_index}-long", item["id"]))


def qwen_cache_path(chunk_index: int, trajectory_index: int) -> Path:
    return QWEN_CACHE_ROOT / f"chunk-{chunk_index:02d}" / f"trajectory-{trajectory_index:04d}.json"


def build_qwen_cache(chunk_index: int) -> dict[str, Any]:
    output = REPORT_DIR / f"chunk-{chunk_index:02d}/qwen-opd-manifest.json"
    if output.is_file() and load_json(output).get("status") == "PASS":
        report = load_json(output)
        print(f"P11_6_FAST_CHUNK_{chunk_index}_QWEN_CACHE=RESUMED", flush=True)
        return report
    gpu = gpu_start_state(stage="qwen-cache", chunk_index=chunk_index)
    if not gpu["allowed"]:
        return {"stage": "P11.6-FAST-QWEN-CACHE", "status": "DEFERRED_GPU_MEMORY_UNSAFE", "gpu": gpu}
    quota = chunk_quotas(chunk_index)["qwen"]
    weights = current_weights_for_chunk(chunk_index)
    source_hash = checkpoint_fingerprint(weights)
    prompts = load_prompts()
    candidates = candidate_records(prompts, chunk_index)
    long_records = long_prompt_records(prompts, chunk_index, {item["id"] for item in candidates})
    tokenizer = probe.load_tokenizer()
    directory = QWEN_CACHE_ROOT / f"chunk-{chunk_index:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    records = [load_json(path) for path in sorted(directory.glob("trajectory-*.json"))]
    if any(item["source_checkpoint_fingerprint"] != source_hash for item in records):
        raise ValueError("P11.6 cached rollout source changed")
    generated = sum(int(item["rollout_tokens"]) for item in records)
    if generated > quota:
        raise ValueError("P11.6 cached Qwen tokens exceed chunk quota")
    student_load: Any = "RESUMED_CACHED_ROLLOUTS"
    if generated < quota:
        model, attnres, scope, student_load = load_student(weights)
        model.eval()
        attnres.eval()
        index = len(records)
        long_index = max(0, index - len(candidates))
        while generated < quota:
            if index < len(candidates):
                record = candidates[index]
                role = "kimi_candidate"
                maximum = min(QWEN_CANDIDATE_ROLLOUT_TOKENS, quota - generated)
            else:
                record = long_records[long_index % len(long_records)]
                long_index += 1
                role = "main_opd"
                maximum = min(QWEN_LONG_ROLLOUT_TOKENS, quota - generated)
            item = probe.rollout_one(
                model,
                tokenizer,
                record,
                maximum_tokens=maximum,
                namespace=f"p11-6-fast-chunk-{chunk_index}-trajectory-{index}",
            )
            item.update(
                {
                    "schema_version": 1,
                    "stage": "P11.6-FAST-QWEN-EXACT-OPD-CACHE",
                    "chunk_index": chunk_index,
                    "trajectory_index": index,
                    "role": role,
                    "source_checkpoint": str(weights.parent),
                    "source_checkpoint_fingerprint": source_hash,
                    "qwen_scored": False,
                }
            )
            atomic_json(qwen_cache_path(chunk_index, index), item)
            records.append(item)
            generated += int(item["rollout_tokens"])
            index += 1
            if index % 8 == 0 or generated == quota:
                print(
                    f"P11_6_FAST_CHUNK_{chunk_index}_ROLLOUT_TOKENS={generated}/{quota}",
                    flush=True,
                )
        del model, attnres, scope
        release()

    pending = [item for item in records if not item.get("qwen_scored")]
    qwen_backend: Any = "RESUMED_CACHED_QWEN_SCORES"
    if pending:
        from transformers import AutoModelForCausalLM

        qwen = AutoModelForCausalLM.from_pretrained(
            QWEN, dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=False
        )
        qwen_backend = probe.force_qwen_torch_gdn_backend(qwen)
        qwen = qwen.cuda().eval()
        for number, item in enumerate(pending, start=1):
            tokens = torch.tensor([item["full_token_ids"]], dtype=torch.long, device="cuda")
            teacher = qwen_token_logprobs(
                qwen, tokens, target_start=int(item["prompt_tokens"])
            ).detach().float().cpu()[0]
            old = torch.tensor(item["old_student_logprobs"], dtype=torch.float32)
            if teacher.shape != old.shape:
                raise ValueError("P11.6 Qwen/student score shape mismatch")
            difference = teacher - old
            item.update(
                {
                    "qwen_teacher_logprobs": teacher.tolist(),
                    "qwen_student_advantage_mean": float(difference.mean()),
                    "qwen_student_abs_logprob_gap_mean": float(difference.abs().mean()),
                    "disagreement_score": float(
                        difference.abs().mean() + difference.mean().clamp_min(0)
                    ),
                    "qwen_scored": True,
                    "qwen_scored_at": now(),
                }
            )
            atomic_json(qwen_cache_path(chunk_index, int(item["trajectory_index"])), item)
            if number % 8 == 0 or number == len(pending):
                print(
                    f"P11_6_FAST_CHUNK_{chunk_index}_QWEN_SCORE={number}/{len(pending)}",
                    flush=True,
                )
        del qwen
        release()
    records = [load_json(path) for path in sorted(directory.glob("trajectory-*.json"))]
    expected_candidates = 4 * QWEN_CANDIDATE_POOL_PER_DOMAIN if refresh_required_for_chunk(chunk_index) else 0
    checks = {
        "qwen_tokens_exact": sum(int(item["rollout_tokens"]) for item in records) == quota,
        "candidate_pool_exact": sum(item["role"] == "kimi_candidate" for item in records)
        == expected_candidates,
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
            "cache_path": str(qwen_cache_path(chunk_index, int(item["trajectory_index"])).relative_to(ROOT)),
            "cache_sha256": sha256_file(qwen_cache_path(chunk_index, int(item["trajectory_index"]))),
        }
        for item in records
    ]
    report = {
        "stage": "P11.6-FAST-QWEN-EXACT-OPD-CACHE",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "completed_at": now(),
        "chunk_index": chunk_index,
        "source_checkpoint": str(weights.parent),
        "source_checkpoint_fingerprint": source_hash,
        "qwen_token_quota": quota,
        "student_load": student_load,
        "qwen_backend": qwen_backend,
        "trajectories": trajectories,
        "checks": checks,
        "heldout_accessed": False,
        "result_marker": f"P11_6_FAST_CHUNK_{chunk_index}_QWEN_CACHE={'PASS' if all(checks.values()) else 'FAIL'}",
    }
    atomic_json(output, report)
    print(report["result_marker"], flush=True)
    return report


def refresh_selection(refresh_index: int, chunk_index: int) -> dict[str, Any]:
    output = REPORT_DIR / f"refresh-{refresh_index}/kimi-selection.json"
    if output.is_file() and load_json(output).get("status") == "PASS":
        return load_json(output)
    manifest = load_json(REPORT_DIR / f"chunk-{chunk_index:02d}/qwen-opd-manifest.json")
    candidates = [
        load_json(ROOT / item["cache_path"])
        for item in manifest["trajectories"]
        if item["role"] == "kimi_candidate"
    ]
    selected = select_high_information(candidates, per_domain=KIMI_ATTEMPTS_PER_DOMAIN)
    checks = {
        "selected_64": len(selected) == KIMI_MAX_ATTEMPTS_PER_REFRESH,
        "balanced_16_per_domain": all(
            sum(item["domain"] == domain for item in selected) == KIMI_ATTEMPTS_PER_DOMAIN
            for domain in DOMAINS
        ),
        "current_policy_exact": all(
            item["source_checkpoint_fingerprint"] == manifest["source_checkpoint_fingerprint"]
            for item in selected
        ),
        "heldout_not_accessed": True,
    }
    report = {
        "stage": "P11.6-FAST-KIMI-HIGH-INFORMATION-SELECTION",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "completed_at": now(),
        "refresh_index": refresh_index,
        "chunk_index": chunk_index,
        "selection_method": "TOP16_PER_DOMAIN_BY_FROZEN_HIGH_INFORMATION_SELECTOR",
        "selected": selected,
        "checks": checks,
        "api_calls": 0,
        "heldout_accessed": False,
    }
    atomic_json(output, report)
    return report


def all_fast_attempts() -> list[dict[str, Any]]:
    return [load_json(path) for path in sorted(KIMI_CACHE_ROOT.glob("refresh-*/attempt-*.json"))]


def correction_is_trivial(student: str, corrected: str) -> tuple[bool, float]:
    left = " ".join(student.casefold().split())
    right = " ".join(corrected.casefold().split())
    ratio = difflib.SequenceMatcher(a=left, b=right, autojunk=False).ratio()
    edit_fraction = 1.0 - ratio
    return edit_fraction <= 0.02, edit_fraction


def run_kimi_refresh(refresh_index: int, chunk_index: int) -> dict[str, Any]:
    output = REPORT_DIR / f"refresh-{refresh_index}/kimi-refresh.json"
    if output.is_file() and load_json(output).get("status") == "PASS":
        report = load_json(output)
        print(f"P11_6_FAST_KIMI_REFRESH_{refresh_index}=RESUMED", flush=True)
        return report
    selection = refresh_selection(refresh_index, chunk_index)
    if selection["status"] != "PASS":
        return selection
    config = KimiAPIConfig.from_json(API_CONFIG)
    if config.reasoning_effort != "low" or config.max_retries != 0:
        raise ValueError("P11.6 requires low reasoning and zero automatic retry")
    ledger = KimiAPICostLedger(REPORT_ROOT, config)
    teacher = KimiAPITeacher(config, ledger=ledger)
    tokenizer = probe.load_tokenizer()
    directory = KIMI_CACHE_ROOT / f"refresh-{refresh_index}"
    directory.mkdir(parents=True, exist_ok=True)
    last_started: float | None = None
    stop_reason: str | None = None
    rolling: list[dict[str, Any]] = []
    for slot, selected in enumerate(selection["selected"]):
        path = directory / f"attempt-{slot:02d}.json"
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
            attempts = [load_json(value) for value in sorted(directory.glob("attempt-*.json"))]
            if len(attempts) % 16 == 0:
                status = kimi_refresh_early_stop(attempts)
                rolling.append(status)
                if status["stop"]:
                    stop_reason = "ROLLING_16_CONJUNCTIVE_EARLY_STOP"
                    break
            continue
        all_attempts = all_fast_attempts()
        messages, response_format = teacher.structured_trajectory_contract(
            selected["prompt"], selected["student_text"]
        )
        upper = probe.request_upper_bound(config, messages, response_format)
        cumulative = accounted_ledger_cost(REPORT_ROOT)
        gate = fast_api_call_allowed(
            cumulative_cost_cny=float(cumulative["accounted_cost_cny"]),
            next_upper_bound_cny=upper,
            refresh_attempts=len(list(directory.glob("attempt-*.json"))),
            new_attempts=len(all_attempts),
        )
        if not gate["allowed"]:
            stop_reason = str(gate["stop_reason"])
            break
        if last_started is not None:
            remaining = RATE_LIMIT_SECONDS - (time.monotonic() - last_started)
            if remaining > 0:
                print(f"P11_6_FAST_KIMI_RATE_LIMIT_SECONDS={remaining:.3f}", flush=True)
                time.sleep(remaining)
        fingerprint = sha256_json(
            {
                "contract": P11_6_FAST_CONTRACT,
                "refresh": refresh_index,
                "slot": slot,
                "selected_id": selected["id"],
                "student_text_sha256": selected["student_text_sha256"],
                "model": config.model,
                "reasoning_effort": config.reasoning_effort,
            }
        )
        pending = {
            "schema_version": 1,
            "stage": "P11.6-FAST-KIMI-REVIEW",
            "status": "REQUEST_PENDING",
            "refresh_index": refresh_index,
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
            trivial, edit_fraction = correction_is_trivial(selected["student_text"], corrected)
            corrected_tokens = (
                int(
                    tokenizer(
                        corrected,
                        add_special_tokens=False,
                        return_tensors="pt",
                    ).input_ids.numel()
                )
                if corrected
                else 0
            )
            pending.update(
                {
                    "status": "SUCCEEDED",
                    "decision": "keep" if trajectory.mode == "KEEP" else "correct",
                    "corrected_answer": corrected,
                    "corrected_target_tokens": corrected_tokens,
                    "trivial_correction": trivial,
                    "normalized_edit_fraction": edit_fraction,
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
                f"P11_6_FAST_REFRESH_{refresh_index}_KIMI_SLOT={slot + 1}/64 "
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
        attempts = [load_json(value) for value in sorted(directory.glob("attempt-*.json"))]
        if len(attempts) % 16 == 0:
            status = kimi_refresh_early_stop(attempts)
            rolling.append(status)
            print(
                f"P11_6_FAST_REFRESH_{refresh_index}_ROLLING_16="
                f"{'STOP' if status['stop'] else 'CONTINUE'}",
                flush=True,
            )
            if status["stop"]:
                stop_reason = "ROLLING_16_CONJUNCTIVE_EARLY_STOP"
                break
    attempts = [load_json(path) for path in sorted(directory.glob("attempt-*.json"))]
    decisions = Counter(item.get("decision", "failed") for item in attempts)
    complete_reasons = {
        "ROLLING_16_CONJUNCTIVE_EARLY_STOP",
        "P11_CUMULATIVE_API_HARD_CAP",
        "P11_6_MAX_NEW_KIMI_ATTEMPTS",
        "P11_6_REFRESH_ATTEMPT_CAP",
    }
    complete = len(attempts) == KIMI_MAX_ATTEMPTS_PER_REFRESH or stop_reason in complete_reasons
    ledger_summary = accounted_ledger_cost(REPORT_ROOT)
    checks = {
        "attempts_not_above_refresh_cap": len(attempts) <= KIMI_MAX_ATTEMPTS_PER_REFRESH,
        "new_attempts_not_above_128": len(all_fast_attempts()) <= KIMI_MAX_NEW_ATTEMPTS,
        "p11_cumulative_cost_not_above_15": ledger_summary["accounted_cost_cny"]
        <= P11_CUMULATIVE_API_HARD_CAP_CNY,
        "reasoning_content_not_persisted": all(
            item.get("reasoning_content_persisted") is False for item in attempts
        ),
        "no_automatic_retry": True,
        "heldout_not_accessed": True,
    }
    status = "PASS" if complete and all(checks.values()) else "PAUSED"
    report = {
        "stage": "P11.6-FAST-KIMI-REFRESH",
        "status": status,
        "completed_at": now(),
        "refresh_index": refresh_index,
        "chunk_index": chunk_index,
        "attempts": len(attempts),
        "successful": sum(item.get("status") == "SUCCEEDED" for item in attempts),
        "keep": decisions["keep"],
        "correct": decisions["correct"],
        "correction_rate": (
            decisions["correct"] / (decisions["keep"] + decisions["correct"])
            if decisions["keep"] + decisions["correct"]
            else None
        ),
        "rolling_statistics": rolling,
        "stop_reason": stop_reason,
        "p11_cumulative_api_ledger": ledger_summary,
        "attempt_paths": [
            str(path.relative_to(ROOT)) for path in sorted(directory.glob("attempt-*.json"))
        ],
        "checks": checks,
        "heldout_accessed": False,
        "result_marker": f"P11_6_FAST_KIMI_REFRESH_{refresh_index}={status}",
    }
    atomic_json(output, report)
    print(report["result_marker"], flush=True)
    return report


def refresh_attempts(refresh_index: int) -> list[dict[str, Any]]:
    report = load_json(REPORT_DIR / f"refresh-{refresh_index}/kimi-refresh.json")
    return [load_json(ROOT / value) for value in report["attempt_paths"]]


def save_snapshot(
    directory: Path,
    *,
    scope: Any,
    optimizer: Any,
    scaler: Any,
    chunk_index: int,
    completed_bucket: int,
    global_optimizer_step: int,
    completed_main_tokens: int,
    training_curve: list[dict[str, Any]],
    audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    weights = directory / "trainable.safetensors"
    temporary_weights = weights.with_suffix(".safetensors.tmp")
    save_file(probe.trainable_state(scope), str(temporary_weights))
    temporary_weights.replace(weights)
    training_state = directory / "training-state.pt"
    atomic_torch_save(
        training_state,
        {
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "chunk_index": chunk_index,
            "completed_bucket": completed_bucket,
            "global_optimizer_step": global_optimizer_step,
            "completed_main_tokens": completed_main_tokens,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state": torch.cuda.get_rng_state_all(),
        },
    )
    manifest = {
        "stage": "P11.6-FAST-TRAINING-CHECKPOINT",
        "status": "PASS",
        "completed_at": now(),
        "chunk_index": chunk_index,
        "completed_bucket": completed_bucket,
        "global_optimizer_step": global_optimizer_step,
        "completed_main_tokens": completed_main_tokens,
        "trainable_sha256": sha256_file(weights),
        "training_state_sha256": sha256_file(training_state),
        "training_curve": training_curve,
        "audit": audit,
        "heldout_accessed": False,
    }
    atomic_json(directory / "checkpoint-manifest.json", manifest)
    return manifest


def restore_snapshot(
    directory: Path,
    *,
    scope: Any,
    masters: Any,
    optimizer: Any,
    scaler: Any,
    source_is_p11_5a: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_json(directory / "checkpoint-manifest.json")
    weights = directory / "trainable.safetensors"
    training_state = directory / "training-state.pt"
    if (
        sha256_file(weights) != manifest["trainable_sha256"]
        or sha256_file(training_state) != manifest["training_state_sha256"]
    ):
        raise ValueError("P11.6 snapshot hash mismatch")
    probe.load_trainable_state(scope, weights)
    state = torch.load(training_state, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_random_state"])
    torch.cuda.set_rng_state_all(state["cuda_random_state"])
    # Snapshot files hold model weights and Adam moments, not FP32 master
    # parameter values.  Refreshing after every restore is required for exact
    # interruption recovery; otherwise the next step starts from stale masters.
    masters.refresh_from_model()
    audit = {
        "source_is_p11_5a": source_is_p11_5a,
        "model_weights_hash_verified": True,
        "optimizer_state_hash_verified": True,
        "optimizer_state_tensors": len(optimizer.state),
        "fp32_masters_refreshed_after_restore": True,
        "scaler_restored": True,
        "rng_restored": True,
    }
    return state, audit


def raw_dataset_offset() -> int:
    return sum(
        len(raw_lm_target_lengths(probe.wave_main_quotas(index)["raw_lm"]))
        for index in range(probe.WAVE_COUNT)
    )


def raw_chunk_offset(chunk_index: int) -> int:
    return raw_dataset_offset() + sum(
        len(raw_lm_target_lengths(chunk_quotas(index)["raw_lm"]))
        for index in range(chunk_index)
    )


def run_training_chunk(chunk_index: int) -> dict[str, Any]:
    output = REPORT_DIR / f"chunk-{chunk_index:02d}/training-segment.json"
    if output.is_file() and load_json(output).get("status") == "PASS":
        report = load_json(output)
        print(f"P11_6_FAST_CHUNK_{chunk_index}_TRAINING=RESUMED", flush=True)
        return report
    qwen_manifest = load_json(REPORT_DIR / f"chunk-{chunk_index:02d}/qwen-opd-manifest.json")
    if qwen_manifest.get("status") != "PASS":
        raise ValueError("P11.6 training requires completed Qwen cache")
    refresh_index = refresh_index_for_chunk(chunk_index)
    refresh_report = load_json(REPORT_DIR / f"refresh-{refresh_index}/kimi-refresh.json")
    if refresh_report.get("status") != "PASS":
        raise ValueError("P11.6 training requires completed Kimi refresh")
    gpu = gpu_start_state(stage="training-segment", chunk_index=chunk_index)
    if not gpu["allowed"]:
        return {"stage": "P11.6-FAST-TRAINING", "status": "DEFERRED_GPU_MEMORY_UNSAFE", "gpu": gpu}
    quota = chunk_quotas(chunk_index)
    qwen_records = [load_json(ROOT / item["cache_path"]) for item in qwen_manifest["trajectories"]]
    qwen_lengths = [int(item["rollout_tokens"]) for item in qwen_records]
    qwen_buckets = assign_update_buckets(qwen_lengths, UPDATE_BUCKETS_PER_CHUNK)
    raw_lengths = raw_lm_target_lengths(quota["raw_lm"])
    raw_buckets = assign_update_buckets(raw_lengths, UPDATE_BUCKETS_PER_CHUNK)
    attempts = refresh_attempts(refresh_index)
    corrections = [
        item
        for item in attempts
        if item.get("status") == "SUCCEEDED" and item.get("decision") == "correct"
    ]

    source_weights = current_weights_for_chunk(chunk_index)
    model, attnres, scope, student_load = load_student(source_weights)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    masters, optimizer, _unused_scaler = probe.build_optimizer(scope, config)
    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=float(config["optimization"]["fp16_initial_loss_scale"]),
        growth_interval=int(config["optimization"]["fp16_loss_scale_growth_interval"]),
    )
    progress = progress_manifest()
    start_bucket = 0
    training_curve: list[dict[str, Any]] = []
    if progress is None:
        if chunk_index != 0:
            raise ValueError("P11.6 cannot start a later chunk without progress")
        state, restore_audit = restore_snapshot(
            P11_5_SOURCE,
            scope=scope,
            masters=masters,
            optimizer=optimizer,
            scaler=scaler,
            source_is_p11_5a=True,
        )
    else:
        state, restore_audit = restore_snapshot(
            PROGRESS_ROOT,
            scope=scope,
            masters=masters,
            optimizer=optimizer,
            scaler=scaler,
        )
        if int(progress["chunk_index"]) == chunk_index:
            start_bucket = int(progress["completed_bucket"])
            training_curve = list(progress.get("training_curve", []))
        elif int(progress["chunk_index"]) != chunk_index - 1 or int(progress["completed_bucket"]) != 16:
            raise ValueError("P11.6 progress does not precede requested chunk")
    expected_step = SOURCE_OPTIMIZER_STEPS + chunk_index * UPDATE_BUCKETS_PER_CHUNK + start_bucket
    global_step = int(state["global_optimizer_step"])
    if global_step != expected_step:
        raise ValueError(f"P11.6 optimizer step mismatch: {global_step} != {expected_step}")
    source_groups = load_json(PREFLIGHT_REPORT)["source_optimizer_groups"]
    saved_lora_lr = float(source_groups[0]["learning_rate"])
    saved_direct_lr = float(source_groups[1]["learning_rate"])
    terminal_lora_lr = float(config["optimization"]["lora_learning_rate"]) * float(
        config["optimization"]["minimum_learning_rate_ratio"]
    )
    terminal_direct_lr = float(config["optimization"]["direct_learning_rate"]) * float(
        config["optimization"]["minimum_learning_rate_ratio"]
    )
    model.train()
    attnres.train()
    model.model.gradient_checkpointing = True
    raw_dataset = PackedTokenDataset(RAW_DATA, "train")
    tokenizer = probe.load_tokenizer()
    initial_values = load_file(str(P11_5_STEP_ZERO))
    recipe = config["optimization"]
    print("P11_6_FAST_TRAINING_STARTED=true", flush=True)
    for bucket in range(start_bucket, UPDATE_BUCKETS_PER_CHUNK):
        extension_step = global_step - SOURCE_OPTIMIZER_STEPS
        lora_lr = continuation_cosine_learning_rate(
            extension_step,
            extension_steps=NEW_OPTIMIZER_STEPS,
            saved_learning_rate=saved_lora_lr,
            terminal_learning_rate=terminal_lora_lr,
        )
        direct_lr = continuation_cosine_learning_rate(
            extension_step,
            extension_steps=NEW_OPTIMIZER_STEPS,
            saved_learning_rate=saved_direct_lr,
            terminal_learning_rate=terminal_direct_lr,
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
            loss = probe.chunked_opd_loss(
                hidden,
                tokens,
                target_start=int(item["prompt_tokens"]),
                advantage=advantage,
                lm_head_weight=model.lm_head.weight,
            )
            weight = (
                STREAM_LOSS_WEIGHTS["qwen"]
                * UPDATE_BUCKETS_PER_CHUNK
                * int(item["rollout_tokens"])
                / quota["qwen"]
            )
            scaler.scale(weight * loss).backward()
            loss_log["qwen"].append(float(loss.detach()))
            del tokens, hidden, old, teacher, advantage, loss
        for local_index in raw_buckets[bucket]:
            target_tokens = raw_lengths[local_index]
            dataset_index = raw_chunk_offset(chunk_index) + local_index
            tokens = (
                torch.from_numpy(raw_dataset[dataset_index][: target_tokens + 1].copy())
                .unsqueeze(0)
                .cuda()
            )
            hidden = model.model(tokens, use_cache=False, return_dict=True).last_hidden_state
            loss = chunked_causal_lm_loss(
                hidden,
                tokens,
                model.lm_head.weight,
                chunk_size=16,
                checkpoint_chunks=True,
            )
            weight = (
                STREAM_LOSS_WEIGHTS["raw_lm"]
                * UPDATE_BUCKETS_PER_CHUNK
                * target_tokens
                / quota["raw_lm"]
            )
            scaler.scale(weight * loss).backward()
            loss_log["raw_lm"].append(float(loss.detach()))
            del tokens, hidden, loss
        if corrections:
            item = corrections[(chunk_index * UPDATE_BUCKETS_PER_CHUNK + bucket) % len(corrections)]
            prompt = tokenizer(
                item["prompt"], return_tensors="pt", truncation=True, max_length=MAX_PROMPT_TOKENS
            ).input_ids
            target = tokenizer(
                item["corrected_answer"],
                add_special_tokens=False,
                return_tensors="pt",
                truncation=True,
                max_length=768,
            ).input_ids
            if target.numel() > 0:
                tokens = torch.cat((prompt, target), dim=1).cuda()
                hidden = model.model(tokens, use_cache=False, return_dict=True).last_hidden_state
                loss = probe.chunked_target_ce(
                    hidden,
                    tokens,
                    target_start=int(prompt.shape[1]),
                    lm_head_weight=model.lm_head.weight,
                )
                scaler.scale(STREAM_LOSS_WEIGHTS["kimi"] * loss).backward()
                loss_log["kimi"].append(float(loss.detach()))
                del tokens, hidden, loss
            del prompt, target
        if not all(math.isfinite(value) for values in loss_log.values() for value in values):
            raise FloatingPointError("P11.6 encountered non-finite loss")
        masters.copy_scaled_grads_from_model()
        scaler.unscale_(optimizer)
        parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        parameter_names = probe.optimizer_parameter_names(scope, masters)
        coverage = probe.audit_gradient_coverage(parameters, parameter_names)
        unexpected = probe.unexpected_gradient_names(model, attnres, scope)
        gradients = [parameter for parameter in parameters if parameter.grad is not None]
        nonfinite = [
            parameter_names[id(parameter)]
            for parameter in gradients
            if not bool(torch.isfinite(parameter.grad).all())
        ]
        gradient_norm_by_family = probe.optimizer_gradient_norm_by_family(scope, masters)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            gradients, float(recipe["gradient_clip_norm"])
        )
        if (
            not coverage["complete"]
            or coverage["optimizer_tensors"] != 563
            or coverage["gradient_tensors"] != 563
            or unexpected
        ):
            raise RuntimeError(
                f"P11.6 gradient contract failed: coverage={coverage} unexpected={unexpected[:8]}"
            )
        if nonfinite or not math.isfinite(float(gradient_norm)):
            raise FloatingPointError(f"P11.6 non-finite gradients: {nonfinite[:8]}")
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        masters.copy_to_model()
        global_step += 1
        point = {
            "chunk_index": chunk_index,
            "bucket": bucket + 1,
            "global_optimizer_step": global_step,
            "cumulative_main_tokens": cumulative_main_quotas(chunk_index)["main"]
            + (bucket + 1) * (TOKENS_PER_CHUNK // UPDATE_BUCKETS_PER_CHUNK),
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
        training_curve.append(point)
        save_snapshot(
            PROGRESS_ROOT,
            scope=scope,
            optimizer=optimizer,
            scaler=scaler,
            chunk_index=chunk_index,
            completed_bucket=bucket + 1,
            global_optimizer_step=global_step,
            completed_main_tokens=int(point["cumulative_main_tokens"]),
            training_curve=training_curve,
        )
        print(
            f"P11_6_FAST_CHUNK={chunk_index} BUCKET={bucket + 1}/16 "
            f"GLOBAL_STEP={global_step} GRADIENTS=563/563",
            flush=True,
        )

    cumulative = cumulative_main_quotas(chunk_index + 1)["main"]
    audit: dict[str, Any] | None = None
    if is_recovery_marker(chunk_index + 1):
        model.eval()
        attnres.eval()
        diagnostic_tokens = (
            torch.from_numpy(PackedTokenDataset(RAW_DATA, "dev")[0][:128].copy())
            .unsqueeze(0)
            .cuda()
        )
        routing = probe.routing_diagnostic(model, attnres, diagnostic_tokens)
        updates = probe.parameter_update_by_family(scope, initial_values)
        compensation = probe.compensation_warning(updates, routing)
        frozen = probe.frozen_scope_audit(model, scope) if cumulative in FROZEN_KDA_AUDIT_MAIN_TOKENS else None
        source_frozen = load_json(SOURCE_AUDIT_REPORT)["source_frozen_kda"]
        audit = {
            "routing": routing,
            "parameter_update_by_family": updates,
            "kda_freeze_compensation": compensation,
            "frozen_kda": frozen,
            "frozen_kda_exact": (
                frozen is None or frozen["tensor_hash"] == source_frozen["tensor_hash"]
            ),
        }
        if compensation["status"] == "HIGH":
            atomic_json(
                REPORT_DIR / "p11-6-fast-stop-kda-compensation-high.json",
                {
                    "stage": "P11.6-FAST-STOP",
                    "status": "STOP_KDA_COMPENSATION_HIGH",
                    "completed_at": now(),
                    "chunk_index": chunk_index,
                    "cumulative_main_tokens": cumulative,
                    "audit": audit,
                    "heldout_accessed": False,
                },
            )
            raise RuntimeError("P11.6 KDA compensation reached HIGH")
        if not audit["frozen_kda_exact"]:
            raise RuntimeError("P11.6 frozen KDA hash changed")
        recovery = RECOVERY_ROOT / f"main-tokens-{cumulative:07d}"
        save_snapshot(
            recovery,
            scope=scope,
            optimizer=optimizer,
            scaler=scaler,
            chunk_index=chunk_index,
            completed_bucket=UPDATE_BUCKETS_PER_CHUNK,
            global_optimizer_step=global_step,
            completed_main_tokens=cumulative,
            training_curve=training_curve,
            audit=audit,
        )
    checks = {
        "qwen_tokens_exact": sum(qwen_lengths) == quota["qwen"],
        "raw_tokens_exact": sum(raw_lengths) == quota["raw_lm"],
        "main_tokens_exact": sum(qwen_lengths) + sum(raw_lengths) == TOKENS_PER_CHUNK,
        "gradient_coverage_563": all(point["gradient_tensors"] == 563 for point in training_curve),
        "missing_zero": all(point["missing_gradient_count"] == 0 for point in training_curve),
        "unexpected_zero": all(point["unexpected_gradient_count"] == 0 for point in training_curve),
        "losses_finite": all(
            math.isfinite(value)
            for point in training_curve
            for values in point["losses"].values()
            for value in values
        ),
        "optimizer_step_exact": global_step
        == SOURCE_OPTIMIZER_STEPS + (chunk_index + 1) * UPDATE_BUCKETS_PER_CHUNK,
        "lr_boundary_continuous": (
            chunk_index > 0
            or training_curve[0]["lora_learning_rate"] == saved_lora_lr
            and training_curve[0]["direct_learning_rate"] == saved_direct_lr
        ),
        "kda_compensation_not_high": audit is None
        or audit["kda_freeze_compensation"]["status"] != "HIGH",
        "heldout_not_accessed": True,
    }
    report = {
        "stage": "P11.6-FAST-TRAINING-CHUNK",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "completed_at": now(),
        "chunk_index": chunk_index,
        "main_tokens_before": cumulative_main_quotas(chunk_index)["main"],
        "main_tokens_after": cumulative,
        "new_qwen_tokens": sum(qwen_lengths),
        "new_raw_lm_tokens": sum(raw_lengths),
        "kimi_refresh_index": refresh_index,
        "kimi_corrected_trajectories_available": len(corrections),
        "kimi_tokens_in_main_denominator": False,
        "kimi_loss_reduction": "PER_TRAJECTORY_MEAN_TARGET_CE",
        "source_checkpoint": str(source_weights.parent),
        "student_load": student_load,
        "restore_audit": restore_audit,
        "schedule": {
            "mode": "CONTINUATION_COSINE_FROM_PERSISTED_LR_TO_ORIGINAL_MINIMUM",
            "warmup_restarted": False,
            "optimizer_restarted": False,
            "saved_lora_lr": saved_lora_lr,
            "saved_direct_lr": saved_direct_lr,
            "terminal_lora_lr": terminal_lora_lr,
            "terminal_direct_lr": terminal_direct_lr,
        },
        "training_curve": training_curve,
        "recovery_checkpoint_only": is_recovery_marker(chunk_index + 1),
        "runtime_audit": audit,
        "checks": checks,
        "heldout_accessed": False,
        "intermediate_full_dev_performed": False,
        "intermediate_capability_evaluation_performed": False,
        "intermediate_merge_reload_performed": False,
        "intermediate_generation_performed": False,
        "result_marker": f"P11_6_FAST_CHUNK_{chunk_index}_TRAINING={'PASS' if all(checks.values()) else 'FAIL'}",
    }
    atomic_json(output, report)
    print(report["result_marker"], flush=True)
    del model, attnres, scope
    release()
    return report


@torch.inference_mode()
def qwen_teacher_audit(model: torch.nn.Module, records: list[dict[str, Any]]) -> dict[str, Any]:
    trajectories = []
    for item in records:
        tokens = torch.tensor([item["full_token_ids"]], dtype=torch.long, device="cuda")
        hidden = model.model(tokens, use_cache=False, return_dict=True).last_hidden_state
        predictors = hidden[:, int(item["prompt_tokens"]) - 1 : -1]
        targets = tokens[:, int(item["prompt_tokens"]):]
        values = []
        for start in range(0, targets.shape[1], 16):
            end = min(start + 16, targets.shape[1])
            logits = F.linear(predictors[:, start:end], model.lm_head.weight).float()
            values.append(
                torch.log_softmax(logits, dim=-1)
                .gather(-1, targets[:, start:end].unsqueeze(-1))
                .squeeze(-1)
                .cpu()
            )
        current = torch.cat(values, dim=1)[0]
        teacher = torch.tensor(item["qwen_teacher_logprobs"])
        trajectories.append(
            {
                "trajectory_index": int(item["trajectory_index"]),
                "mean_abs_teacher_gap": float((teacher - current).abs().mean()),
                "mean_signed_teacher_gap": float((teacher - current).mean()),
            }
        )
    return {
        "trajectories": trajectories,
        "mean_abs_teacher_gap": sum(item["mean_abs_teacher_gap"] for item in trajectories)
        / len(trajectories),
    }


def final_evaluation() -> dict[str, Any]:
    if EVALUATION_REPORT.is_file() and load_json(EVALUATION_REPORT).get("status") == "PASS":
        report = load_json(EVALUATION_REPORT)
        print("P11_6_FAST_FINAL_EVALUATION=RESUMED", flush=True)
        return report
    gpu = gpu_start_state(stage="final-evaluation")
    if not gpu["allowed"]:
        return {"stage": "P11.6-FAST-FINAL-EVALUATION", "status": "DEFERRED_GPU_MEMORY_UNSAFE", "gpu": gpu}
    progress = progress_manifest()
    if progress is None or progress.get("completed_main_tokens") != TARGET_MAIN_TOKENS:
        raise ValueError("P11.6 final evaluation requires the complete 1M progress snapshot")
    chunk_reports = [
        load_json(REPORT_DIR / f"chunk-{index:02d}/training-segment.json")
        for index in range(CHUNK_COUNT)
    ]
    if not all(item.get("status") == "PASS" for item in chunk_reports):
        raise ValueError("P11.6 final evaluation requires all 12 chunks PASS")
    qwen_manifest = load_json(REPORT_DIR / "chunk-00/qwen-opd-manifest.json")
    qwen_records = [
        load_json(ROOT / item["cache_path"])
        for item in qwen_manifest["trajectories"][:8]
    ]
    kimi_records = [
        item
        for item in refresh_attempts(0)
        if item.get("status") == "SUCCEEDED" and item.get("decision") == "correct"
    ][:16]
    tokenizer = probe.load_tokenizer()
    source_model, source_runtime, source_scope, source_load = load_student()
    source_model.eval()
    source_runtime.eval()
    qwen_source = qwen_teacher_audit(source_model, qwen_records)
    kimi_source = probe.correction_metrics(source_model, tokenizer, kimi_records)
    del source_model, source_runtime, source_scope
    release()

    final_model, final_runtime, final_scope, final_load = load_student(
        PROGRESS_ROOT / "trainable.safetensors"
    )
    final_model.eval()
    final_runtime.eval()
    dev = PackedTokenDataset(RAW_DATA, "dev")
    full_dev_ce = probe.corpus_ce(final_model, dev, label="P11_6_FAST_FINAL")
    capability = probe.general_capability(final_model, tokenizer, label="P11_6_FAST_FINAL")
    qwen_final = qwen_teacher_audit(final_model, qwen_records)
    kimi_final = probe.correction_metrics(final_model, tokenizer, kimi_records)
    tokens = torch.from_numpy(dev[0][:128].copy()).unsqueeze(0).cuda()
    routing = probe.routing_diagnostic(final_model, final_runtime, tokens)
    initial = load_file(str(P11_5_STEP_ZERO))
    updates = probe.parameter_update_by_family(final_scope, initial)
    compensation = probe.compensation_warning(updates, routing)
    frozen = probe.frozen_scope_audit(final_model, final_scope)
    source_frozen = load_json(SOURCE_AUDIT_REPORT)["source_frozen_kda"]
    baseline_256 = load_json(P11_5_REPORT)
    baseline_p10 = load_json(P11_5_BASELINE)
    qwen_nonregressed = (
        qwen_final["mean_abs_teacher_gap"] <= qwen_source["mean_abs_teacher_gap"]
    )
    kimi_source_ce = kimi_source["mean_corrected_ce"]
    kimi_final_ce = kimi_final["mean_corrected_ce"]
    kimi_nonregressed = (
        kimi_source_ce is None
        or kimi_final_ce is not None
        and kimi_final_ce <= kimi_source_ce
    )
    teacher_improved = (
        qwen_final["mean_abs_teacher_gap"] < qwen_source["mean_abs_teacher_gap"]
        or kimi_source_ce is not None
        and kimi_final_ce is not None
        and kimi_final_ce < kimi_source_ce
    )
    runtime_checks = {
        "all_chunks_pass": all(item["status"] == "PASS" for item in chunk_reports),
        "all_training_finite": all(
            math.isfinite(float(point["gradient_norm_before_clip"]))
            for item in chunk_reports
            for point in item["training_curve"]
        ),
        "gradient_coverage_563": all(
            point["gradient_tensors"] == 563 and point["optimizer_tensors"] == 563
            for item in chunk_reports
            for point in item["training_curve"]
        ),
        "missing_zero": all(
            point["missing_gradient_count"] == 0
            for item in chunk_reports
            for point in item["training_curve"]
        ),
        "unexpected_zero": all(
            point["unexpected_gradient_count"] == 0
            for item in chunk_reports
            for point in item["training_curve"]
        ),
        "frozen_kda_hash_unchanged": frozen["tensor_hash"] == source_frozen["tensor_hash"],
        "kda_compensation_not_high": compensation["status"] != "HIGH",
        "attnres_active": routing["active"],
        "heldout_not_accessed": True,
    }
    decision = final_decision(
        final_general_composite=float(capability["general_composite"]),
        baseline_general_composite=float(
            baseline_256["general_capability"]["general_composite"]
        ),
        final_domains=capability["domain_accuracy"],
        baseline_domains=baseline_256["general_capability"]["domain_accuracy"],
        full_dev_ce=full_dev_ce,
        qwen_teacher_nonregressed=qwen_nonregressed,
        kimi_teacher_nonregressed=kimi_nonregressed,
        teacher_improved=teacher_improved,
        runtime_passed=all(runtime_checks.values()),
    )
    report = {
        "stage": "P11.6-FAST-FINAL-EVALUATION",
        "status": "PASS",
        "completed_at": now(),
        "checkpoint": str(PROGRESS_ROOT),
        "full_dev_ce": full_dev_ce,
        "p10_t_full_dev_ce": P10_FULL_DEV_CE,
        "p11_5a_full_dev_ce": baseline_256["full_dev_ce"],
        "full_dev_ce_delta_vs_p10": full_dev_ce - P10_FULL_DEV_CE,
        "full_dev_ce_hard_max": FULL_DEV_CE_MAX,
        "general_capability": capability,
        "p11_5a_general_capability": baseline_256["general_capability"],
        "p10_t_general_capability": baseline_p10["general_capability"],
        "fixed_teacher_audit": {
            "qwen_records": len(qwen_records),
            "kimi_records": len(kimi_records),
            "qwen_source": qwen_source,
            "qwen_final": qwen_final,
            "kimi_source": kimi_source,
            "kimi_final": kimi_final,
            "qwen_nonregressed": qwen_nonregressed,
            "kimi_nonregressed": kimi_nonregressed,
            "teacher_improved": teacher_improved,
        },
        "routing": routing,
        "parameter_update_by_family": updates,
        "kda_freeze_compensation": compensation,
        "frozen_kda": frozen,
        "runtime_checks": runtime_checks,
        "decision": decision,
        "source_load": source_load,
        "final_load": final_load,
        "heldout_accessed": False,
        "result_marker": f"P11_6_FAST_FINAL_EVALUATION={decision['decision']}",
    }
    atomic_json(EVALUATION_REPORT, report)
    print(report["result_marker"], flush=True)
    del final_model, final_runtime, final_scope, tokens
    release()
    return report


@torch.inference_mode()
def short_logits(model: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    return model(tokens, use_cache=False, return_dict=True).logits[:, -1].detach().cpu()


def materialize_selected(evaluation: dict[str, Any]) -> dict[str, Any]:
    decision = evaluation["decision"]
    selected = (
        PROGRESS_ROOT
        if decision["selected_checkpoint_role"] == "P11_6_1M"
        else P11_5_SOURCE
    )
    selected_manifest = load_json(selected / "checkpoint-manifest.json")
    selected_hash = selected_manifest["trainable_sha256"]
    candidate_manifest_path = CANDIDATE / "p11-6-fast-candidate-manifest.json"
    resume_pending_candidate = False
    if candidate_manifest_path.is_file():
        existing = load_json(candidate_manifest_path)
        if existing.get("status") == "PASS" and existing.get("selected_trainable_sha256") == selected_hash:
            print("P11_6_FAST_CANDIDATE=RESUMED", flush=True)
            return existing
        if (
            existing.get("status") == "PENDING_RELOAD_EVALUATION"
            and existing.get("selected_trainable_sha256") == selected_hash
        ):
            resume_pending_candidate = True
        else:
            raise FileExistsError("P11.6 candidate exists with a different provenance")
    model, runtime, scope, load = load_student(selected / "trainable.safetensors")
    model.eval()
    runtime.eval()
    replay_tokens = (
        torch.from_numpy(PackedTokenDataset(RAW_DATA, "dev")[0][:32].copy())
        .unsqueeze(0)
        .cuda()
    )
    selected_logits = short_logits(model, replay_tokens)
    frozen_before = probe.frozen_scope_audit(model, scope)
    frozen_names = list(scope.manifest.get("frozen_audit_parameter_names", []))
    if not resume_pending_candidate:
        merge = merge_all_lora(scope.lora)
        if any("lora" in name.lower() for name, _ in model.named_parameters()):
            raise RuntimeError("P11.6 LoRA parameters remain after merge")
        merged_logits = short_logits(model, replay_tokens)
        merged_logits_sha256 = tensor_sha256(merged_logits)
        selected_to_merged = {
            "selected_logits_sha256": tensor_sha256(selected_logits),
            "merged_logits_sha256": merged_logits_sha256,
            "exact": torch.equal(selected_logits, merged_logits),
            "max_abs": float(
                (selected_logits.float() - merged_logits.float()).abs().max()
            ),
            "relative_l2": float(
                (selected_logits.float() - merged_logits.float()).norm()
                / selected_logits.float().norm().clamp_min(1.0e-30)
            ),
            "diagnostic_only": True,
        }
        staging = CANDIDATE.with_name(CANDIDATE.name + ".staging")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        overrides = candidate_override_state(model, scope)
        save_file(overrides, str(staging / "model-dense-overrides.safetensors"))
        runtime_state = {
            name: value.detach().cpu() for name, value in runtime.state_dict().items()
        }
        save_file(runtime_state, str(staging / "attnres.safetensors"))
        runtime_config = {
            "architecture": "DENSEK3_CORE_KDA_MLA_ATTNRES_SITU_GENERAL_DUAL_TEACHER_OPD",
            "stage": "P11.6-FAST",
            "source_p10_t": str(probe.P10_SOURCE),
            "source_p11_5a": str(P11_5_SOURCE),
            "selected_checkpoint": str(selected),
            "selected_checkpoint_role": decision["selected_checkpoint_role"],
            "cumulative_main_tokens": (
                TARGET_MAIN_TOKENS
                if decision["selected_checkpoint_role"] == "P11_6_1M"
                else SOURCE_MAIN_TOKENS
            ),
            "lora_used_for_optimization": True,
            "lora_present_at_runtime": False,
            "titan_dtype": "fp16",
            "canonical_bf16_replay": "PENDING",
            "heldout_accessed": False,
        }
        atomic_json(staging / "p11-6-fast-runtime-config.json", runtime_config)
        files = {
            path.name: sha256_file(path)
            for path in staging.iterdir()
            if path.is_file()
        }
        atomic_json(
            staging / "p11-6-fast-candidate-manifest.json",
            {
                "stage": "TITAN_FP16_PROVISIONAL_P11_GENERAL_CANDIDATE",
                "status": "PENDING_RELOAD_EVALUATION",
                "selected_checkpoint": str(selected),
                "selected_checkpoint_role": decision["selected_checkpoint_role"],
                "selected_trainable_sha256": selected_hash,
                "files": files,
                "lora_merge": merge,
                "merged_logits_sha256": merged_logits_sha256,
                "selected_to_merged_numerics": selected_to_merged,
                "frozen_scope_audit_before": frozen_before,
                "decision": decision,
                "heldout_accessed": False,
            },
        )
        staging.replace(CANDIDATE)
        del overrides, merged_logits
    else:
        merged_logits_sha256 = load_json(candidate_manifest_path)[
            "merged_logits_sha256"
        ]
    del model, runtime, scope, replay_tokens, selected_logits
    release()

    reloaded, reloaded_runtime, reloaded_load = load_p10_t_student()
    apply_dense_overrides(reloaded, CANDIDATE / "model-dense-overrides.safetensors")
    reloaded_runtime.load_state_dict(load_file(str(CANDIDATE / "attnres.safetensors")))
    enable_p7_latent_cache_runtime(P6_MLA_LAYERS)
    reloaded.eval()
    reloaded_runtime.eval()
    replay_tokens = (
        torch.from_numpy(PackedTokenDataset(RAW_DATA, "dev")[0][:32].copy())
        .unsqueeze(0)
        .cuda()
    )
    after = short_logits(reloaded, replay_tokens)
    dev = PackedTokenDataset(RAW_DATA, "dev")
    full_dev_ce = probe.corpus_ce(reloaded, dev, label="P11_6_FAST_RELOAD")
    tokenizer = probe.load_tokenizer()
    general = probe.general_capability(
        reloaded, tokenizer, label="P11_6_FAST_RELOAD"
    )
    disable_block_attnres_runtime(reloaded)
    generated = probe.generation(reloaded, tokenizer, reloaded_runtime)
    cache_limits = yaml.safe_load(
        (
            TITAN / "configs/p6-titan-fp16-cache-compat-2026-08-19-01.yaml"
        ).read_text(encoding="utf-8")
    )["evaluation"]["fallback_requires_all"]
    cache = probe.cache_equivalence(reloaded, reloaded_runtime, cache_limits)
    enable_block_attnres_runtime(reloaded, reloaded_runtime)
    routing = probe.routing_diagnostic(reloaded, reloaded_runtime, replay_tokens)
    frozen_after = frozen_parameter_audit(reloaded, frozen_names)
    checks = {
        "merged_dense_reload_logits_exact": tensor_sha256(after)
        == merged_logits_sha256,
        "reload_full_dev_finite": math.isfinite(full_dev_ce),
        "reload_full_dev_lm_healthy": full_dev_ce <= FULL_DEV_CE_MAX,
        "reload_general_finite": math.isfinite(float(general["general_composite"])),
        "runtime_lora_absent": not any(
            "lora" in name.lower() for name, _ in reloaded.named_parameters()
        ),
        "attnres_active": routing["active"],
        "cache_pass": cache["status"] == "PASS",
        "generation_pass": all(item["finite"] for item in generated),
        "frozen_kda_exact": frozen_before["tensor_hash"] == frozen_after["tensor_hash"],
        "heldout_not_accessed": True,
    }
    manifest = load_json(candidate_manifest_path)
    manifest.update(
        {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "completed_at": now(),
            "selected_trainable_sha256": selected_hash,
            "merge_reload": {
                "merged_dense_logits_exact": tensor_sha256(after)
                == merged_logits_sha256,
                "merged_logits_sha256": merged_logits_sha256,
                "reloaded_logits_sha256": tensor_sha256(after),
                "reload_full_dev_ce": full_dev_ce,
            },
            "general_capability": general,
            "routing": routing,
            "cache": cache,
            "generation": generated,
            "frozen_scope_audit_after": frozen_after,
            "reload_report": reloaded_load,
            "checks": checks,
            "lora_present_at_runtime": False,
        }
    )
    atomic_json(candidate_manifest_path, manifest)
    del reloaded, reloaded_runtime, replay_tokens, after
    release()
    return manifest


def closeout() -> dict[str, Any]:
    if FINAL_REPORT.is_file() and load_json(FINAL_REPORT).get("status") == "PASS":
        report = load_json(FINAL_REPORT)
        print("P11_6_FAST_CLOSEOUT=RESUMED", flush=True)
        return report
    evaluation = final_evaluation()
    if evaluation.get("status") != "PASS":
        return evaluation
    gpu = gpu_start_state(stage="closeout")
    if not gpu["allowed"]:
        return {"stage": "P11.6-FAST-CLOSEOUT", "status": "DEFERRED_GPU_MEMORY_UNSAFE", "gpu": gpu}
    candidate = materialize_selected(evaluation)
    decision = evaluation["decision"]
    passed = candidate["status"] == "PASS"
    ledger = accounted_ledger_cost(REPORT_ROOT)
    reports = [
        load_json(REPORT_DIR / f"chunk-{index:02d}/training-segment.json")
        for index in range(CHUNK_COUNT)
    ]
    result_status = (
        "PASS"
        if passed
        else "FAIL"
    )
    report = {
        "stage": "P11.6-FAST",
        "status": result_status,
        "completed_at": now(),
        "contract": P11_6_FAST_CONTRACT,
        "decision": decision["decision"],
        "p11_6_status": (
            "PASS"
            if decision["decision"] == "STRONG_GO"
            else "SAFE_BUT_NO_MEASURABLE_EXTERNAL_GAIN"
            if decision["decision"] == "SAFE_BUT_SATURATED"
            else "NO_GO_SCALEUP"
        ),
        "source_checkpoint": str(P11_5_SOURCE),
        "selected_checkpoint_role": decision["selected_checkpoint_role"],
        "selected_candidate": str(CANDIDATE),
        "main_training_tokens": {
            "source": SOURCE_MAIN_TOKENS,
            "target": TARGET_MAIN_TOKENS,
            "new_qwen": sum(item["new_qwen_tokens"] for item in reports),
            "new_raw_lm": sum(item["new_raw_lm_tokens"] for item in reports),
        },
        "final_evaluation": evaluation,
        "candidate_manifest": candidate,
        "p11_cumulative_api_ledger": ledger,
        "p11_general_distillation_frozen": passed,
        "p11_6_2m_automatically_allowed": False,
        "p11_7_allowed": False,
        "p12_long_context_allowed": passed and decision["p12_long_context_allowed"],
        "heldout_accessed": False,
        "canonical_p11_status": "PENDING_NATIVE_BF16_FULL_SCOPE_RETRAIN",
        "result_marker": (
            f"P11_6_FAST={decision['decision']}"
            if passed
            else "P11_6_FAST=CLOSEOUT_FAIL"
        ),
    }
    atomic_json(FINAL_REPORT, report)
    print(report["result_marker"], flush=True)
    print(f"P12_LONG_CONTEXT_ALLOWED={str(report['p12_long_context_allowed']).lower()}", flush=True)
    print("P11_6_2M_AUTOMATICALLY_ALLOWED=false", flush=True)
    print("P11_7_ALLOWED=false", flush=True)
    return report


def run_all() -> dict[str, Any]:
    first = preflight()
    if first.get("status") != "PASS":
        return first
    audit = source_runtime_audit()
    if audit.get("status") != "PASS":
        return audit
    for chunk_index in range(CHUNK_COUNT):
        cache = build_qwen_cache(chunk_index)
        if cache.get("status") != "PASS":
            return cache
        if refresh_required_for_chunk(chunk_index):
            refresh = run_kimi_refresh(refresh_index_for_chunk(chunk_index), chunk_index)
            if refresh.get("status") != "PASS":
                return refresh
        training = run_training_chunk(chunk_index)
        if training.get("status") != "PASS":
            return training
    return closeout()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("preflight", "source-audit", "qwen", "refresh", "chunk", "closeout", "all"),
        required=True,
    )
    parser.add_argument("--chunk-index", type=int)
    parser.add_argument("--refresh-index", type=int)
    args = parser.parse_args()
    envelope = {
        "stage": "P11.6-FAST",
        "status": "RUNNING",
        "started_at": now(),
        "phase": args.phase,
        "chunk_index": args.chunk_index,
        "refresh_index": args.refresh_index,
        "heldout_accessed": False,
        "p11_6_2m_automatically_allowed": False,
        "p11_7_allowed": False,
        "p12_long_context_allowed": False,
    }
    try:
        if args.phase in {"qwen", "chunk"} and (
            args.chunk_index is None or not 0 <= args.chunk_index < CHUNK_COUNT
        ):
            raise ValueError("--chunk-index 0..11 is required")
        if args.phase == "refresh" and (
            args.refresh_index not in (0, 1)
            or args.chunk_index not in (0, CHUNK_COUNT // 2)
        ):
            raise ValueError("refresh requires --refresh-index 0/1 and --chunk-index 0/6")
        if args.phase == "preflight":
            result = preflight()
        elif args.phase == "source-audit":
            result = source_runtime_audit()
        elif args.phase == "qwen":
            result = build_qwen_cache(int(args.chunk_index))
        elif args.phase == "refresh":
            result = run_kimi_refresh(int(args.refresh_index), int(args.chunk_index))
        elif args.phase == "chunk":
            result = run_training_chunk(int(args.chunk_index))
        elif args.phase == "closeout":
            result = closeout()
        else:
            result = run_all()
    except Exception as error:
        result = {
            **envelope,
            "status": "ERROR",
            "completed_at": now(),
            "exception_type": type(error).__name__,
            "exception": str(error),
            "result_marker": "P11_6_FAST=ERROR",
        }
        atomic_json(LAST_ERROR_REPORT, result)
        print(f"P11_6_FAST_EXCEPTION={type(error).__name__}: {error}", flush=True)
    return 0 if result.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
