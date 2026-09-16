"""Pure P11.5a scheduling, cost, and loss contracts.

The runtime runner is intentionally separate from these helpers so the exact
token/accounting rules can be tested without loading either 4B model.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch

P11_5A_AMENDMENT = "P11_5A_PROBE_CONTRACT_2026_08_21"
MAIN_TOKENS = 262_144
QWEN_TOKENS = 157_286
RAW_LM_TOKENS = 104_858
WAVE_COUNT = 4
UPDATE_BUCKETS_PER_WAVE = 16
KIMI_MAX_REVIEW_ATTEMPTS = 64
KIMI_REVIEWS_PER_WAVE = 16
KIMI_REVIEWS_PER_DOMAIN_PER_WAVE = 4
P11_API_TOTAL_HARD_CAP_CNY = 5.0
DAILY_API_TOTAL_HARD_CAP_CNY = 20.0
QWEN_LONG_ROLLOUT_TOKENS = 512
QWEN_CANDIDATE_ROLLOUT_TOKENS = 16
QWEN_CANDIDATES_PER_DOMAIN = 12
RAW_SEQUENCE_TARGET_TOKENS = 2047
STREAM_LOSS_WEIGHTS = {"qwen": 0.36, "raw_lm": 0.24, "kimi": 0.40}
P11_FP16_INITIAL_LOSS_SCALE = 1024.0
P11_FP16_LOSS_SCALE_GROWTH_INTERVAL = 2000
P11_GPU_MEMORY_GATE_AMENDMENT = "P11_5A_TITAN_MEMORY_ENVELOPE_GATE_2026_08_23"
P11_GPU_HISTORICAL_PEAK_MEMORY_MIB = 8_970
P11_GPU_SAFETY_MARGIN_MIB = 6_144
P11_GPU_MAX_START_TEMPERATURE_C = 75


def gpu_memory_envelope(
    *,
    visible: bool,
    temperature_c: int | None,
    memory_used_mib: int | None,
    memory_total_mib: int | None,
) -> dict[str, Any]:
    """Return the Titan P11 start gate without requiring an idle GPU."""
    finite_memory = (
        memory_used_mib is not None
        and memory_total_mib is not None
        and 0 <= memory_used_mib <= memory_total_mib
    )
    required_free_mib = (
        P11_GPU_HISTORICAL_PEAK_MEMORY_MIB + P11_GPU_SAFETY_MARGIN_MIB
    )
    free_memory_mib = (
        memory_total_mib - memory_used_mib if finite_memory else None
    )
    checks = {
        "gpu_visible": visible,
        "temperature_safe": temperature_c is not None
        and temperature_c <= P11_GPU_MAX_START_TEMPERATURE_C,
        "memory_fields_valid": finite_memory,
        "historical_peak_plus_margin_fits": free_memory_mib is not None
        and free_memory_mib >= required_free_mib,
    }
    return {
        "amendment_id": P11_GPU_MEMORY_GATE_AMENDMENT,
        "mode": "HISTORICAL_PEAK_PLUS_SAFETY_MARGIN",
        "allowed": all(checks.values()),
        "checks": checks,
        "memory_used_mib": memory_used_mib,
        "memory_total_mib": memory_total_mib,
        "memory_free_mib": free_memory_mib,
        "historical_peak_memory_mib": P11_GPU_HISTORICAL_PEAK_MEMORY_MIB,
        "safety_margin_mib": P11_GPU_SAFETY_MARGIN_MIB,
        "required_free_mib": required_free_mib,
        "requires_gpu_idle": False,
    }


def rounded_fraction(total: int, numerator: int, denominator: int) -> int:
    """Round non-negative rational values half-up, not with banker's rounding."""
    if total < 0 or numerator < 0 or denominator <= 0 or numerator > denominator:
        raise ValueError("Invalid cumulative fraction")
    return (total * numerator * 2 + denominator) // (2 * denominator)


def cumulative_main_quotas(completed_waves: int) -> dict[str, int]:
    if not 0 <= completed_waves <= WAVE_COUNT:
        raise ValueError("completed_waves is outside P11.5a")
    main = MAIN_TOKENS * completed_waves // WAVE_COUNT
    qwen = rounded_fraction(QWEN_TOKENS, completed_waves, WAVE_COUNT)
    raw_lm = main - qwen
    return {"main": main, "qwen": qwen, "raw_lm": raw_lm}


def wave_main_quotas(wave_index: int) -> dict[str, int]:
    if not 0 <= wave_index < WAVE_COUNT:
        raise ValueError("wave_index is outside P11.5a")
    before = cumulative_main_quotas(wave_index)
    after = cumulative_main_quotas(wave_index + 1)
    return {name: after[name] - before[name] for name in after}


def qwen_rollout_plan(quota: int) -> list[dict[str, int | str]]:
    """Plan 48 short selection rollouts followed by long OPD trajectories.

    Runtime EOS can shorten a trajectory. The runner then appends deterministic
    long trajectories until the exact quota is reached; this static plan is the
    no-EOS upper-efficiency plan and the first 48 roles remain invariant.
    """
    short_count = QWEN_CANDIDATES_PER_DOMAIN * 4
    short_tokens = short_count * QWEN_CANDIDATE_ROLLOUT_TOKENS
    if quota < short_tokens:
        raise ValueError("Qwen wave quota cannot hold the selection pool")
    plan: list[dict[str, int | str]] = [
        {"role": "kimi_candidate", "maximum_tokens": QWEN_CANDIDATE_ROLLOUT_TOKENS}
        for _ in range(short_count)
    ]
    remaining = quota - short_tokens
    while remaining:
        length = min(QWEN_LONG_ROLLOUT_TOKENS, remaining)
        plan.append({"role": "main_opd", "maximum_tokens": length})
        remaining -= length
    return plan


def raw_lm_target_lengths(quota: int) -> list[int]:
    if quota <= 0:
        raise ValueError("Raw-LM quota must be positive")
    full, remainder = divmod(quota, RAW_SEQUENCE_TARGET_TOKENS)
    lengths = [RAW_SEQUENCE_TARGET_TOKENS] * full
    if remainder:
        lengths.append(remainder)
    if sum(lengths) != quota or any(not 0 < value <= RAW_SEQUENCE_TARGET_TOKENS for value in lengths):
        raise AssertionError("Raw-LM exact-token partition failed")
    return lengths


def assign_update_buckets(token_lengths: list[int], bucket_count: int = UPDATE_BUCKETS_PER_WAVE) -> list[list[int]]:
    """Greedily balance indivisible trajectories by supervised-token count."""
    if bucket_count <= 0 or any(value <= 0 for value in token_lengths):
        raise ValueError("Invalid update-bucket inputs")
    buckets: list[list[int]] = [[] for _ in range(bucket_count)]
    totals = [0] * bucket_count
    for index, length in sorted(enumerate(token_lengths), key=lambda item: (-item[1], item[0])):
        target = min(range(bucket_count), key=lambda bucket: (totals[bucket], bucket))
        buckets[target].append(index)
        totals[target] += length
    return buckets


def trajectory_mean(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean over one trajectory's target tokens, never a token-sum loss."""
    if mask is not None:
        if mask.shape != values.shape:
            raise ValueError("Trajectory mask shape mismatch")
        values = values[mask]
    if values.numel() == 0:
        raise ValueError("Trajectory target is empty")
    return values.mean()


def equal_trajectory_mean(losses: list[torch.Tensor]) -> torch.Tensor:
    """Give every corrected trajectory exactly equal base weight."""
    if not losses:
        raise ValueError("No corrected trajectories")
    return torch.stack([value.reshape(()) for value in losses]).mean()


def accounted_ledger_cost(report_dir: Path) -> dict[str, Any]:
    """Conservatively sum every P11 ledger day inside the project report dir."""
    files = sorted(report_dir.glob("kimi-api-cost-*.json"))
    total = 0.0
    entries = 0
    by_status: Counter[str] = Counter()
    file_reports: list[dict[str, Any]] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        file_total = 0.0
        for entry in payload.get("entries", []):
            value = float(entry.get("cost_cny", entry.get("reserved_upper_bound_cny", 0.0)))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid API ledger cost in {path.name}")
            file_total += value
            entries += 1
            by_status[str(entry.get("status", "UNKNOWN"))] += 1
        total += file_total
        file_reports.append({"path": str(path), "accounted_cost_cny": file_total})
    return {
        "accounted_cost_cny": total,
        "entry_count": entries,
        "by_status": dict(sorted(by_status.items())),
        "files": file_reports,
    }


def api_call_allowed(*, cumulative_cost_cny: float, next_upper_bound_cny: float, attempts: int) -> dict[str, Any]:
    if attempts < 0 or not all(math.isfinite(value) and value >= 0 for value in (cumulative_cost_cny, next_upper_bound_cny)):
        raise ValueError("Invalid P11 API accounting")
    review_room = attempts < KIMI_MAX_REVIEW_ATTEMPTS
    budget_room = cumulative_cost_cny + next_upper_bound_cny <= P11_API_TOTAL_HARD_CAP_CNY
    return {
        "allowed": review_room and budget_room,
        "review_attempt_room": review_room,
        "budget_room": budget_room,
        "stop_reason": (
            None
            if review_room and budget_room
            else ("P11_CUMULATIVE_API_HARD_CAP" if not budget_room else "KIMI_MAX_REVIEW_ATTEMPTS")
        ),
        "projected_cumulative_cost_cny": cumulative_cost_cny + next_upper_bound_cny,
    }


def contract_manifest() -> dict[str, Any]:
    waves = []
    for index in range(WAVE_COUNT):
        quota = wave_main_quotas(index)
        waves.append(
            {
                "wave_index": index,
                "starts_at_main_tokens": cumulative_main_quotas(index)["main"],
                "ends_at_main_tokens": cumulative_main_quotas(index + 1)["main"],
                "main_quotas": quota,
                "kimi_max_attempts": KIMI_REVIEWS_PER_WAVE,
                "kimi_max_attempts_per_domain": KIMI_REVIEWS_PER_DOMAIN_PER_WAVE,
                "qwen_static_rollout_plan": qwen_rollout_plan(quota["qwen"]),
                "raw_lm_target_lengths": raw_lm_target_lengths(quota["raw_lm"]),
            }
        )
    checks = {
        "main_exact": cumulative_main_quotas(WAVE_COUNT)["main"] == MAIN_TOKENS,
        "qwen_exact": cumulative_main_quotas(WAVE_COUNT)["qwen"] == QWEN_TOKENS,
        "raw_lm_exact": cumulative_main_quotas(WAVE_COUNT)["raw_lm"] == RAW_LM_TOKENS,
        "wave_attempt_cap_64": WAVE_COUNT * KIMI_REVIEWS_PER_WAVE == KIMI_MAX_REVIEW_ATTEMPTS,
        "loss_weights_sum_one": abs(sum(STREAM_LOSS_WEIGHTS.values()) - 1.0) < 1.0e-12,
    }
    return {
        "amendment": P11_5A_AMENDMENT,
        "main_training_tokens": MAIN_TOKENS,
        "qwen_exact_opd_tokens": QWEN_TOKENS,
        "raw_lm_tokens": RAW_LM_TOKENS,
        "kimi_tokens_in_main_denominator": False,
        "kimi_loss_reduction": "PER_TRAJECTORY_MEAN_TARGET_CE",
        "stream_loss_weights": STREAM_LOSS_WEIGHTS,
        "stream_loss_weight_derivation": "LEGACY_40_PERCENT_KIMI_PLUS_RENORMALIZED_60_PERCENT_MAIN_STREAM",
        "update_buckets_per_wave": UPDATE_BUCKETS_PER_WAVE,
        "api_total_hard_cap_cny": P11_API_TOTAL_HARD_CAP_CNY,
        "daily_api_total_hard_cap_cny": DAILY_API_TOTAL_HARD_CAP_CNY,
        "waves": waves,
        "checks": checks,
        "passed": all(checks.values()),
    }


__all__ = [
    "DAILY_API_TOTAL_HARD_CAP_CNY",
    "KIMI_MAX_REVIEW_ATTEMPTS",
    "KIMI_REVIEWS_PER_DOMAIN_PER_WAVE",
    "KIMI_REVIEWS_PER_WAVE",
    "MAIN_TOKENS",
    "P11_5A_AMENDMENT",
    "P11_API_TOTAL_HARD_CAP_CNY",
    "P11_FP16_INITIAL_LOSS_SCALE",
    "P11_FP16_LOSS_SCALE_GROWTH_INTERVAL",
    "P11_GPU_HISTORICAL_PEAK_MEMORY_MIB",
    "P11_GPU_MAX_START_TEMPERATURE_C",
    "P11_GPU_MEMORY_GATE_AMENDMENT",
    "P11_GPU_SAFETY_MARGIN_MIB",
    "QWEN_TOKENS",
    "RAW_LM_TOKENS",
    "STREAM_LOSS_WEIGHTS",
    "UPDATE_BUCKETS_PER_WAVE",
    "WAVE_COUNT",
    "accounted_ledger_cost",
    "api_call_allowed",
    "assign_update_buckets",
    "contract_manifest",
    "cumulative_main_quotas",
    "equal_trajectory_mean",
    "gpu_memory_envelope",
    "qwen_rollout_plan",
    "raw_lm_target_lengths",
    "trajectory_mean",
    "wave_main_quotas",
]
