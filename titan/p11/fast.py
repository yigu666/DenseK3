"""Pure scheduling and decision contracts for Titan P11.6 FAST.

The module intentionally has no 4B-model dependency.  Runtime code lives in
``titan/scripts/run_p11_fast.py``; these helpers make the exact token, API,
learning-rate continuation, and final-decision rules independently testable.
"""

from __future__ import annotations

import math
from typing import Any

P11_6_FAST_CONTRACT = "P11_6_FAST_GENERAL_DUAL_TEACHER_OPD_2026_08_24"

SOURCE_MAIN_TOKENS = 262_144
TARGET_MAIN_TOKENS = 1_048_576
NEW_MAIN_TOKENS = 786_432
NEW_QWEN_TOKENS = 471_859
NEW_RAW_LM_TOKENS = 314_573

CHUNK_COUNT = 12
TOKENS_PER_CHUNK = 65_536
UPDATE_BUCKETS_PER_CHUNK = 16
TOKENS_PER_UPDATE_BUCKET = 4_096
SOURCE_OPTIMIZER_STEPS = 64
NEW_OPTIMIZER_STEPS = 192
TARGET_OPTIMIZER_STEPS = 256

RECOVERY_MAIN_TOKEN_MARKERS = (
    393_216,
    524_288,
    655_360,
    786_432,
    917_504,
    1_048_576,
)
GRADIENT_AUDIT_STEPS = (64, 160, 256)
FROZEN_KDA_AUDIT_MAIN_TOKENS = (262_144, 655_360, 1_048_576)

KIMI_REFRESH_START_MAIN_TOKENS = (262_144, 655_360)
KIMI_MAX_ATTEMPTS_PER_REFRESH = 64
KIMI_MAX_NEW_ATTEMPTS = 128
KIMI_ATTEMPTS_PER_DOMAIN = 16
KIMI_ROLLING_REVIEW_WINDOW = 16
P11_CUMULATIVE_API_HARD_CAP_CNY = 15.0
DAILY_API_HARD_CAP_CNY = 20.0
FAST_GPU_MEMORY_GATE_AMENDMENT = "P11_6_FAST_TITAN_MEMORY_ONLY_GATE_2026_08_25"
FAST_GPU_SAFETY_MARGIN_MIB = 3_072

P10_FULL_DEV_CE = 2.389_680_047_457_61
FULL_DEV_CE_MAX = P10_FULL_DEV_CE + 0.03
P11_5A_GENERAL_COMPOSITE = 0.65625


def cumulative_new_quotas(completed_chunks: int) -> dict[str, int]:
    """Return exact cumulative P11.6-only quotas after ``completed_chunks``."""
    if not 0 <= completed_chunks <= CHUNK_COUNT:
        raise ValueError("completed_chunks is outside P11.6 FAST")
    main = TOKENS_PER_CHUNK * completed_chunks
    # Difference-of-floors gives exact totals, exact 65,536-token chunks, and
    # never permits Qwen/Raw independent rounding to create a 65,537 chunk.
    qwen = NEW_QWEN_TOKENS * completed_chunks // CHUNK_COUNT
    raw_lm = main - qwen
    return {"main": main, "qwen": qwen, "raw_lm": raw_lm}


def cumulative_main_quotas(completed_chunks: int) -> dict[str, int]:
    values = cumulative_new_quotas(completed_chunks)
    return {
        "main": SOURCE_MAIN_TOKENS + values["main"],
        "new_main": values["main"],
        "new_qwen": values["qwen"],
        "new_raw_lm": values["raw_lm"],
    }


def chunk_quotas(chunk_index: int) -> dict[str, int]:
    if not 0 <= chunk_index < CHUNK_COUNT:
        raise ValueError("chunk_index is outside P11.6 FAST")
    before = cumulative_new_quotas(chunk_index)
    after = cumulative_new_quotas(chunk_index + 1)
    return {name: after[name] - before[name] for name in after}


def refresh_index_for_chunk(chunk_index: int) -> int:
    if not 0 <= chunk_index < CHUNK_COUNT:
        raise ValueError("chunk_index is outside P11.6 FAST")
    return 0 if chunk_index < CHUNK_COUNT // 2 else 1


def refresh_required_for_chunk(chunk_index: int) -> bool:
    return chunk_index in (0, CHUNK_COUNT // 2)


def is_recovery_marker(completed_chunks: int) -> bool:
    return cumulative_main_quotas(completed_chunks)["main"] in RECOVERY_MAIN_TOKEN_MARKERS


def continuation_cosine_learning_rate(
    extension_step: int,
    *,
    extension_steps: int,
    saved_learning_rate: float,
    terminal_learning_rate: float,
) -> float:
    """Continue from the persisted LR without a boundary jump.

    P11.5a ended one discrete update before its analytic cosine endpoint.  A
    fresh 256-step cosine would jump upward at step 65.  P11.6 instead starts
    exactly at the optimizer's persisted LR and smoothly finishes at the same
    minimum LR that the original schedule targeted.
    """
    if extension_steps <= 0 or not 0 <= extension_step < extension_steps:
        raise ValueError("extension step is outside the continuation horizon")
    if not (
        math.isfinite(saved_learning_rate)
        and math.isfinite(terminal_learning_rate)
        and saved_learning_rate > 0
        and terminal_learning_rate > 0
        and saved_learning_rate >= terminal_learning_rate
    ):
        raise ValueError("invalid continuation learning-rate endpoints")
    if extension_steps == 1:
        return terminal_learning_rate
    progress = extension_step / (extension_steps - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return terminal_learning_rate + (saved_learning_rate - terminal_learning_rate) * cosine


def fast_api_call_allowed(
    *,
    cumulative_cost_cny: float,
    next_upper_bound_cny: float,
    refresh_attempts: int,
    new_attempts: int,
) -> dict[str, Any]:
    values = (cumulative_cost_cny, next_upper_bound_cny)
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("invalid P11.6 API cost")
    if refresh_attempts < 0 or new_attempts < 0:
        raise ValueError("invalid P11.6 API attempt count")
    refresh_room = refresh_attempts < KIMI_MAX_ATTEMPTS_PER_REFRESH
    total_room = new_attempts < KIMI_MAX_NEW_ATTEMPTS
    budget_room = cumulative_cost_cny + next_upper_bound_cny <= P11_CUMULATIVE_API_HARD_CAP_CNY
    allowed = refresh_room and total_room and budget_room
    if allowed:
        stop_reason = None
    elif not budget_room:
        stop_reason = "P11_CUMULATIVE_API_HARD_CAP"
    elif not total_room:
        stop_reason = "P11_6_MAX_NEW_KIMI_ATTEMPTS"
    else:
        stop_reason = "P11_6_REFRESH_ATTEMPT_CAP"
    return {
        "allowed": allowed,
        "refresh_attempt_room": refresh_room,
        "new_attempt_room": total_room,
        "budget_room": budget_room,
        "stop_reason": stop_reason,
        "projected_cumulative_cost_cny": cumulative_cost_cny + next_upper_bound_cny,
    }


def kimi_refresh_early_stop(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate the frozen rolling-16 Kimi refresh stop policy."""
    succeeded = [item for item in decisions if item.get("status") == "SUCCEEDED"]
    correct = [item for item in succeeded if item.get("decision") == "correct"]
    keep = [item for item in succeeded if item.get("decision") == "keep"]
    denominator = len(correct) + len(keep)
    correction_rate = len(correct) / denominator if denominator else None
    keep_rate = len(keep) / denominator if denominator else None
    trivial = [item for item in correct if bool(item.get("trivial_correction"))]
    # With zero corrections the "mostly trivial corrections" clause is
    # vacuously satisfied; together with 0% CORRECT and 100% KEEP this is the
    # strongest possible signal that another 48 paid reviews have low value.
    trivial_fraction = (
        len(trivial) / len(correct) if correct else (1.0 if denominator else None)
    )
    window_complete = len(decisions) > 0 and len(decisions) % KIMI_ROLLING_REVIEW_WINDOW == 0
    conditions = {
        "CORRECTION_RATE_LT_15_PERCENT": (
            correction_rate is not None and correction_rate < 0.15
        ),
        "CORRECTIONS_MOSTLY_TRIVIAL": (
            trivial_fraction is not None and trivial_fraction >= 0.75
        ),
        "KEEP_STRONGLY_DOMINATES": keep_rate is not None and keep_rate >= 0.80,
    }
    # The frozen user contract uses a conjunctive early-stop rule.  A single
    # low correction-rate window is not enough by itself to stop a refresh.
    stop = window_complete and all(conditions.values())
    reasons = list(conditions) if stop else []
    return {
        "stop": stop,
        "window_complete": window_complete,
        "attempts_observed": len(decisions),
        "successful_decisions": denominator,
        "correction_rate": correction_rate,
        "keep_rate": keep_rate,
        "trivial_correction_fraction": trivial_fraction,
        "reasons": reasons,
    }


def final_decision(
    *,
    final_general_composite: float,
    baseline_general_composite: float,
    final_domains: dict[str, float],
    baseline_domains: dict[str, float],
    full_dev_ce: float,
    qwen_teacher_nonregressed: bool,
    kimi_teacher_nonregressed: bool,
    teacher_improved: bool,
    runtime_passed: bool,
) -> dict[str, Any]:
    domains_nonregressed = all(
        final_domains.get(name, -math.inf) >= value
        for name, value in baseline_domains.items()
    )
    domains_unchanged = final_domains == baseline_domains
    lm_stable = math.isfinite(full_dev_ce) and full_dev_ce <= FULL_DEV_CE_MAX
    teacher_safe = qwen_teacher_nonregressed and kimi_teacher_nonregressed
    common = lm_stable and teacher_safe and runtime_passed
    if (
        common
        and final_general_composite > baseline_general_composite
        and domains_nonregressed
    ):
        decision = "STRONG_GO"
        selected = "P11_6_1M"
    elif (
        common
        and final_general_composite == baseline_general_composite
        and domains_unchanged
        and teacher_improved
    ):
        decision = "SAFE_BUT_SATURATED"
        selected = "P11_6_1M"
    else:
        decision = "NO_GO"
        selected = "P11_5A_256K_FALLBACK"
    return {
        "decision": decision,
        "selected_checkpoint_role": selected,
        "checks": {
            "full_dev_lm_stable": lm_stable,
            "teacher_nonregressed": teacher_safe,
            "runtime_passed": runtime_passed,
            "domains_nonregressed": domains_nonregressed,
            "domains_unchanged": domains_unchanged,
            "teacher_improved": teacher_improved,
        },
        "p11_6_pass": decision in {"STRONG_GO", "SAFE_BUT_SATURATED"},
        "p11_general_frozen": True,
        # A NO_GO scale-up selects the already-PASS 256K checkpoint.  The
        # contract explicitly ends general OPD at 1M and moves to the next
        # research question rather than spending more tokens rescuing scale-up.
        "p12_long_context_allowed": True,
        "p11_6_2m_automatically_allowed": False,
        "p11_7_allowed": False,
    }


def contract_manifest() -> dict[str, Any]:
    chunks = []
    for index in range(CHUNK_COUNT):
        chunks.append(
            {
                "chunk_index": index,
                "starts_at_main_tokens": cumulative_main_quotas(index)["main"],
                "ends_at_main_tokens": cumulative_main_quotas(index + 1)["main"],
                "new_quotas": chunk_quotas(index),
                "refresh_index": refresh_index_for_chunk(index),
                "kimi_refresh_required": refresh_required_for_chunk(index),
                "recovery_checkpoint": is_recovery_marker(index + 1),
            }
        )
    qwen = sum(item["new_quotas"]["qwen"] for item in chunks)
    raw = sum(item["new_quotas"]["raw_lm"] for item in chunks)
    checks = {
        "source_256k": SOURCE_MAIN_TOKENS == 262_144,
        "target_1m": TARGET_MAIN_TOKENS == 1_048_576,
        "new_main_exact": sum(item["new_quotas"]["main"] for item in chunks) == NEW_MAIN_TOKENS,
        "new_qwen_exact": qwen == NEW_QWEN_TOKENS,
        "new_raw_exact": raw == NEW_RAW_LM_TOKENS,
        "every_chunk_65536": all(item["new_quotas"]["main"] == TOKENS_PER_CHUNK for item in chunks),
        "every_chunk_streams_sum": all(
            item["new_quotas"]["qwen"] + item["new_quotas"]["raw_lm"] == TOKENS_PER_CHUNK
            for item in chunks
        ),
        "optimizer_steps_exact": SOURCE_OPTIMIZER_STEPS + NEW_OPTIMIZER_STEPS == TARGET_OPTIMIZER_STEPS,
        "two_refreshes": sum(item["kimi_refresh_required"] for item in chunks) == 2,
        "final_marker_exact": RECOVERY_MAIN_TOKEN_MARKERS[-1] == TARGET_MAIN_TOKENS,
        "automatic_2m_false": True,
    }
    return {
        "contract": P11_6_FAST_CONTRACT,
        "source_main_tokens": SOURCE_MAIN_TOKENS,
        "target_main_tokens": TARGET_MAIN_TOKENS,
        "new_main_tokens": NEW_MAIN_TOKENS,
        "new_qwen_tokens": NEW_QWEN_TOKENS,
        "new_raw_lm_tokens": NEW_RAW_LM_TOKENS,
        "chunks": chunks,
        "recovery_main_token_markers": list(RECOVERY_MAIN_TOKEN_MARKERS),
        "kimi_refresh_start_main_tokens": list(KIMI_REFRESH_START_MAIN_TOKENS),
        "p11_cumulative_api_hard_cap_cny": P11_CUMULATIVE_API_HARD_CAP_CNY,
        "daily_api_hard_cap_cny": DAILY_API_HARD_CAP_CNY,
        "checks": checks,
        "passed": all(checks.values()),
    }


__all__ = [
    "CHUNK_COUNT",
    "DAILY_API_HARD_CAP_CNY",
    "FROZEN_KDA_AUDIT_MAIN_TOKENS",
    "FULL_DEV_CE_MAX",
    "FAST_GPU_MEMORY_GATE_AMENDMENT",
    "FAST_GPU_SAFETY_MARGIN_MIB",
    "GRADIENT_AUDIT_STEPS",
    "KIMI_ATTEMPTS_PER_DOMAIN",
    "KIMI_MAX_ATTEMPTS_PER_REFRESH",
    "KIMI_MAX_NEW_ATTEMPTS",
    "KIMI_REFRESH_START_MAIN_TOKENS",
    "NEW_MAIN_TOKENS",
    "NEW_OPTIMIZER_STEPS",
    "NEW_QWEN_TOKENS",
    "NEW_RAW_LM_TOKENS",
    "P10_FULL_DEV_CE",
    "P11_5A_GENERAL_COMPOSITE",
    "P11_6_FAST_CONTRACT",
    "P11_CUMULATIVE_API_HARD_CAP_CNY",
    "RECOVERY_MAIN_TOKEN_MARKERS",
    "SOURCE_MAIN_TOKENS",
    "SOURCE_OPTIMIZER_STEPS",
    "TARGET_MAIN_TOKENS",
    "TARGET_OPTIMIZER_STEPS",
    "TOKENS_PER_CHUNK",
    "TOKENS_PER_UPDATE_BUCKET",
    "UPDATE_BUCKETS_PER_CHUNK",
    "chunk_quotas",
    "continuation_cosine_learning_rate",
    "contract_manifest",
    "cumulative_main_quotas",
    "cumulative_new_quotas",
    "fast_api_call_allowed",
    "final_decision",
    "is_recovery_marker",
    "kimi_refresh_early_stop",
    "refresh_index_for_chunk",
    "refresh_required_for_chunk",
]
