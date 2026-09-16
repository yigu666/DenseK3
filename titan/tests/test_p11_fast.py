from __future__ import annotations

import math

import pytest
from p11.fast import (
    CHUNK_COUNT,
    FULL_DEV_CE_MAX,
    NEW_QWEN_TOKENS,
    NEW_RAW_LM_TOKENS,
    SOURCE_MAIN_TOKENS,
    TARGET_MAIN_TOKENS,
    chunk_quotas,
    continuation_cosine_learning_rate,
    contract_manifest,
    cumulative_main_quotas,
    fast_api_call_allowed,
    final_decision,
    kimi_refresh_early_stop,
    refresh_required_for_chunk,
)


def test_fast_token_contract_is_exact() -> None:
    manifest = contract_manifest()
    assert manifest["passed"]
    assert cumulative_main_quotas(0)["main"] == SOURCE_MAIN_TOKENS
    assert cumulative_main_quotas(CHUNK_COUNT)["main"] == TARGET_MAIN_TOKENS
    assert sum(chunk_quotas(i)["qwen"] for i in range(CHUNK_COUNT)) == NEW_QWEN_TOKENS
    assert sum(chunk_quotas(i)["raw_lm"] for i in range(CHUNK_COUNT)) == NEW_RAW_LM_TOKENS
    assert all(sum(chunk_quotas(i)[name] for name in ("qwen", "raw_lm")) == 65_536 for i in range(CHUNK_COUNT))
    assert [i for i in range(CHUNK_COUNT) if refresh_required_for_chunk(i)] == [0, 6]


def test_continuation_lr_has_no_boundary_jump_and_reaches_floor() -> None:
    saved = 1.006599637669943e-6
    floor = 1.0e-6
    values = [
        continuation_cosine_learning_rate(
            step,
            extension_steps=192,
            saved_learning_rate=saved,
            terminal_learning_rate=floor,
        )
        for step in range(192)
    ]
    assert values[0] == saved
    assert values[-1] == floor
    assert all(left >= right for left, right in zip(values, values[1:], strict=False))


def test_continuation_lr_rejects_upward_terminal() -> None:
    with pytest.raises(ValueError):
        continuation_cosine_learning_rate(
            0,
            extension_steps=192,
            saved_learning_rate=1e-6,
            terminal_learning_rate=2e-6,
        )


def test_api_gate_uses_cumulative_15_cny_cap() -> None:
    assert fast_api_call_allowed(
        cumulative_cost_cny=2.67,
        next_upper_bound_cny=0.1,
        refresh_attempts=0,
        new_attempts=0,
    )["allowed"]
    blocked = fast_api_call_allowed(
        cumulative_cost_cny=14.99,
        next_upper_bound_cny=0.02,
        refresh_attempts=0,
        new_attempts=0,
    )
    assert not blocked["allowed"]
    assert blocked["stop_reason"] == "P11_CUMULATIVE_API_HARD_CAP"


def test_kimi_early_stop_only_at_rolling_16_boundaries() -> None:
    keeps = [{"status": "SUCCEEDED", "decision": "keep"}] * 15
    assert not kimi_refresh_early_stop(keeps)["stop"]
    all_keep = kimi_refresh_early_stop(
        keeps + [{"status": "SUCCEEDED", "decision": "keep"}]
    )
    assert all_keep["stop"]
    assert all_keep["trivial_correction_fraction"] == 1.0
    decisions = [
        {
            "status": "SUCCEEDED",
            "decision": "correct" if index == 0 else "keep",
            "trivial_correction": index == 0,
        }
        for index in range(16)
    ]
    stopped = kimi_refresh_early_stop(decisions)
    assert stopped["stop"]
    assert set(stopped["reasons"]) == {
        "CORRECTION_RATE_LT_15_PERCENT",
        "CORRECTIONS_MOSTLY_TRIVIAL",
        "KEEP_STRONGLY_DOMINATES",
    }


def test_final_decision_strong_safe_and_no_go() -> None:
    domains = {"a": 0.5, "b": 0.75}
    common = dict(
        baseline_general_composite=0.625,
        baseline_domains=domains,
        full_dev_ce=FULL_DEV_CE_MAX,
        qwen_teacher_nonregressed=True,
        kimi_teacher_nonregressed=True,
        teacher_improved=True,
        runtime_passed=True,
    )
    strong = final_decision(
        final_general_composite=0.75,
        final_domains={"a": 0.75, "b": 0.75},
        **common,
    )
    assert strong["decision"] == "STRONG_GO"
    safe = final_decision(final_general_composite=0.625, final_domains=domains, **common)
    assert safe["decision"] == "SAFE_BUT_SATURATED"
    no_go = final_decision(
        final_general_composite=0.625,
        final_domains=domains,
        **{**common, "full_dev_ce": math.nextafter(FULL_DEV_CE_MAX, math.inf)},
    )
    assert no_go["decision"] == "NO_GO"
    assert no_go["p12_long_context_allowed"]
