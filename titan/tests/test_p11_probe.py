from __future__ import annotations

import json

import pytest
import torch
from p11.probe import (
    KIMI_MAX_REVIEW_ATTEMPTS,
    MAIN_TOKENS,
    P11_FP16_INITIAL_LOSS_SCALE,
    P11_FP16_LOSS_SCALE_GROWTH_INTERVAL,
    P11_GPU_HISTORICAL_PEAK_MEMORY_MIB,
    P11_GPU_MEMORY_GATE_AMENDMENT,
    P11_GPU_SAFETY_MARGIN_MIB,
    QWEN_TOKENS,
    RAW_LM_TOKENS,
    accounted_ledger_cost,
    api_call_allowed,
    assign_update_buckets,
    contract_manifest,
    cumulative_main_quotas,
    equal_trajectory_mean,
    gpu_memory_envelope,
    qwen_rollout_plan,
    raw_lm_target_lengths,
    trajectory_mean,
    wave_main_quotas,
)


def test_exact_probe_and_wave_token_quotas() -> None:
    assert cumulative_main_quotas(4) == {
        "main": MAIN_TOKENS,
        "qwen": QWEN_TOKENS,
        "raw_lm": RAW_LM_TOKENS,
    }
    waves = [wave_main_quotas(index) for index in range(4)]
    assert [item["main"] for item in waves] == [65_536] * 4
    assert [item["qwen"] for item in waves] == [39_322, 39_321, 39_322, 39_321]
    assert [item["raw_lm"] for item in waves] == [26_214, 26_215, 26_214, 26_215]


@pytest.mark.parametrize("wave", range(4))
def test_static_rollout_and_raw_plans_are_exact(wave: int) -> None:
    quota = wave_main_quotas(wave)
    qwen = qwen_rollout_plan(quota["qwen"])
    raw = raw_lm_target_lengths(quota["raw_lm"])
    assert sum(int(item["maximum_tokens"]) for item in qwen) == quota["qwen"]
    assert sum(item["role"] == "kimi_candidate" for item in qwen) == 48
    assert sum(raw) == quota["raw_lm"]
    assert max(raw) <= 2047


def test_bucket_assignment_is_complete_and_deterministic() -> None:
    lengths = [512] * 75 + [16] * 48 + [154]
    first = assign_update_buckets(lengths)
    second = assign_update_buckets(lengths)
    assert first == second
    assert sorted(index for bucket in first for index in bucket) == list(range(len(lengths)))


def test_kimi_loss_is_trajectory_mean_not_token_sum() -> None:
    short = trajectory_mean(torch.tensor([2.0, 4.0]))
    long = trajectory_mean(torch.tensor([3.0] * 100))
    combined = equal_trajectory_mean([short, long])
    assert float(short) == 3.0
    assert float(long) == 3.0
    assert float(combined) == 3.0


def test_api_budget_precedes_review_count() -> None:
    assert api_call_allowed(cumulative_cost_cny=4.0, next_upper_bound_cny=0.5, attempts=63)["allowed"]
    budget_stop = api_call_allowed(cumulative_cost_cny=4.8, next_upper_bound_cny=0.3, attempts=2)
    assert not budget_stop["allowed"]
    assert budget_stop["stop_reason"] == "P11_CUMULATIVE_API_HARD_CAP"
    attempt_stop = api_call_allowed(cumulative_cost_cny=1.0, next_upper_bound_cny=0.1, attempts=KIMI_MAX_REVIEW_ATTEMPTS)
    assert not attempt_stop["allowed"]
    assert attempt_stop["stop_reason"] == "KIMI_MAX_REVIEW_ATTEMPTS"


def test_cumulative_ledger_includes_success_and_uncertainty(tmp_path) -> None:
    (tmp_path / "kimi-api-cost-2026-08-21.json").write_text(
        json.dumps(
            {
                "entries": [
                    {"status": "SUCCEEDED", "cost_cny": 0.2},
                    {"status": "UNCERTAIN_CHARGE_RESERVED", "reserved_upper_bound_cny": 0.4},
                ]
            }
        ),
        encoding="utf-8",
    )
    report = accounted_ledger_cost(tmp_path)
    assert report["accounted_cost_cny"] == pytest.approx(0.6)
    assert report["entry_count"] == 2


def test_contract_manifest_is_self_consistent() -> None:
    report = contract_manifest()
    assert report["passed"]
    assert all(report["checks"].values())
    assert report["kimi_tokens_in_main_denominator"] is False


def test_titan_accumulation_loss_scale_is_conservative() -> None:
    assert P11_FP16_INITIAL_LOSS_SCALE == 1024.0
    assert P11_FP16_INITIAL_LOSS_SCALE < 65536.0
    assert P11_FP16_LOSS_SCALE_GROWTH_INTERVAL > 64


def test_gpu_memory_envelope_allows_safe_shared_gpu() -> None:
    report = gpu_memory_envelope(
        visible=True,
        temperature_c=72,
        memory_used_mib=7_724,
        memory_total_mib=24_576,
    )
    assert report["amendment_id"] == P11_GPU_MEMORY_GATE_AMENDMENT
    assert report["allowed"]
    assert report["requires_gpu_idle"] is False
    assert report["required_free_mib"] == (
        P11_GPU_HISTORICAL_PEAK_MEMORY_MIB + P11_GPU_SAFETY_MARGIN_MIB
    )


@pytest.mark.parametrize(
    ("temperature_c", "memory_used_mib", "allowed"),
    [(76, 0, False), (72, 9_500, False), (72, 9_462, True)],
)
def test_gpu_memory_envelope_rejects_unsafe_starts(
    temperature_c: int, memory_used_mib: int, allowed: bool
) -> None:
    report = gpu_memory_envelope(
        visible=True,
        temperature_c=temperature_c,
        memory_used_mib=memory_used_mib,
        memory_total_mib=24_576,
    )
    assert report["allowed"] is allowed
