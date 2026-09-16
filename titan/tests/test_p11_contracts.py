from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

# ruff: noqa: E402 -- Titan overlay import is deliberate.
TITAN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TITAN))

from p11 import (
    MAIN_TOKEN_SHARES,
    KimiAPIConfig,
    KimiAPICostLedger,
    KimiAPITeacher,
    SupervisedTokenRouter,
    build_p11_asset_preflight,
    deterministic_candidate_pool,
    exact_token_advantage,
    exact_token_opd_loss,
    parse_kimi_structured_trajectory,
    parse_kimi_trajectory,
    select_high_information,
    stable_id,
    stable_select,
    summarize_reviews,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_token_opd_clips_and_backpropagates() -> None:
    teacher = torch.tensor([[10.0, -9.0, 0.5]])
    old = torch.tensor([[0.0, 0.0, 0.25]])
    advantage = exact_token_advantage(teacher, old)
    assert torch.equal(advantage, torch.tensor([[5.0, -5.0, 0.25]]))
    current = torch.tensor([[-1.0, -2.0, -3.0]], requires_grad=True)
    loss = exact_token_opd_loss(current, advantage, torch.tensor([[1, 1, 0]]))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.equal(current.grad, torch.tensor([[-2.5, 2.5, 0.0]]))


def test_cross_tokenizer_response_contract_rejects_token_leakage() -> None:
    assert parse_kimi_trajectory("KEEP").mode == "KEEP"
    corrected = parse_kimi_trajectory("CORRECTED_RESPONSE\nA clean semantic answer.")
    assert corrected.corrected_text == "A clean semantic answer."
    with pytest.raises(ValueError):
        parse_kimi_trajectory("CORRECTED_RESPONSE\n<|assistant|> leaked")
    with pytest.raises(ValueError):
        parse_kimi_trajectory("free-form answer")

    assert parse_kimi_structured_trajectory('{"decision":"keep","answer":""}').mode == "KEEP"
    structured = parse_kimi_structured_trajectory('{"decision":"correct","answer":"A clean answer."}')
    assert structured.corrected_text == "A clean answer."
    with pytest.raises(ValueError):
        parse_kimi_structured_trajectory('{"decision":"keep","answer":"not empty"}')


def test_router_accounts_supervised_tokens_not_prompts() -> None:
    router = SupervisedTokenRouter()
    assert router.choose() == "qwen"
    router.record("qwen", 600)
    router.record("raw_lm", 400)
    assert router.report()["target"] == MAIN_TOKEN_SHARES
    assert router.report()["actual"] == {"qwen": 0.6, "raw_lm": 0.4}


def test_cost_calibration_selection_and_summary_are_balanced() -> None:
    domains = (
        "GENERAL_KNOWLEDGE",
        "GENERAL_REASONING",
        "MULTILINGUAL_LANGUAGE",
        "INSTRUCTION_SYNTHESIS",
    )
    rows = [{"id": f"{domain}-{index}", "domain": domain} for domain in domains for index in range(20)]
    pool = deterministic_candidate_pool(rows)
    assert len(pool) == 48
    scored = [{**item, "disagreement_score": float(index)} for index, item in enumerate(pool)]
    selected = select_high_information(scored)
    assert len(selected) == 24
    reviews = [
        {
            "status": "SUCCEEDED",
            "domain": item["domain"],
            "decision": "correct" if index % 2 else "keep",
            "corrected_target_tokens": 4 if index % 2 else 0,
            "cost_cny": 0.01,
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "reasoning_tokens": 3,
            },
            "reasoning_content_persisted": False,
            "reasoning_content_trained": False,
            "structured_output_valid": True,
        }
        for index, item in enumerate(selected)
    ]
    summary = summarize_reviews(reviews)
    assert summary["passed"]
    assert summary["correction_count"] == 12
    assert summary["corrected_target_tokens"] == 48


def test_kimi_structured_contract_is_short_and_strict() -> None:
    messages, response_format = KimiAPITeacher.structured_trajectory_contract("Question?", "Student answer.")
    assert len(messages) == 2
    assert response_format["type"] == "json_schema"
    schema = response_format["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["decision", "answer"]


def test_asset_preflight_blocks_missing_real_kimi_and_general_data(tmp_path: Path) -> None:
    p10_final = {
        "status": "PASS",
        "p11_titan_migration_allowed": True,
        "canonical_p10_status": "PENDING_NATIVE_BF16_FULL_SCOPE_RETRAIN",
        "p10_titan_is_canonical": False,
    }
    _write_json(
        tmp_path / "titan/manifests/reproduction/p10-t/p10-t-final-check.json",
        p10_final,
    )
    candidate = tmp_path / "outputs/densek3-core/checkpoints/p10-t/densek3-4b-core-k3-joint-reduced-titan"
    candidate.mkdir(parents=True)
    runtime = candidate / "p10-t-runtime-config.json"
    runtime.write_text("{}", encoding="utf-8")
    _write_json(
        candidate / "p10-t-candidate-manifest.json",
        {
            "status": "PASS",
            "decision": "PASS",
            "heldout_accessed": False,
            "lora_present_at_runtime": False,
            "files": {runtime.name: _sha256(runtime)},
        },
    )
    qwen = tmp_path / "models/Qwen3.5-4B-Base"
    qwen.mkdir(parents=True)
    (qwen / "tokenizer.json").write_text("{}", encoding="utf-8")
    (qwen / "model-1.safetensors").write_bytes(b"qwen")
    _write_json(
        qwen / "model.safetensors.index.json",
        {"weight_map": {"x": "model-1.safetensors"}},
    )
    _write_json(
        tmp_path / "references/Kimi-K3-HF-reference/model.safetensors.index.json",
        {"metadata": {"total_size": 100}, "weight_map": {"x": "model-1.safetensors"}},
    )
    _write_json(
        tmp_path / "titan/configs/p11-kimi-api.json",
        {
            "api_key_env": "P11_TEST_MOONSHOT_KEY",
            "base_url": "https://api.moonshot.cn/v1/chat/completions",
            "cached_input_cny_per_million": 2.0,
            "daily_budget_cny": 20.0,
            "input_cny_per_million": 20.0,
            "logprobs": False,
            "model": "kimi-k3",
            "output_cny_per_million": 100.0,
            "per_call_upper_bound_cny": 0.25,
            "reasoning_effort": "low",
            "teacher_signal": "TEXT_SPACE_TRAJECTORY_ONLY",
            "teacher_loading": "API_ONLY_NO_LOCAL_WEIGHTS",
            "top_logprobs": 0,
        },
    )
    report = build_p11_asset_preflight(tmp_path)
    assert report["status"] == "BLOCKED_ASSET_CONTRACT"
    assert report["qwen_teacher"]["eligible"]
    assert "KIMI_K3_TEACHER_ACCESS" in report["blockers"]
    assert "P11_GENERAL_CORPUS_AND_BENCHMARK" in report["blockers"]
    assert not report["training_allowed"]


def test_kimi_api_descriptor_and_logprob_extraction(tmp_path: Path) -> None:
    descriptor = tmp_path / "kimi.json"
    _write_json(
        descriptor,
        {
            "api_key_env": "P11_TEST_MOONSHOT_KEY",
            "base_url": "https://api.moonshot.cn/v1/chat/completions",
            "cached_input_cny_per_million": 2.0,
            "daily_budget_cny": 20.0,
            "input_cny_per_million": 20.0,
            "logprobs": True,
            "model": "kimi-k3",
            "output_cny_per_million": 100.0,
            "per_call_upper_bound_cny": 0.25,
            "reasoning_effort": "low",
            "top_logprobs": 5,
        },
    )
    config = KimiAPIConfig.from_json(descriptor)
    assert config.model == "kimi-k3"
    values = KimiAPITeacher._extract_logprobs(
        {"message": {"logprobs": {"content": [{"token": "KEEP", "logprob": -0.01, "top_logprobs": []}]}}}
    )
    assert values[0]["token"] == "KEEP"
    with pytest.raises(ValueError):
        _write_json(descriptor, {"api_key": "must-not-be-stored"})
        KimiAPIConfig.from_json(descriptor)


def test_kimi_cost_ledger_enforces_and_persists_sanitized_cost(tmp_path: Path) -> None:
    descriptor = tmp_path / "kimi.json"
    _write_json(
        descriptor,
        {
            "api_key_env": "P11_TEST_MOONSHOT_KEY",
            "base_url": "https://api.moonshot.cn/v1/chat/completions",
            "cached_input_cny_per_million": 2.0,
            "daily_budget_cny": 20.0,
            "input_cny_per_million": 20.0,
            "logprobs": True,
            "max_retries": 0,
            "model": "kimi-k3",
            "output_cny_per_million": 100.0,
            "per_call_upper_bound_cny": 0.25,
            "reasoning_effort": "low",
            "top_logprobs": 5,
        },
    )
    config = KimiAPIConfig.from_json(descriptor)
    ledger = KimiAPICostLedger(tmp_path / "reports", config)
    reservation = ledger.reserve("fingerprint-only", 100, 10)
    cost = ledger.finalize(
        reservation,
        {"prompt_tokens": 100, "cached_tokens": 50, "completion_tokens": 10},
    )
    assert cost == pytest.approx(0.0021)
    ledger_text = next((tmp_path / "reports").glob("kimi-api-cost-*.json")).read_text(encoding="utf-8")
    assert "fingerprint-only" in ledger_text
    assert "prompt text" not in ledger_text
    assert "reasoning_content" not in ledger_text
    assert "api_key" not in ledger_text


def test_p11_general_selection_is_deterministic_disjoint_and_grouped() -> None:
    rows = [{"id": index, "group": f"g{index % 3}", "payload": f"v{index}"} for index in range(12)]
    first = stable_select(
        rows,
        3,
        namespace="unit",
        key=lambda row: row["id"],
        unique_group=lambda row: row["group"],
    )
    second = stable_select(
        reversed(rows),
        3,
        namespace="unit",
        key=lambda row: row["id"],
        unique_group=lambda row: row["group"],
    )
    assert [row["id"] for row in first] == [row["id"] for row in second]
    excluded = {row["_p11_source_id"] for row in first}
    remaining = stable_select(
        rows,
        3,
        namespace="unit",
        key=lambda row: row["id"],
        excluded_ids=excluded,
    )
    assert not excluded & {row["_p11_source_id"] for row in remaining}
    assert stable_id("unit", 1) == stable_id("unit", 1)
