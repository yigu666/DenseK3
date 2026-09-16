"""Run the bounded P11.4b Kimi cost calibration on high-information trajectories."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
TITAN = ROOT / "titan"
SCRIPTS = TITAN / "scripts"
sys.path[:0] = [str(TITAN), str(SCRIPTS)]

from p7_latent_cache import TitanP7HybridCache  # noqa: E402
from p11 import (  # noqa: E402
    CANDIDATES_PER_DOMAIN,
    P11_COST_AMENDMENT,
    REVIEW_COUNT,
    REVIEWS_PER_DOMAIN,
    STAGE_API_BUDGET_CNY,
    KimiAPIConfig,
    KimiAPICostLedger,
    KimiAPITeacher,
    deterministic_candidate_pool,
    select_high_information,
    summarize_reviews,
)
from p11.runtime import (  # noqa: E402
    QWEN,
    TOKENIZER,
    force_qwen_torch_gdn_backend,
    load_p10_t_student,
    qwen_token_logprobs,
    token_hash,
)
from run_p10 import gpu_start_state  # noqa: E402

REPORT_DIR = TITAN / "manifests/reproduction/p11-t"
SELECTION_REPORT = REPORT_DIR / "p11-4b-high-information-selection.json"
PROGRESS_REPORT = REPORT_DIR / "p11-4b-kimi-cost-calibration-progress.json"
FINAL_REPORT = REPORT_DIR / "p11-4b-kimi-cost-calibration.json"
PROMPTS = ROOT / "data/p11-general/prompts/general-train-prompts-v1.jsonl"
CACHE_DIR = ROOT / "data/p11-general/teacher-cache/kimi-cost-calibration"
API_CONFIG = TITAN / "configs/p11-kimi-api.json"
P11_4_REPORT = REPORT_DIR / "p11-4-dual-teacher-joint-preflight.json"
ROLLOUT_TOKENS = 16
MAX_PROMPT_TOKENS = 256
MAX_COMPLETION_TOKENS = 768
RATE_LIMIT_MIN_INTERVAL_SECONDS = 21.0
SEED = 20260821


def now() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def release(*values: object) -> None:
    del values
    gc.collect()
    torch.cuda.empty_cache()


def load_tokenizer() -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=False)


def load_prompts() -> list[dict[str, Any]]:
    return [json.loads(line) for line in PROMPTS.read_text(encoding="utf-8").splitlines() if line.strip()]


def stable_seed(record_id: str) -> int:
    digest = hashlib.sha256(f"{SEED}:{record_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


@torch.inference_mode()
def student_rollout_record(
    model: torch.nn.Module,
    tokenizer: Any,
    record: dict[str, Any],
) -> dict[str, Any]:
    prompt_ids = tokenizer(
        record["prompt"],
        return_tensors="pt",
        truncation=True,
        max_length=MAX_PROMPT_TOKENS,
    ).input_ids.cuda()
    prompt_length = int(prompt_ids.shape[1])
    cache = TitanP7HybridCache(model.config)
    output = model(
        prompt_ids,
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )
    generator = torch.Generator(device="cuda").manual_seed(stable_seed(record["id"]))
    generated: list[int] = []
    old_logprobs: list[float] = []
    entropies: list[float] = []
    for _ in range(ROLLOUT_TOKENS):
        logits = output.logits[:, -1].float()
        log_probabilities = torch.log_softmax(logits, dim=-1)
        probabilities = torch.exp(log_probabilities)
        entropies.append(float(-(probabilities * log_probabilities).sum()))
        token = torch.multinomial(probabilities, 1, generator=generator)
        generated.append(int(token.item()))
        old_logprobs.append(float(log_probabilities.gather(-1, token).item()))
        if generated[-1] == tokenizer.eos_token_id:
            break
        output = model(
            token,
            past_key_values=output.past_key_values,
            use_cache=True,
            return_dict=True,
        )
    if not generated:
        raise ValueError("P11.4b student rollout produced no tokens")
    generated_tensor = torch.tensor([generated], dtype=torch.long)
    full_tokens = torch.cat((prompt_ids.cpu(), generated_tensor), dim=1)
    repetition = 1.0 - len(set(generated)) / len(generated)
    return {
        "id": record["id"],
        "domain": record["domain"],
        "language": record["language"],
        "prompt_sha256": sha256_text(record["prompt"]),
        "prompt_tokens": prompt_length,
        "rollout_tokens": len(generated),
        "rollout_token_ids": generated,
        "full_token_ids": full_tokens[0].tolist(),
        "rollout_token_hash": token_hash(generated_tensor),
        "student_text": tokenizer.decode(generated, skip_special_tokens=True),
        "student_text_sha256": sha256_text(tokenizer.decode(generated, skip_special_tokens=True)),
        "old_student_logprobs": old_logprobs,
        "student_mean_token_entropy": sum(entropies) / len(entropies),
        "student_mean_negative_logprob": -sum(old_logprobs) / len(old_logprobs),
        "repetition_fraction": repetition,
    }


def build_selection() -> dict[str, Any]:
    prerequisite = json.loads(P11_4_REPORT.read_text(encoding="utf-8"))
    if prerequisite.get("status") != "PASS":
        raise ValueError("P11.4b requires frozen PASS P11.4")
    state = gpu_start_state()
    if not state["idle"]:
        return {
            "stage": "P11.4b-HIGH-INFORMATION-SELECTION",
            "status": "DEFERRED_GPU_NOT_IDLE",
            "completed_at": now(),
            "gpu": state,
            "api_calls": 0,
            "api_cost_cny": 0.0,
        }
    prompts = load_prompts()
    pool = deterministic_candidate_pool(prompts)
    prompt_by_id = {item["id"]: item for item in prompts}
    tokenizer = load_tokenizer()
    student, attnres, student_load = load_p10_t_student()
    student.eval()
    attnres.eval()
    rollouts = []
    for index, record in enumerate(pool, start=1):
        rollouts.append(student_rollout_record(student, tokenizer, record))
        if index % 4 == 0:
            print(f"P11_4B_STUDENT_ROLLOUT_PROGRESS={index}/{len(pool)}", flush=True)
    del student, attnres
    release()

    from transformers import AutoModelForCausalLM

    qwen = AutoModelForCausalLM.from_pretrained(
        QWEN,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    qwen_backend = force_qwen_torch_gdn_backend(qwen)
    qwen = qwen.cuda()
    qwen.eval()
    scored: list[dict[str, Any]] = []
    for index, rollout in enumerate(rollouts, start=1):
        tokens = torch.tensor([rollout["full_token_ids"]], dtype=torch.long, device="cuda")
        qwen_values = qwen_token_logprobs(qwen, tokens, target_start=int(rollout["prompt_tokens"])).detach().float().cpu()[0]
        old_values = torch.tensor(rollout["old_student_logprobs"], dtype=torch.float32)
        if qwen_values.shape != old_values.shape:
            raise ValueError("P11.4b Qwen/student scoring shape mismatch")
        advantage = qwen_values - old_values
        mean_abs_gap = float(advantage.abs().mean())
        mean_advantage = float(advantage.mean())
        rollout.update(
            {
                "qwen_teacher_logprobs": qwen_values.tolist(),
                "qwen_student_advantage_mean": mean_advantage,
                "qwen_student_abs_logprob_gap_mean": mean_abs_gap,
                "disagreement_score": mean_abs_gap + max(0.0, mean_advantage),
            }
        )
        scored.append(rollout)
        if index % 4 == 0:
            print(f"P11_4B_QWEN_SCORING_PROGRESS={index}/{len(rollouts)}", flush=True)
    del qwen
    release()
    selected = select_high_information(scored)
    selected_ids = {item["id"] for item in selected}
    selected_records = []
    for item in selected:
        prompt = prompt_by_id[item["id"]]
        selected_records.append(
            {
                **item,
                "prompt": prompt["prompt"],
                "source": prompt["source"],
                "source_record_id": prompt["source_record_id"],
            }
        )
    checks = {
        "p11_4_pass": prerequisite["status"] == "PASS",
        "candidate_pool_48": len(pool) == CANDIDATES_PER_DOMAIN * 4,
        "scored_pool_48": len(scored) == CANDIDATES_PER_DOMAIN * 4,
        "selected_24": len(selected_records) == REVIEW_COUNT,
        "selected_ids_unique": len(selected_ids) == REVIEW_COUNT,
        "balanced_6_per_domain": all(
            sum(item["domain"] == domain for item in selected_records) == REVIEWS_PER_DOMAIN
            for domain in {item["domain"] for item in selected_records}
        )
        and len({item["domain"] for item in selected_records}) == 4,
        "all_scores_finite": all(math.isfinite(float(item["disagreement_score"])) for item in scored),
        "api_not_called": True,
        "heldout_not_accessed": True,
    }
    report = {
        "stage": "P11.4b-HIGH-INFORMATION-SELECTION",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "completed_at": now(),
        "amendment": P11_COST_AMENDMENT,
        "selection_method": ("TOP_PER_DOMAIN_BY_MEAN_ABS_QWEN_STUDENT_LOGPROB_GAP_PLUS_POSITIVE_MEAN_ADVANTAGE"),
        "candidate_pool_per_domain": CANDIDATES_PER_DOMAIN,
        "reviews_per_domain": REVIEWS_PER_DOMAIN,
        "rollout_tokens_max": ROLLOUT_TOKENS,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "student_load": student_load,
        "qwen_backend": qwen_backend,
        "scored_candidates": scored,
        "selected": selected_records,
        "checks": checks,
        "api_calls": 0,
        "api_cost_cny": 0.0,
        "heldout_accessed": False,
        "result_marker": (
            "P11_4B_HIGH_INFORMATION_SELECTION=PASS" if all(checks.values()) else "P11_4B_HIGH_INFORMATION_SELECTION=FAIL"
        ),
    }
    atomic_json(SELECTION_REPORT, report)
    return report


def request_upper_bound(
    config: KimiAPIConfig,
    messages: list[dict[str, str]],
    response_format: dict[str, Any],
) -> float:
    payload = {
        "model": config.model,
        "messages": messages,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "reasoning_effort": config.reasoning_effort,
        "stream": False,
        "response_format": response_format,
    }
    input_upper = len(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode())
    return (input_upper * config.input_cny_per_million + MAX_COMPLETION_TOKENS * config.output_cny_per_million) / 1_000_000


def cached_review(path: Path, selected: dict[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("selected_id") != selected["id"]:
        raise ValueError("P11.4b cache selected identity mismatch")
    if payload.get("student_text_sha256") != selected["student_text_sha256"]:
        raise ValueError("P11.4b cache student identity mismatch")
    return payload


def run_reviews(selection: dict[str, Any]) -> dict[str, Any]:
    config = KimiAPIConfig.from_json(API_CONFIG)
    if config.max_retries != 0 or config.reasoning_effort != "low":
        raise ValueError("P11.4b requires low reasoning and zero automatic retries")
    teacher = KimiAPITeacher(config, ledger=KimiAPICostLedger(REPORT_DIR, config))
    tokenizer = load_tokenizer()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    reviews: list[dict[str, Any]] = []
    api_calls_this_run = 0
    cache_hits = 0
    previous_progress = json.loads(PROGRESS_REPORT.read_text(encoding="utf-8")) if PROGRESS_REPORT.is_file() else {}
    uncertain_reserved_cost_cny = float(previous_progress.get("uncertain_reserved_cost_cny", 0.0))
    if previous_progress.get("status") == "REQUEST_PENDING":
        uncertain_reserved_cost_cny += float(previous_progress.get("pending_upper_bound_cny", 0.0))
    last_request_started: float | None = None
    for index, selected in enumerate(selection["selected"], start=1):
        messages, response_format = teacher.structured_trajectory_contract(selected["prompt"], selected["student_text"])
        fingerprint = sha256_text(
            json.dumps(
                {
                    "amendment": P11_COST_AMENDMENT,
                    "selected_id": selected["id"],
                    "student_text_sha256": selected["student_text_sha256"],
                    "model": config.model,
                    "reasoning_effort": config.reasoning_effort,
                    "response_format": response_format,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        cache_path = CACHE_DIR / f"{fingerprint}.json"
        cached = cached_review(cache_path, selected)
        if cached is not None:
            review = cached
            cache_hits += 1
        else:
            upper = request_upper_bound(config, messages, response_format)
            stage_spent = sum(float(item["cost_cny"]) for item in reviews)
            if stage_spent + uncertain_reserved_cost_cny + upper > STAGE_API_BUDGET_CNY:
                raise RuntimeError("P11.4b stage API hard cap would be exceeded by the next request")
            if last_request_started is not None:
                remaining = RATE_LIMIT_MIN_INTERVAL_SECONDS - (time.monotonic() - last_request_started)
                if remaining > 0:
                    print(
                        f"P11_4B_RATE_LIMIT_PACING_SECONDS={remaining:.3f}",
                        flush=True,
                    )
                    time.sleep(remaining)
            atomic_json(
                PROGRESS_REPORT,
                {
                    "stage": "P11.4b-KIMI-COST-CALIBRATION",
                    "status": "REQUEST_PENDING",
                    "completed_reviews": len(reviews),
                    "pending_selected_id": selected["id"],
                    "pending_upper_bound_cny": upper,
                    "uncertain_reserved_cost_cny": uncertain_reserved_cost_cny,
                    "accounted_stage_cost_cny": stage_spent + uncertain_reserved_cost_cny + upper,
                    "stage_api_budget_cap_cny": STAGE_API_BUDGET_CNY,
                    "reasoning_content_persisted": False,
                    "api_secret_persisted": False,
                },
            )
            last_request_started = time.monotonic()
            trajectory, result = teacher.structured_trajectory(
                selected["prompt"],
                selected["student_text"],
                max_completion_tokens=MAX_COMPLETION_TOKENS,
            )
            decision = "keep" if trajectory.mode == "KEEP" else "correct"
            corrected_text = trajectory.corrected_text or ""
            corrected_ids = tokenizer(
                corrected_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids
            corrected_tokens = int(corrected_ids.numel()) if corrected_text else 0
            review = {
                "schema_version": 1,
                "stage": "P11.4b-KIMI-COST-CALIBRATION",
                "status": "SUCCEEDED",
                "amendment": P11_COST_AMENDMENT,
                "selected_id": selected["id"],
                "domain": selected["domain"],
                "prompt_sha256": selected["prompt_sha256"],
                "student_text": selected["student_text"],
                "student_text_sha256": selected["student_text_sha256"],
                "disagreement_score": selected["disagreement_score"],
                "decision": decision,
                "corrected_answer": corrected_text,
                "corrected_target_tokens": corrected_tokens,
                "teacher_response_sha256": sha256_text(result.text),
                "structured_output_valid": True,
                "teacher_model": result.model,
                "reasoning_effort": config.reasoning_effort,
                "reasoning_content_present": result.reasoning_content_present,
                "reasoning_content_characters": result.reasoning_content_characters,
                "reasoning_content_persisted": False,
                "reasoning_content_trained": False,
                "usage": result.usage,
                "finish_reason": result.finish_reason,
                "cost_cny": result.cost_cny,
                "request_upper_bound_cny": upper,
                "cache_path": str(cache_path.relative_to(ROOT)),
                "completed_at": now(),
            }
            atomic_json(cache_path, review)
            api_calls_this_run += 1
        reviews.append(review)
        atomic_json(
            PROGRESS_REPORT,
            {
                "stage": "P11.4b-KIMI-COST-CALIBRATION",
                "status": "RUNNING",
                "completed_reviews": len(reviews),
                "api_calls_this_run": api_calls_this_run,
                "cache_hits": cache_hits,
                "stage_cost_cny": sum(float(item["cost_cny"]) for item in reviews),
                "uncertain_reserved_cost_cny": uncertain_reserved_cost_cny,
                "accounted_stage_cost_cny": sum(float(item["cost_cny"]) for item in reviews) + uncertain_reserved_cost_cny,
                "stage_api_budget_cap_cny": STAGE_API_BUDGET_CNY,
                "reasoning_content_persisted": False,
                "api_secret_persisted": False,
            },
        )
        print(
            f"P11_4B_KIMI_REVIEW_PROGRESS={index}/{REVIEW_COUNT} "
            f"DECISION={review['decision'].upper()} COST_CNY={float(review['cost_cny']):.6f}",
            flush=True,
        )
    summary = summarize_reviews(reviews)
    summary["uncertain_reserved_cost_cny"] = uncertain_reserved_cost_cny
    summary["accounted_stage_cost_cny"] = float(summary["total_cost_cny"]) + uncertain_reserved_cost_cny
    summary["checks"]["accounted_stage_cost_within_5_cny"] = summary["accounted_stage_cost_cny"] <= STAGE_API_BUDGET_CNY
    summary["passed"] = all(summary["checks"].values())
    passed = summary["passed"]
    report = {
        "stage": "P11.4b-KIMI-COST-CALIBRATION",
        "status": "PASS" if passed else "FAIL",
        "completed_at": now(),
        "amendment": P11_COST_AMENDMENT,
        "general_objective_unchanged": True,
        "teachers_unchanged": True,
        "trainable_scope_unchanged": True,
        "architecture_unchanged": True,
        "kimi_signal_unit": "REVIEWED_ON_POLICY_TRAJECTORIES",
        "kimi_fixed_supervised_token_share_disabled": True,
        "kimi_selection": "HIGH_INFORMATION_STUDENT_TRAJECTORIES",
        "kimi_reasoning_effort": "low",
        "kimi_auto_retry": False,
        "request_rate_limit_rpm": 3,
        "minimum_request_interval_seconds": RATE_LIMIT_MIN_INTERVAL_SECONDS,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "manual_resume_after_rate_limit": uncertain_reserved_cost_cny > 0,
        "kimi_reasoning_content_trained": False,
        "kimi_reasoning_content_persisted": False,
        "structured_output": "JSON_SCHEMA",
        "selection_report": str(SELECTION_REPORT.relative_to(ROOT)),
        "selection_report_sha256": sha256_bytes(SELECTION_REPORT.read_bytes()),
        "review_cache_paths": [item["cache_path"] for item in reviews],
        "summary": summary,
        "api_calls_this_run": api_calls_this_run,
        "cache_hits": cache_hits,
        "stage_api_budget_cap_cny": STAGE_API_BUDGET_CNY,
        "daily_api_hard_cap_cny": config.daily_budget_cny,
        "training_performed": False,
        "optimizer_step_performed": False,
        "training_tokens": 0,
        "heldout_accessed": False,
        "p11_5_total_training_tokens": 262144,
        "p11_5_main_token_mix": {"qwen": 0.60, "raw_lm": 0.40},
        "p11_5_kimi_mode": "SPARSE_CORRECTIVE_SIDE_STREAM",
        "p11_5_probe_allowed": "AWAITING_COST_CALIBRATION_REVIEW" if passed else False,
        "p12_titan_migration_allowed": False,
        "result_marker": ("P11_4B_COST_CALIBRATION=PASS" if passed else "P11_4B_COST_CALIBRATION=FAIL"),
    }
    atomic_json(FINAL_REPORT, report)
    return report


def main() -> int:
    envelope = {
        "stage": "P11.4b-KIMI-COST-CALIBRATION",
        "status": "RUNNING",
        "started_at": now(),
        "amendment": P11_COST_AMENDMENT,
        "stage_api_budget_cap_cny": STAGE_API_BUDGET_CNY,
        "daily_api_hard_cap_cny": 20.0,
        "training_performed": False,
        "heldout_accessed": False,
        "p12_titan_migration_allowed": False,
    }
    atomic_json(FINAL_REPORT, envelope)
    try:
        if SELECTION_REPORT.is_file():
            selection = json.loads(SELECTION_REPORT.read_text(encoding="utf-8"))
            if selection.get("status") != "PASS":
                raise ValueError("Existing P11.4b selection is not PASS")
            print("P11_4B_HIGH_INFORMATION_SELECTION=RESUMED", flush=True)
        else:
            selection = build_selection()
            print(selection.get("result_marker", selection["status"]), flush=True)
            if selection["status"] != "PASS":
                atomic_json(FINAL_REPORT, {**envelope, **selection})
                return 2
        report = run_reviews(selection)
    except Exception as error:
        report = {
            **envelope,
            "status": "ERROR",
            "completed_at": now(),
            "exception_type": type(error).__name__,
            "exception": str(error),
            "training_performed": False,
            "optimizer_step_performed": False,
            "training_tokens": 0,
            "heldout_accessed": False,
            "p11_5_probe_allowed": False,
            "p12_titan_migration_allowed": False,
            "result_marker": "P11_4B_COST_CALIBRATION=FAIL",
        }
        atomic_json(FINAL_REPORT, report)
    print(report["result_marker"], flush=True)
    print(f"P11_5_PROBE_ALLOWED={report['p11_5_probe_allowed']}", flush=True)
    print("P12_TITAN_MIGRATION_ALLOWED=false", flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
