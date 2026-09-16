"""Pure contracts for the P11.4b Kimi cost-calibration stage."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

DOMAINS = (
    "GENERAL_KNOWLEDGE",
    "GENERAL_REASONING",
    "MULTILINGUAL_LANGUAGE",
    "INSTRUCTION_SYNTHESIS",
)
REVIEW_COUNT = 24
REVIEWS_PER_DOMAIN = 6
CANDIDATES_PER_DOMAIN = 12
STAGE_API_BUDGET_CNY = 5.0


def _rank_key(record: dict[str, Any], namespace: str) -> str:
    return hashlib.sha256(f"P11_COST_AMENDMENT_2026_08_21\0{namespace}\0{record['id']}".encode()).hexdigest()


def deterministic_candidate_pool(
    records: Iterable[dict[str, Any]],
    *,
    per_domain: int = CANDIDATES_PER_DOMAIN,
) -> list[dict[str, Any]]:
    """Choose an equal, deterministic free-scoring pool across four domains."""
    if per_domain < REVIEWS_PER_DOMAIN:
        raise ValueError("Candidate pool must contain at least six records per domain")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        domain = str(record.get("domain"))
        if domain in DOMAINS:
            grouped[domain].append(record)
    selected: list[dict[str, Any]] = []
    for domain in DOMAINS:
        values = sorted(grouped[domain], key=lambda item: _rank_key(item, domain))
        if len(values) < per_domain:
            raise ValueError(f"P11 prompt pool is too small for {domain}")
        selected.extend(dict(item) for item in values[:per_domain])
    return selected


def select_high_information(
    scored: Iterable[dict[str, Any]],
    *,
    per_domain: int = REVIEWS_PER_DOMAIN,
) -> list[dict[str, Any]]:
    """Select the largest exact-token Qwen/student disagreement in every domain."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in scored:
        domain = str(record.get("domain"))
        if domain not in DOMAINS:
            raise ValueError(f"Unexpected P11 domain: {domain}")
        score = record.get("disagreement_score")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ValueError("P11 disagreement score must be finite")
        grouped[domain].append(record)
    selected: list[dict[str, Any]] = []
    for domain in DOMAINS:
        values = sorted(
            grouped[domain],
            key=lambda item: (-float(item["disagreement_score"]), str(item["id"])),
        )
        if len(values) < per_domain:
            raise ValueError(f"P11 scored pool is too small for {domain}")
        selected.extend(dict(item) for item in values[:per_domain])
    return selected


def nearest_rank(values: list[int], percentile: float) -> int | None:
    if not values:
        return None
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("Percentile must be between zero and one")
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def summarize_reviews(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize paid reviews without reading or retaining reasoning content."""
    decisions = Counter(str(item["decision"]) for item in records)
    domain_counts = Counter(str(item["domain"]) for item in records)
    corrections = [item for item in records if item["decision"] == "correct"]
    corrected_tokens = [int(item["corrected_target_tokens"]) for item in corrections]
    total_cost = sum(float(item["cost_cny"]) for item in records)
    total_corrected_tokens = sum(corrected_tokens)
    reasoning_values = [
        int(item["usage"]["reasoning_tokens"])
        for item in records
        if isinstance(item.get("usage", {}).get("reasoning_tokens"), int)
    ]
    content_token_values = [
        int(item["usage"]["completion_tokens"]) - int(item["usage"]["reasoning_tokens"])
        for item in records
        if isinstance(item.get("usage", {}).get("completion_tokens"), int)
        and isinstance(item.get("usage", {}).get("reasoning_tokens"), int)
    ]
    reviewed = len(records)
    correction_count = decisions["correct"]
    checks = {
        "review_count_24": reviewed == REVIEW_COUNT,
        "balanced_6_per_domain": all(domain_counts[domain] == REVIEWS_PER_DOMAIN for domain in DOMAINS),
        "decisions_complete": decisions["keep"] + correction_count == reviewed,
        "all_calls_succeeded": all(item.get("status") == "SUCCEEDED" for item in records),
        "all_costs_finite": all(math.isfinite(float(item["cost_cny"])) and float(item["cost_cny"]) >= 0 for item in records),
        "stage_cost_within_5_cny": total_cost <= STAGE_API_BUDGET_CNY,
        "reasoning_content_not_persisted": all(item.get("reasoning_content_persisted") is False for item in records),
        "reasoning_content_not_trained": all(item.get("reasoning_content_trained") is False for item in records),
        "structured_outputs_valid": all(item.get("structured_output_valid") is True for item in records),
    }
    return {
        "reviewed_trajectories": reviewed,
        "keep_count": decisions["keep"],
        "correction_count": correction_count,
        "keep_rate": decisions["keep"] / reviewed if reviewed else None,
        "correction_rate": correction_count / reviewed if reviewed else None,
        "corrected_target_tokens": total_corrected_tokens,
        "median_correction_tokens": nearest_rank(corrected_tokens, 0.5),
        "p90_correction_tokens": nearest_rank(corrected_tokens, 0.9),
        "total_cost_cny": total_cost,
        "cost_per_review_cny": total_cost / reviewed if reviewed else None,
        "cost_per_corrected_trajectory_cny": (total_cost / correction_count if correction_count else None),
        "cost_per_correction_token_cny": (total_cost / total_corrected_tokens if total_corrected_tokens else None),
        "usage": {
            "prompt_tokens": sum(int(item["usage"].get("prompt_tokens", 0)) for item in records),
            "completion_tokens": sum(int(item["usage"].get("completion_tokens", 0)) for item in records),
            "reasoning_tokens": sum(reasoning_values) if len(reasoning_values) == reviewed else None,
            "reasoning_tokens_reported_for_all_calls": len(reasoning_values) == reviewed,
            "final_content_tokens": (sum(content_token_values) if len(content_token_values) == reviewed else None),
            "final_content_tokens_derived_for_all_calls": len(content_token_values) == reviewed,
        },
        "domain_counts": dict(sorted(domain_counts.items())),
        "checks": checks,
        "passed": all(checks.values()),
    }


__all__ = [
    "CANDIDATES_PER_DOMAIN",
    "DOMAINS",
    "REVIEW_COUNT",
    "REVIEWS_PER_DOMAIN",
    "STAGE_API_BUDGET_CNY",
    "deterministic_candidate_pool",
    "nearest_rank",
    "select_high_information",
    "summarize_reviews",
]
