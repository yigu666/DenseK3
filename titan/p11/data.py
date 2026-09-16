"""Deterministic, low-cost P11 general corpus and general-dev materialization."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

P11_DATA_SEED = "DenseK3-P11-general-micro-v1"
LANGUAGE_CONFIGS = ("eng_Latn", "zho_Hans", "fra_Latn", "spa_Latn")
IFEVAL_SUPPORTED = {
    "change_case:english_capital",
    "change_case:english_lowercase",
    "detectable_format:json_format",
    "keywords:existence",
    "keywords:forbidden_words",
    "punctuation:no_comma",
    "startend:quotation",
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def stable_id(namespace: str, value: Any) -> str:
    digest = hashlib.sha256(P11_DATA_SEED.encode() + b"\0" + canonical_bytes(value)).hexdigest()
    return f"{namespace}-{digest[:20]}"


def stable_select(
    rows: Iterable[dict[str, Any]],
    count: int,
    *,
    namespace: str,
    key: Callable[[dict[str, Any]], Any],
    unique_group: Callable[[dict[str, Any]], str] | None = None,
    excluded_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("count must be positive")
    excluded = set() if excluded_ids is None else excluded_ids
    ranked = sorted(((stable_id(namespace, key(row)), row) for row in rows), key=lambda item: item[0])
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row_id, row in ranked:
        if row_id in excluded:
            continue
        if unique_group is not None:
            group = unique_group(row)
            if group in seen:
                continue
            seen.add(group)
        selected.append(dict(row, _p11_source_id=row_id))
        if len(selected) == count:
            return selected
    raise ValueError(f"Only {len(selected)} eligible records exist; {count} required")


def _choice_prompt(question: str, choices: list[str]) -> str:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    rendered = "\n".join(f"{labels[index]}. {choice}" for index, choice in enumerate(choices))
    return (
        f"{question.strip()}\n{rendered}\n"
        "Choose the best answer. Return exactly `ANSWER: X`, replacing X with one option letter."
    )


def mmlu_record(row: dict[str, Any], *, role: str) -> dict[str, Any]:
    choices = [str(value) for value in row["choices"]]
    answer = int(row["answer"])
    source_id = row["_p11_source_id"]
    return {
        "id": f"p11-{role}-{source_id}",
        "role": role,
        "domain": "GENERAL_KNOWLEDGE",
        "source": "cais/mmlu",
        "source_record_id": source_id,
        "subject": str(row["subject"]),
        "language": "en",
        "prompt": _choice_prompt(str(row["question"]), choices),
        "scoring": {"type": "MULTIPLE_CHOICE", "answer": "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[answer]},
    }


def arc_record(row: dict[str, Any], *, role: str) -> dict[str, Any]:
    labels = [str(value) for value in row["choices"]["label"]]
    choices = [str(value) for value in row["choices"]["text"]]
    answer_key = str(row["answerKey"])
    answer_index = labels.index(answer_key)
    source_id = row["_p11_source_id"]
    return {
        "id": f"p11-{role}-{source_id}",
        "role": role,
        "domain": "GENERAL_REASONING",
        "source": "allenai/ai2_arc",
        "source_record_id": str(row["id"]),
        "language": "en",
        "prompt": _choice_prompt(str(row["question"]), choices),
        "scoring": {
            "type": "MULTIPLE_CHOICE",
            "answer": "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[answer_index],
        },
    }


def belebele_record(row: dict[str, Any], *, role: str, config: str) -> dict[str, Any]:
    choices = [str(row[f"mc_answer{index}"]) for index in range(1, 5)]
    answer = int(row["correct_answer_num"]) - 1
    source_id = row["_p11_source_id"]
    question = f"Passage:\n{row['flores_passage']}\n\nQuestion: {row['question']}"
    return {
        "id": f"p11-{role}-{source_id}",
        "role": role,
        "domain": "MULTILINGUAL_LANGUAGE",
        "source": "facebook/belebele",
        # Belebele question_number restarts for every passage and is not a
        # globally unique row identity. The stable source hash includes link +
        # question_number and is also the identity used by the exclusion Gate.
        "source_record_id": f"{config}:{source_id}",
        "language": config,
        "prompt": _choice_prompt(question, choices),
        "scoring": {"type": "MULTIPLE_CHOICE", "answer": "ABCD"[answer]},
    }


def ifeval_record(row: dict[str, Any], *, role: str) -> dict[str, Any]:
    source_id = row["_p11_source_id"]
    return {
        "id": f"p11-{role}-{source_id}",
        "role": role,
        "domain": "INSTRUCTION_SYNTHESIS",
        "source": "google/IFEval",
        "source_record_id": str(row["key"]),
        "language": "en",
        "prompt": str(row["prompt"]),
        "scoring": {
            "type": "IFEVAL_STRICT",
            "instruction_id_list": list(row["instruction_id_list"]),
            "kwargs": list(row["kwargs"]),
        },
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_sha256(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(canonical_bytes(record) + b"\n")
    return digest.hexdigest()


def record_counts(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(record[field]) for record in records).items()))
