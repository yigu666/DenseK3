"""Materialize the deterministic P11-T general micro benchmark and prompt pool."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TITAN = ROOT / "titan"
sys.path.insert(0, str(TITAN))

from p11.data import (  # noqa: E402
    IFEVAL_SUPPORTED,
    LANGUAGE_CONFIGS,
    P11_DATA_SEED,
    arc_record,
    belebele_record,
    canonical_bytes,
    file_sha256,
    ifeval_record,
    mmlu_record,
    record_counts,
    stable_select,
)

OUTPUT = ROOT / "data/p11-general"
HF_MIRROR = "https://hf-mirror.com"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_bytes(record) + b"\n" for record in records))


def _revision(repo: str) -> str:
    from huggingface_hub import HfApi

    revision = HfApi(endpoint=HF_MIRROR).dataset_info(repo).sha
    if not revision:
        raise ValueError(f"Dataset revision is unavailable: {repo}")
    return revision


def _load(repo: str, config: str | None, split: str, revision: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = (
        load_dataset(repo, config, split=split, revision=revision)
        if config
        else load_dataset(repo, split=split, revision=revision)
    )
    return [dict(row) for row in dataset]


def _ifeval_eligible(row: dict[str, Any]) -> bool:
    identifiers = set(str(value) for value in row["instruction_id_list"])
    return bool(identifiers) and identifiers <= IFEVAL_SUPPORTED


def materialize() -> dict[str, Any]:
    if os.environ.get("HF_ENDPOINT", HF_MIRROR) != HF_MIRROR:
        raise ValueError("P11 public datasets must use the approved domestic HF mirror")
    os.environ["HF_ENDPOINT"] = HF_MIRROR
    revisions = {
        repo: _revision(repo)
        for repo in ("cais/mmlu", "allenai/ai2_arc", "facebook/belebele", "google/IFEval")
    }

    mmlu_dev_rows = _load("cais/mmlu", "all", "validation", revisions["cais/mmlu"])
    mmlu_dev = stable_select(
        mmlu_dev_rows,
        8,
        namespace="mmlu-dev",
        key=lambda row: row,
        unique_group=lambda row: str(row["subject"]),
    )
    mmlu_train_rows = _load(
        "cais/mmlu", "all", "auxiliary_train", revisions["cais/mmlu"]
    )
    mmlu_train = stable_select(
        mmlu_train_rows, 416, namespace="mmlu-train", key=lambda row: row
    )

    arc_dev_rows = _load(
        "allenai/ai2_arc", "ARC-Challenge", "validation", revisions["allenai/ai2_arc"]
    )
    arc_dev = stable_select(
        arc_dev_rows, 8, namespace="arc-dev", key=lambda row: row["id"]
    )
    arc_train_rows = _load(
        "allenai/ai2_arc", "ARC-Challenge", "train", revisions["allenai/ai2_arc"]
    )
    arc_train = stable_select(
        arc_train_rows, 256, namespace="arc-train", key=lambda row: row["id"]
    )

    belebele_dev: list[tuple[str, dict[str, Any]]] = []
    belebele_train: list[tuple[str, dict[str, Any]]] = []
    for config in LANGUAGE_CONFIGS:
        rows = _load("facebook/belebele", config, "test", revisions["facebook/belebele"])
        dev = stable_select(
            rows,
            2,
            namespace=f"belebele-{config}",
            key=lambda row: [row["link"], row["question_number"]],
        )
        dev_source_ids = {row["_p11_source_id"] for row in dev}
        train = stable_select(
            rows,
            64,
            namespace=f"belebele-{config}",
            key=lambda row: [row["link"], row["question_number"]],
            excluded_ids=dev_source_ids,
        )
        belebele_dev.extend((config, row) for row in dev)
        belebele_train.extend((config, row) for row in train)

    ifeval_rows = _load("google/IFEval", None, "train", revisions["google/IFEval"])
    eligible_ifeval = [row for row in ifeval_rows if _ifeval_eligible(row)]
    ifeval_dev = stable_select(
        eligible_ifeval, 8, namespace="ifeval", key=lambda row: row["key"]
    )
    ifeval_dev_ids = {row["_p11_source_id"] for row in ifeval_dev}
    ifeval_train = stable_select(
        eligible_ifeval,
        96,
        namespace="ifeval",
        key=lambda row: row["key"],
        excluded_ids=ifeval_dev_ids,
    )

    benchmark = [mmlu_record(row, role="dev") for row in mmlu_dev]
    benchmark += [arc_record(row, role="dev") for row in arc_dev]
    benchmark += [
        belebele_record(row, role="dev", config=config) for config, row in belebele_dev
    ]
    benchmark += [ifeval_record(row, role="dev") for row in ifeval_dev]
    benchmark.sort(key=lambda row: row["id"])

    prompt_pool = [mmlu_record(row, role="train") for row in mmlu_train]
    prompt_pool += [arc_record(row, role="train") for row in arc_train]
    prompt_pool += [
        belebele_record(row, role="train", config=config)
        for config, row in belebele_train
    ]
    prompt_pool += [ifeval_record(row, role="train") for row in ifeval_train]
    prompt_pool.sort(key=lambda row: row["id"])
    if len(benchmark) != 32 or record_counts(benchmark, "domain") != {
        "GENERAL_KNOWLEDGE": 8,
        "GENERAL_REASONING": 8,
        "INSTRUCTION_SYNTHESIS": 8,
        "MULTILINGUAL_LANGUAGE": 8,
    }:
        raise ValueError("P11 general-dev four-bucket contract is not exact")
    if len(prompt_pool) != 1024:
        raise ValueError("P11 general training prompt pool must contain exactly 1024 records")

    benchmark_path = OUTPUT / "eval/general-dev-micro-v1.jsonl"
    prompts_path = OUTPUT / "prompts/general-train-prompts-v1.jsonl"
    _write_jsonl(benchmark_path, benchmark)
    _write_jsonl(prompts_path, prompt_pool)

    p5 = ROOT / "data/p5-probe/packed"
    raw_files = [p5 / name for name in ("corpus-manifest.json", "train.bin", "train.idx")]
    if not all(path.is_file() for path in raw_files):
        raise FileNotFoundError("Frozen P5 train corpus is incomplete")
    raw_lm = {
        "source": "FROZEN_P5_FINEWEB_EDU_TRAIN_ONLY",
        "heldout_bin_opened": False,
        "sequence_length": 2048,
        "sequence_count": 976,
        "effective_tokens": 1_998_848,
        "files": {
            str(path.relative_to(ROOT)): file_sha256(path) for path in raw_files
        },
    }
    dev_source_ids = {(row["source"], row["source_record_id"]) for row in benchmark}
    train_source_ids = {(row["source"], row["source_record_id"]) for row in prompt_pool}
    source_ids_disjoint = not dev_source_ids & train_source_ids
    if not source_ids_disjoint:
        raise ValueError("P11 general-dev and training source identities overlap")
    benchmark_manifest = {
        "stage": "P11.1-GENERAL-DEV-MICRO-V1",
        "status": "PASS",
        "completed_at": datetime.now().astimezone().isoformat(),
        "seed": P11_DATA_SEED,
        "dataset_endpoint": HF_MIRROR,
        "dataset_revisions": revisions,
        "benchmark_path": str(benchmark_path.relative_to(ROOT)),
        "benchmark_sha256": file_sha256(benchmark_path),
        "record_count": len(benchmark),
        "domain_counts": record_counts(benchmark, "domain"),
        "scoring_contracts": ["MULTIPLE_CHOICE", "IFEVAL_STRICT"],
        "checkpoint_selection_role": "P11_MICRO_GENERAL_DEV_PRIMARY",
        "math_code_specialist_weight": 0.0,
        "heldout_accessed": False,
        "result_marker": "P11_GENERAL_BENCHMARK_MATERIALIZATION=PASS",
    }
    corpus_manifest = {
        "stage": "P11.1-GENERAL-CORPUS-AND-ROUTING",
        "status": "PASS",
        "completed_at": datetime.now().astimezone().isoformat(),
        "seed": P11_DATA_SEED,
        "prompt_pool_path": str(prompts_path.relative_to(ROOT)),
        "prompt_pool_sha256": file_sha256(prompts_path),
        "prompt_pool_record_count": len(prompt_pool),
        "prompt_pool_domain_counts": record_counts(prompt_pool, "domain"),
        "general_dev_source_ids_disjoint": source_ids_disjoint,
        "raw_lm": raw_lm,
        "signal_share_unit": "SUPERVISED_TOKENS",
        "signal_shares": {"kimi": 0.40, "qwen": 0.35, "raw_lm": 0.25},
        "probe_target_tokens": {
            "kimi": 104_858,
            "qwen": 91_750,
            "raw_lm": 65_536,
            "total": 262_144,
        },
        "teacher_outputs_materialized": False,
        "training_allowed": False,
        "p11_2_p11_3_micro_gates_allowed": True,
        "heldout_accessed": False,
        "result_marker": "P11_PROBE_CORPUS_MATERIALIZATION=PASS",
    }
    _write_json(OUTPUT / "eval/general-benchmark-manifest.json", benchmark_manifest)
    _write_json(OUTPUT / "packed/corpus-manifest.json", corpus_manifest)
    return {"benchmark": benchmark_manifest, "corpus": corpus_manifest}


def main() -> int:
    try:
        result = materialize()
        print(result["benchmark"]["result_marker"], flush=True)
        print(result["corpus"]["result_marker"], flush=True)
        print("P11_T_TRAINING_ALLOWED=false", flush=True)
        print("P11_2_P11_3_MICRO_GATES_ALLOWED=true", flush=True)
        return 0
    except Exception as error:
        failure = {
            "stage": "P11.1-GENERAL-CORPUS-AND-BENCHMARK",
            "status": "ERROR",
            "completed_at": datetime.now().astimezone().isoformat(),
            "exception_type": type(error).__name__,
            "exception": str(error),
            "heldout_accessed": False,
            "training_allowed": False,
            "result_marker": "P11_GENERAL_DATA_MATERIALIZATION=FAIL",
        }
        _write_json(TITAN / "manifests/reproduction/p11-t/p11-t-data-failure.json", failure)
        print(failure["result_marker"], flush=True)
        print(f"P11_DATA_EXCEPTION={type(error).__name__}: {error}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
