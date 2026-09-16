#!/usr/bin/env python3
"""Inference-only public capability benchmark suite for DenseK3-Core.

The program evaluates exactly two immutable model states: the Qwen donor and
the P11.6 canonical checkpoint.
It deliberately has no training, backward, optimizer, or weight-update path.

Every model/benchmark worker is a separate process.  Results are appended per
case and can be resumed without repeating completed cases.  A shared-GPU
orchestrator admits only one worker at a time after checking physical free
memory; it never inspects, signals, or changes another user's process.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("DENSEK3_EVAL_OUT", ROOT / "results/reproduced")).resolve()
DATA = OUT / "datasets"
QWEN_PATH = Path(os.environ.get("DENSEK3_QWEN_PATH", ROOT / "models/Qwen3.5-4B-Base")).resolve()
P11_PATH = Path(os.environ.get("DENSEK3_P11_PATH", ROOT / "models/densek3-p11.6")).resolve()
P11_LOADER = ROOT / "evaluation/model_loader.py"
MMLU_DATA = DATA / "mmlu-all"
WIKI_DATA = DATA / "wikitext-103-raw-test"
LONGBENCH_DATA = DATA / "longbench-v2-train"
LONGBENCH_REPO = DATA / "longbench-official"
RULER_REPO = DATA / "ruler-official"

MODELS = ("qwen", "p11")
AXES = ("mmlu", "wikitext103", "longbench-v2", "ruler")
GIB = 2**30
RUNTIME_HEADROOM_BYTES = int(1.5 * GIB)
PREFILL_CHUNK_TOKENS = 1024
WIKI_CONTEXT = 4096
WIKI_STRIDE = 2048
WIKI_LOGIT_CHUNK = 64
RULER_LENGTHS = (4096, 8192, 16384, 32768, 65536, 131072)
RULER_TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
)
RULER_GENERATION_TOKENS = {
    "niah_single_1": 128,
    "niah_single_2": 128,
    "niah_single_3": 128,
    "niah_multikey_1": 128,
    "niah_multikey_2": 128,
    "niah_multikey_3": 128,
    "niah_multivalue": 128,
    "niah_multiquery": 128,
    "vt": 30,
    "cwe": 120,
    "fwe": 50,
    "qa_1": 32,
    "qa_2": 32,
}

# Admission estimates include weights plus measured/expected short-lived state.
# A further 1.5 GiB physical headroom is required by the orchestrator.
ADMISSION_BYTES = {
    "mmlu": {"qwen": 12 * GIB, "p11": 13 * GIB},
    "wikitext103": {"qwen": 13 * GIB, "p11": 14 * GIB},
    "longbench-v2": {"qwen": 16 * GIB, "p11": 15 * GIB},
    "ruler": {"qwen": 17 * GIB, "p11": 16 * GIB},
}

MMLU_CATEGORIES = {
    "abstract_algebra": "STEM", "anatomy": "STEM", "astronomy": "STEM",
    "business_ethics": "other", "clinical_knowledge": "other",
    "college_biology": "STEM", "college_chemistry": "STEM",
    "college_computer_science": "STEM", "college_mathematics": "STEM",
    "college_medicine": "other", "college_physics": "STEM",
    "computer_security": "STEM", "conceptual_physics": "STEM",
    "econometrics": "social_sciences", "electrical_engineering": "STEM",
    "elementary_mathematics": "STEM", "formal_logic": "humanities",
    "global_facts": "other", "high_school_biology": "STEM",
    "high_school_chemistry": "STEM", "high_school_computer_science": "STEM",
    "high_school_european_history": "humanities",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_mathematics": "STEM", "high_school_microeconomics": "social_sciences",
    "high_school_physics": "STEM", "high_school_psychology": "social_sciences",
    "high_school_statistics": "STEM", "high_school_us_history": "humanities",
    "high_school_world_history": "humanities", "human_aging": "other",
    "human_sexuality": "social_sciences", "international_law": "humanities",
    "jurisprudence": "humanities", "logical_fallacies": "humanities",
    "machine_learning": "STEM", "management": "other", "marketing": "other",
    "medical_genetics": "other", "miscellaneous": "other", "moral_disputes": "humanities",
    "moral_scenarios": "humanities", "nutrition": "other", "philosophy": "humanities",
    "prehistory": "humanities", "professional_accounting": "other",
    "professional_law": "humanities", "professional_medicine": "other",
    "professional_psychology": "social_sciences", "public_relations": "social_sciences",
    "security_studies": "social_sciences", "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences", "virology": "other",
    "world_religions": "humanities",
}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}-{time.time_ns()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    values = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                values.append(json.loads(line))
    return values


def import_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ensure_dirs() -> None:
    for name in (
        "source-audit", "datasets", "mmlu", "wikitext103", "longbench-v2",
        "ruler", "summary", "logs",
    ):
        (OUT / name).mkdir(parents=True, exist_ok=True)


def git_head(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        return None


def dataset_fingerprint(path: Path) -> dict[str, Any]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    return {
        "path": str(path.relative_to(ROOT)),
        "tree_sha256": sha256_tree(path),
        "file_count": len(files),
        "bytes": sum(item.stat().st_size for item in files),
    }


def prepare_source_audit() -> dict[str, Any]:
    ensure_dirs()
    required = (
        QWEN_PATH, P11_PATH, P11_LOADER,
        MMLU_DATA, WIKI_DATA, LONGBENCH_DATA, LONGBENCH_REPO, RULER_REPO,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing benchmark inputs: {missing}")
    qwen_shards = sorted(QWEN_PATH.glob("*.safetensors"))
    value = {
        "stage": "FINAL_CAPABILITY_BENCHMARK_SOURCE_AUDIT",
        "status": "PASS",
        "result_marker": "FINAL_CAPABILITY_SOURCE_AUDIT=PASS",
        "created_at": now(),
        "models": {
            "qwen": {
                "role": "DONOR_REFERENCE",
                "path": str(QWEN_PATH.relative_to(ROOT)),
                "config_sha256": sha256_file(QWEN_PATH / "config.json"),
                "weight_shards": {item.name: sha256_file(item) for item in qwen_shards},
            },
            "p11": {
                "role": "CURRENT_CANONICAL",
                "path": str(P11_PATH.relative_to(ROOT)),
                "manifest_sha256": sha256_file(P11_PATH / "p11-6-fast-candidate-manifest.json"),
            },
        },
        "datasets": {
            "mmlu": dataset_fingerprint(MMLU_DATA),
            "wikitext103": dataset_fingerprint(WIKI_DATA),
            "longbench_v2": dataset_fingerprint(LONGBENCH_DATA),
            "longbench_official_code": {
                "path": str(LONGBENCH_REPO.relative_to(ROOT)),
                "git_commit": git_head(LONGBENCH_REPO),
                "download_mirror": "https://ghfast.top/https://github.com/THUDM/LongBench.git",
            },
            "ruler_official_code": {
                "path": str(RULER_REPO.relative_to(ROOT)),
                "git_commit": git_head(RULER_REPO),
                "download_mirror": "https://ghfast.top/https://github.com/NVIDIA/RULER.git",
                "official_default_samples_per_task_length": 500,
                "suite_samples_per_task_length": 4,
                "suite_label": "RULER_REDUCED_SAMPLE_EVAL",
                "reason": (
                    "39,000 full cases across six long lengths and three models is disproportionate; "
                    "fixed 4-case strata preserve all 13 tasks and lengths"
                ),
            },
        },
        "contracts": {
            "training": False,
            "backward": False,
            "optimizer": False,
            "weight_update": False,
            "canonical_model_unchanged": "P11.6",
            "shared_gpu": True,
            "shared_speed_label": "DIAGNOSTIC_ONLY_SHARED_GPU",
            "runtime_headroom_bytes": RUNTIME_HEADROOM_BYTES,
            "download_endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        },
    }
    atomic_json(OUT / "source-audit/source-audit.json", value)
    return value


def gpu_rows() -> list[dict[str, int]]:
    command = [
        "nvidia-smi", "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = []
    for line in result.stdout.splitlines():
        fields = [int(part.strip()) for part in line.split(",")]
        rows.append({
            "index": fields[0], "total_mib": fields[1], "used_mib": fields[2],
            "free_mib": fields[3], "utilization": fields[4],
        })
    return rows


def compute_apps() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except Exception:
        return []
    values = []
    for line in result.stdout.splitlines():
        fields = [part.strip() for part in line.split(",", 3)]
        if len(fields) == 4:
            values.append({"gpu_uuid": fields[0], "pid": fields[1], "name": fields[2], "used_mib": fields[3]})
    return values


def allocator_counters(torch: Any) -> dict[str, int]:
    stats = torch.cuda.memory_stats(0)
    return {
        "num_ooms": int(stats.get("num_ooms", 0)),
        "num_alloc_retries": int(stats.get("num_alloc_retries", 0)),
    }


def runtime_headroom(torch: Any) -> dict[str, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    if int(free_bytes) < RUNTIME_HEADROOM_BYTES:
        raise RuntimeError(f"SHARED_GPU_RUNTIME_HEADROOM_LOST free_bytes={int(free_bytes)}")
    return {"free_bytes": int(free_bytes), "total_bytes": int(total_bytes)}


def load_model(model_name: str) -> tuple[Any, Any, Any, dict[str, Any], list[Any]]:
    import torch

    if model_name == "qwen":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_PATH,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).to("cuda:0").eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        metadata = {
            "loader": "TRANSFORMERS_AUTOMODELFORCAUSALLM_TEXT_ONLY",
            "class": type(model).__name__,
            "dtype": "torch.bfloat16",
            "attention_implementation": "sdpa",
            "load_seconds": time.perf_counter() - started,
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        }
        return model, None, tokenizer, metadata, []
    loader = import_module(P11_LOADER, f"densek3_p11_loader_{os.getpid()}_{time.time_ns()}")
    started = time.perf_counter()
    if model_name == "p11":
        model, attnres, tokenizer, metadata, cleanup = loader.load_p11(P11_PATH)
    else:
        raise ValueError(model_name)
    metadata = dict(metadata)
    metadata["load_seconds"] = time.perf_counter() - started
    metadata["public_loader_sha256"] = sha256_file(P11_LOADER)
    return model, attnres, tokenizer, metadata, cleanup


def cleanup_model(model: Any, attnres: Any, cleanup: list[Any]) -> None:
    for callback in reversed(cleanup):
        with contextlib.suppress(Exception):
            callback()
    with contextlib.suppress(Exception):
        if model is not None:
            model.cpu()
    with contextlib.suppress(Exception):
        if attnres is not None:
            attnres.cpu()
    del model, attnres
    gc.collect()
    with contextlib.suppress(Exception):
        import torch

        torch.cuda.empty_cache()


def choice_ids(tokenizer: Any, *, leading_space: bool) -> list[int]:
    values = []
    for label in "ABCD":
        encoded = tokenizer.encode((" " if leading_space else "") + label, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"choice token {label} is not one token: {encoded}")
        values.append(int(encoded[0]))
    return values


def next_token_logits(
    model: Any, input_ids: Any, *, force_chunked: bool = False,
) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as functional

    runtime_headroom(torch)
    length = int(input_ids.shape[1])
    started = time.perf_counter()
    if length <= 8192 and not force_chunked:
        output = model(input_ids=input_ids.to("cuda:0"), use_cache=False, logits_to_keep=1, return_dict=True)
        logits = output.logits[0, -1].float()
        del output
        torch.cuda.synchronize()
        return logits, {
            "input_tokens": length,
            "prefill_seconds": time.perf_counter() - started,
            "cache": False,
            "prefill_mode": "full_no_cache",
        }
    cache = None
    logits = None
    for offset in range(0, length, PREFILL_CHUNK_TOKENS):
        runtime_headroom(torch)
        current = input_ids[:, offset : offset + PREFILL_CHUNK_TOKENS].to("cuda:0")
        output = model.model(input_ids=current, past_key_values=cache, use_cache=True, return_dict=True)
        cache = output.past_key_values
        hidden = output.last_hidden_state
        logits = functional.linear(hidden[:, -1].float(), model.lm_head.weight.float())[0]
        del current, output, hidden
    if logits is None or cache is None:
        raise RuntimeError("empty prompt")
    torch.cuda.synchronize()
    seen_tokens = None
    if hasattr(cache, "latent_contract"):
        seen_tokens = int(cache.latent_contract()["seen_tokens"])
    elif hasattr(cache, "get_seq_length"):
        seen_tokens = int(cache.get_seq_length())
    if seen_tokens is not None and seen_tokens != length:
        raise RuntimeError(f"cache length mismatch {seen_tokens} != {length}")
    del cache
    return logits, {
        "input_tokens": length,
        "prefill_seconds": time.perf_counter() - started,
        "cache": True,
        "cache_seen_tokens": seen_tokens,
        "prefill_mode": "chunked_cache",
    }


def format_mmlu_question(row: dict[str, Any], include_answer: bool) -> str:
    choices = row["choices"]
    lines = [str(row["question"])]
    lines.extend(f"{label}. {choices[index]}" for index, label in enumerate("ABCD"))
    suffix = f"Answer: {'ABCD'[int(row['answer'])]}" if include_answer else "Answer:"
    return "\n".join(lines + [suffix])


def mmlu_prompt(subject: str, dev_rows: list[dict[str, Any]], test_row: dict[str, Any]) -> str:
    pretty = subject.replace("_", " ")
    parts = [f"The following are multiple choice questions (with answers) about {pretty}."]
    parts.extend(format_mmlu_question(row, True) for row in dev_rows)
    parts.append(format_mmlu_question(test_row, False))
    return "\n\n".join(parts)


def summarize_accuracy(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: defaultdict[str, list[bool]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(bool(row["correct"]))
    return {
        name: {"correct": sum(values), "count": len(values), "accuracy": sum(values) / len(values)}
        for name, values in sorted(grouped.items())
    }


def worker_mmlu(model_name: str) -> dict[str, Any]:
    import torch
    from datasets import load_from_disk

    dataset = load_from_disk(str(MMLU_DATA))
    dev_by_subject: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dataset["dev"]:
        dev_by_subject[str(row["subject"])].append(dict(row))
    if set(dev_by_subject) != set(MMLU_CATEGORIES):
        raise RuntimeError("MMLU 57-subject dev inventory mismatch")
    if any(len(rows) != 5 for rows in dev_by_subject.values()):
        raise RuntimeError("MMLU official dev is not exactly 5-shot per subject")
    output = OUT / "mmlu" / f"{model_name}-5shot.json"
    partial = OUT / "mmlu" / f"{model_name}-5shot.cases.jsonl"
    if output.is_file() and read_json(output).get("status") == "PASS":
        return read_json(output)
    existing = {str(row["case_id"]): row for row in read_jsonl(partial)}
    model = attnres = tokenizer = None
    cleanup: list[Any] = []
    model, attnres, tokenizer, metadata, cleanup = load_model(model_name)
    labels = choice_ids(tokenizer, leading_space=True)
    before = allocator_counters(torch)
    torch.cuda.reset_peak_memory_stats()
    wall_started = time.perf_counter()
    try:
        for index, raw in enumerate(dataset["test"], start=1):
            row = dict(raw)
            case_id = f"{row['subject']}:{index - 1}"
            if case_id in existing:
                continue
            prompt = mmlu_prompt(str(row["subject"]), dev_by_subject[str(row["subject"])], row)
            input_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").input_ids
            with torch.inference_mode():
                logits, timing = next_token_logits(
                    model, input_ids,
                    force_chunked=False,
                )
            selected = logits[labels]
            prediction = int(torch.argmax(selected).item())
            answer = int(row["answer"])
            result = {
                "case_id": case_id,
                "subject": str(row["subject"]),
                "category": MMLU_CATEGORIES[str(row["subject"])],
                "answer": answer,
                "prediction": prediction,
                "correct": prediction == answer,
                "choice_logits": [float(value) for value in selected.tolist()],
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                **timing,
            }
            append_jsonl(partial, result)
            existing[case_id] = result
            del input_ids, logits, selected
            if index % 100 == 0:
                print(f"FINAL_BENCH_MMLU_{model_name.upper()}={index}/14042", flush=True)
        rows = list(existing.values())
        if len(rows) != 14042:
            raise RuntimeError(f"MMLU incomplete {len(rows)}/14042")
        after = allocator_counters(torch)
        elapsed = time.perf_counter() - wall_started
        report = {
            "stage": "FINAL_CAPABILITY_MMLU_5SHOT",
            "status": "PASS",
            "result_marker": "FINAL_BENCH_MMLU=PASS",
            "model": model_name,
            "created_at": now(),
            "model_metadata": metadata,
            "dataset": dataset_fingerprint(MMLU_DATA),
            "prompt_contract": "OFFICIAL_MMLU_DEV_5SHOT_NO_CHAT_TEMPLATE_CONDITIONAL_LABEL_LL",
            "choice_token_ids": labels,
            "count": len(rows),
            "correct": sum(bool(row["correct"]) for row in rows),
            "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
            "by_subject": summarize_accuracy(rows, "subject"),
            "by_category": summarize_accuracy(rows, "category"),
            "timing": {
                "wall_seconds": elapsed,
                "prompt_tokens": sum(int(row["input_tokens"]) for row in rows),
                "tokens_per_second": sum(int(row["input_tokens"]) for row in rows) / elapsed,
                "cases_per_second": len(rows) / elapsed,
                "classification": "DIAGNOSTIC_ONLY_SHARED_GPU",
            },
            "gpu": {
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "allocator_delta": {key: after[key] - before[key] for key in before},
            },
            "inference_only": True,
            "training": False,
            "backward": False,
            "optimizer": False,
            "weight_update": False,
        }
        atomic_json(output, report)
        return report
    finally:
        cleanup_model(model, attnres, cleanup)


def wiki_tokens(tokenizer: Any) -> list[int]:
    from datasets import load_from_disk

    dataset = load_from_disk(str(WIKI_DATA))
    text = "\n\n".join(str(row["text"]) for row in dataset)
    return [int(value) for value in tokenizer.encode(text, add_special_tokens=False)]


def wiki_hidden(model: Any, input_ids: Any, *, force_chunked: bool) -> Any:
    """Return all window hidden states, optionally using cache-chunked prefill."""
    import torch

    if not force_chunked:
        return model.model(input_ids=input_ids, use_cache=False, return_dict=True).last_hidden_state
    pieces = []
    cache = None
    for offset in range(0, int(input_ids.shape[1]), PREFILL_CHUNK_TOKENS):
        current = input_ids[:, offset : offset + PREFILL_CHUNK_TOKENS]
        output = model.model(
            input_ids=current,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = output.past_key_values
        pieces.append(output.last_hidden_state)
        del current, output
    if cache is not None and hasattr(cache, "latent_contract"):
        seen = int(cache.latent_contract()["seen_tokens"])
        if seen != int(input_ids.shape[1]):
            raise RuntimeError(f"WikiText cache length mismatch {seen} != {int(input_ids.shape[1])}")
    hidden = torch.cat(pieces, dim=1)
    del pieces, cache
    return hidden


def worker_wikitext(model_name: str) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    output = OUT / "wikitext103" / f"{model_name}.json"
    partial = OUT / "wikitext103" / f"{model_name}.windows.jsonl"
    if output.is_file() and read_json(output).get("status") == "PASS":
        return read_json(output)
    existing = {int(row["target_end"]): row for row in read_jsonl(partial)}
    model = attnres = tokenizer = None
    cleanup: list[Any] = []
    model, attnres, tokenizer, metadata, cleanup = load_model(model_name)
    tokens = wiki_tokens(tokenizer)
    token_digest = hashlib.sha256()
    for value in tokens:
        token_digest.update(int(value).to_bytes(4, "little", signed=False))
    token_hash = token_digest.hexdigest()
    before = allocator_counters(torch)
    torch.cuda.reset_peak_memory_stats()
    wall_started = time.perf_counter()
    try:
        previous_end = 1
        while previous_end < len(tokens):
            target_end = min(previous_end + WIKI_STRIDE, len(tokens))
            if target_end in existing:
                previous_end = target_end
                continue
            window_start = max(0, target_end - WIKI_CONTEXT)
            input_ids = torch.tensor(tokens[window_start:target_end], dtype=torch.long).unsqueeze(0).to("cuda:0")
            runtime_headroom(torch)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                hidden = wiki_hidden(
                    model, input_ids,
                    force_chunked=False,
                )
            local_target_start = max(previous_end, window_start + 1) - window_start
            local_target_end = target_end - window_start
            loss_sum = 0.0
            count = 0
            for begin in range(local_target_start, local_target_end, WIKI_LOGIT_CHUNK):
                end = min(begin + WIKI_LOGIT_CHUNK, local_target_end)
                logits = model.lm_head(hidden[:, begin - 1 : end - 1])
                targets = input_ids[:, begin:end]
                loss = functional.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum"
                )
                loss_sum += float(loss.item())
                count += int(targets.numel())
                del logits, targets, loss
            torch.cuda.synchronize()
            row = {
                "target_start": previous_end,
                "target_end": target_end,
                "window_start": window_start,
                "window_tokens": int(input_ids.shape[1]),
                "counted_tokens": count,
                "loss_sum": loss_sum,
                "seconds": time.perf_counter() - started,
                "finite": math.isfinite(loss_sum),
            }
            append_jsonl(partial, row)
            existing[target_end] = row
            previous_end = target_end
            del input_ids, hidden
            gc.collect()
            torch.cuda.empty_cache()
            if len(existing) % 25 == 0:
                print(f"FINAL_BENCH_WIKI_{model_name.upper()}_WINDOWS={len(existing)}", flush=True)
        rows = [existing[key] for key in sorted(existing)]
        expected = len(tokens) - 1
        counted = sum(int(row["counted_tokens"]) for row in rows)
        if counted != expected:
            raise RuntimeError(f"WikiText token accounting mismatch {counted} != {expected}")
        loss_sum = sum(float(row["loss_sum"]) for row in rows)
        ce = loss_sum / counted
        after = allocator_counters(torch)
        elapsed = time.perf_counter() - wall_started
        report = {
            "stage": "FINAL_CAPABILITY_WIKITEXT103_RAW_TEST",
            "status": "PASS",
            "result_marker": "FINAL_BENCH_WIKITEXT103=PASS",
            "model": model_name,
            "created_at": now(),
            "model_metadata": metadata,
            "dataset": dataset_fingerprint(WIKI_DATA),
            "tokenizer": "COMMON_QWEN3_5_4B_BASE_TOKENIZER",
            "token_ids_sha256": token_hash,
            "window": WIKI_CONTEXT,
            "stride": WIKI_STRIDE,
            "counted_tokens": counted,
            "expected_counted_tokens": expected,
            "cross_entropy": ce,
            "perplexity": math.exp(ce),
            "finite": math.isfinite(ce),
            "timing": {
                "wall_seconds": elapsed,
                "model_seconds": sum(float(row["seconds"]) for row in rows),
                "tokens_per_second": counted / elapsed,
                "classification": "DIAGNOSTIC_ONLY_SHARED_GPU",
            },
            "gpu": {
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "allocator_delta": {key: after[key] - before[key] for key in before},
            },
            "inference_only": True,
            "training": False,
            "backward": False,
            "optimizer": False,
            "weight_update": False,
        }
        atomic_json(output, report)
        return report
    finally:
        cleanup_model(model, attnres, cleanup)


def longbench_template() -> tuple[str, str]:
    path = LONGBENCH_REPO / "prompts/0shot.txt"
    original = path.read_text(encoding="utf-8")
    marker = 'Format your response as follows: "The correct answer is (insert answer here)".'
    if marker not in original:
        raise RuntimeError("LongBench-v2 official 0shot prompt changed")
    scoring = original.replace(marker, "The correct answer is (")
    return scoring, sha256_file(path)


def longbench_prompt(template: str, row: dict[str, Any]) -> str:
    return (
        template.replace("$DOC$", str(row["context"]).strip())
        .replace("$Q$", str(row["question"]).strip())
        .replace("$C_A$", str(row["choice_A"]).strip())
        .replace("$C_B$", str(row["choice_B"]).strip())
        .replace("$C_C$", str(row["choice_C"]).strip())
        .replace("$C_D$", str(row["choice_D"]).strip())
    )


def longbench_bucket(length: int) -> str:
    if length <= 32768:
        return "LE_32K"
    if length <= 65536:
        return "GT_32K_LE_64K"
    return "GT_64K_LE_128K"


def worker_longbench(model_name: str) -> dict[str, Any]:
    import torch
    from datasets import load_from_disk

    output = OUT / "longbench-v2" / f"{model_name}.json"
    partial = OUT / "longbench-v2" / f"{model_name}.cases.jsonl"
    if output.is_file() and read_json(output).get("status") == "PASS":
        return read_json(output)
    existing = {str(row["case_id"]): row for row in read_jsonl(partial)}
    dataset = load_from_disk(str(LONGBENCH_DATA))
    template, template_hash = longbench_template()
    model = attnres = tokenizer = None
    cleanup: list[Any] = []
    model, attnres, tokenizer, metadata, cleanup = load_model(model_name)
    labels = choice_ids(tokenizer, leading_space=False)
    before = allocator_counters(torch)
    torch.cuda.reset_peak_memory_stats()
    wall_started = time.perf_counter()
    excluded: list[dict[str, Any]] = []
    eligible_count = 0
    try:
        for raw in dataset:
            row = dict(raw)
            case_id = str(row["_id"])
            prompt = longbench_prompt(template, row)
            input_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").input_ids
            length = int(input_ids.shape[1])
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if length > 131072:
                excluded.append({"case_id": case_id, "input_tokens": length, "prompt_sha256": prompt_hash})
                del input_ids
                continue
            eligible_count += 1
            if case_id in existing:
                del input_ids
                continue
            with torch.inference_mode():
                logits, timing = next_token_logits(
                    model, input_ids,
                    force_chunked=False,
                )
            selected = logits[labels]
            prediction = int(torch.argmax(selected).item())
            answer = "ABCD".index(str(row["answer"]).strip())
            result = {
                "case_id": case_id,
                "domain": str(row["domain"]),
                "sub_domain": str(row["sub_domain"]),
                "difficulty": str(row["difficulty"]),
                "reported_length": str(row["length"]),
                "length_bucket": longbench_bucket(length),
                "answer": answer,
                "prediction": prediction,
                "correct": prediction == answer,
                "choice_logits": [float(value) for value in selected.tolist()],
                "prompt_sha256": prompt_hash,
                **timing,
            }
            append_jsonl(partial, result)
            existing[case_id] = result
            del input_ids, logits, selected
            gc.collect()
            torch.cuda.empty_cache()
            if eligible_count % 10 == 0:
                print(
                    f"FINAL_BENCH_LONGBENCH_{model_name.upper()}={eligible_count} eligible processed",
                    flush=True,
                )
        rows = list(existing.values())
        if len(rows) != eligible_count:
            raise RuntimeError(f"LongBench-v2 incomplete {len(rows)}/{eligible_count}")
        after = allocator_counters(torch)
        elapsed = time.perf_counter() - wall_started
        report = {
            "stage": "FINAL_CAPABILITY_LONGBENCH_V2",
            "status": "PASS",
            "result_marker": "FINAL_BENCH_LONGBENCH_V2=PASS",
            "model": model_name,
            "created_at": now(),
            "model_metadata": metadata,
            "dataset": dataset_fingerprint(LONGBENCH_DATA),
            "official_code_commit": git_head(LONGBENCH_REPO),
            "official_prompt_sha256": template_hash,
            "prompt_contract": "OFFICIAL_0SHOT_FULL_CONTEXT_WITH_CONDITIONAL_A_B_C_D_PREFIX",
            "choice_token_ids": labels,
            "official_total": len(dataset),
            "eligible_le_131072": len(rows),
            "excluded_gt_131072": len(excluded),
            "excluded_cases": excluded,
            "correct": sum(bool(row["correct"]) for row in rows),
            "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
            "by_length_bucket": summarize_accuracy(rows, "length_bucket"),
            "by_domain": summarize_accuracy(rows, "domain"),
            "by_sub_domain": summarize_accuracy(rows, "sub_domain"),
            "by_difficulty": summarize_accuracy(rows, "difficulty"),
            "timing": {
                "wall_seconds": elapsed,
                "prompt_tokens": sum(int(row["input_tokens"]) for row in rows),
                "tokens_per_second": sum(int(row["input_tokens"]) for row in rows) / elapsed,
                "cases_per_second": len(rows) / elapsed,
                "classification": "DIAGNOSTIC_ONLY_SHARED_GPU",
            },
            "gpu": {
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "allocator_delta": {key: after[key] - before[key] for key in before},
            },
            "inference_only": True,
            "training": False,
            "backward": False,
            "optimizer": False,
            "weight_update": False,
        }
        atomic_json(output, report)
        return report
    finally:
        cleanup_model(model, attnres, cleanup)


def ruler_metric(task: str, predictions: list[str], references: list[list[str]]) -> float:
    if task in ("qa_1", "qa_2"):
        scores = [
            max(1.0 if value.lower() in pred.lower() else 0.0 for value in refs)
            for pred, refs in zip(predictions, references, strict=True)
        ]
    else:
        scores = [
            sum(1.0 if value.lower() in pred.lower() else 0.0 for value in refs) / len(refs)
            for pred, refs in zip(predictions, references, strict=True)
        ]
    return round(sum(scores) / len(scores) * 100, 2)


def ruler_data_file(length: int, task: str) -> Path:
    return DATA / "ruler-generated-reduced" / str(length) / task / "validation.jsonl"


def greedy_generate(model: Any, tokenizer: Any, input_ids: Any, max_new_tokens: int) -> tuple[str, dict[str, Any]]:
    import torch
    import torch.nn.functional as functional

    runtime_headroom(torch)
    cache = None
    logits = None
    length = int(input_ids.shape[1])
    torch.cuda.synchronize()
    started = time.perf_counter()
    for offset in range(0, length, PREFILL_CHUNK_TOKENS):
        runtime_headroom(torch)
        current = input_ids[:, offset : offset + PREFILL_CHUNK_TOKENS].to("cuda:0")
        output = model.model(input_ids=current, past_key_values=cache, use_cache=True, return_dict=True)
        cache = output.past_key_values
        hidden = output.last_hidden_state
        logits = functional.linear(hidden[:, -1].float(), model.lm_head.weight.float())
        del current, output, hidden
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - started
    if cache is None or logits is None:
        raise RuntimeError("empty RULER prompt")
    generated: list[int] = []
    decode_started = time.perf_counter()
    eos_ids = {value for value in (tokenizer.eos_token_id,) if value is not None}
    for _ in range(max_new_tokens):
        token = int(torch.argmax(logits[0]).item())
        if token in eos_ids:
            break
        generated.append(token)
        current = torch.tensor([[token]], device="cuda:0", dtype=torch.long)
        output = model.model(input_ids=current, past_key_values=cache, use_cache=True, return_dict=True)
        cache = output.past_key_values
        hidden = output.last_hidden_state
        logits = functional.linear(hidden[:, -1].float(), model.lm_head.weight.float())
        del current, output, hidden
    torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - decode_started
    del cache, logits
    return tokenizer.decode(generated, skip_special_tokens=True), {
        "input_tokens": length,
        "generated_tokens": len(generated),
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
    }


def worker_ruler(model_name: str, length: int) -> dict[str, Any]:
    import torch

    if length not in RULER_LENGTHS:
        raise ValueError(length)
    output = OUT / "ruler" / model_name / f"context-{length}.json"
    partial = OUT / "ruler" / model_name / f"context-{length}.cases.jsonl"
    if output.is_file() and read_json(output).get("status") == "PASS":
        return read_json(output)
    for task in RULER_TASKS:
        if not ruler_data_file(length, task).is_file():
            raise FileNotFoundError(f"RULER generated input missing: {ruler_data_file(length, task)}")
    existing = {str(row["case_id"]): row for row in read_jsonl(partial)}
    model = attnres = tokenizer = None
    cleanup: list[Any] = []
    model, attnres, tokenizer, metadata, cleanup = load_model(model_name)
    before = allocator_counters(torch)
    torch.cuda.reset_peak_memory_stats()
    wall_started = time.perf_counter()
    try:
        all_rows: list[dict[str, Any]] = []
        for task in RULER_TASKS:
            source = read_jsonl(ruler_data_file(length, task))
            if len(source) != 4:
                raise RuntimeError(f"RULER reduced stratum {task}/{length} is not 4 cases")
            for item in source:
                case_id = f"{task}:{length}:{item['index']}"
                if case_id in existing:
                    all_rows.append(existing[case_id])
                    continue
                prompt = str(item["input"]) + str(item.get("answer_prefix", ""))
                input_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").input_ids
                with torch.inference_mode():
                    prediction, timing = greedy_generate(
                        model, tokenizer, input_ids, RULER_GENERATION_TOKENS[task]
                    )
                references = [str(value) for value in item["outputs"]]
                result = {
                    "case_id": case_id,
                    "task": task,
                    "length": length,
                    "prediction": prediction.strip(),
                    "references": references,
                    "input_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    **timing,
                }
                append_jsonl(partial, result)
                existing[case_id] = result
                all_rows.append(result)
                del input_ids
                gc.collect()
                torch.cuda.empty_cache()
                print(f"FINAL_BENCH_RULER_{model_name.upper()}_{length}={len(all_rows)}/52", flush=True)
        by_task = {}
        for task in RULER_TASKS:
            rows = [row for row in all_rows if row["task"] == task]
            by_task[task] = {
                "count": len(rows),
                "score": ruler_metric(task, [row["prediction"] for row in rows], [row["references"] for row in rows]),
            }
        after = allocator_counters(torch)
        elapsed = time.perf_counter() - wall_started
        report = {
            "stage": "FINAL_CAPABILITY_RULER_REDUCED",
            "status": "PASS",
            "result_marker": "FINAL_BENCH_RULER_REDUCED=PASS",
            "model": model_name,
            "context_length": length,
            "created_at": now(),
            "model_metadata": metadata,
            "official_code_commit": git_head(RULER_REPO),
            "sample_contract": "RULER_REDUCED_SAMPLE_EVAL_4_PER_TASK_LENGTH_FIXED_ONCE",
            "tasks": list(RULER_TASKS),
            "case_count": len(all_rows),
            "by_task": by_task,
            "average_score": statistics.fmean(value["score"] for value in by_task.values()),
            "timing": {
                "wall_seconds": elapsed,
                "input_tokens": sum(int(row["input_tokens"]) for row in all_rows),
                "generated_tokens": sum(int(row["generated_tokens"]) for row in all_rows),
                "prefill_tokens_per_second": sum(int(row["input_tokens"]) for row in all_rows)
                / sum(float(row["prefill_seconds"]) for row in all_rows),
                "decode_tokens_per_second": sum(int(row["generated_tokens"]) for row in all_rows)
                / max(sum(float(row["decode_seconds"]) for row in all_rows), 1e-9),
                "classification": "DIAGNOSTIC_ONLY_SHARED_GPU",
            },
            "gpu": {
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "allocator_delta": {key: after[key] - before[key] for key in before},
            },
            "inference_only": True,
            "training": False,
            "backward": False,
            "optimizer": False,
            "weight_update": False,
        }
        atomic_json(output, report)
        return report
    finally:
        cleanup_model(model, attnres, cleanup)


def write_error(axis: str, model_name: str, error: BaseException, length: int | None = None) -> Path:
    suffix = f"-{length}" if length is not None else ""
    path = OUT / "logs" / f"error-{axis}-{model_name}{suffix}-{int(time.time())}.json"
    value = {
        "stage": "FINAL_CAPABILITY_BENCHMARK_ATTEMPT_ERROR",
        "status": "ERROR",
        "axis": axis,
        "model": model_name,
        "length": length,
        "created_at": now(),
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "oom": "out of memory" in str(error).lower(),
        "gpu_state": gpu_rows(),
        "compute_apps": compute_apps(),
        "inference_only": True,
        "training": False,
        "backward": False,
        "optimizer": False,
        "weight_update": False,
    }
    atomic_json(path, value)
    return path


def worker(args: argparse.Namespace) -> int:
    ensure_dirs()
    try:
        if args.axis == "mmlu":
            result = worker_mmlu(args.model)
        elif args.axis == "wikitext103":
            result = worker_wikitext(args.model)
        elif args.axis == "longbench-v2":
            result = worker_longbench(args.model)
        elif args.axis == "ruler":
            if args.length is None:
                raise ValueError("RULER worker requires --length")
            result = worker_ruler(args.model, args.length)
        else:
            raise ValueError(args.axis)
        print(result["result_marker"], flush=True)
        return 0
    except Exception as error:
        path = write_error(args.axis, args.model, error, args.length)
        print(f"FINAL_BENCH_ERROR={path} {type(error).__name__}: {error}", flush=True)
        return 2


def result_path(axis: str, model_name: str, length: int | None = None) -> Path:
    if axis == "mmlu":
        return OUT / "mmlu" / f"{model_name}-5shot.json"
    if axis == "wikitext103":
        return OUT / "wikitext103" / f"{model_name}.json"
    if axis == "longbench-v2":
        return OUT / "longbench-v2" / f"{model_name}.json"
    if axis == "ruler" and length is not None:
        return OUT / "ruler" / model_name / f"context-{length}.json"
    raise ValueError((axis, model_name, length))


def choose_gpu(axis: str, model_name: str) -> tuple[int, list[dict[str, int]]] | None:
    required = ADMISSION_BYTES[axis][model_name] + RUNTIME_HEADROOM_BYTES
    rows = sorted(gpu_rows(), key=lambda row: row["free_mib"], reverse=True)
    for row in rows:
        if row["free_mib"] * 2**20 >= required:
            return row["index"], rows
    return None


def orchestrate(args: argparse.Namespace) -> int:
    prepare_source_audit()
    jobs: list[tuple[str, str, int | None]] = []
    for axis in ("mmlu", "wikitext103", "longbench-v2"):
        for model_name in MODELS:
            jobs.append((axis, model_name, None))
    for length in RULER_LENGTHS:
        for model_name in MODELS:
            jobs.append(("ruler", model_name, length))
    state_path = OUT / "summary/orchestrator-state.json"
    state = {
        "stage": "FINAL_CAPABILITY_BENCHMARK_ORCHESTRATOR",
        "status": "RUNNING",
        "created_at": now(),
        "pid": os.getpid(),
        "jobs_total": len(jobs),
        "jobs": [],
        "inference_only": True,
    }
    atomic_json(state_path, state)
    for axis, model_name, length in jobs:
        target = result_path(axis, model_name, length)
        if target.is_file() and read_json(target).get("status") == "PASS":
            state["jobs"].append({"axis": axis, "model": model_name, "length": length, "status": "PASS_REUSED"})
            atomic_json(state_path, state)
            continue
        while True:
            selected = choose_gpu(axis, model_name)
            if selected is not None:
                device, snapshot = selected
                break
            snapshot = gpu_rows()
            state["waiting"] = {"axis": axis, "model": model_name, "length": length, "gpu": snapshot, "at": now()}
            atomic_json(state_path, state)
            print(f"FINAL_BENCH_WAIT_GPU axis={axis} model={model_name} state={snapshot}", flush=True)
            time.sleep(args.poll_seconds)
        command = [
            sys.executable, str(Path(__file__).resolve()), "worker", "--axis", axis,
            "--model", model_name,
        ]
        if length is not None:
            command += ["--length", str(length)]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(device)
        environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        started = now()
        print(f"FINAL_BENCH_START axis={axis} model={model_name} length={length} physical_gpu={device}", flush=True)
        result = subprocess.run(command, env=environment)
        job = {
            "axis": axis, "model": model_name, "length": length, "physical_gpu": device,
            "started_at": started, "ended_at": now(), "returncode": result.returncode,
            "status": "PASS" if result.returncode == 0 else "ERROR_CONTINUED",
            "gpu_at_admission": snapshot,
        }
        state["jobs"].append(job)
        state.pop("waiting", None)
        atomic_json(state_path, state)
    state["status"] = "COMPLETE_WITH_ERRORS" if any(job["status"] == "ERROR_CONTINUED" for job in state["jobs"]) else "PASS"
    state["ended_at"] = now()
    atomic_json(state_path, state)
    print(f"FINAL_CAPABILITY_ORCHESTRATOR={state['status']}", flush=True)
    return 0


def benchmark_summary() -> dict[str, Any]:
    report: dict[str, Any] = {
        "stage": "FINAL_CAPABILITY_BENCHMARK_SUMMARY",
        "created_at": now(),
        "models": {},
        "canonical_model": "P11.6",
    }
    for model_name in MODELS:
        values: dict[str, Any] = {}
        for axis in ("mmlu", "wikitext103", "longbench-v2"):
            path = result_path(axis, model_name)
            values[axis] = read_json(path) if path.is_file() else {"status": "NOT_RUN"}
        ruler = {}
        for length in RULER_LENGTHS:
            path = result_path("ruler", model_name, length)
            ruler[str(length)] = read_json(path) if path.is_file() else {"status": "NOT_RUN"}
        values["ruler"] = ruler
        report["models"][model_name] = values
    statuses = []
    for value in report["models"].values():
        statuses.extend(value[axis].get("status") for axis in ("mmlu", "wikitext103", "longbench-v2"))
        statuses.extend(item.get("status") for item in value["ruler"].values())
    report["status"] = "PASS" if statuses and all(value == "PASS" for value in statuses) else "PARTIAL"
    report["result_marker"] = f"FINAL_CAPABILITY_BENCHMARKS={report['status']}"
    atomic_json(OUT / "summary/final-capability-summary.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--axis", choices=AXES, required=True)
    worker_parser.add_argument("--model", choices=MODELS, required=True)
    worker_parser.add_argument("--length", type=int)
    orchestrator_parser = subparsers.add_parser("orchestrate")
    orchestrator_parser.add_argument("--poll-seconds", type=int, default=60)
    subparsers.add_parser("summarize")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        value = prepare_source_audit()
        print(value["result_marker"], flush=True)
        return 0
    if args.command == "worker":
        return worker(args)
    if args.command == "orchestrate":
        return orchestrate(args)
    if args.command == "summarize":
        value = benchmark_summary()
        print(value["result_marker"], flush=True)
        return 0
    raise ValueError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
