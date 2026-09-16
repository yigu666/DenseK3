#!/usr/bin/env python3
"""Prepare the fixed reduced RULER suite using domestic dataset mirrors.

The NVIDIA RULER generator remains unmodified.  This helper only converts
mirror-fetched source corpora into the exact JSON structures expected by the
official generator and invokes that generator with four fixed samples for each
of the 13 official tasks at all six required lengths.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("DENSEK3_EVAL_OUT", ROOT / "results/reproduced")).resolve()
DATA = OUT / "datasets"
RULER = DATA / "ruler-official"
JSON_DIR = RULER / "scripts/data/synthetic/json"
GENERATED = DATA / "ruler-generated-reduced"
QWEN = Path(os.environ.get("DENSEK3_QWEN_PATH", ROOT / "models/Qwen3.5-4B-Base")).resolve()
PREPARE = RULER / "scripts/data/prepare.py"
TASKS = (
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
LENGTHS = (4096, 8192, 16384, 32768, 65536, 131072)
SAMPLES = 4


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}-{time.time_ns()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_source_corpora() -> dict[str, Any]:
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = Path(
        snapshot_download(
            repo_id="codemaivanngu/NVIDIA-RULER-offline",
            repo_type="dataset",
            allow_patterns=["NVIDIA-RULER-offline.tar.gz"],
        )
    )
    archive = snapshot / "NVIDIA-RULER-offline.tar.gz"
    names = ("english_words.json", "PaulGrahamEssays.json", "squad.json", "hotpotqa.json")
    with tarfile.open(archive, "r:gz") as bundle:
        for name in names:
            member_name = f"NVIDIA-RULER/scripts/data/synthetic/json/{name}"
            member = bundle.getmember(member_name)
            source = bundle.extractfile(member)
            if source is None:
                raise RuntimeError(f"cannot extract {member_name}")
            destination = JSON_DIR / name
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_bytes(source.read())
            temporary.replace(destination)
    return {
        "mirror_dataset": "codemaivanngu/NVIDIA-RULER-offline",
        "archive_sha256": sha256_file(archive),
        "files": {name: sha256_file(JSON_DIR / name) for name in names},
    }


def generate() -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True, trust_remote_code=False)
    outputs: dict[str, Any] = {}
    environment = os.environ.copy()
    environment["PATH"] = f"{Path(sys.executable).parent}:{environment.get('PATH', '')}"
    for length in LENGTHS:
        for task in TASKS:
            destination = GENERATED / str(length) / task / "validation.jsonl"
            if not destination.is_file() or sum(1 for _ in destination.open(encoding="utf-8")) != SAMPLES:
                command = [
                    sys.executable,
                    str(PREPARE),
                    "--save_dir",
                    str(GENERATED / str(length)),
                    "--benchmark",
                    "synthetic",
                    "--task",
                    task,
                    "--subset",
                    "validation",
                    "--tokenizer_path",
                    str(QWEN),
                    "--tokenizer_type",
                    "hf",
                    "--max_seq_length",
                    str(length),
                    "--model_template_type",
                    "base",
                    "--num_samples",
                    str(SAMPLES),
                    "--random_seed",
                    "42",
                ]
                subprocess.run(command, check=True, cwd=PREPARE.parent, env=environment)
            rows = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines() if line]
            if len(rows) != SAMPLES:
                raise RuntimeError(f"wrong generated count {task}/{length}: {len(rows)}")
            actual_lengths = []
            for row in rows:
                prompt = str(row["input"]) + str(row.get("answer_prefix", ""))
                actual_lengths.append(len(tokenizer.encode(prompt, add_special_tokens=True)))
            if max(actual_lengths) > length:
                raise RuntimeError(f"generated prompt exceeds requested length {task}/{length}: {actual_lengths}")
            outputs[f"{length}/{task}"] = {
                "sha256": sha256_file(destination),
                "rows": len(rows),
                "actual_prompt_token_min": min(actual_lengths),
                "actual_prompt_token_max": max(actual_lengths),
            }
            print(f"FINAL_BENCH_RULER_DATA={length}/{task}", flush=True)
    return outputs


def main() -> int:
    if not PREPARE.is_file():
        raise FileNotFoundError(PREPARE)
    sources = write_source_corpora()
    generated = generate()
    report = {
        "stage": "FINAL_CAPABILITY_RULER_REDUCED_DATA_PREPARATION",
        "status": "PASS",
        "result_marker": "FINAL_BENCH_RULER_DATA=PASS",
        "created_at": now(),
        "download_endpoint": "https://hf-mirror.com",
        "official_ruler_commit": subprocess.run(
            ["git", "-C", str(RULER), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip(),
        "samples_per_task_length": SAMPLES,
        "tasks": list(TASKS),
        "lengths": list(LENGTHS),
        "sources": sources,
        "generated": generated,
    }
    atomic_json(DATA / "ruler-generated-reduced-manifest.json", report)
    print(report["result_marker"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
