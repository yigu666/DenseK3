"""Download public evaluation assets without committing dataset content."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download

QWEN_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
LONGBENCH_COMMIT = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
RULER_COMMIT = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"


def clone_at(url: str, destination: Path, commit: str) -> None:
    if not destination.exists():
        subprocess.run(["git", "clone", url, str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "fetch", "--all", "--tags"], check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", "--detach", commit], check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = Path(os.environ.get("DENSEK3_EVAL_OUT", root / "results/reproduced")).resolve()
    data = output / "datasets"
    data.mkdir(parents=True, exist_ok=True)

    if not args.skip_model:
        snapshot_download(
            repo_id="Qwen/Qwen3.5-4B-Base",
            revision=QWEN_REVISION,
            local_dir=root / "models/Qwen3.5-4B-Base",
        )

    load_dataset("cais/mmlu", "all").save_to_disk(data / "mmlu-all")
    load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test").save_to_disk(
        data / "wikitext-103-raw-test"
    )
    load_dataset("THUDM/LongBench-v2", split="train").save_to_disk(data / "longbench-v2-train")
    clone_at("https://github.com/THUDM/LongBench.git", data / "longbench-official", LONGBENCH_COMMIT)
    clone_at("https://github.com/NVIDIA/RULER.git", data / "ruler-official", RULER_COMMIT)
    print(f"DENSEK3_EVALUATION_ASSETS_READY={data}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

