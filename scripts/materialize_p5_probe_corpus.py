"""Materialize and freeze the exact 2.396M-token P5.3-Probe corpus."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from densek3_core.recovery.p5_corpus import materialize_corpus_twice, verify_frozen_corpus

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data/p5-probe"
DEFAULT_TOKENIZER_DIR = ROOT / "outputs/densek3-core/checkpoints/p4/densek3-4b-qwen-compat"
DEFAULT_OUTPUT = ROOT / "outputs/densek3-core/manifests/p5/p5-probe-corpus-gate.json"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report: dict[str, Any] = {
        "stage": "P5-Probe-corpus",
        "status": "FAIL",
        "p5_probe_corpus_frozen": False,
        "p5_3_probe_training_allowed": False,
    }
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir.resolve(), trust_remote_code=False)
        materialization = materialize_corpus_twice(
            args.data_dir.resolve(),
            tokenizer,
            args.tokenizer_dir.resolve(),
        )
        report = verify_frozen_corpus(args.data_dir.resolve())
        report["materialization"] = {
            "packed_dir": materialization["packed_dir"],
            "sha256sums_path": materialization["sha256sums_path"],
            "re_materialization_hash_stable": materialization["re_materialization_hash_stable"],
        }
        report["checks"]["re_materialization_hash_stable"] = materialization[
            "re_materialization_hash_stable"
        ]
        report["status"] = "PASS" if all(report["checks"].values()) else "FAIL"
        report["p5_probe_corpus_frozen"] = report["status"] == "PASS"
        report["p5_3_probe_training_allowed"] = report["status"] == "PASS"
        report["result_marker"] = f"P5_PROBE_CORPUS_MATERIALIZATION={report['status']}"
        report["completed_at"] = now()
        if report["status"] != "PASS":
            raise ValueError(f"P5 Probe corpus Gate failed: {report['checks']}")
    except Exception as exc:  # noqa: BLE001 - corpus Gate must always persist its failure
        report.update(
            {
                "status": "FAIL",
                "p5_probe_corpus_frozen": False,
                "p5_3_probe_training_allowed": False,
                "result_marker": "P5_PROBE_CORPUS_MATERIALIZATION=FAIL",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
                "completed_at": now(),
            }
        )
    save_json(args.output.resolve(), report)
    print(report["result_marker"], flush=True)
    print(f"P5_PROBE_CORPUS_FROZEN={str(report['p5_probe_corpus_frozen']).lower()}", flush=True)
    print(
        f"P5_3_PROBE_TRAINING_ALLOWED={str(report['p5_3_probe_training_allowed']).lower()}",
        flush=True,
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
