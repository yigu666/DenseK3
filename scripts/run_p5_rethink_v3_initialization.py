"""Run the no-training P5 RETHINK v3 I0/I1 initialization probe."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from densek3_core.evaluation.p4_runtime import load_densek3_model
from densek3_core.recovery.p5_corpus import PackedTokenDataset
from densek3_core.recovery.p5_factorial import apply_factorial_case, first_generation_divergence
from densek3_core.recovery.p5_rethink_v3 import (
    P5_RETHINK_V3_FORMULA_HASH,
    apply_analytic_i1,
)
from densek3_core.recovery.p5_transition import (
    KDAParameterizationObserver,
    hidden_drift,
    iter_finite_reports,
    token_decision_drift,
)
from densek3_core.transplant.full_model import verify_p4_artifact

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/training/p5-rethink-v3-initialization.yaml"
DEFAULT_P4_ARTIFACT = ROOT / "outputs/densek3-core/checkpoints/p4/densek3-4b-qwen-compat"
DEFAULT_DATA_DIR = ROOT / "data/p5-probe"
DEFAULT_PROMPTS = ROOT / "data/p4-smoke-prompts.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/densek3-core/manifests/p5/rethink-v3"
DEFAULT_V2_REPORT = ROOT / "outputs/densek3-core/manifests/p5/rethink-v2/p5-rethink-v2-phase2-final.json"
CANONICAL_PROMPT = (
    "Dense recurrent language models can combine efficient state updates with occasional "
    "softmax attention. 请简要解释这种混合结构，并给出一个代码示例。"
)


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


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--p4-artifact-dir", type=Path, default=DEFAULT_P4_ARTIFACT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--v2-report", type=Path, default=DEFAULT_V2_REPORT)
    return parser.parse_args()


def validate_config(config: dict[str, Any]) -> dict[str, bool]:
    forbidden = config["forbidden"]
    blocked = config["blocked"]
    checks = {
        "stage": config["stage"] == "P5-RETHINK-V3-INITIALIZATION-PROBE",
        "frozen": config["status"] == "FROZEN",
        "priority": config["priority"] == "INITIALIZATION_FIRST",
        "architecture": config["architecture"]
        == {
            "decay_formula": "kimi_k3_lower_bounded_sigmoid",
            "lower_bound": -5.0,
            "output_gate": "silu",
            "topology": "24_kda_8_gqa",
        },
        "only_i0_i1": config["initializations"]["I2"]["status"]
        == "BLOCKED_NOT_REQUIRED_BEFORE_I1_RESULT_REVIEW",
        "full_dev_primary": config["evaluation"]["primary_metric"] == "full_dev_ce"
        and config["evaluation"]["dev_sequences"] == 97,
        "all_mutations_forbidden": all(forbidden.values()),
        "all_downstream_blocked": all(value is False for value in blocked.values()),
    }
    if not all(checks.values()):
        raise ValueError(f"P5 RETHINK v3 config failed: {checks}")
    return checks


def validate_v2_failure(report: dict[str, Any]) -> dict[str, bool]:
    checks = {
        "stage": report["stage"] == "P5-RETHINK-V2-PHASE2",
        "failure": report["status"] == "FAIL_DEV_GATE"
        and report["result_marker"] == "P5_RETHINK_V2_PHASE2=FAIL_DEV_GATE",
        "both_failed": all(report["cells"][cell]["gate"]["decision"] == "FAIL_DEV_GATE" for cell in ("V2-B", "V2-C")),
        "no_candidate": report["selection"]["selected_cell"] is None
        and report["heldout_candidate_count"] == 0,
        "heldout": report["heldout_accessed"] is False,
        "blocked": report["p5_full_allowed"] is False
        and report["p5_frozen"] is False
        and report["p6_allowed"] is False,
    }
    if not all(checks.values()):
        raise ValueError(f"P5 RETHINK v2 failure is not frozen: {checks}")
    return checks


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("P5 RETHINK v3 initialization probe requires the authoritative CUDA server")
    return torch.device("cuda")


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_prompts(path: Path) -> list[dict[str, str]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if {item["id"] for item in records} != {"english", "chinese", "code", "factual", "longer"}:
        raise ValueError("P5 RETHINK v3 must reuse exactly five frozen P4 prompts")
    return records


@torch.no_grad()
def full_dev_ce(model: torch.nn.Module, dev: PackedTokenDataset, *, chunk_size: int = 64) -> float:
    if dev.split != "dev" or len(dev) != 97:
        raise ValueError("P5 RETHINK v3 requires exactly 97 frozen dev sequences")
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    for index in range(len(dev)):
        input_ids = torch.from_numpy(dev[index].copy()).unsqueeze(0).to(next(model.parameters()).device)
        hidden = model.model(input_ids=input_ids, use_cache=False, return_dict=True).last_hidden_state
        targets = input_ids[:, 1:]
        shifted = hidden[:, :-1]
        for start in range(0, shifted.shape[1], chunk_size):
            end = min(start + chunk_size, shifted.shape[1])
            logits = F.linear(shifted[:, start:end].float(), model.lm_head.weight.float())
            total_loss += float(
                F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets[:, start:end].reshape(-1), reduction="sum")
            )
            total_tokens += targets[:, start:end].numel()
        if (index + 1) % 16 == 0 or index + 1 == len(dev):
            print(f"P5_RETHINK_V3_FULL_DEV_PROGRESS={index + 1}/97", flush=True)
    return total_loss / total_tokens


@torch.no_grad()
def generation_observation(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[dict[str, str]],
) -> list[dict[str, Any]]:
    records = []
    device = next(model.parameters()).device
    for item in prompts:
        encoded = tokenizer(item["prompt"], return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        generated = model.greedy_generate(
            input_ids,
            attention_mask=encoded["attention_mask"].to(device),
            max_new_tokens=8,
        )
        new_tokens = generated[:, input_ids.shape[1] :].cpu().flatten().tolist()
        counts = {token: new_tokens.count(token) for token in set(new_tokens)}
        records.append(
            {
                "id": item["id"],
                "input_ids": input_ids.cpu().tolist(),
                "new_tokens": new_tokens,
                "text": tokenizer.decode(generated[0], skip_special_tokens=False),
                "not_collapsed": bool(new_tokens)
                and max(counts.values()) / len(new_tokens) <= 0.75
                and len(counts) / len(new_tokens) >= 0.25,
            }
        )
    return records


def compact_parameterization(layers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "layer_index": item["layer_index"],
            "mode": item["mode"],
            "lower_bound": item["lower_bound"],
            "log_decay": item["log_decay"],
            "retention": item["retention"],
        }
        for item in layers
    ]


@torch.no_grad()
def observe(
    model: torch.nn.Module,
    tokenizer: Any,
    dev: PackedTokenDataset,
    prompts: list[dict[str, str]],
) -> dict[str, Any]:
    device = next(model.parameters()).device
    encoded = tokenizer(CANONICAL_PROMPT, return_tensors="pt", truncation=True, max_length=64)
    input_ids = encoded["input_ids"].to(device)
    with KDAParameterizationObserver(model) as observer:
        output = model(
            input_ids=input_ids,
            attention_mask=encoded["attention_mask"].to(device),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    logits = output.logits.detach().cpu().float()
    hidden_states = tuple(value.detach().cpu() for value in output.hidden_states)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), input_ids.cpu()[:, 1:].reshape(-1))
    parameterization = compact_parameterization(observer.layers)
    dev_ce = full_dev_ce(model, dev)
    generations = generation_observation(model, tokenizer, prompts)
    finite = (
        math.isfinite(dev_ce)
        and math.isfinite(float(ce))
        and bool(torch.isfinite(logits).all())
        and all(bool(torch.isfinite(value).all()) for value in hidden_states)
        and all(iter_finite_reports(parameterization))
    )
    return {
        "input_ids": input_ids.cpu(),
        "logits": logits,
        "hidden_states": hidden_states,
        "public": {
            "full_dev_ce": dev_ce,
            "canonical_ce": float(ce),
            "parameterization": parameterization,
            "generations": generations,
            "generation_not_collapsed": all(item["not_collapsed"] for item in generations),
            "finite": finite,
        },
    }


def compare(reference: dict[str, Any], actual: dict[str, Any], mixer_types: list[str]) -> dict[str, Any]:
    if not torch.equal(reference["input_ids"], actual["input_ids"]):
        raise ValueError("P5 RETHINK v3 canonical inputs differ")
    curve = []
    for layer_index in range(32):
        metric = hidden_drift(reference["hidden_states"][layer_index + 1], actual["hidden_states"][layer_index + 1])
        metric.update({"layer_index": layer_index, "layer_type": mixer_types[layer_index]})
        curve.append(metric)
    generation = []
    for expected, observed in zip(
        reference["public"]["generations"],
        actual["public"]["generations"],
        strict=True,
    ):
        generation.append(
            {
                "id": expected["id"],
                **first_generation_divergence(expected["new_tokens"], observed["new_tokens"]),
                "actual_text": observed["text"],
            }
        )
    return {
        "per_layer_hidden_drift": curve,
        "final_hidden": curve[-1],
        "logits": hidden_drift(reference["logits"], actual["logits"]),
        "token_decisions": token_decision_drift(reference["logits"], actual["logits"]),
        "generation_divergence": generation,
    }


def main() -> int:
    args = parse_args()
    for name in ("config", "p4_artifact_dir", "data_dir", "prompts", "output_dir", "v2_report"):
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "p5-rethink-v3-initialization-probe.json"
    print("P5_RETHINK_V3_INITIALIZATION_PROBE=RUNNING", flush=True)
    print("TRAINING_ALLOWED=false", flush=True)
    print("HELDOUT_ACCESSED=false", flush=True)
    print("P6_ALLOWED=false", flush=True)
    try:
        from transformers import AutoTokenizer

        started = time.perf_counter()
        config = load_yaml(args.config)
        config_checks = validate_config(config)
        v2 = load_json(args.v2_report)
        v2_checks = validate_v2_failure(v2)
        artifact = verify_p4_artifact(args.p4_artifact_dir, verify_tensor_hashes=False)
        device = require_cuda()
        tokenizer = AutoTokenizer.from_pretrained(args.p4_artifact_dir, trust_remote_code=False)
        dev = PackedTokenDataset(args.data_dir / "packed", "dev")
        prompts = load_prompts(args.prompts)

        reference_model = load_densek3_model(args.p4_artifact_dir, device=device, dtype=torch.bfloat16)
        apply_factorial_case(reference_model, "A")
        reference = observe(reference_model, tokenizer, dev, prompts)
        reference_public = reference["public"]

        i0_model = load_densek3_model(args.p4_artifact_dir, device=device, dtype=torch.bfloat16)
        apply_factorial_case(i0_model, "B")
        i0 = observe(i0_model, tokenizer, dev, prompts)
        i0_public = {**i0["public"], **compare(reference, i0, i0_model.config.mixer_types)}
        del i0_model, i0
        clear_cuda()

        i1_model = load_densek3_model(args.p4_artifact_dir, device=device, dtype=torch.bfloat16)
        i1_initialization = apply_analytic_i1(i1_model)
        apply_factorial_case(i1_model, "B")
        i1 = observe(i1_model, tokenizer, dev, prompts)
        i1_public = {**i1["public"], **compare(reference, i1, i1_model.config.mixer_types)}
        del i1_model, i1
        clear_cuda()

        i1_improvement = i0_public["full_dev_ce"] - i1_public["full_dev_ce"]
        selected = "I1" if i1_improvement > 0 else "I0"
        report = {
            "schema_version": 1,
            "stage": "P5-RETHINK-V3-INITIALIZATION-PROBE",
            "status": "PASS",
            "config_checks": config_checks,
            "v2_failure_checks": v2_checks,
            "artifact": artifact,
            "formula_hash": P5_RETHINK_V3_FORMULA_HASH,
            "reference_p4": reference_public,
            "initializations": {
                "I0": {
                    "definition": "CURRENT_DIRECT_PARAMETER_INHERITANCE",
                    "training_performed": False,
                    "calibration_used": False,
                    **i0_public,
                },
                "I1": {
                    "definition": "RETENTION_MATCHED_ANALYTIC_LOCAL_VALUE_AND_SLOPE",
                    "training_performed": False,
                    "calibration_used": False,
                    "initialization_manifest": i1_initialization,
                    **i1_public,
                },
                "I2": {
                    "status": "BLOCKED_NOT_RUN",
                    "reason": "I1_RESULT_REQUIRES_REVIEW_BEFORE_ANY_CALIBRATION_FITTING",
                },
            },
            "comparison": {
                "primary_metric": "full_dev_ce",
                "i0_full_dev_ce": i0_public["full_dev_ce"],
                "i1_full_dev_ce": i1_public["full_dev_ce"],
                "i1_improvement_vs_i0": i1_improvement,
                "selected_zero_shot_initialization": selected,
                "selection_is_diagnostic_only": True,
            },
            "training_performed": False,
            "optimizer_created": False,
            "heldout_accessed": False,
            "payload_splits_opened": ["dev"],
            "i2_allowed": False,
            "p5_v3_training_allowed": False,
            "p5_full_allowed": False,
            "p5_frozen": False,
            "p6_allowed": False,
            "result_marker": "P5_RETHINK_V3_INITIALIZATION_PROBE=PASS",
            "elapsed_seconds": time.perf_counter() - started,
            "completed_at": now(),
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
            },
        }
        if not (reference_public["finite"] and i0_public["finite"] and i1_public["finite"]):
            raise ValueError("P5 RETHINK v3 initialization probe produced non-finite evidence")
        save_json(report_path, report)
        print(report["result_marker"], flush=True)
        print(f"P5_RETHINK_V3_I0_FULL_DEV_CE={i0_public['full_dev_ce']:.12f}", flush=True)
        print(f"P5_RETHINK_V3_I1_FULL_DEV_CE={i1_public['full_dev_ce']:.12f}", flush=True)
        print(f"P5_RETHINK_V3_I1_IMPROVEMENT={i1_improvement:.12f}", flush=True)
        print(f"P5_RETHINK_V3_SELECTED_ZERO_SHOT={selected}", flush=True)
        print("TRAINING_ALLOWED=false", flush=True)
        print("HELDOUT_ACCESSED=false", flush=True)
        print("P6_ALLOWED=false", flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001 - every technical failure must persist
        failure = {
            "stage": "P5-RETHINK-V3-INITIALIZATION-PROBE",
            "status": "ERROR",
            "technical_execution": "INVALID",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "training_performed": False,
            "heldout_accessed": False,
            "p5_v3_training_allowed": False,
            "p5_full_allowed": False,
            "p5_frozen": False,
            "p6_allowed": False,
            "result_marker": "P5_RETHINK_V3_INITIALIZATION_PROBE=ERROR",
            "completed_at": now(),
        }
        save_json(report_path, failure)
        print(failure["result_marker"], flush=True)
        print(f"P5_RETHINK_V3_EXCEPTION={type(exc).__name__}: {exc}", flush=True)
        print("TRAINING_ALLOWED=false", flush=True)
        print("HELDOUT_ACCESSED=false", flush=True)
        print("P6_ALLOWED=false", flush=True)
        return 1
    finally:
        clear_cuda()


if __name__ == "__main__":
    raise SystemExit(main())
