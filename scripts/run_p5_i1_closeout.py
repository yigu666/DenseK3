"""Run the unique-candidate, no-training P5 I1 heldout and native checkpoint closeout."""

from __future__ import annotations

import argparse
import gc
import json
import math
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from densek3_core.evaluation.p4_runtime import (
    load_densek3_model,
    load_densek3_model_from_pretrained,
)
from densek3_core.kda.reference import kimi_k3_decay
from densek3_core.modeling.modeling_densek3 import DenseK3KDA
from densek3_core.recovery.p5_closeout import (
    P5_ARCHITECTURE,
    apply_native_k3_decay_silu,
    save_native_checkpoint,
    sha256_file,
    tensor_sha256,
    verify_native_checkpoint,
)
from densek3_core.recovery.p5_corpus import PackedTokenDataset, verify_frozen_corpus
from densek3_core.recovery.p5_i1_closeout import (
    evaluate_i1_closeout_gate,
    i1_checkpoint_provenance,
    i1_heldout_gate,
    validate_i1_closeout_config,
    validate_i1_probe,
    verify_i1_replay_binding,
)
from densek3_core.recovery.p5_rethink_v3 import (
    apply_analytic_i1,
)
from densek3_core.transplant.full_model import verify_p4_artifact

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/training/p5-rethink-v3-closeout.yaml"
DEFAULT_P4_ARTIFACT = ROOT / "outputs/densek3-core/checkpoints/p4/densek3-4b-qwen-compat"
DEFAULT_PROBE_REPORT = (
    ROOT / "outputs/densek3-core/manifests/p5/rethink-v3/p5-rethink-v3-initialization-probe.json"
)
DEFAULT_DATA_DIR = ROOT / "data/p5-probe"
DEFAULT_PROMPTS = ROOT / "data/p4-smoke-prompts.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/densek3-core/manifests/p5/i1-closeout"
DEFAULT_CHECKPOINT_DIR = ROOT / "outputs/densek3-core/checkpoints/p5/densek3-4b-k3-decay-silu-i1"
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


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return value


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("P5 I1 closeout requires the authoritative CUDA server")
    return torch.device("cuda")


def load_prompts(path: Path) -> list[dict[str, str]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if {item["id"] for item in records} != {"english", "chinese", "code", "factual", "longer"}:
        raise ValueError("P5 I1 closeout requires exactly the frozen five-category prompts")
    return records


@torch.no_grad()
def corpus_ce(model: torch.nn.Module, dataset: PackedTokenDataset, *, chunk_size: int = 64) -> float:
    if dataset.split != "heldout" or len(dataset) != 97:
        raise ValueError("P5 I1 closeout must open exactly 97 frozen heldout sequences")
    device = next(model.parameters()).device
    losses = []
    model.eval()
    for index in range(len(dataset)):
        tokens = torch.from_numpy(dataset[index].copy()).unsqueeze(0).to(device)
        hidden = model.model(input_ids=tokens, use_cache=False, return_dict=True).last_hidden_state[:, :-1]
        targets = tokens[:, 1:]
        total = targets.numel()
        sequence_loss = torch.zeros((), device=device, dtype=torch.float32)
        for start in range(0, hidden.shape[1], chunk_size):
            end = min(start + chunk_size, hidden.shape[1])
            # Match the frozen P4 heldout baseline: BF16 LM-head GEMM, then FP32 CE.
            logits = F.linear(hidden[:, start:end], model.lm_head.weight).float()
            sequence_loss += F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets[:, start:end].reshape(-1),
                reduction="sum",
            )
        losses.append(float(sequence_loss / total))
        if (index + 1) % 16 == 0 or index + 1 == len(dataset):
            print(f"P5_I1_HELDOUT_PROGRESS={index + 1}/97", flush=True)
    return sum(losses) / len(losses)


@torch.no_grad()
def generation_report(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[dict[str, str]],
) -> dict[str, Any]:
    device = next(model.parameters()).device
    records = []
    for item in prompts:
        encoded = tokenizer(item["prompt"], return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        generated = model.greedy_generate(
            input_ids,
            attention_mask=encoded["attention_mask"].to(device),
            max_new_tokens=8,
        )
        tokens = generated[:, input_ids.shape[1] :].cpu().flatten().tolist()
        counts = Counter(tokens)
        not_collapsed = bool(tokens) and max(counts.values()) / len(tokens) <= 0.75 and len(counts) / len(tokens) >= 0.25
        records.append(
            {
                "id": item["id"],
                "new_tokens": tokens,
                "text": tokenizer.decode(generated[0], skip_special_tokens=False),
                "not_collapsed": not_collapsed,
            }
        )
    return {"records": records, "not_collapsed": all(item["not_collapsed"] for item in records)}


@torch.no_grad()
def bf16_forward_observation(model: torch.nn.Module, tokenizer: Any) -> dict[str, Any]:
    device = next(model.parameters()).device
    encoded = tokenizer(CANONICAL_PROMPT, return_tensors="pt", truncation=True, max_length=64)
    output = model(
        input_ids=encoded["input_ids"].to(device),
        attention_mask=encoded["attention_mask"].to(device),
        use_cache=False,
        return_dict=True,
    )
    return {
        "finite": bool(torch.isfinite(output.logits).all()),
        "shape": list(output.logits.shape),
        "logits_hash": tensor_sha256(output.logits),
    }


def native_semantics(model: torch.nn.Module) -> dict[str, Any]:
    kda = [layer.self_attn for layer in model.model.layers if isinstance(layer.self_attn, DenseK3KDA)]
    probe = torch.tensor([[[[-1000.0, 0.0, 1000.0]]]], device=next(model.parameters()).device)
    a_log = torch.zeros(1, device=probe.device)
    dt_bias = torch.zeros(3, device=probe.device)
    decay = kimi_k3_decay(probe, a_log, dt_bias, lower_bound=-5.0)
    checks = {
        "kda_24": len(kda) == 24,
        "gqa_8": len(model.config.gqa_indices) == 8,
        "k3_mode": model.config.kda_gate_mode == "kimi_k3",
        "lower_bound": model.config.kda_gate_lower_bound == -5.0,
        "decay_range": bool((decay >= -5.0).all() and (decay <= 0.0).all()),
        "silu_gate": model.config.kda_output_gate_activation == "silu"
        and all(module.o_norm.activation == "silu" for module in kda),
        "no_alpha": model.config.kda_output_gate_alpha is None
        and all(module.o_norm.alpha is None for module in kda),
    }
    return {"passed": all(checks.values()), "checks": checks}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--p4-artifact-dir", type=Path, default=DEFAULT_P4_ARTIFACT)
    parser.add_argument("--probe-report", type=Path, default=DEFAULT_PROBE_REPORT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in ("config", "p4_artifact_dir", "probe_report", "data_dir", "prompts", "output_dir", "checkpoint_dir"):
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    heldout_path = args.output_dir / "p5-i1-heldout.json"
    freeze_path = args.output_dir / "p5-i1-freeze-report.json"
    report_path = args.output_dir / "p5-i1-closeout-report.json"
    final_path = args.output_dir / "p5-final-check.json"
    print("P5_I1_CLOSEOUT=RUNNING", flush=True)
    print("TRAINING_PERFORMED=false", flush=True)
    print("I2_USED=false", flush=True)
    print("P6_ALLOWED=false", flush=True)
    try:
        from transformers import AutoTokenizer

        config_checks = validate_i1_closeout_config(load_yaml(args.config))
        probe = load_json(args.probe_report)
        probe_checks = validate_i1_probe(probe)
        corpus = verify_frozen_corpus(args.data_dir)
        if corpus["p5_probe_corpus_frozen"] is not True:
            raise ValueError("P5 I1 closeout corpus verification failed")
        p4_artifact = verify_p4_artifact(args.p4_artifact_dir, verify_tensor_hashes=True)
        device = require_cuda()
        tokenizer = AutoTokenizer.from_pretrained(args.p4_artifact_dir, trust_remote_code=False)
        prompts = load_prompts(args.prompts)

        # This is the only heldout candidate. No donor, I0, I2, or trained checkpoint is loaded.
        model = load_densek3_model(args.p4_artifact_dir, device=device, dtype=torch.bfloat16)
        initialization = apply_analytic_i1(model)
        apply_native_k3_decay_silu(model)
        model.config.p5_initialization = "retention_matched_analytic"
        model.config.p5_initialization_formula_hash = initialization["formula_hash"]
        model.config.p5_recovery_training_required = False
        heldout_resumed = heldout_path.is_file()
        if heldout_resumed:
            heldout = load_json(heldout_path)
            heldout_resume_checks = {
                "stage": heldout.get("stage") == "P5-I1-HELDOUT",
                "candidate": heldout.get("candidate") == "I1"
                and heldout.get("candidate_count") == 1,
                "sequences": heldout.get("sequences") == 97,
                "finite": heldout.get("finite") is True,
                "gate_replay": heldout.get("gate")
                == i1_heldout_gate(float(heldout.get("gate", {}).get("p5_ce"))),
                "no_alternates": heldout.get("other_candidates_evaluated") == [],
                "not_selection": heldout.get("heldout_used_for_selection") is False,
            }
            if not all(heldout_resume_checks.values()):
                raise ValueError(f"P5 I1 heldout report cannot be resumed: {heldout_resume_checks}")
            heldout["resume"] = {"used": True, "checks": heldout_resume_checks}
            print("P5_I1_HELDOUT=RESUMED_WITHOUT_DATA_ACCESS", flush=True)
        else:
            if (args.checkpoint_dir / "p5-checkpoint-manifest.json").is_file():
                raise ValueError("P5 I1 checkpoint exists before the one-time heldout Gate")
            heldout_ce = corpus_ce(model, PackedTokenDataset(args.data_dir / "packed", "heldout"))
            heldout = {
                "stage": "P5-I1-HELDOUT",
                "status": "PASS" if i1_heldout_gate(heldout_ce)["passed"] else "FAIL",
                "candidate": "I1",
                "candidate_count": 1,
                "sequences": 97,
                "finite": math.isfinite(heldout_ce),
                "gate": i1_heldout_gate(heldout_ce),
                "other_candidates_evaluated": [],
                "heldout_used_for_selection": False,
                "resume": {"used": False},
                "completed_at": now(),
            }
            save_json(heldout_path, heldout)
        print(f"P5_I1_HELDOUT_GATE={'PASS' if heldout['gate']['passed'] else 'FAIL'}", flush=True)
        print(f"P5_I1_HELDOUT_CE={heldout['gate']['p5_ce']:.12f}", flush=True)
        if not heldout["gate"]["passed"]:
            final = {
                "stage": "P5",
                "status": "FAIL_HELDOUT_GENERALIZATION",
                "p5_i1_heldout_gate": False,
                "checkpoint_materialized": False,
                "p5_frozen": False,
                "p6_allowed": False,
                "result_marker": "P5_FINAL_CHECK=FAIL",
                "completed_at": now(),
            }
            save_json(final_path, final)
            save_json(report_path, {**final, "heldout": heldout})
            print("P5_FINAL_CHECK=FAIL", flush=True)
            print("P5_FROZEN=false", flush=True)
            print("P6_ALLOWED=false", flush=True)
            return 1

        provenance = i1_checkpoint_provenance(
            initialization,
            initialization_evidence_sha256=sha256_file(args.probe_report),
        )
        manifest_path = args.checkpoint_dir / "p5-checkpoint-manifest.json"
        if manifest_path.is_file():
            if not heldout_resumed:
                raise ValueError("P5 I1 checkpoint cannot predate the resumable heldout report")
            manifest = load_json(manifest_path)
            checkpoint_write_resumed = True
            print("P5_I1_CHECKPOINT_WRITE=RESUMED", flush=True)
        else:
            manifest = save_native_checkpoint(
                model,
                args.p4_artifact_dir,
                args.checkpoint_dir,
                source_p4_parameter_hash=p4_artifact["parameter_hash"],
                initialization_provenance=provenance,
            )
            checkpoint_write_resumed = False
        del model
        clear_cuda()

        # Replay through a second independent formal P4 load.  The formal
        # loader promotes A_log/dt_bias to FP32; reading their safetensors
        # storage dtype directly would not reproduce the actual I1 mapping.
        replay_model = load_densek3_model(
            args.p4_artifact_dir,
            device=device,
            dtype=torch.bfloat16,
        )
        replay_manifest = apply_analytic_i1(replay_model)
        replay_parameters = dict(replay_model.named_parameters())
        replay_tensors = {
            name: replay_parameters[name]
            for name in replay_manifest["changed_tensor_names"]
        }
        mapping_replay = verify_i1_replay_binding(
            manifest,
            replay_tensors,
            replay_manifest,
            expected_p4_parameter_hash=p4_artifact["parameter_hash"],
        )
        if not mapping_replay["passed"]:
            raise ValueError(f"P5 I1 deterministic replay failed: {mapping_replay['checks']}")
        del replay_tensors, replay_parameters, replay_model
        clear_cuda()
        verification = verify_native_checkpoint(args.checkpoint_dir, verify_tensors=True)
        if verification["status"] != "PASS":
            raise ValueError(f"P5 I1 checkpoint verification failed: {verification['checks']}")

        indexed = load_densek3_model(args.checkpoint_dir, device=device, dtype=torch.bfloat16)
        indexed_observation = bf16_forward_observation(indexed, tokenizer)
        del indexed
        clear_cuda()
        reloaded = load_densek3_model_from_pretrained(
            args.checkpoint_dir,
            device=device,
            dtype=torch.bfloat16,
        )
        reloaded_observation = bf16_forward_observation(reloaded, tokenizer)
        save_reload_checks = {
            "logits_exact": indexed_observation["logits_hash"] == reloaded_observation["logits_hash"],
            "shape_exact": indexed_observation["shape"] == reloaded_observation["shape"],
            "finite": indexed_observation["finite"] and reloaded_observation["finite"],
        }
        architecture_checks = {
            "tensor_count": manifest["target_tensor_count"] == 546,
            "parameter_count": manifest["target_unique_parameters"] == 4_224_329_984,
            "topology": len(reloaded.config.kda_indices) == 24 and len(reloaded.config.gqa_indices) == 8,
            "native_parameterization": reloaded.config.kda_parameterization == "p5_k3_decay_silu",
            "analytic_initialization": getattr(reloaded.config, "p5_initialization", None)
            == "retention_matched_analytic",
            "formula_hash": getattr(reloaded.config, "p5_initialization_formula_hash", None)
            == initialization["formula_hash"],
            "training_not_required": getattr(
                reloaded.config,
                "p5_recovery_training_required",
                None,
            )
            is False,
            "provenance": manifest["initialization_provenance"]["type"] == "RETENTION_MATCHED_ANALYTIC",
            "no_probe_source": "probe_a_source" not in manifest,
            "no_probe_b_field": "probe_b_parameters_applied" not in manifest,
        }
        semantics = native_semantics(reloaded)
        generation = generation_report(reloaded, tokenizer, prompts)
        encoded = tokenizer(CANONICAL_PROMPT, return_tensors="pt", truncation=True, max_length=64)
        tokens = encoded["input_ids"].to(device)
        mask = encoded["attention_mask"].to(device)
        cached = reloaded(tokens, attention_mask=mask, use_cache=True, return_dict=True)
        prefill_seen = cached.past_key_values.seen_tokens
        next_token = cached.logits[:, -1].argmax(dim=-1, keepdim=True)
        continuation = reloaded(
            next_token,
            attention_mask=torch.ones((1, tokens.shape[1] + 1), device=device, dtype=mask.dtype),
            past_key_values=cached.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        cache_checks = {
            "prefill_seen_tokens": prefill_seen == tokens.shape[1],
            "continuation_shape": list(continuation.logits.shape) == [1, 1, reloaded.config.vocab_size],
            "continuation_finite": bool(torch.isfinite(continuation.logits).all()),
            "continuation_seen_tokens": continuation.past_key_values.seen_tokens == tokens.shape[1] + 1,
        }
        artifact = {
            "stage": "P5.5-I1",
            "checkpoint_directory": str(args.checkpoint_dir),
            "verification": verification,
            "source_p4_verification": p4_artifact,
            "mapping_replay": mapping_replay,
            "architecture": {"checks": architecture_checks, "passed": all(architecture_checks.values())},
            "native_semantics": semantics,
            "save_reload": {"checks": save_reload_checks, "passed": all(save_reload_checks.values())},
            "bf16_forward": {
                "indexed": indexed_observation,
                "reloaded": reloaded_observation,
                "passed": indexed_observation["finite"] and reloaded_observation["finite"],
            },
            "cache": {"checks": cache_checks, "passed": all(cache_checks.values())},
            "generation": generation,
            "training_performed": False,
            "checkpoint_write_resumed": checkpoint_write_resumed,
            "optimizer_state_present": False,
            "initialization_provenance": provenance,
        }
        artifact["passed"] = all(
            (
                verification["status"] == "PASS",
                p4_artifact["status"] == "PASS" and p4_artifact["tensor_hashes_verified"] == 546,
                mapping_replay["passed"],
                artifact["architecture"]["passed"],
                artifact["native_semantics"]["passed"],
                artifact["save_reload"]["passed"],
                artifact["bf16_forward"]["passed"],
                artifact["cache"]["passed"],
                artifact["generation"]["not_collapsed"],
            )
        )
        save_json(
            freeze_path,
            {"status": "PASS" if artifact["passed"] else "FAIL", **artifact, "completed_at": now()},
        )
        gate = evaluate_i1_closeout_gate(heldout, artifact)
        report = {
            "stage": "P5-I1-CLOSEOUT",
            "status": "PASS" if gate["passed"] else "FAIL",
            "architecture": P5_ARCHITECTURE,
            "initialization": "RETENTION_MATCHED_ANALYTIC",
            "config_checks": config_checks,
            "probe_checks": probe_checks,
            "heldout": heldout,
            "artifact": artifact,
            "gate": gate,
            "research_record": {
                "P5_DIRECT_INHERITANCE": "REJECTED",
                "P5_TRAINED_RECOVERY_V1": "FAILED_GENERALIZATION",
                "P5_TRAINED_RECOVERY_V2": "FAILED_GENERALIZATION",
                "P5_ANALYTIC_RETENTION_MATCHING": "PASS",
            },
            "p5_recovery_training_required": False,
            "p5_frozen": gate["passed"],
            "p6_allowed": gate["passed"],
            "result_marker": f"P5_FINAL_CHECK={'PASS' if gate['passed'] else 'FAIL'}",
            "completed_at": now(),
        }
        save_json(report_path, report)
        final = {
            "stage": "P5",
            "status": report["status"],
            "p5_i1_heldout_gate": heldout["gate"]["passed"],
            "p5_5_freeze": artifact["passed"],
            "architecture": "K3_DECAY_PLUS_SILU",
            "initialization": "RETENTION_MATCHED_ANALYTIC",
            "recovery_training_required": False,
            "p5_frozen": gate["passed"],
            "p6_allowed": gate["passed"],
            "checkpoint_directory": str(args.checkpoint_dir),
            "result_marker": report["result_marker"],
            "completed_at": now(),
        }
        save_json(final_path, final)
        print(f"P5_05_SAVE_RELOAD_FREEZE={'PASS' if final['p5_5_freeze'] else 'FAIL'}", flush=True)
        print(report["result_marker"], flush=True)
        print("P5_ARCHITECTURE=K3_DECAY_PLUS_SILU", flush=True)
        print("P5_INITIALIZATION=RETENTION_MATCHED_ANALYTIC", flush=True)
        print("P5_RECOVERY_TRAINING_REQUIRED=false", flush=True)
        print(f"P5_FROZEN={'true' if gate['passed'] else 'false'}", flush=True)
        print(f"P6_ALLOWED={'true' if gate['passed'] else 'false'}", flush=True)
        return 0 if gate["passed"] else 1
    except Exception as exc:  # noqa: BLE001 - every closeout failure must persist
        failure = {
            "stage": "P5-I1-CLOSEOUT",
            "status": "ERROR",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "training_performed": False,
            "p5_frozen": False,
            "p6_allowed": False,
            "result_marker": "P5_FINAL_CHECK=FAIL",
            "completed_at": now(),
        }
        save_json(report_path, failure)
        save_json(final_path, failure)
        print("P5_FINAL_CHECK=FAIL", flush=True)
        print("P5_FROZEN=false", flush=True)
        print("P6_ALLOWED=false", flush=True)
        print(f"P5_I1_CLOSEOUT_EXCEPTION={type(exc).__name__}: {exc}", flush=True)
        return 1
    finally:
        clear_cuda()


if __name__ == "__main__":
    raise SystemExit(main())
