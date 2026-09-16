"""Run the bounded 128K-token Titan P8 routing-learnability probe."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
TITAN = ROOT / "titan"
sys.path.insert(0, str(TITAN))

from attnres import (  # noqa: E402
    BlockAttnResReference,
    disable_block_attnres_runtime,
    enable_block_attnres_runtime,
)
from kda_sm75_backend import enable_sm75_fused_recurrent_fallback  # noqa: E402
from p7_latent_cache import (  # noqa: E402
    disable_p7_latent_cache_runtime,
    enable_p7_latent_cache_runtime,
)
from run_p8_reference import (  # noqa: E402
    CANDIDATE,
    DATA,
    cache_equivalence,
    corpus_ce,
    generation,
    gpu_start_state,
)

from densek3_core.evaluation.p4_runtime import load_densek3_model  # noqa: E402
from densek3_core.recovery.p5_corpus import (  # noqa: E402
    P5_SEQUENCE_LENGTH,
    PackedTokenDataset,
    verify_frozen_corpus,
)
from densek3_core.recovery.p5_probe import (  # noqa: E402
    chunked_causal_lm_loss,
    gradient_report,
    learning_rate_at_step,
)
from densek3_core.recovery.p6_closeout import (  # noqa: E402
    checkpoint_identity,
    sha256_file,
)

CONFIG = TITAN / "configs/p8-titan-block-attnres.yaml"
CACHE_CONTRACT = TITAN / "configs/p6-titan-fp16-cache-compat-2026-08-19-01.yaml"
REFERENCE_FINAL = TITAN / "manifests/reproduction/p8/p8-reference-final-check.json"
ZERO_REPORT = TITAN / "manifests/reproduction/p8/p8-2-zero-init-shock.json"
REPORT_DIR = TITAN / "manifests/reproduction/p8/routing-probe"
CHECKPOINT_DIR = ROOT / "outputs/densek3-core/checkpoints/p8/routing-probe"
CANDIDATE_DIR = ROOT / "outputs/densek3-core/checkpoints/p8/densek3-4b-block-attnres-titan"
FINAL = REPORT_DIR / "p8-routing-probe-final.json"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def tensor_dict_sha256(values: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def runtime_state(runtime: BlockAttnResReference) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().contiguous()
        for name, value in runtime.state_dict().items()
    }


def save_checkpoint(
    step: int,
    effective_tokens: int,
    runtime: BlockAttnResReference,
    optimizer: torch.optim.Optimizer,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    directory = CHECKPOINT_DIR / f"step-{step:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    state = runtime_state(runtime)
    weights = directory / "attnres.safetensors"
    save_file(state, str(weights))
    optimizer_path = directory / "optimizer.pt"
    torch.save(optimizer.state_dict(), optimizer_path)
    manifest = {
        "stage": "TITAN_P8_3_ROUTING_PROBE",
        "step": step,
        "effective_tokens": effective_tokens,
        "path": str(directory.resolve()),
        "attnres_sha256": sha256_file(weights),
        "attnres_tensor_hash": tensor_dict_sha256(state),
        "optimizer_sha256": sha256_file(optimizer_path),
        "evaluation": evaluation,
        "source_checkpoint": str(CANDIDATE.resolve()),
        "source_weights_modified": False,
    }
    atomic_json(directory / "checkpoint-manifest.json", manifest)
    return manifest


@torch.inference_mode()
def routing_diagnostic(
    model: torch.nn.Module,
    runtime: BlockAttnResReference,
    tokens: torch.Tensor,
) -> dict[str, Any]:
    runtime.clear_diagnostics()
    runtime.capture_diagnostics = True
    runtime.capture_tensor_metrics = False
    runtime.capture_activations = False
    enable_block_attnres_runtime(model, runtime)
    try:
        hidden = model.model(tokens, use_cache=False, return_dict=True).last_hidden_state
    finally:
        disable_block_attnres_runtime(model)
        runtime.capture_diagnostics = False
    records = list(runtime.routing_records)
    if len(records) != 65:
        raise RuntimeError(f"Expected 65 routing records, observed {len(records)}")
    query = runtime.query_report()
    entropy_gaps = [float(item["entropy_gap_mean"]) for item in records]
    alpha_excess = [
        float(item["alpha_max_mean"]) - 1.0 / int(item["source_count"])
        for item in records
    ]
    content_std = [float(item["content_std"]) for item in records]
    summary = {
        "query": query,
        "routing_records": records,
        "entropy_gap_mean": float(np.mean(entropy_gaps)),
        "entropy_gap_max": max(entropy_gaps),
        "alpha_excess_mean": float(np.mean(alpha_excess)),
        "alpha_excess_max": max(alpha_excess),
        "content_std_max": max(content_std),
        "nonuniform_sites": sum(value > 1e-6 for value in entropy_gaps),
        "content_dependent_sites": sum(value > 1e-7 for value in content_std),
        "finite": bool(torch.isfinite(hidden).all()),
        "hidden_rms": float(hidden.float().square().mean().sqrt()),
    }
    runtime.clear_diagnostics()
    return summary


def trainable_contract(
    model: torch.nn.Module,
    runtime: BlockAttnResReference,
) -> dict[str, Any]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in runtime.parameters():
        parameter.requires_grad_(True)
    source_trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    runtime_trainable = [name for name, value in runtime.named_parameters() if value.requires_grad]
    return {
        "source_trainable": source_trainable,
        "source_frozen": not source_trainable,
        "runtime_trainable_count": sum(
            value.numel() for value in runtime.parameters() if value.requires_grad
        ),
        "runtime_trainable_tensors": len(runtime_trainable),
        "query_tensors": sum(name.endswith("query") for name in runtime_trainable),
        "norm_tensors": sum(name.endswith("source_norm.weight") for name in runtime_trainable),
        "scope_exact": len(runtime_trainable) == 130,
    }


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    runtime: BlockAttnResReference,
    tokenizer: Any,
    dev: PackedTokenDataset,
    diagnostic_tokens: torch.Tensor,
    cache_limits: dict[str, Any],
    *,
    step: int,
    effective_tokens: int,
    known_dev_ce: float | None = None,
) -> dict[str, Any]:
    model.eval()
    runtime.eval()
    if known_dev_ce is None:
        enable_block_attnres_runtime(model, runtime)
        try:
            dev_ce = corpus_ce(model, dev, label=f"ROUTING_{effective_tokens}")
        finally:
            disable_block_attnres_runtime(model)
    else:
        dev_ce = known_dev_ce
    routing = routing_diagnostic(model, runtime, diagnostic_tokens)
    generated = generation(model, tokenizer, runtime)
    cache = cache_equivalence(model, runtime, cache_limits)
    return {
        "step": step,
        "effective_tokens": effective_tokens,
        "full_dev_ce": dev_ce,
        "routing": routing,
        "generation": generated,
        "generation_finite": all(item["finite"] for item in generated),
        "cache": cache,
        "finite": math.isfinite(dev_ce) and routing["finite"],
    }


def materialize_candidate(
    runtime: BlockAttnResReference,
    best: dict[str, Any],
    source_identity: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    state = runtime_state(runtime)
    weights = CANDIDATE_DIR / "attnres.safetensors"
    save_file(state, str(weights))
    reloaded = BlockAttnResReference(
        runtime.hidden_size,
        runtime.eps,
        block_size_residuals=runtime.block_size_residuals,
    )
    reloaded.load_state_dict(load_file(str(weights)))
    reload_state = runtime_state(reloaded)
    checks = {
        "save_reload_exact": tensor_dict_sha256(state) == tensor_dict_sha256(reload_state),
        "source_checkpoint_identity": checkpoint_identity(CANDIDATE) == source_identity,
        "source_weights_copied": False,
        "rank_unchanged": config["frozen"]["mla_rank"] == 512,
        "p7_cache_required": config["frozen"]["p7_true_latent_cache"] is True,
        "heldout_accessed": False,
    }
    manifest = {
        "stage": "TITAN_P8_4_PROVISIONAL_CANDIDATE",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "architecture": "24_KDA_PLUS_8_MLA_PLUS_BLOCK_ATTNRES_N8",
        "source_checkpoint": str(CANDIDATE.resolve()),
        "source_checkpoint_identity": source_identity,
        "attnres_weights": str(weights.resolve()),
        "attnres_sha256": sha256_file(weights),
        "attnres_tensor_hash": tensor_dict_sha256(state),
        "best_evaluation": best,
        "checks": checks,
        "titan_only": True,
        "canonical_bf16_replay": "PENDING",
        "canonical_frozen": False,
    }
    atomic_json(CANDIDATE_DIR / "p8-candidate-manifest.json", manifest)
    return manifest


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    state = gpu_start_state()
    envelope: dict[str, Any] = {
        "stage": "TITAN_P8_3_ROUTING_LEARNABILITY",
        "status": "RUNNING",
        "started_at": now(),
        "gpu": state,
        "heldout_accessed": False,
        "source_weights_modified": False,
        "p8_3b_extension_allowed": False,
        "p9_titan_migration_allowed": False,
    }
    atomic_json(FINAL, envelope)
    try:
        if not state["idle"]:
            envelope["status"] = "DEFERRED_GPU_NOT_IDLE"
            atomic_json(FINAL, envelope)
            print("TITAN_P8_3=DEFERRED_GPU_NOT_IDLE")
            return 3
        config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        reference = load_json(REFERENCE_FINAL)
        zero = load_json(ZERO_REPORT)
        cache_limits = yaml.safe_load(CACHE_CONTRACT.read_text(encoding="utf-8"))[
            "evaluation"
        ]["fallback_requires_all"]
        frozen_corpus = verify_frozen_corpus(DATA)
        probe = config["training_probe"]
        prerequisite_checks = {
            "reference_pass": reference["status"] == "PASS",
            "p8_2_strong_go": zero["decision"] == "STRONG_GO",
            "training_allowed": reference["p8_3_training_allowed"] is True,
            "corpus_frozen": frozen_corpus["p5_probe_corpus_frozen"] is True,
            "sequence_length": probe["sequence_length"] == P5_SEQUENCE_LENGTH,
            "budget": probe["max_effective_tokens"] == 131_072,
            "range": probe["train_sequence_range"] == [0, 64],
            "objective": probe["objective"] == "causal_lm_ce_only",
            "heldout_forbidden": probe["heldout_accessed"] is False,
        }
        if not all(prerequisite_checks.values()):
            raise ValueError(f"P8.3 prerequisites failed: {prerequisite_checks}")
        random.seed(20260820)
        np.random.seed(20260820)
        torch.manual_seed(20260820)
        torch.cuda.manual_seed_all(20260820)
        enable_sm75_fused_recurrent_fallback()
        enable_p7_latent_cache_runtime()
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            ROOT / "outputs/densek3-core/checkpoints/p5/densek3-4b-k3-decay-silu-i1",
            trust_remote_code=False,
        )
        model = load_densek3_model(CANDIDATE, device="cuda", dtype=torch.float16)
        runtime = BlockAttnResReference(
            model.config.hidden_size,
            model.config.rms_norm_eps,
            block_size_residuals=int(config["architecture"]["block_size_residuals"]),
        ).cuda()
        scope = trainable_contract(model, runtime)
        if not (scope["source_frozen"] and scope["scope_exact"]):
            raise ValueError(f"P8.3 trainable scope failed: {scope}")
        source_identity = checkpoint_identity(CANDIDATE)
        optimizer = torch.optim.AdamW(
            runtime.parameters(),
            lr=float(probe["learning_rate"]),
            weight_decay=float(probe["weight_decay"]),
        )
        if optimizer.state:
            raise RuntimeError("P8.3 optimizer must start fresh")
        scaler = torch.amp.GradScaler("cuda")
        train = PackedTokenDataset(DATA / "packed", "train")
        dev = PackedTokenDataset(DATA / "packed", "dev")
        diagnostic_tokens = torch.from_numpy(dev[0][:128].copy()).unsqueeze(0).cuda()
        target_steps = {
            int(tokens) // P5_SEQUENCE_LENGTH
            for tokens in probe["evaluation_effective_tokens"]
        }
        prior_zero_ce = float(zero["p8_zero_full_dev_ce"])
        evaluations = [
            evaluate(
                model,
                runtime,
                tokenizer,
                dev,
                diagnostic_tokens,
                cache_limits,
                step=0,
                effective_tokens=0,
            )
        ]
        zero_ce = float(evaluations[0]["full_dev_ce"])
        zero_replay_delta = zero_ce - prior_zero_ce
        zero_replay_consistent = abs(zero_replay_delta) <= 1.0e-3
        if not zero_replay_consistent:
            raise ValueError(
                "P8.3 streaming-runtime zero-init replay diverged from frozen P8.2: "
                f"prior={prior_zero_ce:.12f}, current={zero_ce:.12f}, "
                f"delta={zero_replay_delta:+.12f}"
            )
        checkpoints = [save_checkpoint(0, 0, runtime, optimizer, evaluations[0])]
        training_curve = []
        total_steps = int(probe["max_effective_tokens"]) // P5_SEQUENCE_LENGTH
        train_start, train_end = (int(value) for value in probe["train_sequence_range"])
        if train_end - train_start != total_steps:
            raise ValueError("P8.3 train range and token budget disagree")
        model.train()
        model.model.gradient_checkpointing = True
        runtime.train()
        started = time.perf_counter()
        for step in range(total_steps):
            tokens = torch.from_numpy(train[train_start + step].copy()).unsqueeze(0).cuda()
            lr = learning_rate_at_step(
                step,
                total_steps,
                base_learning_rate=float(probe["learning_rate"]),
                warmup_ratio=float(probe["warmup_ratio"]),
                minimum_ratio=float(probe["minimum_learning_rate_ratio"]),
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            enable_block_attnres_runtime(model, runtime)
            try:
                hidden = model.model(tokens, use_cache=False, return_dict=True).last_hidden_state
            finally:
                disable_block_attnres_runtime(model)
            loss = chunked_causal_lm_loss(
                hidden,
                tokens,
                model.lm_head.weight,
                chunk_size=64,
                checkpoint_chunks=True,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite P8 loss at step {step + 1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradients = gradient_report(
                [value for value in runtime.parameters() if value.requires_grad],
                float(probe["gradient_clip_norm"]),
            )
            if not gradients["passed"]:
                raise FloatingPointError(f"Invalid P8 gradients: {gradients}")
            scaler.step(optimizer)
            scaler.update()
            completed = step + 1
            effective_tokens = completed * P5_SEQUENCE_LENGTH
            training_curve.append(
                {
                    "step": completed,
                    "effective_tokens": effective_tokens,
                    "sequence_index": train_start + step,
                    "learning_rate": lr,
                    "causal_lm_ce": float(loss.detach()),
                    "gradient": gradients,
                    "query_norm_max": runtime.query_report()["max"],
                }
            )
            del tokens, hidden, loss
            if completed % 4 == 0:
                print(
                    f"TITAN_P8_3_TRAIN STEP={completed}/{total_steps} "
                    f"TOKENS={effective_tokens} CE={training_curve[-1]['causal_lm_ce']:.6f}",
                    flush=True,
                )
            if completed in target_steps:
                gc.collect()
                torch.cuda.empty_cache()
                evaluation = evaluate(
                    model,
                    runtime,
                    tokenizer,
                    dev,
                    diagnostic_tokens,
                    cache_limits,
                    step=completed,
                    effective_tokens=effective_tokens,
                )
                evaluations.append(evaluation)
                checkpoints.append(
                    save_checkpoint(completed, effective_tokens, runtime, optimizer, evaluation)
                )
                print(
                    f"TITAN_P8_3_EVAL TOKENS={effective_tokens} "
                    f"DEV_CE={evaluation['full_dev_ce']:.9f} "
                    f"QUERY_MAX={evaluation['routing']['query']['max']:.6g} "
                    f"NONUNIFORM_SITES={evaluation['routing']['nonuniform_sites']}",
                    flush=True,
                )
                model.train()
                runtime.train()
        best_index = min(range(len(evaluations)), key=lambda index: evaluations[index]["full_dev_ce"])
        best = evaluations[best_index]
        best_checkpoint = checkpoints[best_index]
        best_state = load_file(str(Path(best_checkpoint["path"]) / "attnres.safetensors"))
        runtime.load_state_dict(best_state)
        improvement = best["full_dev_ce"] - zero_ce
        routing = best["routing"]
        routing_learned = (
            routing["query"]["max"] > 0.0
            and routing["nonuniform_sites"] > 0
            and routing["content_dependent_sites"] > 0
            and routing["alpha_excess_max"] > 0.0
        )
        safety = (
            best["finite"]
            and best["generation_finite"]
            and best["cache"]["status"] == "PASS"
            and checkpoint_identity(CANDIDATE) == source_identity
        )
        if improvement <= -0.01 and routing_learned and safety:
            decision = "GO"
        elif -0.01 < improvement <= 0.01 and routing_learned and safety:
            decision = "WEAK_MODIFY"
        elif improvement > 0.03 or not routing_learned or not safety:
            decision = "DROP_NO_GO"
        else:
            decision = "REVIEW_REQUIRED"
        candidate = None
        if decision == "GO":
            candidate = materialize_candidate(runtime, best, source_identity, config)
            if candidate["status"] != "PASS":
                raise ValueError("P8.4 provisional candidate materialization failed")
        report = {
            "stage": "TITAN_P8_3_ROUTING_LEARNABILITY",
            "status": "PASS" if safety else "FAIL",
            "decision": decision,
            "prerequisite_checks": prerequisite_checks,
            "trainable_scope": scope,
            "optimizer": {
                "type": "AdamW",
                "fresh": True,
                "learning_rate": probe["learning_rate"],
                "weight_decay": probe["weight_decay"],
                "warmup_ratio": probe["warmup_ratio"],
                "minimum_learning_rate_ratio": probe["minimum_learning_rate_ratio"],
                "gradient_clip_norm": probe["gradient_clip_norm"],
                "grad_scaler": True,
            },
            "objective": "causal_lm_ce_only",
            "training_curve": training_curve,
            "evaluations": evaluations,
            "checkpoints": checkpoints,
            "best_checkpoint": best_checkpoint,
            "best_evaluation": best,
            "zero_init_dev_ce": zero_ce,
            "prior_p8_2_zero_init_dev_ce": prior_zero_ce,
            "zero_init_replay_delta_ce": zero_replay_delta,
            "zero_init_replay_consistent": zero_replay_consistent,
            "best_delta_ce_vs_zero": improvement,
            "routing_learned": routing_learned,
            "safety": safety,
            "source_checkpoint_identity_before": source_identity,
            "source_checkpoint_identity_after": checkpoint_identity(CANDIDATE),
            "candidate": candidate,
            "wall_seconds": time.perf_counter() - started,
            "heldout_accessed": False,
            "source_weights_modified": False,
            "extension_performed": False,
        }
        atomic_json(REPORT_DIR / "p8-routing-probe.json", report)
        envelope.update(
            {
                "status": report["status"],
                "completed_at": now(),
                "decision": decision,
                "routing_learned": routing_learned,
                "best_checkpoint": best_checkpoint["path"],
                "best_effective_tokens": best["effective_tokens"],
                "best_full_dev_ce": best["full_dev_ce"],
                "best_delta_ce_vs_zero": improvement,
                "p8_3b_extension_allowed": decision == "WEAK_MODIFY",
                "p9_titan_migration_allowed": decision == "GO",
                "p8_canonical_frozen": False,
                "p8_formal_status": "TITAN_PROVISIONAL_PENDING_NATIVE_BF16",
            }
        )
        atomic_json(FINAL, envelope)
        print(f"TITAN_P8_3_ROUTING_LEARNABILITY={report['status']}", flush=True)
        print(f"TITAN_P8_3_DECISION={decision}", flush=True)
        print(f"P9_TITAN_MIGRATION_ALLOWED={str(decision == 'GO').lower()}", flush=True)
        print("P9_CANONICAL_ALLOWED=false", flush=True)
        return 0 if report["status"] == "PASS" else 1
    except Exception as exc:
        envelope.update(
            {
                "status": "ERROR",
                "completed_at": now(),
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        atomic_json(FINAL, envelope)
        print(f"TITAN_P8_3_EXCEPTION={type(exc).__name__}: {exc}", flush=True)
        return 1
    finally:
        disable_p7_latent_cache_runtime()


if __name__ == "__main__":
    raise SystemExit(main())
