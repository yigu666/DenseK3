"""Run the frozen P6.2b-1 L3/L23 attention-function calibration pilot."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from densek3_core.evaluation.p4_runtime import load_densek3_model
from densek3_core.modeling.modeling_densek3 import DenseK3GQA
from densek3_core.recovery.p5_corpus import PackedTokenDataset, verify_frozen_corpus
from densek3_core.recovery.p6_function_calibration import (
    TRAINABLE_PARAMETER_NAMES,
    capture_frozen_p5_inputs,
    differentiable_function_outputs,
    freeze_local_calibration_scope,
    local_go_no_go,
    local_validation_metrics,
    normalized_mse,
    select_document_disjoint_sequences,
)
from densek3_core.recovery.p6_initialization import (
    activation_weighted_rank_factorization,
    apply_rmsnorm_scale,
    covariance_from_batches,
    fit_scale_matched_rmsnorm,
)
from densek3_core.recovery.p6_mla_probe import (
    P6NoPEFullAttention,
    P6NoPEMLA,
    tensor_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/architecture/p6-attention-function-local-calibration.yaml"
DEFAULT_P5 = ROOT / "outputs/densek3-core/checkpoints/p5/densek3-4b-k3-decay-silu-i1"
DEFAULT_P6_2 = ROOT / "outputs/densek3-core/manifests/p6/p6-2/p6-2-scaleout.json"
DEFAULT_P6_2A = ROOT / "outputs/densek3-core/manifests/p6/p6-2a/p6-2a-capacity.json"
DEFAULT_DATA = ROOT / "data/p5-probe"
DEFAULT_OUTPUT = ROOT / "outputs/densek3-core/manifests/p6/p6-2b"
DEFAULT_ARTIFACT = ROOT / "outputs/densek3-core/artifacts/p6/p6-2b-pilot"
PILOT_LAYERS = [3, 23]
INITIALIZATION = "ACTIVATION_WEIGHTED_RANK512_PLUS_SCALE_MATCHED_LATENT_NORM"


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


def validate_config(config: dict[str, Any]) -> dict[str, bool]:
    checks = {
        "stage": config["stage"] == "P6.2b-1" and config["status"] == "FROZEN",
        "source": config["source"]["initialization"] == INITIALIZATION
        and config["source"]["teacher"] == "FULL_RANK_NOPE_B",
        "pilot": config["scope"]["pilot_layers"] == PILOT_LAYERS,
        "architecture": config["scope"]["formal_kv_lora_rank"] == 512
        and config["scope"]["no_pe"] is True
        and config["scope"]["qk_norm"] is False,
        "scope": config["scope"]["global_recovery_training"] is False
        and config["scope"]["local_calibration_fitting"] is True
        and config["scope"]["frozen_semantics"]
        == [
            "q_projection",
            "gate_projection",
            "output_projection",
            "all_remaining_model_parameters",
        ],
        "data": config["data"]["split"] == "train"
        and config["data"]["sequence_prefix"] == 256
        and config["data"]["fit_sequences"] == 128
        and config["data"]["validation_sequences"] == 32
        and config["data"]["fit_activation_tokens_per_layer"] == 32_768
        and config["data"]["validation_activation_tokens_per_layer"] == 8_192
        and config["data"]["document_disjoint"] is True
        and config["data"]["full_dev_accessed"] is False
        and config["data"]["heldout_accessed"] is False
        and config["data"]["canonical_accessed"] is False,
        "optimizer": config["optimization"]
        == {
            "optimizer": "AdamW",
            "learning_rate": 0.0002,
            "betas": [0.9, 0.95],
            "epsilon": 1e-8,
            "weight_decay": 0.0,
            "gradient_clip": 1.0,
            "maximum_steps_per_layer": 128,
            "evaluation_steps": [0, 32, 64, 96, 128],
            "scheduler": "constant",
            "seed": 20260814,
            "sweep": False,
            "fresh_optimizer_per_layer": True,
            "master_parameter_dtype": "float32",
            "formal_validation_dtype": "bfloat16",
        },
        "loss": config["loss"]
        == {
            "attention_core_normalized_mse_weight": 1.0,
            "mixer_output_normalized_mse_weight": 1.0,
            "kv_reconstruction_weight": 0.0,
            "lm_ce_weight": 0.0,
            "full_model_hidden_weight": 0.0,
            "selection_metric": "validation_attention_core_plus_mixer_relative_l2",
        },
        "artifact": config["artifact"]["local_trainable_only"] is True
        and config["artifact"]["full_model_checkpoint"] is False
        and config["artifact"]["optimizer_state"] is False,
        "forbidden": all(config["forbidden"].values()),
    }
    if not all(checks.values()):
        raise ValueError(f"P6.2b-1 frozen config failed: {checks}")
    return checks


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trainable_state(module: P6NoPEMLA) -> dict[str, torch.Tensor]:
    named = dict(module.named_parameters())
    return {
        name: named[name].detach().cpu().float().clone() for name in TRAINABLE_PARAMETER_NAMES
    }


@torch.no_grad()
def load_trainable_state(module: P6NoPEMLA, state: dict[str, torch.Tensor]) -> None:
    named = dict(module.named_parameters())
    if set(state) != set(TRAINABLE_PARAMETER_NAMES):
        raise ValueError("P6.2b trainable state keys violate the frozen scope")
    for name in TRAINABLE_PARAMETER_NAMES:
        named[name].copy_(state[name].to(device=named[name].device, dtype=named[name].dtype))


def selection_key(metrics: dict[str, Any]) -> tuple[float, float, float]:
    core = float(metrics["attention_core"]["relative_l2_error"])
    mixer = float(metrics["mixer_output"]["relative_l2_error"])
    return core + mixer, mixer, core


def evaluate_state(
    donor: DenseK3GQA,
    initial_student: P6NoPEMLA,
    state: dict[str, torch.Tensor],
    residual_batches: list[torch.Tensor],
    normalized_batches: list[torch.Tensor],
) -> dict[str, Any]:
    student = copy.deepcopy(initial_student).eval()
    load_trainable_state(student, state)
    metrics = local_validation_metrics(donor, student, residual_batches, normalized_batches)
    del student
    torch.cuda.empty_cache()
    return metrics


def calibrate_layer(
    donor: DenseK3GQA,
    initial_student: P6NoPEMLA,
    fit_normalized: list[torch.Tensor],
    validation_residual: list[torch.Tensor],
    validation_normalized: list[torch.Tensor],
    optimization: dict[str, Any],
) -> dict[str, Any]:
    device = next(initial_student.parameters()).device
    master = copy.deepcopy(initial_student).float().eval()
    scope = freeze_local_calibration_scope(master)
    teacher = P6NoPEFullAttention(donor, split_q_gate=False, apply_qk_norm=True).float().eval()
    teacher.requires_grad_(False)
    frozen_names = ("q_proj.weight", "g_proj.weight", "o_proj.weight")
    frozen_before = {
        name: tensor_sha256(dict(master.named_parameters())[name]) for name in frozen_names
    }
    named = dict(master.named_parameters())
    optimizer = torch.optim.AdamW(
        [named[name] for name in TRAINABLE_PARAMETER_NAMES],
        lr=optimization["learning_rate"],
        betas=tuple(optimization["betas"]),
        eps=optimization["epsilon"],
        weight_decay=optimization["weight_decay"],
    )
    initial_state = trainable_state(master)
    baseline = evaluate_state(
        donor,
        initial_student,
        initial_state,
        validation_residual,
        validation_normalized,
    )
    best_state = initial_state
    best_metrics = baseline
    best_step = 0
    evaluations = [{"step": 0, "selection_key": list(selection_key(baseline)), "metrics": baseline}]
    training_history = []
    evaluation_steps = set(optimization["evaluation_steps"])
    maximum_steps = optimization["maximum_steps_per_layer"]
    if len(fit_normalized) != maximum_steps:
        raise ValueError("P6.2b one-pass fit batches must equal the fixed optimizer-step cap")
    for step, normalized_cpu in enumerate(fit_normalized, start=1):
        normalized = normalized_cpu.to(device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            reference = differentiable_function_outputs(teacher, normalized)
        actual = differentiable_function_outputs(master, normalized)
        core_loss = normalized_mse(actual["attention_core"], reference["attention_core"])
        mixer_loss = normalized_mse(actual["mixer_output"], reference["mixer_output"])
        loss = core_loss + mixer_loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("P6.2b local function loss is non-finite")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [named[name] for name in TRAINABLE_PARAMETER_NAMES],
            optimization["gradient_clip"],
        )
        optimizer.step()
        training_history.append(
            {
                "step": step,
                "effective_activation_tokens": step * normalized.shape[1],
                "loss": float(loss.detach()),
                "attention_core_normalized_mse": float(core_loss.detach()),
                "mixer_output_normalized_mse": float(mixer_loss.detach()),
                "gradient_norm_before_clip": float(gradient_norm),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if step in evaluation_steps:
            state = trainable_state(master)
            metrics = evaluate_state(
                donor,
                initial_student,
                state,
                validation_residual,
                validation_normalized,
            )
            key = selection_key(metrics)
            evaluations.append({"step": step, "selection_key": list(key), "metrics": metrics})
            print(
                f"P6_2B_LAYER_{donor.layer_idx}_VALIDATION_STEP={step} "
                f"CORE={metrics['attention_core']['relative_l2_error']:.9g} "
                f"MIXER={metrics['mixer_output']['relative_l2_error']:.9g}",
                flush=True,
            )
            if key < selection_key(best_metrics):
                best_state = state
                best_metrics = metrics
                best_step = step
    frozen_after = {
        name: tensor_sha256(dict(master.named_parameters())[name]) for name in frozen_names
    }
    frozen_checks = {name: frozen_before[name] == frozen_after[name] for name in frozen_names}
    if not all(frozen_checks.values()):
        raise ValueError(f"P6.2b frozen projection changed: {frozen_checks}")
    del optimizer, teacher, master
    torch.cuda.empty_cache()
    return {
        "scope": scope,
        "fresh_optimizer": True,
        "optimizer_state_loaded": False,
        "optimizer": optimization,
        "baseline_validation": baseline,
        "evaluations": evaluations,
        "training_history": training_history,
        "best_step": best_step,
        "best_effective_activation_tokens": best_step * validation_normalized[0].shape[1],
        "best_validation": best_metrics,
        "best_state": best_state,
        "frozen_projection_hash_checks": frozen_checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--p5-checkpoint", type=Path, default=DEFAULT_P5)
    parser.add_argument("--p6-2-report", type=Path, default=DEFAULT_P6_2)
    parser.add_argument("--p6-2a-report", type=Path, default=DEFAULT_P6_2A)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in (
        "config",
        "p5_checkpoint",
        "p6_2_report",
        "p6_2a_report",
        "data_dir",
        "output_dir",
        "artifact_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "p6-2b-local-calibration.json"
    final_path = args.output_dir / "p6-2b-final-check.json"
    print("P6_2B_LOCAL_FUNCTION_CALIBRATION=RUNNING", flush=True)
    print("P6_2B_PILOT_LAYERS=3,23", flush=True)
    print("P6_FORMAL_KV_LORA_RANK=512", flush=True)
    print("P6_3_ALLOWED=false", flush=True)
    print("P7_ALLOWED=false", flush=True)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("P6.2b requires the authoritative CUDA server")
        config = load_yaml(args.config)
        config_checks = validate_config(config)
        p6_2 = load_json(args.p6_2_report)
        p6_2a = load_json(args.p6_2a_report)
        source_checks = {
            "p6_2_redesign": p6_2["status"] == "REDESIGN"
            and p6_2["p6_3_allowed"] is False,
            "p6_2a_complete": p6_2a["status"] == "PASS_ATTRIBUTION_COMPLETE"
            and p6_2a["formal_kv_lora_rank"] == 512,
            "capacity_hypothesis_not_supported": p6_2a["aggregate"][
                "median_mixer_rank_elasticity"
            ]
            < 0.25,
            "distribution_shift_not_supported": p6_2a["aggregate"][
                "median_mixer_scaleout_amplification"
            ]
            < 1.25,
            "heldout_never_accessed": p6_2["heldout_accessed"] is False
            and p6_2a["heldout_accessed"] is False,
            "same_p5": Path(p6_2["source_paths"]["p5_checkpoint"]).resolve()
            == args.p5_checkpoint
            and Path(p6_2a["source_paths"]["p5_checkpoint"]).resolve()
            == args.p5_checkpoint,
        }
        if not all(source_checks.values()):
            raise ValueError(f"P6.2b source checks failed: {source_checks}")
        corpus = verify_frozen_corpus(args.data_dir)
        corpus_manifest = load_json(args.data_dir / "packed/corpus-manifest.json")
        split = select_document_disjoint_sequences(
            corpus_manifest,
            fit_sequences=config["data"]["fit_sequences"],
            validation_sequences=config["data"]["validation_sequences"],
        )
        if corpus["p5_probe_corpus_frozen"] is not True or split["document_overlap_count"]:
            raise ValueError("P6.2b frozen corpus or document split validation failed")

        torch.manual_seed(config["optimization"]["seed"])
        torch.cuda.manual_seed_all(config["optimization"]["seed"])
        device = torch.device("cuda")
        train = PackedTokenDataset(args.data_dir / "packed", "train")
        model = load_densek3_model(args.p5_checkpoint, device=device, dtype=torch.bfloat16)
        model.requires_grad_(False)
        initialization_inputs = capture_frozen_p5_inputs(
            model,
            train,
            PILOT_LAYERS,
            list(range(8)),
            token_prefix=None,
            marker="P6_2B_C3_INPUT_CAPTURE",
        )
        if initialization_inputs["manifest"]["token_sha256"] != p6_2["calibration_manifest"][
            "token_sha256"
        ]:
            raise ValueError("P6.2b C3 initialization tokens do not replay P6.2")
        fit_inputs = capture_frozen_p5_inputs(
            model,
            train,
            PILOT_LAYERS,
            split["fit_sequence_indices"],
            token_prefix=config["data"]["sequence_prefix"],
            marker="P6_2B_FIT_INPUT_CAPTURE",
        )
        validation_inputs = capture_frozen_p5_inputs(
            model,
            train,
            PILOT_LAYERS,
            split["validation_sequence_indices"],
            token_prefix=config["data"]["sequence_prefix"],
            marker="P6_2B_VALIDATION_INPUT_CAPTURE",
        )
        if fit_inputs["manifest"]["effective_activation_tokens_per_layer"] != 32_768:
            raise ValueError("P6.2b fit activation-token budget changed")
        if validation_inputs["manifest"]["effective_activation_tokens_per_layer"] != 8_192:
            raise ValueError("P6.2b validation activation-token budget changed")

        layer_reports: dict[str, dict[str, Any]] = {}
        baseline_for_decision: dict[int, dict[str, Any]] = {}
        calibrated_for_decision: dict[int, dict[str, Any]] = {}
        artifact_tensors: dict[str, torch.Tensor] = {}
        for layer_index in PILOT_LAYERS:
            print(f"P6_2B_LAYER_{layer_index}=INITIALIZING_C3", flush=True)
            donor = model.model.layers[layer_index].self_attn
            if not isinstance(donor, DenseK3GQA):
                raise ValueError(f"P6.2b layer {layer_index} is not a frozen P5 GQA donor")
            initialization_normalized = initialization_inputs["normalized"][layer_index]
            covariance, covariance_report = covariance_from_batches(
                initialization_normalized,
                device=device,
            )
            factorization = activation_weighted_rank_factorization(
                donor.k_proj.weight,
                donor.v_proj.weight,
                covariance,
                num_heads=donor.num_heads,
                num_kv_heads=donor.num_key_value_heads,
                head_dim=donor.head_dim,
                rank=512,
            )
            initial_student = P6NoPEMLA(donor, factorization).eval()
            with torch.no_grad():
                latent_batches = [
                    initial_student.kv_a_proj(batch.to(device)).cpu()
                    for batch in initialization_normalized
                ]
            gamma, scale_report = fit_scale_matched_rmsnorm(
                latent_batches,
                eps=initial_student.kv_a_layernorm.eps,
            )
            apply_rmsnorm_scale(initial_student.kv_a_layernorm, gamma)
            scale_report["applied_gamma_sha256_bf16"] = tensor_sha256(
                initial_student.kv_a_layernorm.weight
            )
            frozen_c3 = p6_2["per_layer_conversion"][str(layer_index)]
            c3_replay = {
                "kv_a_fp32": factorization.report["kv_a_sha256_fp32"]
                == frozen_c3["factorization"]["kv_a_sha256_fp32"],
                "kv_b_fp32": factorization.report["kv_b_sha256_fp32"]
                == frozen_c3["factorization"]["kv_b_sha256_fp32"],
                "gamma_bf16": scale_report["applied_gamma_sha256_bf16"]
                == frozen_c3["scale_matching"]["applied_gamma_sha256_bf16"],
            }
            if not all(c3_replay.values()):
                raise ValueError(f"P6.2b layer {layer_index} C3 replay failed: {c3_replay}")
            print(f"P6_2B_LAYER_{layer_index}=CALIBRATING", flush=True)
            calibration = calibrate_layer(
                donor,
                initial_student,
                fit_inputs["normalized"][layer_index],
                validation_inputs["residual"][layer_index],
                validation_inputs["normalized"][layer_index],
                config["optimization"],
            )
            baseline_for_decision[layer_index] = calibration["baseline_validation"]
            calibrated_for_decision[layer_index] = calibration["best_validation"]
            best_state = calibration.pop("best_state")
            state_hashes = {}
            for name, value in best_state.items():
                artifact_name = f"model.layers.{layer_index}.self_attn.{name}"
                artifact_value = value.to(torch.bfloat16).contiguous()
                artifact_tensors[artifact_name] = artifact_value
                state_hashes[name] = tensor_sha256(artifact_value)
            layer_reports[str(layer_index)] = {
                "layer_index": layer_index,
                "c3_replay": c3_replay,
                "covariance": covariance_report,
                "factorization": factorization.report,
                "scale_matching": scale_report,
                "calibration": calibration,
                "best_trainable_tensor_hashes_bf16": state_hashes,
                "formal_semantics": {
                    "rank": initial_student.kv_lora_rank,
                    "no_pe": initial_student.position_encoding == "none",
                    "qk_norm": initial_student.apply_qk_norm,
                    "latent_norm": initial_student.apply_latent_norm,
                    "split_q_gate": initial_student.split_q_gate,
                },
            }
            print(
                f"P6_2B_LAYER_{layer_index}=BEST STEP={calibration['best_step']} "
                f"CORE={calibration['best_validation']['attention_core']['relative_l2_error']:.9g} "
                f"MIXER={calibration['best_validation']['mixer_output']['relative_l2_error']:.9g}",
                flush=True,
            )
            del covariance, factorization, latent_batches, gamma, initial_student
            torch.cuda.empty_cache()

        decision = local_go_no_go(
            baseline_for_decision,
            calibrated_for_decision,
            attention_core_improvement_min=config["gate"][
                "attention_core_improvement_fraction_min"
            ],
        )
        args.artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = args.artifact_dir / "p6-2b-pilot-trainable.safetensors"
        temporary_artifact = artifact_path.with_suffix(".safetensors.tmp")
        save_file(artifact_tensors, str(temporary_artifact))
        temporary_artifact.replace(artifact_path)
        artifact_report = {
            "path": str(artifact_path),
            "sha256": sha256_file(artifact_path),
            "tensor_count": len(artifact_tensors),
            "tensor_names": sorted(artifact_tensors),
            "full_model_checkpoint": False,
            "local_calibration_trainable_only": True,
            "optimizer_state_saved": False,
            "formal_candidate": False,
        }
        gate_checks = {
            "config_frozen": all(config_checks.values()),
            "source_frozen": all(source_checks.values()),
            "document_overlap_zero": split["document_overlap_count"] == 0,
            "fit_token_budget_exact": fit_inputs["manifest"][
                "effective_activation_tokens_per_layer"
            ]
            == 32_768,
            "validation_token_budget_exact": validation_inputs["manifest"][
                "effective_activation_tokens_per_layer"
            ]
            == 8_192,
            "only_allowed_parameters_trainable": all(
                layer["calibration"]["scope"]["trainable_parameter_names"]
                == list(TRAINABLE_PARAMETER_NAMES)
                for layer in layer_reports.values()
            ),
            "frozen_projections_exact": all(
                all(layer["calibration"]["frozen_projection_hash_checks"].values())
                for layer in layer_reports.values()
            ),
            "formal_semantics_unchanged": all(
                layer["formal_semantics"]
                == {
                    "rank": 512,
                    "no_pe": True,
                    "qk_norm": False,
                    "latent_norm": True,
                    "split_q_gate": True,
                }
                for layer in layer_reports.values()
            ),
            "full_dev_not_accessed": True,
            "heldout_not_accessed": True,
            "canonical_not_accessed": True,
            "full_model_checkpoint_not_written": True,
        }
        if not all(gate_checks.values()):
            raise ValueError(f"P6.2b audit checks failed: {gate_checks}")
        passed = decision["passed"]
        report = {
            "schema_version": 1,
            "stage": "P6.2b-1",
            "status": "GO" if passed else "NO_GO",
            "mode": "ATTENTION_FUNCTION_AWARE_LOCAL_CALIBRATION",
            "source_paths": {
                "p5_checkpoint": str(args.p5_checkpoint),
                "p6_2_report": str(args.p6_2_report),
                "p6_2a_report": str(args.p6_2a_report),
                "frozen_corpus": str(args.data_dir),
            },
            "config_checks": config_checks,
            "source_checks": source_checks,
            "document_disjoint_split": split,
            "initialization_input_manifest": initialization_inputs["manifest"],
            "fit_input_manifest": fit_inputs["manifest"],
            "validation_input_manifest": validation_inputs["manifest"],
            "layers": layer_reports,
            "decision": decision,
            "audit_checks": gate_checks,
            "artifact": artifact_report,
            "formal_kv_lora_rank": 512,
            "global_recovery_training_performed": False,
            "local_calibration_fitting_performed": True,
            "full_dev_accessed": False,
            "heldout_accessed": False,
            "canonical_accessed": False,
            "full_model_checkpoint_written": False,
            "p6_2b_scaleout_allowed": passed,
            "p6_3_allowed": False,
            "p7_allowed": False,
            "result_marker": (
                "P6_2B_LOCAL_FUNCTION_CALIBRATION=GO"
                if passed
                else "P6_2B_LOCAL_FUNCTION_CALIBRATION=NO_GO"
            ),
            "completed_at": now(),
        }
        save_json(report_path, report)
        final = {
            "stage": "P6.2b-1",
            "status": report["status"],
            "mode": report["mode"],
            "pilot_layers": PILOT_LAYERS,
            "formal_kv_lora_rank": 512,
            "decision": decision,
            "artifact": artifact_report,
            "full_dev_accessed": False,
            "heldout_accessed": False,
            "full_model_checkpoint_written": False,
            "p6_2b_scaleout_allowed": passed,
            "p6_3_allowed": False,
            "p7_allowed": False,
            "result_marker": report["result_marker"],
            "completed_at": now(),
        }
        save_json(final_path, final)
        print(report["result_marker"], flush=True)
        print(f"P6_2B_SCALEOUT_ALLOWED={str(passed).lower()}", flush=True)
        print("P6_3_ALLOWED=false", flush=True)
        print("P7_ALLOWED=false", flush=True)
        return 0 if passed else 1
    except Exception as exc:  # noqa: BLE001 - persist every pilot failure
        failure = {
            "stage": "P6.2b-1",
            "status": "ERROR",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "formal_kv_lora_rank": 512,
            "global_recovery_training_performed": False,
            "full_dev_accessed": False,
            "heldout_accessed": False,
            "full_model_checkpoint_written": False,
            "p6_2b_scaleout_allowed": False,
            "p6_3_allowed": False,
            "p7_allowed": False,
            "result_marker": "P6_2B_LOCAL_FUNCTION_CALIBRATION=ERROR",
            "completed_at": now(),
        }
        save_json(report_path, failure)
        save_json(final_path, failure)
        print(failure["result_marker"], flush=True)
        print(f"P6_2B_EXCEPTION={type(exc).__name__}: {exc}", flush=True)
        print("P6_3_ALLOWED=false", flush=True)
        print("P7_ALLOWED=false", flush=True)
        return 1
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
