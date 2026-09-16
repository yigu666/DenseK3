"""Run the one-time P6.2b-2 fixed-budget extension with exact optimizer replay."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
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
    extension_outcome,
    freeze_local_calibration_scope,
    local_go_no_go,
    normalized_mse,
    select_document_disjoint_sequences,
)
from densek3_core.recovery.p6_initialization import (
    activation_weighted_rank_factorization,
    apply_rmsnorm_scale,
    covariance_from_batches,
    fit_scale_matched_rmsnorm,
)
from densek3_core.recovery.p6_mla_probe import P6NoPEFullAttention, P6NoPEMLA, tensor_sha256

ROOT = Path(__file__).resolve().parents[1]
BASE_RUNNER_PATH = ROOT / "scripts/run_p6_function_calibration.py"
BASE_SPEC = importlib.util.spec_from_file_location("p6_2b_1_runner", BASE_RUNNER_PATH)
if BASE_SPEC is None or BASE_SPEC.loader is None:
    raise ImportError(f"Cannot load P6.2b-1 runner: {BASE_RUNNER_PATH}")
BASE = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(BASE)

DEFAULT_CONFIG = ROOT / "configs/architecture/p6-attention-function-budget-extension.yaml"
DEFAULT_BASE_CONFIG = ROOT / "configs/architecture/p6-attention-function-local-calibration.yaml"
DEFAULT_REPORT = ROOT / "outputs/densek3-core/manifests/p6/p6-2b/p6-2b-local-calibration.json"
DEFAULT_ARTIFACT = (
    ROOT / "outputs/densek3-core/artifacts/p6/p6-2b-pilot/p6-2b-pilot-trainable.safetensors"
)
DEFAULT_DATA = ROOT / "data/p5-probe"
DEFAULT_OUTPUT = ROOT / "outputs/densek3-core/manifests/p6/p6-2b-2"
DEFAULT_EXTENSION_ARTIFACT = ROOT / "outputs/densek3-core/artifacts/p6/p6-2b-2-pilot"
PILOT_LAYERS = [3, 23]


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_config(config: dict[str, Any]) -> dict[str, bool]:
    checks = {
        "stage": config["stage"] == "P6.2b-2" and config["status"] == "FROZEN",
        "source": config["source"]["required_p6_2b_1_status"] == "NO_GO",
        "scope": config["scope"]
        == {
            "pilot_layers": PILOT_LAYERS,
            "formal_kv_lora_rank": 512,
            "one_time_extension": True,
            "previous_steps": 128,
            "extension_steps": 128,
            "hard_total_steps": 256,
            "previous_activation_tokens_per_layer": 32_768,
            "extension_activation_tokens_per_layer": 32_768,
            "hard_total_activation_tokens_per_layer": 65_536,
            "further_budget_extension_allowed": False,
        },
        "resume": config["resume"]
        == {
            "optimizer": "AdamW",
            "scheduler": "constant",
            "deterministic_replay_required": True,
            "replay_all_128_training_metrics_exact": True,
            "replay_step128_bf16_tensor_hashes_exact": True,
            "replay_step128_validation_exact": True,
            "fresh_optimizer_at_extension_forbidden": True,
            "scheduler_semantically_continuable_required": True,
        },
        "data": config["data"]["fit_validation_split_identical"] is True
        and config["data"]["fit_order_identical"] is True
        and config["data"]["extension_repeats_same_fixed_fit_order_once"] is True
        and config["data"]["evaluation_steps"] == [160, 192, 224, 256]
        and config["data"]["full_dev_accessed"] is False
        and config["data"]["heldout_accessed"] is False
        and config["data"]["canonical_accessed"] is False,
        "artifact": config["artifact"]["local_trainable_only"] is True
        and config["artifact"]["full_model_checkpoint"] is False
        and config["artifact"]["optimizer_state"] is False,
        "forbidden": all(config["forbidden"].values()),
    }
    if not all(checks.values()):
        raise ValueError(f"P6.2b-2 frozen config failed: {checks}")
    return checks


def optimizer_state_hashes(
    optimizer: torch.optim.Optimizer,
    named: dict[str, torch.nn.Parameter],
) -> dict[str, dict[str, Any]]:
    result = {}
    for name in TRAINABLE_PARAMETER_NAMES:
        state = optimizer.state[named[name]]
        result[name] = {
            "step": float(state["step"].detach().cpu()),
            "exp_avg_sha256": tensor_sha256(state["exp_avg"]),
            "exp_avg_sq_sha256": tensor_sha256(state["exp_avg_sq"]),
        }
    return result


def exact_training_record(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return actual == expected


def replay_and_extend_layer(
    donor: DenseK3GQA,
    initial_student: P6NoPEMLA,
    fit_normalized: list[torch.Tensor],
    validation_residual: list[torch.Tensor],
    validation_normalized: list[torch.Tensor],
    base_layer_report: dict[str, Any],
    optimization: dict[str, Any],
    extension_evaluation_steps: list[int],
) -> dict[str, Any]:
    device = next(initial_student.parameters()).device
    master = copy.deepcopy(initial_student).float().eval()
    scope = freeze_local_calibration_scope(master)
    teacher = P6NoPEFullAttention(donor, split_q_gate=False, apply_qk_norm=True).float().eval()
    teacher.requires_grad_(False)
    named = dict(master.named_parameters())
    frozen_names = ("q_proj.weight", "g_proj.weight", "o_proj.weight")
    frozen_before = {name: tensor_sha256(named[name]) for name in frozen_names}
    optimizer = torch.optim.AdamW(
        [named[name] for name in TRAINABLE_PARAMETER_NAMES],
        lr=optimization["learning_rate"],
        betas=tuple(optimization["betas"]),
        eps=optimization["epsilon"],
        weight_decay=optimization["weight_decay"],
    )
    baseline_state = BASE.trainable_state(master)
    reconstructed_baseline = BASE.evaluate_state(
        donor,
        initial_student,
        baseline_state,
        validation_residual,
        validation_normalized,
    )
    if reconstructed_baseline != base_layer_report["calibration"]["baseline_validation"]:
        raise ValueError("P6.2b-2 baseline validation replay is not exact")

    previous_history = base_layer_report["calibration"]["training_history"]
    previous_evaluations = {
        int(item["step"]): item for item in base_layer_report["calibration"]["evaluations"]
    }
    replay_checks = {
        "all_128_training_metrics_exact": True,
        "previous_validation_metrics_exact": True,
        "step128_bf16_tensor_hashes_exact": False,
        "step128_best_checkpoint": base_layer_report["calibration"]["best_step"] == 128,
        "constant_scheduler_semantically_continuable": optimization["scheduler"] == "constant",
    }
    extension_history = []
    extension_evaluations = []
    best_state = None
    best_metrics = None
    best_step = 128
    optimizer_state_at_resume = None
    original_evaluation_steps = {32, 64, 96, 128}
    extension_evaluation_steps_set = set(extension_evaluation_steps)
    for step in range(1, 257):
        normalized_cpu = fit_normalized[(step - 1) % len(fit_normalized)]
        normalized = normalized_cpu.to(device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            reference = differentiable_function_outputs(teacher, normalized)
        actual = differentiable_function_outputs(master, normalized)
        core_loss = normalized_mse(actual["attention_core"], reference["attention_core"])
        mixer_loss = normalized_mse(actual["mixer_output"], reference["mixer_output"])
        loss = core_loss + mixer_loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("P6.2b-2 local function loss is non-finite")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [named[name] for name in TRAINABLE_PARAMETER_NAMES],
            optimization["gradient_clip"],
        )
        optimizer.step()
        record = {
            "step": step,
            "effective_activation_tokens": step * normalized.shape[1],
            "loss": float(loss.detach()),
            "attention_core_normalized_mse": float(core_loss.detach()),
            "mixer_output_normalized_mse": float(mixer_loss.detach()),
            "gradient_norm_before_clip": float(gradient_norm),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        if step <= 128:
            if not exact_training_record(record, previous_history[step - 1]):
                replay_checks["all_128_training_metrics_exact"] = False
                raise ValueError(
                    f"OPTIMIZER_STATE_RECONSTRUCTION_NOT_EXACT at layer {donor.layer_idx} step {step}"
                )
        else:
            extension_history.append(record)

        if step in original_evaluation_steps or step in extension_evaluation_steps_set:
            state = BASE.trainable_state(master)
            metrics = BASE.evaluate_state(
                donor,
                initial_student,
                state,
                validation_residual,
                validation_normalized,
            )
            if step in original_evaluation_steps:
                if metrics != previous_evaluations[step]["metrics"]:
                    replay_checks["previous_validation_metrics_exact"] = False
                    raise ValueError(
                        f"OPTIMIZER_STATE_RECONSTRUCTION_NOT_EXACT validation layer "
                        f"{donor.layer_idx} step {step}"
                    )
            else:
                extension_evaluations.append(
                    {
                        "step": step,
                        "selection_key": list(BASE.selection_key(metrics)),
                        "metrics": metrics,
                    }
                )
                print(
                    f"P6_2B_2_LAYER_{donor.layer_idx}_VALIDATION_STEP={step} "
                    f"CORE={metrics['attention_core']['relative_l2_error']:.9g} "
                    f"MIXER={metrics['mixer_output']['relative_l2_error']:.9g}",
                    flush=True,
                )
            if step == 128:
                replay_hashes = {
                    name: tensor_sha256(value.to(torch.bfloat16))
                    for name, value in state.items()
                }
                replay_checks["step128_bf16_tensor_hashes_exact"] = (
                    replay_hashes == base_layer_report["best_trainable_tensor_hashes_bf16"]
                )
                if not replay_checks["step128_bf16_tensor_hashes_exact"]:
                    raise ValueError(
                        f"OPTIMIZER_STATE_RECONSTRUCTION_NOT_EXACT tensor hashes layer "
                        f"{donor.layer_idx}"
                    )
                optimizer_state_at_resume = optimizer_state_hashes(optimizer, named)
                best_state = state
                best_metrics = metrics
            elif step > 128 and BASE.selection_key(metrics) < BASE.selection_key(best_metrics):
                best_state = state
                best_metrics = metrics
                best_step = step

    if best_state is None or best_metrics is None or optimizer_state_at_resume is None:
        raise RuntimeError("P6.2b-2 did not reconstruct the step-128 resume state")
    frozen_after = {name: tensor_sha256(named[name]) for name in frozen_names}
    frozen_checks = {name: frozen_before[name] == frozen_after[name] for name in frozen_names}
    if not all(frozen_checks.values()):
        raise ValueError(f"P6.2b-2 frozen projection changed: {frozen_checks}")
    del optimizer, teacher, master
    torch.cuda.empty_cache()
    return {
        "scope": scope,
        "replay_checks": replay_checks,
        "optimizer_state_at_resumed_step128": optimizer_state_at_resume,
        "scheduler": {
            "type": "constant",
            "stateful": False,
            "semantically_continuable": True,
            "learning_rate": optimization["learning_rate"],
        },
        "extension_training_history": extension_history,
        "extension_evaluations": extension_evaluations,
        "best_step_within_256": best_step,
        "best_effective_activation_tokens": best_step * validation_normalized[0].shape[1],
        "best_validation": best_metrics,
        "best_state": best_state,
        "frozen_projection_hash_checks": frozen_checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--p6-2b-1-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--p6-2b-1-artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_EXTENSION_ARTIFACT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in (
        "config",
        "base_config",
        "p6_2b_1_report",
        "p6_2b_1_artifact",
        "data_dir",
        "output_dir",
        "artifact_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "p6-2b-2-extension.json"
    final_path = args.output_dir / "p6-2b-2-final-check.json"
    print("P6_2B_2_BUDGET_EXTENSION=RUNNING", flush=True)
    print("P6_2B_1_LOCAL_FUNCTION_CALIBRATION=NO_GO", flush=True)
    print("P6_FORMAL_KV_LORA_RANK=512", flush=True)
    print("P6_3_ALLOWED=false", flush=True)
    print("P7_ALLOWED=false", flush=True)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("P6.2b-2 requires the authoritative CUDA server")
        config = load_yaml(args.config)
        base_config = load_yaml(args.base_config)
        config_checks = validate_config(config)
        if not all(BASE.validate_config(base_config).values()):
            raise ValueError("P6.2b-2 base config no longer replays P6.2b-1")
        base_report = load_json(args.p6_2b_1_report)
        source_checks = {
            "base_no_go": base_report["status"] == "NO_GO"
            and base_report["p6_2b_scaleout_allowed"] is False,
            "best_steps_128": all(
                base_report["layers"][str(index)]["calibration"]["best_step"] == 128
                for index in PILOT_LAYERS
            ),
            "artifact_exact": sha256_file(args.p6_2b_1_artifact)
            == base_report["artifact"]["sha256"],
            "full_dev_unread": base_report["full_dev_accessed"] is False,
            "heldout_unread": base_report["heldout_accessed"] is False,
            "formal_rank": base_report["formal_kv_lora_rank"] == 512,
        }
        if not all(source_checks.values()):
            raise ValueError(f"P6.2b-2 source checks failed: {source_checks}")
        optimization = base_report["layers"]["3"]["calibration"]["optimizer"]
        if optimization != base_report["layers"]["23"]["calibration"]["optimizer"]:
            raise ValueError("P6.2b-2 layer optimizers differ")
        if optimization["scheduler"] != "constant":
            raise ValueError("SCHEDULER_EXTENSION_NOT_SEMANTICALLY_CONTINUABLE")

        corpus = verify_frozen_corpus(args.data_dir)
        corpus_manifest = load_json(args.data_dir / "packed/corpus-manifest.json")
        split = select_document_disjoint_sequences(
            corpus_manifest,
            fit_sequences=base_config["data"]["fit_sequences"],
            validation_sequences=base_config["data"]["validation_sequences"],
        )
        split_checks = {
            "corpus_frozen": corpus["p5_probe_corpus_frozen"] is True,
            "fit_indices": split["fit_sequence_indices"]
            == base_report["document_disjoint_split"]["fit_sequence_indices"],
            "validation_indices": split["validation_sequence_indices"]
            == base_report["document_disjoint_split"]["validation_sequence_indices"],
            "document_overlap_zero": split["document_overlap_count"] == 0,
        }
        if not all(split_checks.values()):
            raise ValueError(f"P6.2b-2 split replay failed: {split_checks}")

        torch.manual_seed(optimization["seed"])
        torch.cuda.manual_seed_all(optimization["seed"])
        device = torch.device("cuda")
        train = PackedTokenDataset(args.data_dir / "packed", "train")
        p5_checkpoint = Path(base_report["source_paths"]["p5_checkpoint"]).resolve()
        model = load_densek3_model(p5_checkpoint, device=device, dtype=torch.bfloat16)
        model.requires_grad_(False)
        initialization_inputs = capture_frozen_p5_inputs(
            model,
            train,
            PILOT_LAYERS,
            list(range(8)),
            token_prefix=None,
            marker="P6_2B_2_C3_INPUT_CAPTURE",
        )
        fit_inputs = capture_frozen_p5_inputs(
            model,
            train,
            PILOT_LAYERS,
            split["fit_sequence_indices"],
            token_prefix=base_config["data"]["sequence_prefix"],
            marker="P6_2B_2_FIT_INPUT_CAPTURE",
        )
        validation_inputs = capture_frozen_p5_inputs(
            model,
            train,
            PILOT_LAYERS,
            split["validation_sequence_indices"],
            token_prefix=base_config["data"]["sequence_prefix"],
            marker="P6_2B_2_VALIDATION_INPUT_CAPTURE",
        )
        input_replay = {
            "initialization": initialization_inputs["manifest"]
            == base_report["initialization_input_manifest"],
            "fit": fit_inputs["manifest"] == base_report["fit_input_manifest"],
            "validation": validation_inputs["manifest"]
            == base_report["validation_input_manifest"],
        }
        if not all(input_replay.values()):
            raise ValueError(f"P6.2b-2 input replay failed: {input_replay}")

        layer_reports = {}
        baseline_for_decision = {}
        calibrated_for_decision = {}
        artifact_tensors = {}
        for layer_index in PILOT_LAYERS:
            donor = model.model.layers[layer_index].self_attn
            if not isinstance(donor, DenseK3GQA):
                raise ValueError(f"P6.2b-2 layer {layer_index} is not frozen P5 GQA")
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
            print(f"P6_2B_2_LAYER_{layer_index}=REPLAYING_1_TO_128", flush=True)
            extension = replay_and_extend_layer(
                donor,
                initial_student,
                fit_inputs["normalized"][layer_index],
                validation_inputs["residual"][layer_index],
                validation_inputs["normalized"][layer_index],
                base_report["layers"][str(layer_index)],
                optimization,
                config["data"]["evaluation_steps"],
            )
            best_state = extension.pop("best_state")
            baseline_for_decision[layer_index] = base_report["layers"][str(layer_index)][
                "calibration"
            ]["baseline_validation"]
            calibrated_for_decision[layer_index] = extension["best_validation"]
            state_hashes = {}
            for name, value in best_state.items():
                artifact_name = f"model.layers.{layer_index}.self_attn.{name}"
                artifact_value = value.to(torch.bfloat16).contiguous()
                artifact_tensors[artifact_name] = artifact_value
                state_hashes[name] = tensor_sha256(artifact_value)
            layer_reports[str(layer_index)] = {
                "layer_index": layer_index,
                "covariance": covariance_report,
                "factorization": factorization.report,
                "scale_matching": scale_report,
                "extension": extension,
                "best_trainable_tensor_hashes_bf16": state_hashes,
            }
            print(
                f"P6_2B_2_LAYER_{layer_index}=BEST "
                f"STEP={extension['best_step_within_256']} "
                f"CORE={extension['best_validation']['attention_core']['relative_l2_error']:.9g} "
                f"MIXER={extension['best_validation']['mixer_output']['relative_l2_error']:.9g}",
                flush=True,
            )
            del covariance, factorization, latent_batches, gamma, initial_student
            torch.cuda.empty_cache()

        local_decision = local_go_no_go(
            baseline_for_decision,
            calibrated_for_decision,
            attention_core_improvement_min=config["gate"][
                "attention_core_improvement_fraction_min"
            ],
        )
        layer23_by_step = {
            int(item["step"]): float(item["metrics"]["mixer_output"]["relative_l2_error"])
            for item in layer_reports["23"]["extension"]["extension_evaluations"]
            if int(item["step"]) in {192, 224, 256}
        }
        outcome = extension_outcome(
            local_decision,
            layer23_by_step,
            plateau_last64_absolute_improvement_max=config["gate"][
                "plateau_last64_absolute_improvement_max"
            ],
        )
        args.artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = args.artifact_dir / "p6-2b-2-best-trainable.safetensors"
        temporary_artifact = artifact_path.with_suffix(".safetensors.tmp")
        save_file(artifact_tensors, str(temporary_artifact))
        temporary_artifact.replace(artifact_path)
        artifact_report = {
            "path": str(artifact_path),
            "sha256": sha256_file(artifact_path),
            "tensor_count": len(artifact_tensors),
            "tensor_names": sorted(artifact_tensors),
            "full_model_checkpoint": False,
            "formal_candidate": False,
            "optimizer_state_saved": False,
        }
        audit_checks = {
            "config_frozen": all(config_checks.values()),
            "source_frozen": all(source_checks.values()),
            "split_exact": all(split_checks.values()),
            "inputs_exact": all(input_replay.values()),
            "optimizer_trajectory_reconstructed_exactly": all(
                all(layer["extension"]["replay_checks"].values())
                for layer in layer_reports.values()
            ),
            "scheduler_semantically_continuable": all(
                layer["extension"]["scheduler"]["semantically_continuable"]
                for layer in layer_reports.values()
            ),
            "frozen_projections_exact": all(
                all(layer["extension"]["frozen_projection_hash_checks"].values())
                for layer in layer_reports.values()
            ),
            "hard_cap_exact": all(
                layer["extension"]["extension_training_history"][-1]["step"] == 256
                and layer["extension"]["extension_training_history"][-1][
                    "effective_activation_tokens"
                ]
                == 65_536
                for layer in layer_reports.values()
            ),
            "full_dev_not_accessed": True,
            "heldout_not_accessed": True,
            "canonical_not_accessed": True,
            "full_model_checkpoint_not_written": True,
            "further_extension_forbidden": True,
        }
        if not all(audit_checks.values()):
            raise ValueError(f"P6.2b-2 audit checks failed: {audit_checks}")
        passed = outcome["status"] == "GO"
        report = {
            "schema_version": 1,
            "stage": "P6.2b-2",
            "status": outcome["status"],
            "mode": "ONE_TIME_FIXED_BUDGET_EXTENSION",
            "p6_2b_1_status_unchanged": "NO_GO",
            "source_paths": {
                "p6_2b_1_report": str(args.p6_2b_1_report),
                "p6_2b_1_artifact": str(args.p6_2b_1_artifact),
                "p5_checkpoint": str(p5_checkpoint),
                "frozen_corpus": str(args.data_dir),
            },
            "config_checks": config_checks,
            "source_checks": source_checks,
            "split_checks": split_checks,
            "input_replay": input_replay,
            "layers": layer_reports,
            "local_decision": local_decision,
            "outcome": outcome,
            "audit_checks": audit_checks,
            "artifact": artifact_report,
            "formal_kv_lora_rank": 512,
            "total_activation_tokens_per_layer": 65_536,
            "total_steps_per_layer": 256,
            "full_dev_accessed": False,
            "heldout_accessed": False,
            "canonical_accessed": False,
            "full_model_checkpoint_written": False,
            "further_budget_extension_allowed": False,
            "p6_2b_scaleout_allowed": passed,
            "p6_3_allowed": False,
            "p7_allowed": False,
            "result_marker": f"P6_2B_2={outcome['status']}",
            "completed_at": now(),
        }
        save_json(report_path, report)
        final = {
            "stage": "P6.2b-2",
            "status": outcome["status"],
            "mode": report["mode"],
            "p6_2b_1_status_unchanged": "NO_GO",
            "local_decision": local_decision,
            "outcome": outcome,
            "artifact": artifact_report,
            "formal_kv_lora_rank": 512,
            "full_dev_accessed": False,
            "heldout_accessed": False,
            "full_model_checkpoint_written": False,
            "further_budget_extension_allowed": False,
            "p6_2b_scaleout_allowed": passed,
            "p6_3_allowed": False,
            "p7_allowed": False,
            "result_marker": report["result_marker"],
            "completed_at": now(),
        }
        save_json(final_path, final)
        print(report["result_marker"], flush=True)
        print(f"P6_2B_SCALEOUT_ALLOWED={str(passed).lower()}", flush=True)
        print("P6_2B_FURTHER_BUDGET_EXTENSION_ALLOWED=false", flush=True)
        print("P6_3_ALLOWED=false", flush=True)
        print("P7_ALLOWED=false", flush=True)
        return 0 if passed else 1
    except Exception as exc:  # noqa: BLE001 - persist every extension failure
        failure = {
            "stage": "P6.2b-2",
            "status": "ERROR",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "p6_2b_1_status_unchanged": "NO_GO",
            "formal_kv_lora_rank": 512,
            "full_dev_accessed": False,
            "heldout_accessed": False,
            "full_model_checkpoint_written": False,
            "further_budget_extension_allowed": False,
            "p6_2b_scaleout_allowed": False,
            "p6_3_allowed": False,
            "p7_allowed": False,
            "result_marker": "P6_2B_2=ERROR",
            "completed_at": now(),
        }
        save_json(report_path, failure)
        save_json(final_path, failure)
        print(failure["result_marker"], flush=True)
        print(f"P6_2B_2_EXCEPTION={type(exc).__name__}: {exc}", flush=True)
        print("P6_2B_SCALEOUT_ALLOWED=false", flush=True)
        print("P6_3_ALLOWED=false", flush=True)
        print("P7_ALLOWED=false", flush=True)
        return 1
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
