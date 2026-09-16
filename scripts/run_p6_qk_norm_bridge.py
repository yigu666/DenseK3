"""Run the two-phase P6.2c frozen donor Q/K RMSNorm semantic bridge."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

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
    qk_norm_bridge_phase_a_decision,
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
DEFAULT_CONFIG = ROOT / "configs/architecture/p6-qk-norm-semantic-bridge.yaml"
DEFAULT_BASE_CONFIG = ROOT / "configs/architecture/p6-attention-function-local-calibration.yaml"
DEFAULT_B1_REPORT = (
    ROOT / "outputs/densek3-core/manifests/p6/p6-2b/p6-2b-local-calibration.json"
)
DEFAULT_B2_REPORT = (
    ROOT / "outputs/densek3-core/manifests/p6/p6-2b-2/p6-2b-2-extension.json"
)
DEFAULT_B2_ARTIFACT = (
    ROOT
    / "outputs/densek3-core/artifacts/p6/p6-2b-2-pilot/p6-2b-2-best-trainable.safetensors"
)
DEFAULT_DATA = ROOT / "data/p5-probe"
DEFAULT_OUTPUT = ROOT / "outputs/densek3-core/manifests/p6/p6-2c"
DEFAULT_ARTIFACT = ROOT / "outputs/densek3-core/artifacts/p6/p6-2c-pilot"
PILOT_LAYERS = [3, 23]
QK_FROZEN_PARAMETER_NAMES = ("q_norm.weight", "k_norm.weight")


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
    phase_b = config["phase_b"]
    checks = {
        "stage": config["stage"] == "P6.2c" and config["status"] == "FROZEN",
        "source": config["source"]["p6_2b_1_required_status"] == "NO_GO"
        and config["source"]["p6_2b_2_required_status"] == "NO_GO_WITHIN_64K"
        and config["source"]["initialization"]
        == "ACTIVATION_WEIGHTED_RANK512_PLUS_SCALE_MATCHED_LATENT_NORM"
        and config["source"]["teacher"] == "FULL_RANK_NOPE_B",
        "scope": config["scope"]
        == {
            "pilot_layers": PILOT_LAYERS,
            "formal_kv_lora_rank": 512,
            "no_pe": True,
            "donor_qk_norm_preserved": True,
            "donor_qk_norm_frozen": True,
            "global_recovery_training": False,
            "eight_layer_scaleout": False,
        },
        "phase_a": config["phase_a"]["training"] is False
        and config["phase_a"]["n2_diagnostic_only"] is True
        and config["phase_a"]["layer_23_mixer_improvement_fraction_min"] == 0.10
        and config["phase_a"]["layer_23_mixer_relative_l2_signal_max"] == 0.70
        and config["phase_a"]["full_dev_accessed"] is False
        and config["phase_a"]["heldout_accessed"] is False
        and config["phase_a"]["canonical_accessed"] is False,
        "phase_b": phase_b["condition"] == "PHASE_A_POSITIVE_SIGNAL"
        and phase_b["fresh_c3_initialization"] is True
        and phase_b["p6_2b_checkpoint_resume"] is False
        and phase_b["optimizer"] == "AdamW"
        and phase_b["learning_rate"] == 0.0002
        and phase_b["betas"] == [0.9, 0.95]
        and phase_b["epsilon"] == 1.0e-8
        and phase_b["weight_decay"] == 0.0
        and phase_b["gradient_clip"] == 1.0
        and phase_b["maximum_steps_per_layer"] == 256
        and phase_b["maximum_activation_tokens_per_layer"] == 65_536
        and phase_b["evaluation_steps"] == [0, 32, 64, 96, 128, 160, 192, 224, 256]
        and phase_b["scheduler"] == "constant"
        and phase_b["seed"] == 20260814
        and phase_b["best_checkpoint_within_budget"] is True,
        "data": config["data"]["split"] == "train"
        and config["data"]["sequence_prefix"] == 256
        and config["data"]["c3_initialization_sequences"] == 8
        and config["data"]["fit_sequences"] == 128
        and config["data"]["validation_sequences"] == 32
        and config["data"]["fit_order_repeated_exactly_twice"] is True
        and config["data"]["document_disjoint"] is True
        and config["data"]["full_dev_accessed"] is False
        and config["data"]["heldout_accessed"] is False
        and config["data"]["canonical_accessed"] is False,
        "loss": config["loss"]
        == {
            "attention_core_normalized_mse_weight": 1.0,
            "mixer_output_normalized_mse_weight": 1.0,
            "all_other_weights": 0.0,
        },
        "trainable": config["trainable"]["parameter_names"]
        == list(TRAINABLE_PARAMETER_NAMES),
        "gate": config["gate"]["layer_23_mixer_relative_l2_max"] == 0.40
        and config["gate"]["formal_comparison_baseline"]
        == "N0_C3_QK_NORM_REMOVED"
        and config["gate"]["layer_23_mixer_improvement_fraction_min"] == 0.45
        and config["gate"]["layer_3_mixer_relative_l2_max"] == 0.32
        and config["gate"]["layer_3_mixer_degradation_fraction_max"] == 0.20
        and config["gate"]["attention_core_improvement_fraction_min"] == 0.20
        and config["gate"]["finite_required"] is True
        and config["gate"]["formal_gate_modified"] is False,
        "artifact": config["artifact"]["local_trainable_only"] is True
        and config["artifact"]["full_model_checkpoint"] is False
        and config["artifact"]["optimizer_state"] is False,
        "forbidden": all(config["forbidden"].values()),
    }
    if not all(checks.values()):
        raise ValueError(f"P6.2c frozen config failed: {checks}")
    return checks


def metrics_are_finite(metrics: dict[str, Any]) -> bool:
    fields = ("attention_core", "mixer_output", "post_attention_residual_hidden")
    for field in fields:
        values = metrics[field]
        if not math.isfinite(float(values["relative_l2_error"])):
            return False
        if any(
            int(values[name]) != 0
            for name in (
                "reference_nan_count",
                "reference_inf_count",
                "actual_nan_count",
                "actual_inf_count",
            )
        ):
            return False
    return True


def trainable_state(module: P6NoPEMLA) -> dict[str, torch.Tensor]:
    named = dict(module.named_parameters())
    return {
        name: named[name].detach().cpu().float().clone() for name in TRAINABLE_PARAMETER_NAMES
    }


@torch.no_grad()
def load_trainable_state(module: P6NoPEMLA, state: dict[str, torch.Tensor]) -> None:
    named = dict(module.named_parameters())
    if set(state) != set(TRAINABLE_PARAMETER_NAMES):
        raise ValueError("P6.2c trainable state keys violate the frozen scope")
    for name in TRAINABLE_PARAMETER_NAMES:
        named[name].copy_(state[name].to(device=named[name].device, dtype=named[name].dtype))


def selection_key(metrics: dict[str, Any]) -> tuple[float, float, float]:
    core = float(metrics["attention_core"]["relative_l2_error"])
    mixer = float(metrics["mixer_output"]["relative_l2_error"])
    return core + mixer, mixer, core


@torch.no_grad()
def evaluate_state(
    donor: DenseK3GQA,
    initial_student: P6NoPEMLA,
    state: dict[str, torch.Tensor],
    residual_batches: list[torch.Tensor],
    normalized_batches: list[torch.Tensor],
) -> dict[str, Any]:
    student = copy.deepcopy(initial_student).eval()
    load_trainable_state(student, state)
    metrics = local_validation_metrics(
        donor,
        student,
        residual_batches,
        normalized_batches,
    )
    del student
    torch.cuda.empty_cache()
    return metrics


def build_c3_students(
    donor: DenseK3GQA,
    initialization_normalized: list[torch.Tensor],
    *,
    device: torch.device,
) -> tuple[P6NoPEMLA, P6NoPEMLA, dict[str, Any]]:
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
    n0 = P6NoPEMLA(donor, factorization, apply_qk_norm=False).eval()
    n1 = P6NoPEMLA(donor, factorization, apply_qk_norm=True).eval()
    with torch.no_grad():
        latent_batches = [
            n0.kv_a_proj(batch.to(device)).cpu() for batch in initialization_normalized
        ]
    gamma, scale_report = fit_scale_matched_rmsnorm(
        latent_batches,
        eps=n0.kv_a_layernorm.eps,
    )
    apply_rmsnorm_scale(n0.kv_a_layernorm, gamma)
    apply_rmsnorm_scale(n1.kv_a_layernorm, gamma)
    scale_report["applied_gamma_sha256_bf16"] = tensor_sha256(
        n0.kv_a_layernorm.weight
    )
    qk_copy_checks = {
        "q_norm_exact": tensor_sha256(n1.q_norm.weight) == tensor_sha256(donor.q_norm.weight),
        "k_norm_exact": tensor_sha256(n1.k_norm.weight) == tensor_sha256(donor.k_norm.weight),
    }
    if not all(qk_copy_checks.values()):
        raise ValueError(f"P6.2c donor Q/K norm copy failed: {qk_copy_checks}")
    construction = {
        "covariance": covariance_report,
        "factorization": factorization.report,
        "scale_matching": scale_report,
        "qk_copy_checks": qk_copy_checks,
        "formal_kv_lora_rank": factorization.kv_a_weight.shape[0],
    }
    del covariance, factorization, latent_batches, gamma
    return n0, n1, construction


def artifact_state_for_layer(
    artifact: dict[str, torch.Tensor],
    layer_index: int,
) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.{layer_index}.self_attn."
    state = {
        name.removeprefix(prefix): value.float()
        for name, value in artifact.items()
        if name.startswith(prefix)
    }
    if set(state) != set(TRAINABLE_PARAMETER_NAMES):
        raise ValueError(f"P6.2c layer {layer_index} P6.2b-2 artifact scope mismatch")
    return state


def calibrate_bridge_layer(
    donor: DenseK3GQA,
    initial_student: P6NoPEMLA,
    fit_normalized: list[torch.Tensor],
    validation_residual: list[torch.Tensor],
    validation_normalized: list[torch.Tensor],
    optimization: dict[str, Any],
    phase_a_n1_metrics: dict[str, Any],
    *,
    selection_mixer_relative_l2_max: float | None = None,
) -> dict[str, Any]:
    device = next(initial_student.parameters()).device
    master = copy.deepcopy(initial_student).float().eval()
    scope = freeze_local_calibration_scope(master)
    if not master.apply_qk_norm:
        raise ValueError("P6.2c Phase B requires preserved Q/K norm semantics")
    teacher = P6NoPEFullAttention(donor, split_q_gate=False, apply_qk_norm=True).float().eval()
    teacher.requires_grad_(False)
    named = dict(master.named_parameters())
    frozen_names = (
        "q_proj.weight",
        "g_proj.weight",
        "o_proj.weight",
        *QK_FROZEN_PARAMETER_NAMES,
    )
    frozen_before = {name: tensor_sha256(named[name]) for name in frozen_names}
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
    if baseline != phase_a_n1_metrics:
        raise ValueError(f"P6.2c layer {donor.layer_idx} fresh N1 baseline is not exact")
    best_state = initial_state
    best_metrics = baseline
    best_step = 0
    unconstrained_best_metrics = baseline
    unconstrained_best_step = 0
    if (
        selection_mixer_relative_l2_max is not None
        and baseline["mixer_output"]["relative_l2_error"]
        > selection_mixer_relative_l2_max
    ):
        raise ValueError("P6 local calibration baseline violates its selection safety limit")
    evaluations = [{"step": 0, "selection_key": list(selection_key(baseline)), "metrics": baseline}]
    history = []
    evaluation_steps = set(optimization["evaluation_steps"])
    maximum_steps = optimization["maximum_steps_per_layer"]
    if len(fit_normalized) != 128 or maximum_steps != 256:
        raise ValueError("P6.2c requires 128 fixed fit batches repeated exactly twice")
    for step in range(1, maximum_steps + 1):
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
            raise FloatingPointError("P6.2c local function loss is non-finite")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [named[name] for name in TRAINABLE_PARAMETER_NAMES],
            optimization["gradient_clip"],
        )
        optimizer.step()
        history.append(
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
                f"P6_2C_LAYER_{donor.layer_idx}_VALIDATION_STEP={step} "
                f"CORE={metrics['attention_core']['relative_l2_error']:.9g} "
                f"MIXER={metrics['mixer_output']['relative_l2_error']:.9g}",
                flush=True,
            )
            if key < selection_key(best_metrics) and (
                selection_mixer_relative_l2_max is None
                or metrics["mixer_output"]["relative_l2_error"]
                <= selection_mixer_relative_l2_max
            ):
                best_state = state
                best_metrics = metrics
                best_step = step
            if key < selection_key(unconstrained_best_metrics):
                unconstrained_best_metrics = metrics
                unconstrained_best_step = step
    frozen_after = {name: tensor_sha256(named[name]) for name in frozen_names}
    frozen_checks = {name: frozen_before[name] == frozen_after[name] for name in frozen_names}
    if not all(frozen_checks.values()):
        raise ValueError(f"P6.2c frozen parameter changed: {frozen_checks}")
    del optimizer, teacher, master
    torch.cuda.empty_cache()
    return {
        "scope": scope,
        "fresh_c3_initialization": True,
        "p6_2b_checkpoint_resumed": False,
        "optimizer": optimization,
        "baseline_validation": baseline,
        "evaluations": evaluations,
        "training_history": history,
        "best_step": best_step,
        "best_effective_activation_tokens": best_step * validation_normalized[0].shape[1],
        "best_validation": best_metrics,
        "best_state": best_state,
        "selection_mixer_relative_l2_max": selection_mixer_relative_l2_max,
        "fresh_n1_step_zero_eligible": True,
        "unconstrained_best_step": unconstrained_best_step,
        "unconstrained_best_validation": unconstrained_best_metrics,
        "frozen_parameter_hash_checks": frozen_checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--p6-2b-1-report", type=Path, default=DEFAULT_B1_REPORT)
    parser.add_argument("--p6-2b-2-report", type=Path, default=DEFAULT_B2_REPORT)
    parser.add_argument("--p6-2b-2-artifact", type=Path, default=DEFAULT_B2_ARTIFACT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in (
        "config",
        "base_config",
        "p6_2b_1_report",
        "p6_2b_2_report",
        "p6_2b_2_artifact",
        "data_dir",
        "output_dir",
        "artifact_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "p6-2c-qk-norm-bridge.json"
    final_path = args.output_dir / "p6-2c-final-check.json"
    report: dict[str, Any] = {
        "schema_version": 1,
        "stage": "P6.2c",
        "status": "RUNNING",
        "mode": "QK_NORM_SEMANTIC_BRIDGE",
        "p6_2b_1_status_unchanged": "NO_GO",
        "p6_2b_2_status_unchanged": "NO_GO_WITHIN_64K",
        "formal_kv_lora_rank": 512,
        "full_dev_accessed": False,
        "heldout_accessed": False,
        "canonical_accessed": False,
        "full_model_checkpoint_written": False,
        "p6_2c_scaleout_allowed": False,
        "p6_3_allowed": False,
        "p7_allowed": False,
    }
    print("P6_2C_QK_NORM_SEMANTIC_BRIDGE=RUNNING", flush=True)
    print("P6_FORMAL_KV_LORA_RANK=512", flush=True)
    print("P6_3_ALLOWED=false", flush=True)
    print("P7_ALLOWED=false", flush=True)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("P6.2c requires the authoritative CUDA server")
        config = load_yaml(args.config)
        base_config = load_yaml(args.base_config)
        config_checks = validate_config(config)
        base_contract_checks = {
            "data": all(
                config["data"][name] == base_config["data"][name]
                for name in ("sequence_prefix", "fit_sequences", "validation_sequences")
            ),
            "optimizer": all(
                config["phase_b"][name] == base_config["optimization"][name]
                for name in (
                    "optimizer",
                    "learning_rate",
                    "betas",
                    "epsilon",
                    "weight_decay",
                    "gradient_clip",
                    "scheduler",
                    "seed",
                )
            ),
            "loss": config["loss"]["attention_core_normalized_mse_weight"]
            == base_config["loss"]["attention_core_normalized_mse_weight"]
            and config["loss"]["mixer_output_normalized_mse_weight"]
            == base_config["loss"]["mixer_output_normalized_mse_weight"],
        }
        if not all(base_contract_checks.values()):
            raise ValueError(f"P6.2c changed the P6.2b contract: {base_contract_checks}")
        b1_report = load_json(args.p6_2b_1_report)
        b2_report = load_json(args.p6_2b_2_report)
        source_checks = {
            "p6_2b_1_no_go": b1_report["status"] == "NO_GO"
            and b1_report["p6_2b_scaleout_allowed"] is False,
            "p6_2b_2_no_go_within_64k": b2_report["status"] == "NO_GO_WITHIN_64K"
            and b2_report["further_budget_extension_allowed"] is False,
            "p6_2b_2_artifact_exact": sha256_file(args.p6_2b_2_artifact)
            == b2_report["artifact"]["sha256"],
            "p6_2b_2_best_step256": all(
                b2_report["layers"][str(index)]["extension"]["best_step_within_256"] == 256
                for index in PILOT_LAYERS
            ),
            "formal_rank": b1_report["formal_kv_lora_rank"] == 512
            and b2_report["formal_kv_lora_rank"] == 512,
            "full_dev_unread": b1_report["full_dev_accessed"] is False
            and b2_report["full_dev_accessed"] is False,
            "heldout_unread": b1_report["heldout_accessed"] is False
            and b2_report["heldout_accessed"] is False,
        }
        if not all(source_checks.values()):
            raise ValueError(f"P6.2c source checks failed: {source_checks}")

        corpus = verify_frozen_corpus(args.data_dir)
        corpus_manifest = load_json(args.data_dir / "packed/corpus-manifest.json")
        split = select_document_disjoint_sequences(
            corpus_manifest,
            fit_sequences=config["data"]["fit_sequences"],
            validation_sequences=config["data"]["validation_sequences"],
        )
        split_checks = {
            "corpus_frozen": corpus["p5_probe_corpus_frozen"] is True,
            "fit_indices_replay": split["fit_sequence_indices"]
            == b1_report["document_disjoint_split"]["fit_sequence_indices"],
            "validation_indices_replay": split["validation_sequence_indices"]
            == b1_report["document_disjoint_split"]["validation_sequence_indices"],
            "document_overlap_zero": split["document_overlap_count"] == 0,
        }
        if not all(split_checks.values()):
            raise ValueError(f"P6.2c split replay failed: {split_checks}")

        optimization = {
            key: config["phase_b"][key]
            for key in (
                "optimizer",
                "learning_rate",
                "betas",
                "epsilon",
                "weight_decay",
                "gradient_clip",
                "maximum_steps_per_layer",
                "evaluation_steps",
                "scheduler",
                "seed",
            )
        }
        prior_optimizer = b1_report["layers"]["3"]["calibration"]["optimizer"]
        unchanged_optimizer_checks = {
            key: optimization[key] == prior_optimizer[key]
            for key in (
                "optimizer",
                "learning_rate",
                "betas",
                "epsilon",
                "weight_decay",
                "gradient_clip",
                "scheduler",
                "seed",
            )
        }
        if not all(unchanged_optimizer_checks.values()):
            raise ValueError(f"P6.2c optimizer contract changed: {unchanged_optimizer_checks}")

        torch.manual_seed(optimization["seed"])
        torch.cuda.manual_seed_all(optimization["seed"])
        device = torch.device("cuda")
        train = PackedTokenDataset(args.data_dir / "packed", "train")
        p5_checkpoint = Path(b1_report["source_paths"]["p5_checkpoint"]).resolve()
        model = load_densek3_model(p5_checkpoint, device=device, dtype=torch.bfloat16)
        model.requires_grad_(False)
        initialization_inputs = capture_frozen_p5_inputs(
            model,
            train,
            PILOT_LAYERS,
            list(range(config["data"]["c3_initialization_sequences"])),
            token_prefix=None,
            marker="P6_2C_C3_INPUT_CAPTURE",
        )
        fit_inputs = capture_frozen_p5_inputs(
            model,
            train,
            PILOT_LAYERS,
            split["fit_sequence_indices"],
            token_prefix=config["data"]["sequence_prefix"],
            marker="P6_2C_FIT_INPUT_CAPTURE",
        )
        validation_inputs = capture_frozen_p5_inputs(
            model,
            train,
            PILOT_LAYERS,
            split["validation_sequence_indices"],
            token_prefix=config["data"]["sequence_prefix"],
            marker="P6_2C_VALIDATION_INPUT_CAPTURE",
        )
        input_replay = {
            "initialization": initialization_inputs["manifest"]
            == b1_report["initialization_input_manifest"],
            "fit": fit_inputs["manifest"] == b1_report["fit_input_manifest"],
            "validation": validation_inputs["manifest"]
            == b1_report["validation_input_manifest"],
        }
        if not all(input_replay.values()):
            raise ValueError(f"P6.2c input replay failed: {input_replay}")

        b2_artifact = load_file(str(args.p6_2b_2_artifact), device="cpu")
        phase_a_layers = {}
        for layer_index in PILOT_LAYERS:
            donor = model.model.layers[layer_index].self_attn
            if not isinstance(donor, DenseK3GQA):
                raise ValueError(f"P6.2c layer {layer_index} is not frozen P5 GQA")
            n0, n1, construction = build_c3_students(
                donor,
                initialization_inputs["normalized"][layer_index],
                device=device,
            )
            n2 = copy.deepcopy(n1).eval()
            n2_state = artifact_state_for_layer(b2_artifact, layer_index)
            n2_hash_checks = {
                name: tensor_sha256(value.to(torch.bfloat16))
                == b2_report["layers"][str(layer_index)]["best_trainable_tensor_hashes_bf16"][
                    name
                ]
                for name, value in n2_state.items()
            }
            if not all(n2_hash_checks.values()):
                raise ValueError(f"P6.2c N2 artifact replay failed: {n2_hash_checks}")
            load_trainable_state(n2, n2_state)
            cases = {
                "N0": local_validation_metrics(
                    donor,
                    n0,
                    validation_inputs["residual"][layer_index],
                    validation_inputs["normalized"][layer_index],
                ),
                "N1": local_validation_metrics(
                    donor,
                    n1,
                    validation_inputs["residual"][layer_index],
                    validation_inputs["normalized"][layer_index],
                ),
                "N2": local_validation_metrics(
                    donor,
                    n2,
                    validation_inputs["residual"][layer_index],
                    validation_inputs["normalized"][layer_index],
                ),
            }
            previous_layer = b1_report["layers"][str(layer_index)]
            c3_replay_checks = {
                "covariance_exact": construction["covariance"]
                == previous_layer["covariance"],
                "factorization_exact": construction["factorization"]
                == previous_layer["factorization"],
                "scale_matching_exact": construction["scale_matching"]
                == previous_layer["scale_matching"],
                "n0_validation_exact": cases["N0"]
                == previous_layer["calibration"]["baseline_validation"],
            }
            if not all(c3_replay_checks.values()):
                raise ValueError(
                    f"P6.2c layer {layer_index} C3/N0 replay failed: {c3_replay_checks}"
                )
            finite_checks = {name: metrics_are_finite(metrics) for name, metrics in cases.items()}
            if not all(finite_checks.values()):
                raise FloatingPointError(f"P6.2c layer {layer_index} Phase A is non-finite")
            phase_a_layers[str(layer_index)] = {
                "layer_index": layer_index,
                "construction": construction,
                "cases": cases,
                "finite_checks": finite_checks,
                "c3_replay_checks": c3_replay_checks,
                "n2_diagnostic_only": True,
                "n2_source_step": b2_report["layers"][str(layer_index)]["extension"][
                    "best_step_within_256"
                ],
                "n2_artifact_hash_checks": n2_hash_checks,
            }
            print(
                f"P6_2C_PHASE_A_LAYER_{layer_index} "
                f"N0_MIXER={cases['N0']['mixer_output']['relative_l2_error']:.9g} "
                f"N1_MIXER={cases['N1']['mixer_output']['relative_l2_error']:.9g} "
                f"N2_MIXER={cases['N2']['mixer_output']['relative_l2_error']:.9g}",
                flush=True,
            )
            del n0, n1, n2, n2_state
            torch.cuda.empty_cache()

        phase_a_decision = qk_norm_bridge_phase_a_decision(
            phase_a_layers["23"]["cases"]["N0"],
            phase_a_layers["23"]["cases"]["N1"],
            mixer_improvement_fraction_min=config["phase_a"][
                "layer_23_mixer_improvement_fraction_min"
            ],
            mixer_relative_l2_signal_max=config["phase_a"][
                "layer_23_mixer_relative_l2_signal_max"
            ],
        )
        report.update(
            {
                "source_paths": {
                    "p5_checkpoint": str(p5_checkpoint),
                    "p6_2b_1_report": str(args.p6_2b_1_report),
                    "p6_2b_2_report": str(args.p6_2b_2_report),
                    "p6_2b_2_artifact": str(args.p6_2b_2_artifact),
                    "frozen_corpus": str(args.data_dir),
                },
                "config_checks": config_checks,
                "base_contract_checks": base_contract_checks,
                "source_checks": source_checks,
                "split_checks": split_checks,
                "input_replay": input_replay,
                "unchanged_optimizer_checks": unchanged_optimizer_checks,
                "phase_a": {
                    "status": phase_a_decision["status"],
                    "training_performed": False,
                    "layers": phase_a_layers,
                    "decision": phase_a_decision,
                },
            }
        )
        save_json(report_path, report)
        print(f"P6_2C_PHASE_A={phase_a_decision['status']}", flush=True)
        if not phase_a_decision["passed"]:
            report.update(
                {
                    "status": "NO_GO_PHASE_A",
                    "phase_b": {"status": "NOT_RUN_NO_PHASE_A_SIGNAL"},
                    "result_marker": "P6_2C=NO_GO_PHASE_A",
                    "completed_at": now(),
                }
            )
            save_json(report_path, report)
            final = {
                "stage": "P6.2c",
                "status": report["status"],
                "phase_a": phase_a_decision,
                "phase_b_status": "NOT_RUN_NO_PHASE_A_SIGNAL",
                "p6_2c_scaleout_allowed": False,
                "p6_3_allowed": False,
                "p7_allowed": False,
                "result_marker": report["result_marker"],
                "completed_at": now(),
            }
            save_json(final_path, final)
            print(report["result_marker"], flush=True)
            print("P6_2C_SCALEOUT_ALLOWED=false", flush=True)
            return 1

        phase_b_layers = {}
        baseline_for_decision = {}
        calibrated_for_decision = {}
        artifact_tensors = {}
        for layer_index in PILOT_LAYERS:
            donor = model.model.layers[layer_index].self_attn
            n0, initial_student, construction = build_c3_students(
                donor,
                initialization_inputs["normalized"][layer_index],
                device=device,
            )
            del n0
            if construction != phase_a_layers[str(layer_index)]["construction"]:
                raise ValueError(f"P6.2c layer {layer_index} fresh C3 construction changed")
            print(f"P6_2C_PHASE_B_LAYER_{layer_index}=CALIBRATING", flush=True)
            calibration = calibrate_bridge_layer(
                donor,
                initial_student,
                fit_inputs["normalized"][layer_index],
                validation_inputs["residual"][layer_index],
                validation_inputs["normalized"][layer_index],
                optimization,
                phase_a_layers[str(layer_index)]["cases"]["N1"],
            )
            best_state = calibration.pop("best_state")
            baseline_for_decision[layer_index] = phase_a_layers[str(layer_index)][
                "cases"
            ]["N0"]
            calibrated_for_decision[layer_index] = calibration["best_validation"]
            tensor_hashes = {}
            for name, value in best_state.items():
                artifact_name = f"model.layers.{layer_index}.self_attn.{name}"
                artifact_value = value.to(torch.bfloat16).contiguous()
                artifact_tensors[artifact_name] = artifact_value
                tensor_hashes[name] = tensor_sha256(artifact_value)
            phase_b_layers[str(layer_index)] = {
                "layer_index": layer_index,
                "construction_replayed_exactly": True,
                "calibration": calibration,
                "best_trainable_tensor_hashes_bf16": tensor_hashes,
            }
            print(
                f"P6_2C_PHASE_B_LAYER_{layer_index}=BEST "
                f"STEP={calibration['best_step']} "
                f"CORE={calibration['best_validation']['attention_core']['relative_l2_error']:.9g} "
                f"MIXER={calibration['best_validation']['mixer_output']['relative_l2_error']:.9g}",
                flush=True,
            )
            del initial_student, best_state
            torch.cuda.empty_cache()
            report["phase_b"] = {"status": "RUNNING", "layers": phase_b_layers}
            save_json(report_path, report)

        local_decision = local_go_no_go(
            baseline_for_decision,
            calibrated_for_decision,
            attention_core_improvement_min=config["gate"][
                "attention_core_improvement_fraction_min"
            ],
        )
        args.artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = args.artifact_dir / "p6-2c-best-trainable.safetensors"
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
            "base_contract_frozen": all(base_contract_checks.values()),
            "sources_frozen": all(source_checks.values()),
            "split_exact": all(split_checks.values()),
            "inputs_exact": all(input_replay.values()),
            "phase_a_no_training": report["phase_a"]["training_performed"] is False,
            "phase_a_positive_signal": phase_a_decision["passed"] is True,
            "n2_diagnostic_only": all(
                layer["n2_diagnostic_only"] for layer in phase_a_layers.values()
            ),
            "phase_b_fresh_c3": all(
                layer["calibration"]["fresh_c3_initialization"]
                and not layer["calibration"]["p6_2b_checkpoint_resumed"]
                for layer in phase_b_layers.values()
            ),
            "qk_norm_frozen": all(
                layer["calibration"]["frozen_parameter_hash_checks"]["q_norm.weight"]
                and layer["calibration"]["frozen_parameter_hash_checks"]["k_norm.weight"]
                for layer in phase_b_layers.values()
            ),
            "hard_cap_exact": all(
                layer["calibration"]["training_history"][-1]["step"] == 256
                and layer["calibration"]["training_history"][-1][
                    "effective_activation_tokens"
                ]
                == 65_536
                for layer in phase_b_layers.values()
            ),
            "full_dev_not_accessed": True,
            "heldout_not_accessed": True,
            "canonical_not_accessed": True,
            "full_model_checkpoint_not_written": True,
        }
        if not all(audit_checks.values()):
            raise ValueError(f"P6.2c audit checks failed: {audit_checks}")
        passed = local_decision["passed"]
        report.update(
            {
                "status": "GO" if passed else "NO_GO",
                "phase_b": {
                    "status": "GO" if passed else "NO_GO",
                    "fresh_c3_initialization": True,
                    "p6_2b_checkpoint_resumed": False,
                    "training_baseline": "N1_C3_DONOR_QK_NORM_PRESERVED",
                    "formal_gate_baseline": "N0_C3_QK_NORM_REMOVED",
                    "layers": phase_b_layers,
                    "local_decision": local_decision,
                },
                "audit_checks": audit_checks,
                "artifact": artifact_report,
                "p6_2c_scaleout_allowed": passed,
                "result_marker": f"P6_2C={'GO' if passed else 'NO_GO'}",
                "completed_at": now(),
            }
        )
        save_json(report_path, report)
        final = {
            "stage": "P6.2c",
            "status": report["status"],
            "mode": report["mode"],
            "phase_a": phase_a_decision,
            "phase_b_local_decision": local_decision,
            "artifact": artifact_report,
            "formal_kv_lora_rank": 512,
            "full_dev_accessed": False,
            "heldout_accessed": False,
            "canonical_accessed": False,
            "full_model_checkpoint_written": False,
            "p6_2c_scaleout_allowed": passed,
            "p6_3_allowed": False,
            "p7_allowed": False,
            "result_marker": report["result_marker"],
            "completed_at": now(),
        }
        save_json(final_path, final)
        print(report["result_marker"], flush=True)
        print(f"P6_2C_SCALEOUT_ALLOWED={str(passed).lower()}", flush=True)
        print("P6_3_ALLOWED=false", flush=True)
        print("P7_ALLOWED=false", flush=True)
        del model, train, initialization_inputs, fit_inputs, validation_inputs
        gc.collect()
        torch.cuda.empty_cache()
        return 0 if passed else 1
    except Exception as error:  # noqa: BLE001
        report.update(
            {
                "status": "ERROR",
                "exception_type": type(error).__name__,
                "exception": str(error),
                "traceback": traceback.format_exc(),
                "result_marker": "P6_2C=ERROR",
                "p6_2c_scaleout_allowed": False,
                "p6_3_allowed": False,
                "p7_allowed": False,
                "completed_at": now(),
            }
        )
        save_json(report_path, report)
        save_json(
            final_path,
            {
                "stage": "P6.2c",
                "status": "ERROR",
                "exception_type": type(error).__name__,
                "exception": str(error),
                "p6_2c_scaleout_allowed": False,
                "p6_3_allowed": False,
                "p7_allowed": False,
                "result_marker": "P6_2C=ERROR",
                "completed_at": now(),
            },
        )
        print("P6_2C=ERROR", flush=True)
        print("P6_2C_SCALEOUT_ALLOWED=false", flush=True)
        print("P6_3_ALLOWED=false", flush=True)
        print("P7_ALLOWED=false", flush=True)
        print(f"P6_2C_EXCEPTION={type(error).__name__}: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
