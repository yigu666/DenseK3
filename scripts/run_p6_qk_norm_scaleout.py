"""Run P6.2d independent Q/K-norm-preserving calibration and one scaleout."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import statistics
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from densek3_core.evaluation.p4_runtime import load_densek3_model
from densek3_core.modeling.modeling_densek3 import DenseK3GQA, DenseK3MLA
from densek3_core.recovery.p5_corpus import PackedTokenDataset, verify_frozen_corpus
from densek3_core.recovery.p6_function_calibration import (
    TRAINABLE_PARAMETER_NAMES,
    capture_frozen_p5_inputs,
    select_document_disjoint_sequences,
)
from densek3_core.recovery.p6_mla_probe import tensor_sha256
from densek3_core.recovery.p6_scaleout import (
    architecture_shock_recovery,
    collect_frozen_donor_calibration,
    full_model_shock_metrics,
    local_scaleout_safety,
    write_candidate_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load script module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BRIDGE = load_script("p6_2c_bridge_runner", ROOT / "scripts/run_p6_qk_norm_bridge.py")
OLD_SCALEOUT = load_script("p6_2_scaleout_runner", ROOT / "scripts/run_p6_scaleout.py")

DEFAULT_CONFIG = ROOT / "configs/architecture/p6-eight-layer-qk-norm-calibrated-scaleout.yaml"
DEFAULT_BRIDGE_CONFIG = ROOT / "configs/architecture/p6-qk-norm-semantic-bridge.yaml"
DEFAULT_B1_REPORT = (
    ROOT / "outputs/densek3-core/manifests/p6/p6-2b/p6-2b-local-calibration.json"
)
DEFAULT_OLD_P6 = ROOT / "outputs/densek3-core/manifests/p6/p6-2/p6-2-scaleout.json"
DEFAULT_P6C_REPORT = (
    ROOT / "outputs/densek3-core/manifests/p6/p6-2c/p6-2c-qk-norm-bridge.json"
)
DEFAULT_P6C_ARTIFACT = (
    ROOT / "outputs/densek3-core/artifacts/p6/p6-2c-pilot/p6-2c-best-trainable.safetensors"
)
DEFAULT_DATA = ROOT / "data/p5-probe"
DEFAULT_PROMPTS = ROOT / "data/p4-smoke-prompts.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs/densek3-core/manifests/p6/p6-2d"
DEFAULT_LOCAL_ARTIFACT = ROOT / "outputs/densek3-core/artifacts/p6/p6-2d-local"
DEFAULT_CANDIDATE = (
    ROOT
    / "outputs/densek3-core/checkpoints/p6/densek3-4b-8mla-c3-qk-norm-calibrated-candidate"
)
TARGET_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]
REUSED_LAYERS = [3, 23]
NEW_LAYERS = [7, 11, 15, 19, 27, 31]


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
        "stage": config["stage"] == "P6.2d" and config["status"] == "FROZEN",
        "scope": config["scope"]
        == {
            "target_layers": TARGET_LAYERS,
            "reused_layers": REUSED_LAYERS,
            "newly_calibrated_layers": NEW_LAYERS,
            "topology": "24_kda_8_mla",
            "formal_kv_lora_rank": 512,
            "no_pe": True,
            "qk_norm_semantics": "PRESERVE_DONOR_QK_RMSNORM",
            "latent_norm": "SCALE_MATCHED",
            "cache_mode": "expanded_reference",
            "true_latent_cache": False,
            "global_recovery_training": False,
        },
        "initialization": config["initialization"]
        == {
            "algorithm": "ACTIVATION_WEIGHTED_RANK512_PLUS_SCALE_MATCHED_LATENT_NORM",
            "statistics_source": "UNTOUCHED_FROZEN_P5",
            "independent_per_layer": True,
            "sequential_self_conditioning": False,
            "donor_qk_norm_present_from_start": True,
        },
        "data": config["data"]
        == {
            "split": "train",
            "sequence_prefix": 256,
            "c3_initialization_sequences": 8,
            "fit_sequences": 128,
            "validation_sequences": 32,
            "fit_order_repeated_exactly_twice": True,
            "document_disjoint": True,
        },
        "optimization": config["optimization"]["optimizer"] == "AdamW"
        and config["optimization"]["learning_rate"] == 0.0002
        and config["optimization"]["betas"] == [0.9, 0.95]
        and config["optimization"]["epsilon"] == 1.0e-8
        and config["optimization"]["weight_decay"] == 0.0
        and config["optimization"]["gradient_clip"] == 1.0
        and config["optimization"]["maximum_steps_per_layer"] == 256
        and config["optimization"]["maximum_activation_tokens_per_layer"] == 65_536
        and config["optimization"]["evaluation_steps"]
        == [0, 32, 64, 96, 128, 160, 192, 224, 256]
        and config["optimization"]["scheduler"] == "constant"
        and config["optimization"]["seed"] == 20260814
        and config["optimization"]["best_checkpoint_within_budget"] is True,
        "loss": config["loss"]
        == {
            "teacher": "PER_LAYER_FULL_RANK_NOPE_B",
            "attention_core_normalized_mse_weight": 1.0,
            "mixer_output_normalized_mse_weight": 1.0,
            "all_other_weights": 0.0,
        },
        "trainable": config["trainable"]["parameter_names"]
        == list(TRAINABLE_PARAMETER_NAMES),
        "local_safety": config["local_safety"]
        == {
            "fresh_n1_mixer_relative_l2_max": 0.70,
            "fresh_n1_is_step_zero_safety_candidate": True,
            "calibrated_mixer_must_not_exceed_n1": True,
            "calibrated_mixer_relative_l2_max": 0.70,
            "finite_required": True,
            "no_shared_layer23_absolute_gate": True,
        },
        "evaluation": config["evaluation"]
        == {
            "unique_full_model_candidate": True,
            "candidate_checkpoint_written_before_full_dev": True,
            "full_dev_sequences": 97,
            "full_dev_evaluations": 1,
            "full_dev_role": "primary_gate",
            "canonical_accessed": False,
            "heldout_accessed": False,
            "generation_categories": 5,
            "record_full_model_drift": True,
        },
        "gate": config["gate"]["p5_full_dev_ce"] == 2.306309482533407
        and config["gate"]["full_dev_ce_delta_vs_p5_max"] == 0.15
        and config["gate"]["generation_not_collapsed_required"] is True
        and config["gate"]["finite_required"] is True
        and config["gate"]["no_new_drift_thresholds"] is True,
        "comparison": config["comparison"]["old_p6_2_full_dev_ce"]
        == 2.7116558523606864
        and config["comparison"]["report_absolute_ce_recovered"] is True
        and config["comparison"]["report_old_architecture_shock_recovery_fraction"] is True,
        "forbidden": all(config["forbidden"].values()),
    }
    if not all(checks.values()):
        raise ValueError(f"P6.2d frozen config failed: {checks}")
    return checks


def donor_parameter_hashes(donors: dict[int, DenseK3GQA]) -> dict[str, str]:
    return {
        f"model.layers.{layer_index}.self_attn.{name}": tensor_sha256(value)
        for layer_index, donor in donors.items()
        for name, value in donor.named_parameters()
    }


def state_tensor_hashes(state: dict[str, torch.Tensor]) -> dict[str, str]:
    return {name: tensor_sha256(value.to(torch.bfloat16)) for name, value in state.items()}


def metric_improvements(
    n0: dict[str, Any],
    n1: dict[str, Any],
    best: dict[str, Any],
) -> dict[str, float]:
    result = {}
    for field in ("attention_core", "mixer_output", "post_attention_residual_hidden"):
        n0_value = float(n0[field]["relative_l2_error"])
        n1_value = float(n1[field]["relative_l2_error"])
        best_value = float(best[field]["relative_l2_error"])
        result[f"{field}_n0_to_n1"] = (n0_value - n1_value) / max(
            n0_value,
            torch.finfo(torch.float64).eps,
        )
        result[f"{field}_n1_to_best"] = (n1_value - best_value) / max(
            n1_value,
            torch.finfo(torch.float64).eps,
        )
        result[f"{field}_n0_to_best"] = (n0_value - best_value) / max(
            n0_value,
            torch.finfo(torch.float64).eps,
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--p6-2c-config", type=Path, default=DEFAULT_BRIDGE_CONFIG)
    parser.add_argument("--p6-2b-1-report", type=Path, default=DEFAULT_B1_REPORT)
    parser.add_argument("--old-p6-2-report", type=Path, default=DEFAULT_OLD_P6)
    parser.add_argument("--p6-2c-report", type=Path, default=DEFAULT_P6C_REPORT)
    parser.add_argument("--p6-2c-artifact", type=Path, default=DEFAULT_P6C_ARTIFACT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--local-artifact-dir", type=Path, default=DEFAULT_LOCAL_ARTIFACT)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in (
        "config",
        "p6_2c_config",
        "p6_2b_1_report",
        "old_p6_2_report",
        "p6_2c_report",
        "p6_2c_artifact",
        "data_dir",
        "prompts",
        "output_dir",
        "local_artifact_dir",
        "candidate_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "p6-2d-scaleout.json"
    final_path = args.output_dir / "p6-2d-final-check.json"
    report: dict[str, Any] = {
        "schema_version": 1,
        "stage": "P6.2d",
        "status": "RUNNING",
        "mode": "EIGHT_LAYER_INDEPENDENT_QK_NORM_PRESERVING_FUNCTION_CALIBRATION_SCALEOUT",
        "formal_kv_lora_rank": 512,
        "qk_norm_semantics": "PRESERVE_DONOR_QK_RMSNORM",
        "full_dev_evaluations": 0,
        "canonical_accessed": False,
        "heldout_accessed": False,
        "global_recovery_training_performed": False,
        "p6_3_allowed": False,
        "p7_allowed": False,
    }
    print("P6_2D_EIGHT_LAYER_SCALEOUT=RUNNING", flush=True)
    print("P6_FORMAL_KV_LORA_RANK=512", flush=True)
    print("P6_QK_NORM_SEMANTICS=PRESERVE_DONOR_QK_RMSNORM", flush=True)
    print("P6_3_ALLOWED=false", flush=True)
    print("P7_ALLOWED=false", flush=True)
    try:
        from transformers import AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("P6.2d requires the authoritative CUDA server")
        config = load_yaml(args.config)
        config_checks = validate_config(config)
        old_p6 = load_json(args.old_p6_2_report)
        p6c = load_json(args.p6_2c_report)
        b1 = load_json(args.p6_2b_1_report)
        configured_p5_checkpoint = (ROOT / config["source"]["p5_checkpoint"]).resolve()
        source_checks = {
            "old_p6_redesign": old_p6["status"] == "REDESIGN"
            and old_p6["full_dev"]["p6_candidate_ce"]
            == config["comparison"]["old_p6_2_full_dev_ce"],
            "p6c_go": p6c["status"] == "GO"
            and p6c["p6_2c_scaleout_allowed"] is True,
            "p6c_artifact_hash": sha256_file(args.p6_2c_artifact)
            == config["source"]["p6_2c_artifact_sha256"]
            == p6c["artifact"]["sha256"],
            "p6c_config_hash": sha256_file(args.p6_2c_config)
            == config["source"]["p6_2c_config_sha256"],
            "p6c_config_valid": all(p6c["config_checks"].values()),
            "p6c_qk_frozen": all(
                p6c["phase_b"]["layers"][str(index)]["calibration"][
                    "frozen_parameter_hash_checks"
                ][name]
                for index in REUSED_LAYERS
                for name in ("q_norm.weight", "k_norm.weight")
            ),
            "p5_baseline": old_p6["full_dev"]["p5_baseline_ce"]
            == config["gate"]["p5_full_dev_ce"],
            "same_p5": Path(p6c["source_paths"]["p5_checkpoint"]).resolve()
            == Path(b1["source_paths"]["p5_checkpoint"]).resolve(),
            "configured_p5": configured_p5_checkpoint
            == Path(b1["source_paths"]["p5_checkpoint"]).resolve(),
            "no_heldout": p6c["heldout_accessed"] is False
            and old_p6["heldout_accessed"] is False,
        }
        if not all(source_checks.values()):
            raise ValueError(f"P6.2d source checks failed: {source_checks}")

        corpus = verify_frozen_corpus(args.data_dir)
        if corpus["p5_probe_corpus_frozen"] is not True:
            raise ValueError("P6.2d frozen corpus verification failed")
        corpus_manifest = load_json(args.data_dir / "packed/corpus-manifest.json")
        split = select_document_disjoint_sequences(
            corpus_manifest,
            fit_sequences=config["data"]["fit_sequences"],
            validation_sequences=config["data"]["validation_sequences"],
        )
        split_checks = {
            "fit_indices_replay": split["fit_sequence_indices"]
            == b1["document_disjoint_split"]["fit_sequence_indices"],
            "validation_indices_replay": split["validation_sequence_indices"]
            == b1["document_disjoint_split"]["validation_sequence_indices"],
            "document_overlap_zero": split["document_overlap_count"] == 0,
        }
        if not all(split_checks.values()):
            raise ValueError(f"P6.2d split replay failed: {split_checks}")

        optimization = {
            key: config["optimization"][key]
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
        previous_optimizer = b1["layers"]["3"]["calibration"]["optimizer"]
        unchanged_optimization = {
            key: optimization[key] == previous_optimizer[key]
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
        if not all(unchanged_optimization.values()):
            raise ValueError(f"P6.2d optimization changed: {unchanged_optimization}")

        torch.manual_seed(optimization["seed"])
        torch.cuda.manual_seed_all(optimization["seed"])
        device = torch.device("cuda")
        p5_checkpoint = Path(b1["source_paths"]["p5_checkpoint"]).resolve()
        tokenizer = AutoTokenizer.from_pretrained(p5_checkpoint, trust_remote_code=False)
        prompts = OLD_SCALEOUT.load_prompts(args.prompts)
        train = PackedTokenDataset(args.data_dir / "packed", "train")
        model = load_densek3_model(p5_checkpoint, device=device, dtype=torch.bfloat16)
        model.requires_grad_(False)
        calibration = collect_frozen_donor_calibration(
            model,
            train,
            TARGET_LAYERS,
            list(range(config["data"]["c3_initialization_sequences"])),
            selection_prefix=config["data"]["sequence_prefix"],
        )
        fit_inputs = capture_frozen_p5_inputs(
            model,
            train,
            TARGET_LAYERS,
            split["fit_sequence_indices"],
            token_prefix=config["data"]["sequence_prefix"],
            marker="P6_2D_FIT_INPUT_CAPTURE",
        )
        validation_inputs = capture_frozen_p5_inputs(
            model,
            train,
            TARGET_LAYERS,
            split["validation_sequence_indices"],
            token_prefix=config["data"]["sequence_prefix"],
            marker="P6_2D_VALIDATION_INPUT_CAPTURE",
        )
        input_checks = {
            "frozen_donor": calibration["manifest"]["all_statistics_from_untouched_p5_donor"]
            is True,
            "no_sequential_conditioning": calibration["manifest"]["sequential_self_conditioning"]
            is False,
            "fit_source": fit_inputs["manifest"]["source_model"] == "UNTOUCHED_FROZEN_P5",
            "validation_source": validation_inputs["manifest"]["source_model"]
            == "UNTOUCHED_FROZEN_P5",
            "fit_token_replay": fit_inputs["manifest"]["token_sha256"]
            == b1["fit_input_manifest"]["token_sha256"],
            "validation_token_replay": validation_inputs["manifest"]["token_sha256"]
            == b1["validation_input_manifest"]["token_sha256"],
        }
        if not all(input_checks.values()):
            raise ValueError(f"P6.2d frozen input capture failed: {input_checks}")

        donors = {}
        for layer_index in TARGET_LAYERS:
            donor = model.model.layers[layer_index].self_attn
            if not isinstance(donor, DenseK3GQA) or donor.layer_idx != layer_index:
                raise ValueError(f"P6.2d layer {layer_index} is not frozen donor GQA")
            donors[layer_index] = donor
        donor_hashes_before = donor_parameter_hashes(donors)
        p6c_artifact = load_file(str(args.p6_2c_artifact), device="cpu")
        initial_reports = {}
        selected_states: dict[int, dict[str, torch.Tensor]] = {}
        calibration_reports = {}

        for layer_index in TARGET_LAYERS:
            donor = donors[layer_index]
            n0, n1, construction = BRIDGE.build_c3_students(
                donor,
                calibration["normalized"][layer_index],
                device=device,
            )
            n0_metrics = BRIDGE.local_validation_metrics(
                donor,
                n0,
                validation_inputs["residual"][layer_index],
                validation_inputs["normalized"][layer_index],
            )
            n1_metrics = BRIDGE.local_validation_metrics(
                donor,
                n1,
                validation_inputs["residual"][layer_index],
                validation_inputs["normalized"][layer_index],
            )
            finite = BRIDGE.metrics_are_finite(n0_metrics) and BRIDGE.metrics_are_finite(
                n1_metrics
            )
            qk_hashes = {
                "q_norm": tensor_sha256(n1.q_norm.weight),
                "k_norm": tensor_sha256(n1.k_norm.weight),
                "donor_q_norm": tensor_sha256(donor.q_norm.weight),
                "donor_k_norm": tensor_sha256(donor.k_norm.weight),
            }
            qk_hashes["q_exact"] = qk_hashes["q_norm"] == qk_hashes["donor_q_norm"]
            qk_hashes["k_exact"] = qk_hashes["k_norm"] == qk_hashes["donor_k_norm"]
            if not finite or not qk_hashes["q_exact"] or not qk_hashes["k_exact"]:
                raise ValueError(f"P6.2d layer {layer_index} fresh N1 construction failed")
            initial_reports[str(layer_index)] = {
                "construction": construction,
                "n0": n0_metrics,
                "n1": n1_metrics,
                "finite": finite,
                "qk_norm_hashes": qk_hashes,
            }
            if layer_index in REUSED_LAYERS:
                state = BRIDGE.artifact_state_for_layer(p6c_artifact, layer_index)
                tensor_checks = {
                    name: tensor_sha256(value.to(torch.bfloat16))
                    == p6c["phase_b"]["layers"][str(layer_index)][
                        "best_trainable_tensor_hashes_bf16"
                    ][name]
                    for name, value in state.items()
                }
                if not all(tensor_checks.values()):
                    raise ValueError(f"P6.2d layer {layer_index} reused tensor hash failed")
                BRIDGE.load_trainable_state(n1, state)
                best_metrics = BRIDGE.local_validation_metrics(
                    donor,
                    n1,
                    validation_inputs["residual"][layer_index],
                    validation_inputs["normalized"][layer_index],
                )
                expected = p6c["phase_b"]["layers"][str(layer_index)]["calibration"][
                    "best_validation"
                ]
                reuse_checks = {
                    "source_artifact": source_checks["p6c_artifact_hash"],
                    "calibration_config": source_checks["p6c_config_hash"],
                    "tensor_hashes": all(tensor_checks.values()),
                    "qk_norm_frozen": all(
                        p6c["phase_b"]["layers"][str(layer_index)]["calibration"][
                            "frozen_parameter_hash_checks"
                        ][name]
                        for name in ("q_norm.weight", "k_norm.weight")
                    ),
                    "validation_metrics_exact": best_metrics == expected,
                }
                if not all(reuse_checks.values()):
                    raise ValueError(f"P6.2d layer {layer_index} reuse failed: {reuse_checks}")
                selected_states[layer_index] = state
                calibration_reports[str(layer_index)] = {
                    "mode": "REUSED_P6_2C_VALIDATED_ARTIFACT",
                    "best_step": p6c["phase_b"]["layers"][str(layer_index)]["calibration"][
                        "best_step"
                    ],
                    "best_effective_activation_tokens": p6c["phase_b"]["layers"][
                        str(layer_index)
                    ]["calibration"]["best_effective_activation_tokens"],
                    "baseline_validation": n1_metrics,
                    "best_validation": best_metrics,
                    "reuse_checks": reuse_checks,
                    "best_trainable_tensor_hashes_bf16": state_tensor_hashes(state),
                }
            print(
                f"P6_2D_LAYER_{layer_index}_FRESH_N1 "
                f"N0_MIXER={n0_metrics['mixer_output']['relative_l2_error']:.9g} "
                f"N1_MIXER={n1_metrics['mixer_output']['relative_l2_error']:.9g}",
                flush=True,
            )
            del n0, n1
            torch.cuda.empty_cache()

        fresh_n1_checks = {
            str(index): initial_reports[str(index)]["n1"]["mixer_output"][
                "relative_l2_error"
            ]
            <= config["local_safety"]["fresh_n1_mixer_relative_l2_max"]
            for index in TARGET_LAYERS
        }
        report.update(
            {
                "source_paths": {
                    "p5_checkpoint": str(p5_checkpoint),
                    "old_p6_2_report": str(args.old_p6_2_report),
                    "p6_2c_report": str(args.p6_2c_report),
                    "p6_2c_artifact": str(args.p6_2c_artifact),
                    "frozen_corpus": str(args.data_dir),
                },
                "config_checks": config_checks,
                "source_checks": source_checks,
                "split_checks": split_checks,
                "unchanged_optimization": unchanged_optimization,
                "input_checks": input_checks,
                "calibration_manifest": calibration["manifest"],
                "fit_input_manifest": fit_inputs["manifest"],
                "validation_input_manifest": validation_inputs["manifest"],
                "initial_per_layer": initial_reports,
                "fresh_n1_checks": fresh_n1_checks,
                "local_calibration": calibration_reports,
            }
        )
        save_json(report_path, report)
        if not all(fresh_n1_checks.values()):
            report.update(
                {
                    "status": "REDESIGN_FRESH_N1_OUTLIER",
                    "result_marker": "P6_2D_EIGHT_LAYER_SCALEOUT=REDESIGN_FRESH_N1_OUTLIER",
                    "completed_at": now(),
                }
            )
            save_json(report_path, report)
            save_json(
                final_path,
                {
                    "stage": "P6.2d",
                    "status": report["status"],
                    "p6_3_allowed": False,
                    "p7_allowed": False,
                    "result_marker": report["result_marker"],
                    "completed_at": now(),
                },
            )
            print(report["result_marker"], flush=True)
            return 1

        for layer_index in NEW_LAYERS:
            donor = donors[layer_index]
            n0, initial_student, construction = BRIDGE.build_c3_students(
                donor,
                calibration["normalized"][layer_index],
                device=device,
            )
            del n0
            if construction != initial_reports[str(layer_index)]["construction"]:
                raise ValueError(f"P6.2d layer {layer_index} fresh construction changed")
            print(f"P6_2D_LAYER_{layer_index}=CALIBRATING", flush=True)
            calibrated = BRIDGE.calibrate_bridge_layer(
                donor,
                initial_student,
                fit_inputs["normalized"][layer_index],
                validation_inputs["residual"][layer_index],
                validation_inputs["normalized"][layer_index],
                optimization,
                initial_reports[str(layer_index)]["n1"],
                selection_mixer_relative_l2_max=initial_reports[str(layer_index)]["n1"][
                    "mixer_output"
                ]["relative_l2_error"],
            )
            state = calibrated.pop("best_state")
            selected_states[layer_index] = state
            calibration_reports[str(layer_index)] = {
                "mode": "FRESH_N1_INDEPENDENT_LOCAL_CALIBRATION",
                **calibrated,
                "best_trainable_tensor_hashes_bf16": state_tensor_hashes(state),
            }
            print(
                f"P6_2D_LAYER_{layer_index}=BEST STEP={calibrated['best_step']} "
                f"CORE={calibrated['best_validation']['attention_core']['relative_l2_error']:.9g} "
                f"MIXER={calibrated['best_validation']['mixer_output']['relative_l2_error']:.9g}",
                flush=True,
            )
            del initial_student
            torch.cuda.empty_cache()
            report["local_calibration"] = calibration_reports
            save_json(report_path, report)

        donor_hashes_after = donor_parameter_hashes(donors)
        donor_hash_checks = {
            name: donor_hashes_before[name] == donor_hashes_after[name]
            for name in donor_hashes_before
        }
        if not all(donor_hash_checks.values()):
            raise ValueError("P6.2d frozen P5 donor changed during independent calibration")
        safety_input = {
            index: {
                "n1": initial_reports[str(index)]["n1"],
                "best": calibration_reports[str(index)]["best_validation"],
                "finite": BRIDGE.metrics_are_finite(initial_reports[str(index)]["n1"])
                and BRIDGE.metrics_are_finite(
                    calibration_reports[str(index)]["best_validation"]
                ),
            }
            for index in TARGET_LAYERS
        }
        local_safety = local_scaleout_safety(
            safety_input,
            mixer_relative_l2_max=config["local_safety"][
                "calibrated_mixer_relative_l2_max"
            ],
        )
        per_layer_summary = {
            str(index): {
                "n0": initial_reports[str(index)]["n0"],
                "n1": initial_reports[str(index)]["n1"],
                "best": calibration_reports[str(index)]["best_validation"],
                "best_step": calibration_reports[str(index)]["best_step"],
                "best_effective_activation_tokens": calibration_reports[str(index)][
                    "best_effective_activation_tokens"
                ],
                "improvements": metric_improvements(
                    initial_reports[str(index)]["n0"],
                    initial_reports[str(index)]["n1"],
                    calibration_reports[str(index)]["best_validation"],
                ),
                "safety": local_safety["per_layer"][str(index)],
            }
            for index in TARGET_LAYERS
        }
        artifact_tensors = {}
        for layer_index, state in selected_states.items():
            for name, value in state.items():
                artifact_tensors[
                    f"model.layers.{layer_index}.self_attn.{name}"
                ] = value.to(torch.bfloat16).contiguous()
        args.local_artifact_dir.mkdir(parents=True, exist_ok=True)
        local_artifact_path = (
            args.local_artifact_dir / "p6-2d-eight-layer-best-trainable.safetensors"
        )
        temporary_artifact = local_artifact_path.with_suffix(".safetensors.tmp")
        save_file(artifact_tensors, str(temporary_artifact))
        temporary_artifact.replace(local_artifact_path)
        local_artifact = {
            "path": str(local_artifact_path),
            "sha256": sha256_file(local_artifact_path),
            "tensor_count": len(artifact_tensors),
            "tensor_names": sorted(artifact_tensors),
            "full_model_checkpoint": False,
        }
        report.update(
            {
                "local_calibration": calibration_reports,
                "per_layer_summary": per_layer_summary,
                "local_safety": local_safety,
                "frozen_donor_parameter_hash_checks": donor_hash_checks,
                "local_artifact": local_artifact,
            }
        )
        save_json(report_path, report)
        if not local_safety["passed"]:
            report.update(
                {
                    "status": "REDESIGN_LOCAL_SAFETY",
                    "result_marker": "P6_2D_EIGHT_LAYER_SCALEOUT=REDESIGN_LOCAL_SAFETY",
                    "completed_at": now(),
                }
            )
            save_json(report_path, report)
            save_json(
                final_path,
                {
                    "stage": "P6.2d",
                    "status": report["status"],
                    "local_artifact": local_artifact,
                    "p6_3_allowed": False,
                    "p7_allowed": False,
                    "result_marker": report["result_marker"],
                    "completed_at": now(),
                },
            )
            print(report["result_marker"], flush=True)
            return 1

        model.config.mla_qk_norm = True
        native_checks = {}
        for layer_index in TARGET_LAYERS:
            donor = donors[layer_index]
            n0, probe, construction = BRIDGE.build_c3_students(
                donor,
                calibration["normalized"][layer_index],
                device=device,
            )
            del n0
            if construction != initial_reports[str(layer_index)]["construction"]:
                raise ValueError(f"P6.2d layer {layer_index} native construction changed")
            BRIDGE.load_trainable_state(probe, selected_states[layer_index])
            native = DenseK3MLA(model.config, layer_index).to(
                device=device,
                dtype=next(probe.parameters()).dtype,
            )
            native.load_state_dict(probe.state_dict(), strict=True)
            native.eval()
            native_metrics = BRIDGE.local_validation_metrics(
                donor,
                native,
                validation_inputs["residual"][layer_index],
                validation_inputs["normalized"][layer_index],
            )
            expected_metrics = calibration_reports[str(layer_index)]["best_validation"]
            checks = {
                "apply_qk_norm": native.apply_qk_norm is True,
                "q_norm_exact": tensor_sha256(native.q_norm.weight)
                == tensor_sha256(donor.q_norm.weight),
                "k_norm_exact": tensor_sha256(native.k_norm.weight)
                == tensor_sha256(donor.k_norm.weight),
                "local_metrics_exact": native_metrics == expected_metrics,
            }
            if not all(checks.values()):
                raise ValueError(f"P6.2d layer {layer_index} native replay failed: {checks}")
            native_checks[str(layer_index)] = checks
            layer = model.model.layers[layer_index]
            layer.self_attn = native
            layer.layer_type = "mla"
            model.config.mixer_types[layer_index] = "mla"
            model.config.layer_types[layer_index] = "full_attention"
            del probe
            torch.cuda.empty_cache()

        model.config.validate_architecture()
        topology_checks = {
            "kda_24": len(model.config.kda_indices) == 24,
            "mla_8": model.config.mla_indices == TARGET_LAYERS,
            "gqa_0": not model.config.gqa_indices,
            "rank_512": model.config.mla_kv_lora_rank == 512,
            "no_pe": model.config.mla_position_encoding == "none",
            "qk_norm_preserved": model.config.mla_qk_norm is True,
            "expanded_reference_cache": model.config.mla_cache_mode == "expanded_reference",
        }
        if not all(topology_checks.values()):
            raise ValueError(f"P6.2d assembled topology failed: {topology_checks}")
        candidate_provenance = {
            "p5_checkpoint": str(p5_checkpoint),
            "p6_2c_report": str(args.p6_2c_report),
            "p6_2c_artifact_sha256": config["source"]["p6_2c_artifact_sha256"],
            "local_artifact_sha256": local_artifact["sha256"],
            "target_layers": TARGET_LAYERS,
            "reused_layers": REUSED_LAYERS,
            "newly_calibrated_layers": NEW_LAYERS,
            "qk_norm_semantics": "PRESERVE_DONOR_QK_RMSNORM",
            "per_layer_best_steps": {
                str(index): calibration_reports[str(index)]["best_step"]
                for index in TARGET_LAYERS
            },
        }
        candidate_artifact = write_candidate_checkpoint(
            model,
            args.candidate_dir,
            p5_checkpoint,
            candidate_provenance,
            stage="P6.2d-CANDIDATE",
        )
        report.update(
            {
                "native_replay_checks": native_checks,
                "topology_checks": topology_checks,
                "candidate_artifact": candidate_artifact,
                "status": "CANDIDATE_WRITTEN_FULL_DEV_PENDING",
            }
        )
        save_json(report_path, report)
        print("P6_2D_CANDIDATE_CHECKPOINT=WRITTEN", flush=True)

        shock = full_model_shock_metrics(
            model,
            train,
            list(range(config["data"]["c3_initialization_sequences"])),
            calibration["baseline_curves"],
            calibration["baseline_final"],
            prefix=config["data"]["sequence_prefix"],
        )
        dev = PackedTokenDataset(args.data_dir / "packed", "dev")
        candidate_full_dev = OLD_SCALEOUT.full_dev_ce(model, dev)
        report["full_dev_evaluations"] = 1
        generation = OLD_SCALEOUT.generation_report(model, tokenizer, prompts)
        p5_ce = config["gate"]["p5_full_dev_ce"]
        full_dev_delta = candidate_full_dev - p5_ce
        recovery = architecture_shock_recovery(
            p5_ce=p5_ce,
            old_p6_ce=config["comparison"]["old_p6_2_full_dev_ce"],
            new_p6_ce=candidate_full_dev,
        )
        gate_checks = {
            "local_safety": local_safety["passed"] is True,
            "full_dev_ce": full_dev_delta <= config["gate"]["full_dev_ce_delta_vs_p5_max"],
            "finite": math.isfinite(candidate_full_dev)
            and OLD_SCALEOUT.finite_metrics(shock),
            "generation_not_collapsed": generation["not_collapsed"] is True,
            "topology": all(topology_checks.values()),
            "native_replay": all(all(item.values()) for item in native_checks.values()),
            "single_full_dev_evaluation": report["full_dev_evaluations"] == 1,
            "canonical_not_accessed": True,
            "heldout_not_accessed": True,
        }
        passed = all(gate_checks.values())
        mixer_values = [
            per_layer_summary[str(index)]["best"]["mixer_output"]["relative_l2_error"]
            for index in TARGET_LAYERS
        ]
        report.update(
            {
                "status": "PASS" if passed else "REDESIGN_FULL_MODEL",
                "full_model_shock": shock,
                "local_summary": {
                    "median_best_mixer_relative_l2": statistics.median(mixer_values),
                    "worst_best_mixer_relative_l2": max(mixer_values),
                    "worst_layer": TARGET_LAYERS[
                        max(range(len(mixer_values)), key=mixer_values.__getitem__)
                    ],
                },
                "full_dev": {
                    "p5_baseline_ce": p5_ce,
                    "p6_2d_candidate_ce": candidate_full_dev,
                    "delta": full_dev_delta,
                    "maximum_delta": config["gate"]["full_dev_ce_delta_vs_p5_max"],
                    "maximum_ce": p5_ce + config["gate"]["full_dev_ce_delta_vs_p5_max"],
                    "primary_gate": True,
                    "evaluation_count": 1,
                },
                "old_p6_2_comparison": recovery,
                "generation": generation,
                "gate": {"checks": gate_checks, "passed": passed},
                "p6_3_allowed": passed,
                "result_marker": "P6_2D_EIGHT_LAYER_SCALEOUT=PASS"
                if passed
                else "P6_2D_EIGHT_LAYER_SCALEOUT=REDESIGN_FULL_MODEL",
                "completed_at": now(),
            }
        )
        save_json(report_path, report)
        final = {
            "stage": "P6.2d",
            "status": report["status"],
            "formal_kv_lora_rank": 512,
            "qk_norm_semantics": "PRESERVE_DONOR_QK_RMSNORM",
            "candidate_directory": candidate_artifact["directory"],
            "full_dev": report["full_dev"],
            "old_p6_2_comparison": recovery,
            "gate": report["gate"],
            "canonical_accessed": False,
            "heldout_accessed": False,
            "p6_frozen": False,
            "p6_3_allowed": passed,
            "p7_allowed": False,
            "result_marker": report["result_marker"],
            "completed_at": now(),
        }
        save_json(final_path, final)
        print(report["result_marker"], flush=True)
        print(f"P6_2D_FULL_DEV_CE={candidate_full_dev:.12f}", flush=True)
        print(f"P6_2D_FULL_DEV_DELTA_VS_P5={full_dev_delta:+.12f}", flush=True)
        print(
            "P6_2D_OLD_SHOCK_RECOVERY_FRACTION="
            f"{recovery['old_architecture_shock_recovery_fraction']:.12f}",
            flush=True,
        )
        print(f"P6_3_ALLOWED={str(passed).lower()}", flush=True)
        print("P7_ALLOWED=false", flush=True)
        del model, train, dev, calibration, fit_inputs, validation_inputs
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
                "p6_3_allowed": False,
                "p7_allowed": False,
                "result_marker": "P6_2D_EIGHT_LAYER_SCALEOUT=ERROR",
                "completed_at": now(),
            }
        )
        save_json(report_path, report)
        save_json(
            final_path,
            {
                "stage": "P6.2d",
                "status": "ERROR",
                "exception_type": type(error).__name__,
                "exception": str(error),
                "p6_3_allowed": False,
                "p7_allowed": False,
                "result_marker": report["result_marker"],
                "completed_at": now(),
            },
        )
        print("P6_2D_EIGHT_LAYER_SCALEOUT=ERROR", flush=True)
        print("P6_3_ALLOWED=false", flush=True)
        print("P7_ALLOWED=false", flush=True)
        print(f"P6_2D_EXCEPTION={type(error).__name__}: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
