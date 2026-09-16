"""Run the frozen P6.2a teacher-forced rank-capacity diagnostic."""

from __future__ import annotations

import argparse
import gc
import json
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from densek3_core.evaluation.p4_runtime import load_densek3_model
from densek3_core.modeling.modeling_densek3 import DenseK3GQA
from densek3_core.recovery.p5_corpus import PackedTokenDataset, verify_frozen_corpus
from densek3_core.recovery.p6_capacity import (
    aggregate_capacity_diagnostics,
    capacity_diagnostics,
    collect_scaleout_inputs,
    latent_rms_statistics,
)
from densek3_core.recovery.p6_initialization import (
    activation_weighted_rank_factorization,
    apply_rmsnorm_scale,
    covariance_from_batches,
    fit_scale_matched_rmsnorm,
)
from densek3_core.recovery.p6_mla_probe import P6NoPEMLA, tensor_sha256
from densek3_core.recovery.p6_scaleout import (
    collect_frozen_donor_calibration,
    local_layer_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/architecture/p6-teacher-forced-rank-capacity.yaml"
DEFAULT_P5 = ROOT / "outputs/densek3-core/checkpoints/p5/densek3-4b-k3-decay-silu-i1"
DEFAULT_P6_1C = ROOT / "outputs/densek3-core/manifests/p6/p6-1c/p6-1c-initialization.json"
DEFAULT_P6_2 = ROOT / "outputs/densek3-core/manifests/p6/p6-2/p6-2-scaleout.json"
DEFAULT_DATA = ROOT / "data/p5-probe"
DEFAULT_OUTPUT = ROOT / "outputs/densek3-core/manifests/p6/p6-2a"
TARGET_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]
DIAGNOSTIC_RANKS = [512, 768, 1024]
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
        "stage": config["stage"] == "P6.2a" and config["status"] == "FROZEN",
        "source": config["source"]["required_initialization"] == INITIALIZATION,
        "layers": config["scope"]["target_layers"] == TARGET_LAYERS,
        "rank_contract": config["scope"]["formal_kv_lora_rank"] == 512
        and config["scope"]["diagnostic_ranks"] == DIAGNOSTIC_RANKS,
        "no_mutation": config["scope"]["training"] is False
        and config["scope"]["optimizer"] is False
        and config["scope"]["checkpoint_write"] is False
        and config["scope"]["heldout_accessed"] is False,
        "calibration": config["calibration"]
        == {
            "split": "train",
            "sequence_indices": list(range(8)),
            "sequence_length": 2048,
            "evaluation_prefix": 256,
            "statistics_source": "untouched_frozen_p5_donor",
            "teacher_forced_per_layer": True,
            "scaleout_input_capture_rank": 512,
            "covariance_dtype": "float32",
            "factorization_dtype": "float32",
            "deterministic_exact": True,
        },
        "local_only": config["evaluation"]["full_dev"] is False
        and config["evaluation"]["canonical"] is False
        and config["evaluation"]["generation"] is False
        and config["evaluation"]["compare_true_scaleout_inputs"] is True,
        "interpretation": config["interpretation"]["thresholds_are_diagnostic_only"] is True
        and config["interpretation"]["continuous_metrics_are_primary"] is True,
        "forbidden": all(config["forbidden"].values()),
    }
    if not all(checks.values()):
        raise ValueError(f"P6.2a frozen config failed: {checks}")
    return checks


def finite_metrics(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_metrics(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_metrics(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--p5-checkpoint", type=Path, default=DEFAULT_P5)
    parser.add_argument("--p6-1c-report", type=Path, default=DEFAULT_P6_1C)
    parser.add_argument("--p6-2-report", type=Path, default=DEFAULT_P6_2)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in ("config", "p5_checkpoint", "p6_1c_report", "p6_2_report", "data_dir", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "p6-2a-capacity.json"
    final_path = args.output_dir / "p6-2a-final-check.json"
    print("P6_2A_TEACHER_FORCED_RANK_CAPACITY=RUNNING", flush=True)
    print("P6_FORMAL_KV_LORA_RANK=512", flush=True)
    print("P6_TRAINING_ALLOWED=false", flush=True)
    print("P6_3_ALLOWED=false", flush=True)
    print("P7_ALLOWED=false", flush=True)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("P6.2a requires the authoritative CUDA server")
        config = load_yaml(args.config)
        config_checks = validate_config(config)
        p6_1c = load_json(args.p6_1c_report)
        p6_2 = load_json(args.p6_2_report)
        source_checks = {
            "p6_1c": p6_1c["status"] == "PASS"
            and p6_1c["selection"]["selected_case"] == "C3",
            "p6_2_frozen_redesign": p6_2["status"] == "REDESIGN"
            and p6_2["p6_3_allowed"] is False,
            "p6_2_full_dev_failure_unchanged": p6_2["full_dev"]["delta"]
            > p6_2["full_dev"]["maximum_delta"],
            "heldout_never_accessed": p6_1c["heldout_accessed"] is False
            and p6_2["heldout_accessed"] is False,
            "same_p5": Path(p6_1c["source_paths"]["p5_checkpoint"]).resolve()
            == args.p5_checkpoint
            and Path(p6_2["source_paths"]["p5_checkpoint"]).resolve() == args.p5_checkpoint,
        }
        if not all(source_checks.values()):
            raise ValueError(f"P6.2a source checks failed: {source_checks}")
        corpus = verify_frozen_corpus(args.data_dir)
        if corpus["p5_probe_corpus_frozen"] is not True:
            raise ValueError("P6.2a frozen corpus verification failed")

        device = torch.device("cuda")
        train = PackedTokenDataset(args.data_dir / "packed", "train")
        model = load_densek3_model(args.p5_checkpoint, device=device, dtype=torch.bfloat16)
        calibration = collect_frozen_donor_calibration(
            model,
            train,
            TARGET_LAYERS,
            list(range(8)),
            selection_prefix=256,
        )
        calibration_replay = {
            "token_hashes": calibration["manifest"]["token_sha256"]
            == p6_2["calibration_manifest"]["token_sha256"],
            "normalized_input_hashes": calibration["manifest"][
                "per_layer_normalized_input_sha256"
            ]
            == p6_2["calibration_manifest"]["per_layer_normalized_input_sha256"],
            "untouched_donor": calibration["manifest"][
                "all_statistics_from_untouched_p5_donor"
            ]
            is True,
            "no_self_conditioning": calibration["manifest"]["sequential_self_conditioning"]
            is False,
        }
        if not all(calibration_replay.values()):
            raise ValueError(f"P6.2a calibration replay failed: {calibration_replay}")

        donors: dict[int, DenseK3GQA] = {}
        rank512_targets: dict[int, P6NoPEMLA] = {}
        layer_reports: dict[str, dict[str, Any]] = {}
        for ordinal, layer_index in enumerate(TARGET_LAYERS, start=1):
            donor = model.model.layers[layer_index].self_attn
            if not isinstance(donor, DenseK3GQA) or donor.layer_idx != layer_index:
                raise ValueError(f"P6.2a layer {layer_index} is not its frozen P5 GQA donor")
            donors[layer_index] = donor
            normalized_batches = calibration["normalized"][layer_index]
            covariance, covariance_report = covariance_from_batches(
                normalized_batches,
                device=device,
            )
            rank_reports: dict[int, dict[str, Any]] = {}
            for rank in DIAGNOSTIC_RANKS:
                print(
                    f"P6_2A_LAYER_{layer_index}_RANK_{rank}=RUNNING "
                    f"({ordinal}/8)",
                    flush=True,
                )
                factorization = activation_weighted_rank_factorization(
                    donor.k_proj.weight,
                    donor.v_proj.weight,
                    covariance,
                    num_heads=donor.num_heads,
                    num_kv_heads=donor.num_key_value_heads,
                    head_dim=donor.head_dim,
                    rank=rank,
                )
                target = P6NoPEMLA(donor, factorization).eval()
                with torch.no_grad():
                    latent_batches = [
                        target.kv_a_proj(batch.to(device)).cpu() for batch in normalized_batches
                    ]
                gamma, scale_report = fit_scale_matched_rmsnorm(
                    latent_batches,
                    eps=target.kv_a_layernorm.eps,
                )
                apply_rmsnorm_scale(target.kv_a_layernorm, gamma)
                scale_report["applied_gamma_sha256_bf16"] = tensor_sha256(
                    target.kv_a_layernorm.weight
                )
                local_metrics, trace_hash_report = local_layer_metrics(
                    donor,
                    target,
                    calibration["residual_prefix"][layer_index],
                    normalized_batches,
                    prefix=256,
                )
                latent_statistics = latent_rms_statistics(
                    target,
                    normalized_batches,
                    prefix=256,
                )
                rank_reports[rank] = {
                    "rank": rank,
                    "formal_architecture_rank": rank == 512,
                    "diagnostic_only": rank != 512,
                    "factorization": factorization.report,
                    "scale_matching": scale_report,
                    "latent_rms": latent_statistics,
                    "teacher_forced_vs_full_rank_nope": local_metrics,
                    "trace_hashes": trace_hash_report,
                    "finite": finite_metrics(local_metrics)
                    and factorization.report["all_finite"]
                    and scale_report["all_finite"]
                    and latent_statistics["all_finite"],
                }
                print(
                    f"P6_2A_LAYER_{layer_index}_RANK_{rank}=MEASURED "
                    f"MIXER_REL_L2={local_metrics['mixer_output']['relative_l2_error']:.9g}",
                    flush=True,
                )
                if rank == 512:
                    rank512_targets[layer_index] = target
                else:
                    del target
                del factorization, latent_batches, gamma
                torch.cuda.empty_cache()
            frozen_r512 = p6_2["per_layer_conversion"][str(layer_index)]
            replay_checks = {
                "kv_a_fp32": rank_reports[512]["factorization"]["kv_a_sha256_fp32"]
                == frozen_r512["factorization"]["kv_a_sha256_fp32"],
                "kv_b_fp32": rank_reports[512]["factorization"]["kv_b_sha256_fp32"]
                == frozen_r512["factorization"]["kv_b_sha256_fp32"],
                "gamma_bf16": rank_reports[512]["scale_matching"][
                    "applied_gamma_sha256_bf16"
                ]
                == frozen_r512["scale_matching"]["applied_gamma_sha256_bf16"],
                "mixer_metric": rank_reports[512]["teacher_forced_vs_full_rank_nope"][
                    "mixer_output"
                ]["relative_l2_error"]
                == frozen_r512["local_vs_full_rank_nope"]["mixer_output"][
                    "relative_l2_error"
                ],
            }
            layer_reports[str(layer_index)] = {
                "layer_index": layer_index,
                "covariance": covariance_report,
                "teacher_forced_by_rank": rank_reports,
                "p6_2_reported_r512_replay": replay_checks,
                "p6_2_reported_r512_was_already_teacher_forced": True,
            }
            del covariance

        for layer_index in TARGET_LAYERS:
            layer = model.model.layers[layer_index]
            layer.self_attn = rank512_targets[layer_index]
            layer.layer_type = "mla"
            model.config.mixer_types[layer_index] = "mla"
            model.config.layer_types[layer_index] = "full_attention"
        model.config.validate_architecture()
        if model.config.mla_kv_lora_rank != 512:
            raise ValueError("P6.2a in-memory scale-out changed the formal rank")

        scaleout_inputs = collect_scaleout_inputs(
            model,
            train,
            TARGET_LAYERS,
            list(range(8)),
            prefix=256,
        )
        if scaleout_inputs["manifest"]["token_sha256"] != calibration["manifest"]["token_sha256"]:
            raise ValueError("P6.2a donor and scale-out token hashes differ")
        interpretation = config["interpretation"]
        for layer_index in TARGET_LAYERS:
            scaleout_metrics, scaleout_hashes = local_layer_metrics(
                donors[layer_index],
                rank512_targets[layer_index],
                scaleout_inputs["residual"][layer_index],
                scaleout_inputs["normalized"][layer_index],
                prefix=256,
            )
            layer = layer_reports[str(layer_index)]
            by_rank = {
                rank: layer["teacher_forced_by_rank"][rank][
                    "teacher_forced_vs_full_rank_nope"
                ]
                for rank in DIAGNOSTIC_RANKS
            }
            layer["scaleout_input_r512_vs_full_rank_nope"] = scaleout_metrics
            layer["scaleout_input_r512_trace_hashes"] = scaleout_hashes
            layer["diagnostics"] = capacity_diagnostics(
                by_rank,
                scaleout_metrics,
                rank_elasticity_high_min=interpretation["rank_elasticity_high_min"],
                scaleout_amplification_high_min=interpretation[
                    "scaleout_amplification_high_min"
                ],
            )
            layer["finite"] = (
                all(item["finite"] for item in layer["teacher_forced_by_rank"].values())
                and finite_metrics(scaleout_metrics)
                and all(layer["p6_2_reported_r512_replay"].values())
            )

        aggregate = aggregate_capacity_diagnostics(layer_reports)
        gate_checks = {
            "source_frozen": all(source_checks.values()),
            "config_frozen": all(config_checks.values()),
            "calibration_exact_replay": all(calibration_replay.values()),
            "all_layers_all_ranks_finite": all(
                layer["finite"] for layer in layer_reports.values()
            ),
            "all_r512_exactly_replay_p6_2": all(
                all(layer["p6_2_reported_r512_replay"].values())
                for layer in layer_reports.values()
            ),
            "formal_rank_unchanged": model.config.mla_kv_lora_rank == 512,
            "diagnostic_ranks_not_materialized": True,
            "training_not_performed": True,
            "optimizer_not_created": True,
            "full_dev_not_accessed": True,
            "heldout_not_accessed": True,
            "checkpoint_not_written": True,
        }
        passed = all(gate_checks.values())
        if not passed:
            raise ValueError(f"P6.2a diagnostic completeness checks failed: {gate_checks}")
        report = {
            "schema_version": 1,
            "stage": "P6.2a",
            "status": "PASS_ATTRIBUTION_COMPLETE",
            "mode": "TEACHER_FORCED_RANK_CAPACITY_DIAGNOSTIC",
            "source_paths": {
                "p5_checkpoint": str(args.p5_checkpoint),
                "p6_1c_report": str(args.p6_1c_report),
                "p6_2_report": str(args.p6_2_report),
                "frozen_corpus": str(args.data_dir),
            },
            "config_checks": config_checks,
            "source_checks": source_checks,
            "calibration_replay": calibration_replay,
            "donor_calibration_manifest": calibration["manifest"],
            "scaleout_input_manifest": scaleout_inputs["manifest"],
            "formal_kv_lora_rank": 512,
            "diagnostic_ranks": DIAGNOSTIC_RANKS,
            "per_layer": layer_reports,
            "aggregate": aggregate,
            "gate": {"checks": gate_checks, "passed": True},
            "p6_2_status_unchanged": "REDESIGN",
            "training_performed": False,
            "optimizer_created": False,
            "full_dev_accessed": False,
            "heldout_accessed": False,
            "checkpoint_written": False,
            "p6_3_allowed": False,
            "p7_allowed": False,
            "result_marker": "P6_2A_TEACHER_FORCED_RANK_CAPACITY=PASS",
            "completed_at": now(),
        }
        save_json(report_path, report)
        final = {
            "stage": "P6.2a",
            "status": "PASS_ATTRIBUTION_COMPLETE",
            "mode": report["mode"],
            "formal_kv_lora_rank": 512,
            "diagnostic_ranks": DIAGNOSTIC_RANKS,
            "aggregate": aggregate,
            "p6_2_status_unchanged": "REDESIGN",
            "training_performed": False,
            "full_dev_accessed": False,
            "heldout_accessed": False,
            "checkpoint_written": False,
            "p6_3_allowed": False,
            "p7_allowed": False,
            "result_marker": report["result_marker"],
            "completed_at": now(),
        }
        save_json(final_path, final)
        print(report["result_marker"], flush=True)
        print(
            "P6_2A_MEDIAN_MIXER_RANK_ELASTICITY="
            f"{aggregate['median_mixer_rank_elasticity']:.9g}",
            flush=True,
        )
        print(
            "P6_2A_MEDIAN_MIXER_SCALEOUT_AMPLIFICATION="
            f"{aggregate['median_mixer_scaleout_amplification']:.9g}",
            flush=True,
        )
        print("P6_3_ALLOWED=false", flush=True)
        print("P7_ALLOWED=false", flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001 - persist every diagnostic failure
        failure = {
            "stage": "P6.2a",
            "status": "ERROR",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "formal_kv_lora_rank": 512,
            "training_performed": False,
            "full_dev_accessed": False,
            "heldout_accessed": False,
            "checkpoint_written": False,
            "p6_3_allowed": False,
            "p7_allowed": False,
            "result_marker": "P6_2A_TEACHER_FORCED_RANK_CAPACITY=ERROR",
            "completed_at": now(),
        }
        save_json(report_path, failure)
        save_json(final_path, failure)
        print(failure["result_marker"], flush=True)
        print(f"P6_2A_EXCEPTION={type(exc).__name__}: {exc}", flush=True)
        print("P6_3_ALLOWED=false", flush=True)
        print("P7_ALLOWED=false", flush=True)
        return 1
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
