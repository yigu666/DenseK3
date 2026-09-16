"""Run frozen P6.2 C3 conversion at all eight global-attention layers."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from densek3_core.evaluation.p4_runtime import load_densek3_model
from densek3_core.modeling.modeling_densek3 import DenseK3GQA
from densek3_core.recovery.p5_corpus import PackedTokenDataset, verify_frozen_corpus
from densek3_core.recovery.p5_transition import hidden_drift, token_decision_drift
from densek3_core.recovery.p6_initialization import (
    activation_weighted_rank_factorization,
    apply_rmsnorm_scale,
    covariance_from_batches,
    fit_scale_matched_rmsnorm,
)
from densek3_core.recovery.p6_mla_probe import P6NoPEMLA, tensor_sha256
from densek3_core.recovery.p6_scaleout import (
    collect_frozen_donor_calibration,
    full_model_shock_metrics,
    local_layer_metrics,
    write_candidate_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/architecture/p6-eight-layer-mla-scaleout.yaml"
DEFAULT_P5 = ROOT / "outputs/densek3-core/checkpoints/p5/densek3-4b-k3-decay-silu-i1"
DEFAULT_P6_1 = ROOT / "outputs/densek3-core/manifests/p6/p6-single-layer-probe.json"
DEFAULT_P6_1C = ROOT / "outputs/densek3-core/manifests/p6/p6-1c/p6-1c-initialization.json"
DEFAULT_DATA = ROOT / "data/p5-probe"
DEFAULT_PROMPTS = ROOT / "data/p4-smoke-prompts.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs/densek3-core/manifests/p6/p6-2"
DEFAULT_CANDIDATE = ROOT / "outputs/densek3-core/checkpoints/p6/densek3-4b-8mla-c3-candidate"
TARGET_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]
INITIALIZATION = "ACTIVATION_WEIGHTED_RANK512_PLUS_SCALE_MATCHED_LATENT_NORM"
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


def validate_config(config: dict[str, Any]) -> dict[str, bool]:
    checks = {
        "stage": config["stage"] == "P6.2" and config["status"] == "FROZEN",
        "source": config["source"]["required_initialization"] == INITIALIZATION,
        "layers": config["scope"]["target_layers"] == TARGET_LAYERS,
        "architecture": config["scope"]["topology"] == "24_kda_8_mla"
        and config["scope"]["rank"] == 512
        and config["scope"]["no_pe"] is True
        and config["scope"]["training"] is False
        and config["scope"]["cache_mode"] == "expanded_reference"
        and config["scope"]["true_latent_cache"] is False,
        "calibration": config["calibration"]
        == {
            "split": "train",
            "sequence_indices": list(range(8)),
            "sequence_length": 2048,
            "selection_token_prefix": 256,
            "statistics_source": "untouched_frozen_p5_donor",
            "sequential_self_conditioning": False,
            "per_layer_statistics": True,
            "covariance_dtype": "float32",
            "factorization_dtype": "float32",
            "deterministic_exact": True,
        },
        "evaluation": config["evaluation"]["canonical_role"] == "diagnostic_only"
        and config["evaluation"]["full_dev_sequences"] == 97
        and config["evaluation"]["full_dev_role"] == "primary_gate"
        and config["evaluation"]["unique_full_model_candidate"] is True,
        "gate": config["gate"]["full_dev_ce_delta_vs_p5_max"] == 0.15
        and config["gate"]["drift_is_diagnostic_not_thresholded"] is True
        and config["gate"]["per_layer_mixer_is_diagnostic_not_thresholded"] is True,
        "candidate": config["candidate"]["native_mla_required"] is True
        and config["candidate"]["write_only_after_gate_pass"] is True
        and config["candidate"]["formal_save_reload_stage"] == "P6.4",
        "forbidden": all(config["forbidden"].values()),
    }
    if not all(checks.values()):
        raise ValueError(f"P6.2 frozen config failed: {checks}")
    return checks


def load_prompts(path: Path) -> list[dict[str, str]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if {record["id"] for record in records} != {"english", "chinese", "code", "factual", "longer"}:
        raise ValueError("P6.2 requires exactly the frozen five-category prompts")
    return records


@torch.no_grad()
def canonical_observation(model: torch.nn.Module, tokenizer: Any) -> dict[str, Any]:
    device = next(model.parameters()).device
    encoded = tokenizer(CANONICAL_PROMPT, return_tensors="pt", truncation=True, max_length=64)
    input_ids = encoded["input_ids"].to(device)
    output = model(
        input_ids=input_ids,
        attention_mask=encoded["attention_mask"].to(device),
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    logits = output.logits.detach().cpu()
    hidden = tuple(value.detach().cpu() for value in output.hidden_states)
    return {
        "input_ids": input_ids.cpu(),
        "logits": logits,
        "hidden_states": hidden,
        "cross_entropy": float(
            F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), input_ids.cpu()[:, 1:].reshape(-1))
        ),
        "logits_sha256": tensor_sha256(output.logits),
        "finite": bool(torch.isfinite(logits).all())
        and all(bool(torch.isfinite(value).all()) for value in hidden),
    }


def canonical_comparison(reference: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    curve = []
    for index in range(32):
        metric = hidden_drift(reference["hidden_states"][index + 1], actual["hidden_states"][index + 1])
        metric.update({"layer_index": index, "layer_type": "mla" if index in TARGET_LAYERS else "kda"})
        curve.append(metric)
    return {
        "per_layer_hidden_drift": curve,
        "final_hidden": curve[-1],
        "logits": hidden_drift(reference["logits"], actual["logits"]),
        "token_decisions": token_decision_drift(reference["logits"], actual["logits"]),
    }


@torch.no_grad()
def full_dev_ce(model: torch.nn.Module, dev: PackedTokenDataset, *, chunk_size: int = 64) -> float:
    if dev.split != "dev" or len(dev) != 97:
        raise ValueError("P6.2 requires the frozen 97-sequence dev split")
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0
    for index in range(len(dev)):
        tokens = torch.from_numpy(dev[index].copy()).unsqueeze(0).to(device)
        hidden = model.model(input_ids=tokens, use_cache=False, return_dict=True).last_hidden_state[:, :-1]
        labels = tokens[:, 1:]
        for start in range(0, hidden.shape[1], chunk_size):
            end = min(start + chunk_size, hidden.shape[1])
            logits = F.linear(hidden[:, start:end].float(), model.lm_head.weight.float())
            total_loss += float(
                F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels[:, start:end].reshape(-1),
                    reduction="sum",
                )
            )
            total_tokens += labels[:, start:end].numel()
        if (index + 1) % 16 == 0 or index + 1 == len(dev):
            print(f"P6_2_FULL_DEV_PROGRESS={index + 1}/97", flush=True)
    return total_loss / total_tokens


@torch.no_grad()
def generation_report(model: torch.nn.Module, tokenizer: Any, prompts: list[dict[str, str]]) -> dict[str, Any]:
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
        records.append(
            {
                "id": item["id"],
                "new_tokens": tokens,
                "not_collapsed": bool(tokens)
                and max(counts.values()) / len(tokens) <= 0.75
                and len(counts) / len(tokens) >= 0.25,
            }
        )
    return {"records": records, "not_collapsed": all(item["not_collapsed"] for item in records)}


def finite_metrics(value: Any) -> bool:
    if isinstance(value, dict):
        if "reference_nan_count" in value:
            return all(
                value[key] == 0
                for key in (
                    "reference_nan_count",
                    "actual_nan_count",
                    "reference_inf_count",
                    "actual_inf_count",
                )
            )
        return all(finite_metrics(child) for child in value.values())
    if isinstance(value, list):
        return all(finite_metrics(child) for child in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--p5-checkpoint", type=Path, default=DEFAULT_P5)
    parser.add_argument("--p6-1-report", type=Path, default=DEFAULT_P6_1)
    parser.add_argument("--p6-1c-report", type=Path, default=DEFAULT_P6_1C)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in (
        "config",
        "p5_checkpoint",
        "p6_1_report",
        "p6_1c_report",
        "data_dir",
        "prompts",
        "output_dir",
        "candidate_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "p6-2-scaleout.json"
    final_path = args.output_dir / "p6-2-final-check.json"
    print("P6_2_EIGHT_LAYER_MLA_SCALEOUT=RUNNING", flush=True)
    print("P6_3_ALLOWED=false", flush=True)
    print("P7_ALLOWED=false", flush=True)
    try:
        from transformers import AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("P6.2 requires the authoritative CUDA server")
        config = load_yaml(args.config)
        config_checks = validate_config(config)
        p6_1 = load_json(args.p6_1_report)
        p6_1c = load_json(args.p6_1c_report)
        source_checks = {
            "p6_0": p6_1["p6_0"]["status"] == "PASS",
            "p6_1": p6_1["p6_1"]["status"] == "PASS_MEASUREMENT_COMPLETE",
            "p6_1c": p6_1c["status"] == "PASS"
            and p6_1c["selection"]["selected_case"] == "C3"
            and p6_1c["selection"]["selected_initialization"] == INITIALIZATION,
            "p6_2_allowed": p6_1c["p6_2_allowed"] is True,
            "p7_blocked": p6_1c["p7_allowed"] is False,
            "heldout": p6_1["heldout_accessed"] is False and p6_1c["heldout_accessed"] is False,
            "same_p5": Path(p6_1["source_paths"]["p5_checkpoint"]).resolve() == args.p5_checkpoint
            and Path(p6_1c["source_paths"]["p5_checkpoint"]).resolve() == args.p5_checkpoint,
        }
        if not all(source_checks.values()):
            raise ValueError(f"P6.2 source checks failed: {source_checks}")
        corpus = verify_frozen_corpus(args.data_dir)
        if corpus["p5_probe_corpus_frozen"] is not True:
            raise ValueError("P6.2 frozen corpus verification failed")
        device = torch.device("cuda")
        tokenizer = AutoTokenizer.from_pretrained(args.p5_checkpoint, trust_remote_code=False)
        prompts = load_prompts(args.prompts)
        train = PackedTokenDataset(args.data_dir / "packed", "train")
        dev = PackedTokenDataset(args.data_dir / "packed", "dev")
        model = load_densek3_model(args.p5_checkpoint, device=device, dtype=torch.bfloat16)
        baseline_canonical = canonical_observation(model, tokenizer)
        frozen_a = p6_1["p6_1"]["cases"]["A_GQA_PARTIAL_ROPE"]
        baseline_replay = {
            "canonical_logits": baseline_canonical["logits_sha256"] == frozen_a["canonical_logits_sha256"],
            "canonical_ce": baseline_canonical["cross_entropy"] == frozen_a["canonical_ce"],
            "full_dev_available": math.isfinite(frozen_a["full_dev_ce"]),
        }
        if not all(baseline_replay.values()):
            raise ValueError(f"P6.2 P5 baseline replay failed: {baseline_replay}")

        calibration = collect_frozen_donor_calibration(
            model,
            train,
            TARGET_LAYERS,
            list(range(8)),
            selection_prefix=256,
        )
        calibration_replay = {
            "token_hashes": calibration["manifest"]["token_sha256"]
            == p6_1c["calibration"]["token_sha256"],
            "untouched_donor": calibration["manifest"]["all_statistics_from_untouched_p5_donor"] is True,
            "no_self_conditioning": calibration["manifest"]["sequential_self_conditioning"] is False,
        }
        if not all(calibration_replay.values()):
            raise ValueError(f"P6.2 calibration replay failed: {calibration_replay}")

        donors = {}
        for layer_index in TARGET_LAYERS:
            donor = model.model.layers[layer_index].self_attn
            if not isinstance(donor, DenseK3GQA) or donor.layer_idx != layer_index:
                raise ValueError(f"P6.2 layer {layer_index} is not its frozen donor GQA")
            donors[layer_index] = donor
        targets = {}
        layer_reports = {}
        for ordinal, layer_index in enumerate(TARGET_LAYERS, start=1):
            print(f"P6_2_LAYER_{layer_index}_C3=RUNNING ({ordinal}/8)", flush=True)
            donor = donors[layer_index]
            normalized_batches = calibration["normalized"][layer_index]
            covariance, covariance_report = covariance_from_batches(normalized_batches, device=device)
            factorization = activation_weighted_rank_factorization(
                donor.k_proj.weight,
                donor.v_proj.weight,
                covariance,
                num_heads=donor.num_heads,
                num_kv_heads=donor.num_key_value_heads,
                head_dim=donor.head_dim,
                rank=512,
            )
            target = P6NoPEMLA(donor, factorization).eval()
            latent_batches = [target.kv_a_proj(batch.to(device)).cpu() for batch in normalized_batches]
            gamma, scale_report = fit_scale_matched_rmsnorm(
                latent_batches,
                eps=target.kv_a_layernorm.eps,
            )
            apply_rmsnorm_scale(target.kv_a_layernorm, gamma)
            scale_report["applied_gamma_sha256_bf16"] = tensor_sha256(target.kv_a_layernorm.weight)
            local_metrics, trace_hashes = local_layer_metrics(
                donor,
                target,
                calibration["residual_prefix"][layer_index],
                normalized_batches,
                prefix=256,
            )
            layer_reports[str(layer_index)] = {
                "layer_index": layer_index,
                "covariance": covariance_report,
                "factorization": factorization.report,
                "scale_matching": scale_report,
                "local_vs_full_rank_nope": local_metrics,
                "trace_hashes": trace_hashes,
                "finite": finite_metrics(local_metrics)
                and factorization.report["all_finite"]
                and scale_report["all_finite"],
            }
            targets[layer_index] = target
            print(
                f"P6_2_LAYER_{layer_index}_C3=PASS "
                f"MIXER_REL_L2={local_metrics['mixer_output']['relative_l2_error']:.9g}",
                flush=True,
            )
            del covariance, factorization, latent_batches, gamma
            torch.cuda.empty_cache()

        layer3_replay = {
            "kv_a_fp32": layer_reports["3"]["factorization"]["kv_a_sha256_fp32"]
            == p6_1c["factorizations"]["W1"]["kv_a_sha256_fp32"],
            "kv_b_fp32": layer_reports["3"]["factorization"]["kv_b_sha256_fp32"]
            == p6_1c["factorizations"]["W1"]["kv_b_sha256_fp32"],
            "gamma_bf16": layer_reports["3"]["scale_matching"]["applied_gamma_sha256_bf16"]
            == p6_1c["scale_matching"]["C3"]["applied_gamma_sha256_bf16"],
            "mixer_metric": layer_reports["3"]["local_vs_full_rank_nope"]["mixer_output"][
                "relative_l2_error"
            ]
            == p6_1c["cases"]["C3"]["calibration"]["local_vs_B"]["mixer_output"][
                "relative_l2_error"
            ],
        }
        if not all(layer3_replay.values()):
            raise ValueError(f"P6.2 layer-3 C3 replay failed: {layer3_replay}")

        for layer_index in TARGET_LAYERS:
            layer = model.model.layers[layer_index]
            layer.self_attn = targets[layer_index]
            layer.layer_type = "mla"
            model.config.mixer_types[layer_index] = "mla"
            model.config.layer_types[layer_index] = "full_attention"
        model.config.validate_architecture()
        if model.config.mla_indices != TARGET_LAYERS or model.config.gqa_indices:
            raise ValueError("P6.2 assembled topology is not exactly 24 KDA plus 8 MLA")

        shock = full_model_shock_metrics(
            model,
            train,
            list(range(8)),
            calibration["baseline_curves"],
            calibration["baseline_final"],
            prefix=256,
        )
        candidate_canonical = canonical_observation(model, tokenizer)
        canonical_diagnostic = canonical_comparison(baseline_canonical, candidate_canonical)
        candidate_full_dev = full_dev_ce(model, dev)
        generation = generation_report(model, tokenizer, prompts)
        p5_full_dev = float(frozen_a["full_dev_ce"])
        full_dev_delta = candidate_full_dev - p5_full_dev
        mixer_values = [
            layer_reports[str(index)]["local_vs_full_rank_nope"]["mixer_output"]["relative_l2_error"]
            for index in TARGET_LAYERS
        ]
        worst_offset = max(range(len(mixer_values)), key=mixer_values.__getitem__)
        local_summary = {
            "median_mla_mixer_relative_l2": statistics.median(mixer_values),
            "worst_mla_mixer_relative_l2": mixer_values[worst_offset],
            "worst_mla_layer": TARGET_LAYERS[worst_offset],
            "minimum_mla_mixer_relative_l2": min(mixer_values),
            "no_per_layer_acceptance_threshold": True,
        }
        gate_checks = {
            "all_layer_conversions_valid": all(report["finite"] for report in layer_reports.values()),
            "full_dev_ce": full_dev_delta <= config["gate"]["full_dev_ce_delta_vs_p5_max"],
            "finite": math.isfinite(candidate_full_dev)
            and candidate_canonical["finite"]
            and finite_metrics(shock),
            "generation": generation["not_collapsed"] is True,
            "topology": len(model.config.kda_indices) == 24
            and model.config.mla_indices == TARGET_LAYERS
            and not model.config.gqa_indices,
            "expanded_reference_cache": all(
                target.cache_mode == "expanded_reference" for target in targets.values()
            ),
            "heldout_not_accessed": True,
            "training_not_performed": True,
        }
        passed = all(gate_checks.values())
        candidate_artifact = None
        preliminary = {
            "schema_version": 1,
            "stage": "P6.2",
            "status": "PASS_MEASUREMENT_CANDIDATE_WRITE_PENDING" if passed else "REDESIGN",
            "source_paths": {
                "p5_checkpoint": str(args.p5_checkpoint),
                "p6_1_report": str(args.p6_1_report),
                "p6_1c_report": str(args.p6_1c_report),
                "frozen_corpus": str(args.data_dir),
            },
            "config_checks": config_checks,
            "source_checks": source_checks,
            "baseline_replay": baseline_replay,
            "calibration_replay": calibration_replay,
            "calibration_manifest": calibration["manifest"],
            "layer3_c3_replay": layer3_replay,
            "per_layer_conversion": layer_reports,
            "local_summary": local_summary,
            "full_model_shock": shock,
            "canonical_diagnostic_only": {
                "comparison": canonical_diagnostic,
                "candidate_cross_entropy": candidate_canonical["cross_entropy"],
                "candidate_logits_sha256": candidate_canonical["logits_sha256"],
                "used_for_gate": False,
            },
            "full_dev": {
                "p5_baseline_ce": p5_full_dev,
                "p6_candidate_ce": candidate_full_dev,
                "delta": full_dev_delta,
                "maximum_delta": config["gate"]["full_dev_ce_delta_vs_p5_max"],
                "primary_gate": True,
            },
            "generation": generation,
            "gate": {"checks": gate_checks, "passed": passed},
            "training_performed": False,
            "true_latent_cache": False,
            "heldout_accessed": False,
            "p6_frozen": False,
            "p6_3_allowed": False,
            "p7_allowed": False,
            "candidate_artifact": candidate_artifact,
            "completed_at": now(),
        }
        save_json(report_path, preliminary)
        if passed:
            candidate_artifact = write_candidate_checkpoint(
                model,
                args.candidate_dir,
                args.p5_checkpoint,
                {
                    "p5_checkpoint": str(args.p5_checkpoint),
                    "p6_1c_report": str(args.p6_1c_report),
                    "target_layers": TARGET_LAYERS,
                    "per_layer_factor_hashes": {
                        index: {
                            "kv_a": layer_reports[str(index)]["factorization"]["kv_a_sha256_fp32"],
                            "kv_b": layer_reports[str(index)]["factorization"]["kv_b_sha256_fp32"],
                            "gamma_bf16": layer_reports[str(index)]["scale_matching"][
                                "applied_gamma_sha256_bf16"
                            ],
                        }
                        for index in TARGET_LAYERS
                    },
                },
            )
        report = {
            **preliminary,
            "status": "PASS" if passed else "REDESIGN",
            "candidate_artifact": candidate_artifact,
            "p6_3_allowed": passed,
            "result_marker": "P6_2_EIGHT_LAYER_MLA_SCALEOUT=PASS"
            if passed
            else "P6_2_EIGHT_LAYER_MLA_SCALEOUT=REDESIGN",
            "completed_at": now(),
        }
        save_json(report_path, report)
        final = {
            "stage": "P6.2",
            "status": report["status"],
            "initialization": INITIALIZATION,
            "target_layers": TARGET_LAYERS,
            "full_dev_ce_delta_vs_p5": full_dev_delta,
            "candidate_directory": None if candidate_artifact is None else candidate_artifact["directory"],
            "heldout_accessed": False,
            "p6_frozen": False,
            "p6_3_allowed": passed,
            "p7_allowed": False,
            "result_marker": report["result_marker"],
            "completed_at": now(),
        }
        save_json(final_path, final)
        print(report["result_marker"], flush=True)
        print(f"P6_2_FULL_DEV_CE={candidate_full_dev:.12f}", flush=True)
        print(f"P6_2_FULL_DEV_DELTA_VS_P5={full_dev_delta:+.12f}", flush=True)
        print(f"P6_2_WORST_MLA_LAYER={local_summary['worst_mla_layer']}", flush=True)
        print(f"P6_3_ALLOWED={str(passed).lower()}", flush=True)
        print("P7_ALLOWED=false", flush=True)
        return 0 if passed else 1
    except Exception as exc:  # noqa: BLE001 - persist every P6.2 failure
        failure = {
            "stage": "P6.2",
            "status": "ERROR",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "heldout_accessed": False,
            "p6_frozen": False,
            "p6_3_allowed": False,
            "p7_allowed": False,
            "result_marker": "P6_2_EIGHT_LAYER_MLA_SCALEOUT=ERROR",
            "completed_at": now(),
        }
        save_json(report_path, failure)
        save_json(final_path, failure)
        print(failure["result_marker"], flush=True)
        print(f"P6_2_EXCEPTION={type(exc).__name__}: {exc}", flush=True)
        print("P6_3_ALLOWED=false", flush=True)
        print("P7_ALLOWED=false", flush=True)
        return 1
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
