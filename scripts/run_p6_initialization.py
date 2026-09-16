"""Run the frozen P6.1c activation-aware factorization x latent-scale experiment."""

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

from densek3_core.evaluation.p4_runtime import load_densek3_model
from densek3_core.modeling.modeling_densek3 import DenseK3GQA
from densek3_core.recovery.p5_corpus import PackedTokenDataset, verify_frozen_corpus
from densek3_core.recovery.p5_transition import hidden_drift, token_decision_drift
from densek3_core.recovery.p6_attribution import TRACE_FIELDS, trace_case, trace_hashes
from densek3_core.recovery.p6_initialization import (
    DriftAccumulator,
    activation_weighted_rank_factorization,
    apply_rmsnorm_scale,
    covariance_from_batches,
    fit_scale_matched_rmsnorm,
)
from densek3_core.recovery.p6_mla_probe import (
    P6NoPEFullAttention,
    P6NoPEMLA,
    deterministic_rank_factorization,
    tensor_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/architecture/p6-single-layer-mla-initialization.yaml"
DEFAULT_P5 = ROOT / "outputs/densek3-core/checkpoints/p5/densek3-4b-k3-decay-silu-i1"
DEFAULT_P6_1 = ROOT / "outputs/densek3-core/manifests/p6/p6-single-layer-probe.json"
DEFAULT_P6_1B = ROOT / "outputs/densek3-core/manifests/p6/p6-1b/p6-1b-attribution.json"
DEFAULT_DATA = ROOT / "data/p5-probe"
DEFAULT_PROMPTS = ROOT / "data/p4-smoke-prompts.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs/densek3-core/manifests/p6/p6-1c"
CANONICAL_PROMPT = (
    "Dense recurrent language models can combine efficient state updates with occasional "
    "softmax attention. 请简要解释这种混合结构，并给出一个代码示例。"
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
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
        "stage": config["stage"] == "P6.1c" and config["status"] == "FROZEN",
        "source": config["source"]["required_p6_1b_dominant"] == "LOW_RANK_CAPACITY"
        and config["source"]["required_p6_1b_secondary"] == "LATENT_NORMALIZATION",
        "scope": config["scope"]
        == {
            "probe_layer": 3,
            "rank": 512,
            "no_pe": True,
            "split_q_gate": True,
            "qk_norm": False,
            "training": False,
            "other_layers_unchanged": True,
            "scale_to_all_8_layers": False,
            "true_latent_cache": False,
            "heldout_accessed": False,
        },
        "calibration": config["calibration"]
        == {
            "split": "train",
            "sequence_indices": list(range(8)),
            "sequence_length": 2048,
            "selection_token_prefix": 256,
            "covariance_dtype": "float32",
            "factorization_dtype": "float32",
            "deterministic_exact": True,
        },
        "cases": config["cases"]
        == {
            "C0": {"factorization": "weight_svd", "latent_norm": "ones", "frozen_anchor": True},
            "C1": {"factorization": "weight_svd", "latent_norm": "scale_matched", "frozen_anchor": False},
            "C2": {
                "factorization": "activation_weighted_svd",
                "latent_norm": "ones",
                "frozen_anchor": False,
            },
            "C3": {
                "factorization": "activation_weighted_svd",
                "latent_norm": "scale_matched",
                "frozen_anchor": False,
            },
        },
        "selection": config["selection"]["candidates"] == ["C1", "C2", "C3"]
        and config["selection"]["canonical_used_for_selection"] is False
        and config["selection"]["unique_full_dev_candidate"] is True,
        "all_forbidden": all(config["forbidden"].values()),
    }
    if not all(checks.values()):
        raise ValueError(f"P6.1c config failed: {checks}")
    return checks


def load_prompts(path: Path) -> list[dict[str, str]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if {item["id"] for item in records} != {"english", "chinese", "code", "factual", "longer"}:
        raise ValueError("P6.1c must reuse exactly the frozen five-category prompts")
    return records


@torch.no_grad()
def collect_layer3_inputs(
    model: torch.nn.Module,
    train: PackedTokenDataset,
    indices: list[int],
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, Any]]:
    if train.split != "train" or indices != list(range(8)):
        raise ValueError("P6.1c calibration is frozen to train indices 0 through 7")
    device = next(model.parameters()).device
    residual_batches = []
    normalized_batches = []
    token_hashes = []
    layer3 = model.model.layers[3]
    for index in indices:
        tokens = torch.from_numpy(train[index].copy()).unsqueeze(0).to(device)
        captured: list[torch.Tensor] = []

        def capture(
            _module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
            captured_values: list[torch.Tensor] = captured,
        ) -> None:
            captured_values.append(inputs[0].detach())

        handle = layer3.input_layernorm.register_forward_pre_hook(capture)
        try:
            model.model(input_ids=tokens, use_cache=False, return_dict=True)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("P6.1c failed to capture exactly one layer-3 input")
        residual_batches.append(captured[0].cpu())
        normalized_batches.append(layer3.input_layernorm(captured[0]).cpu())
        token_hashes.append(tensor_sha256(tokens))
        print(f"P6_1C_CALIBRATION_CAPTURE={index + 1}/8", flush=True)
    return residual_batches, normalized_batches, {
        "split": "train",
        "sequence_indices": indices,
        "sequence_length": train.sequence_length,
        "token_sha256": token_hashes,
        "residual_input_sha256": [tensor_sha256(value) for value in residual_batches],
        "normalized_input_sha256": [tensor_sha256(value) for value in normalized_batches],
        "tokens": len(indices) * train.sequence_length,
    }


@torch.no_grad()
def fit_case_scale(module: P6NoPEMLA, batches: list[torch.Tensor]) -> dict[str, Any]:
    device = next(module.parameters()).device
    latent_batches = [module.kv_a_proj(batch.to(device)).cpu() for batch in batches]
    gamma, report = fit_scale_matched_rmsnorm(latent_batches, eps=module.kv_a_layernorm.eps)
    apply_rmsnorm_scale(module.kv_a_layernorm, gamma)
    report["latent_batch_sha256"] = [tensor_sha256(value) for value in latent_batches]
    report["applied_gamma_sha256_bf16"] = tensor_sha256(module.kv_a_layernorm.weight)
    return report


def calibration_reference_hidden(
    model: torch.nn.Module,
    batches: PackedTokenDataset,
    indices: list[int],
    *,
    prefix: int,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    device = next(model.parameters()).device
    hidden_batches = []
    token_hashes = []
    for index in indices:
        tokens = torch.from_numpy(batches[index][:prefix].copy()).unsqueeze(0).to(device)
        hidden = model.model(input_ids=tokens, use_cache=False, return_dict=True).last_hidden_state
        hidden_batches.append(hidden.cpu())
        token_hashes.append(tensor_sha256(tokens))
    return hidden_batches, {
        "sequence_indices": indices,
        "selection_token_prefix": prefix,
        "token_sha256": token_hashes,
        "final_hidden_sha256": [tensor_sha256(value) for value in hidden_batches],
    }


@torch.no_grad()
def calibration_model_metrics(
    model: torch.nn.Module,
    batches: PackedTokenDataset,
    indices: list[int],
    reference_hidden_batches: list[torch.Tensor],
    *,
    prefix: int,
    logits_chunk_size: int = 16,
) -> dict[str, Any]:
    """Measure model-level calibration drift without materializing full-vocabulary logits."""
    if len(indices) != len(reference_hidden_batches):
        raise ValueError("P6.1c calibration reference count mismatch")
    device = next(model.parameters()).device
    hidden_accumulator = DriftAccumulator()
    logits_accumulator = DriftAccumulator()
    total_loss = 0.0
    total_tokens = 0
    actual_hidden_hashes = []
    for index, reference_cpu in zip(indices, reference_hidden_batches, strict=True):
        tokens = torch.from_numpy(batches[index][:prefix].copy()).unsqueeze(0).to(device)
        actual_hidden = model.model(input_ids=tokens, use_cache=False, return_dict=True).last_hidden_state
        reference_hidden = reference_cpu.to(device)
        hidden_accumulator.update(reference_hidden, actual_hidden)
        actual_hidden_hashes.append(tensor_sha256(actual_hidden))
        for start in range(0, prefix, logits_chunk_size):
            end = min(start + logits_chunk_size, prefix)
            reference_logits = F.linear(
                reference_hidden[:, start:end],
                model.lm_head.weight,
            ).float()
            actual_logits = F.linear(
                actual_hidden[:, start:end],
                model.lm_head.weight,
            ).float()
            logits_accumulator.update(reference_logits, actual_logits)
            loss_end = min(end, prefix - 1)
            if start < loss_end:
                labels = tokens[:, start + 1 : loss_end + 1]
                total_loss += float(
                    F.cross_entropy(
                        actual_logits[:, : loss_end - start].reshape(-1, actual_logits.shape[-1]),
                        labels.reshape(-1),
                        reduction="sum",
                    )
                )
                total_tokens += labels.numel()
    return {
        "final_hidden": hidden_accumulator.finalize(),
        "logits": logits_accumulator.finalize(),
        "cross_entropy": total_loss / total_tokens,
        "tokens_scored": total_tokens,
        "actual_final_hidden_sha256": actual_hidden_hashes,
    }


@torch.no_grad()
def calibration_local_metrics(
    reference: P6NoPEFullAttention,
    actual: P6NoPEMLA,
    residual_batches: list[torch.Tensor],
    normalized_batches: list[torch.Tensor],
    *,
    prefix: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Aggregate all local trace fields over the frozen selection calibration subset."""
    if len(residual_batches) != len(normalized_batches):
        raise ValueError("P6.1c residual/normalized calibration batch count mismatch")
    device = next(actual.parameters()).device
    fields = (*TRACE_FIELDS, "post_attention_residual_hidden")
    accumulators = {name: DriftAccumulator() for name in fields}
    reference_hashes = []
    actual_hashes = []
    for residual_cpu, normalized_cpu in zip(residual_batches, normalized_batches, strict=True):
        residual = residual_cpu[:, :prefix].to(device)
        normalized = normalized_cpu[:, :prefix].to(device)
        mask = torch.ones(normalized.shape[:2], device=device)
        reference_trace = trace_case(reference, residual, normalized, mask)
        actual_trace = trace_case(actual, residual, normalized, mask)
        for name in fields:
            accumulators[name].update(reference_trace[name], actual_trace[name])
        reference_hashes.append(trace_hashes(reference_trace))
        actual_hashes.append(trace_hashes(actual_trace))
    return (
        {name: accumulator.finalize() for name, accumulator in accumulators.items()},
        {"reference": reference_hashes, "actual": actual_hashes},
    )


@torch.no_grad()
def full_dev_ce(model: torch.nn.Module, dev: PackedTokenDataset, *, chunk_size: int = 64) -> float:
    if dev.split != "dev" or len(dev) != 97:
        raise ValueError("P6.1c requires exactly the frozen 97-sequence dev split")
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
            print(f"P6_1C_FULL_DEV_PROGRESS={index + 1}/97", flush=True)
    return total_loss / total_tokens


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
    logits = output.logits.detach().cpu().float()
    hidden = tuple(value.detach().cpu() for value in output.hidden_states)
    return {
        "input_ids": input_ids.cpu(),
        "logits": logits,
        "hidden_states": hidden,
        "canonical_ce": float(
            F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), input_ids.cpu()[:, 1:].reshape(-1))
        ),
        "canonical_logits_sha256": tensor_sha256(output.logits),
        "finite": bool(torch.isfinite(logits).all())
        and all(bool(torch.isfinite(value).all()) for value in hidden),
    }


def compare_models(reference: dict[str, Any], actual: dict[str, Any], mixer_types: list[str]) -> dict[str, Any]:
    curve = []
    for layer_index in range(32):
        item = hidden_drift(reference["hidden_states"][layer_index + 1], actual["hidden_states"][layer_index + 1])
        item.update({"layer_index": layer_index, "layer_type": mixer_types[layer_index]})
        curve.append(item)
    return {
        "per_layer_hidden_drift": curve,
        "layer_3_block_output": curve[3],
        "final_hidden": curve[-1],
        "logits": hidden_drift(reference["logits"], actual["logits"]),
        "token_decisions": token_decision_drift(reference["logits"], actual["logits"]),
    }


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


def selection_key(case: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    local = case["calibration"]["local_vs_B"]
    model = case["calibration"]["model_vs_A"]
    return (
        local["mixer_output"]["relative_l2_error"],
        local["attention_output"]["relative_l2_error"],
        local["post_attention_residual_hidden"]["relative_l2_error"],
        -model["final_hidden"]["cosine_similarity"],
        -model["logits"]["cosine_similarity"],
        model["cross_entropy"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--p5-checkpoint", type=Path, default=DEFAULT_P5)
    parser.add_argument("--p6-1-report", type=Path, default=DEFAULT_P6_1)
    parser.add_argument("--p6-1b-report", type=Path, default=DEFAULT_P6_1B)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in ("config", "p5_checkpoint", "p6_1_report", "p6_1b_report", "data_dir", "prompts", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "p6-1c-initialization.json"
    final_path = args.output_dir / "p6-1c-final-check.json"
    print("P6_1C_INITIALIZATION=RUNNING", flush=True)
    print("P6_2_ALLOWED=false", flush=True)
    print("P7_ALLOWED=false", flush=True)
    try:
        from transformers import AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("P6.1c requires the authoritative CUDA server")
        config = load_yaml(args.config)
        config_checks = validate_config(config)
        p6_1 = load_json(args.p6_1_report)
        p6_1b = load_json(args.p6_1b_report)
        source_checks = {
            "p6_1": p6_1["status"] == "PASS" and p6_1["p6_1"]["status"] == "PASS_MEASUREMENT_COMPLETE",
            "p6_1b": p6_1b["status"] == "PASS_ATTRIBUTION_COMPLETE"
            and p6_1b["attribution"]["dominant_factor"] == "LOW_RANK_CAPACITY"
            and p6_1b["attribution"]["secondary_factor"] == "LATENT_NORMALIZATION",
            "same_p5": Path(p6_1["source_paths"]["p5_checkpoint"]).resolve() == args.p5_checkpoint
            and Path(p6_1b["source_paths"]["p5_checkpoint"]).resolve() == args.p5_checkpoint,
            "heldout": p6_1["heldout_accessed"] is False and p6_1b["heldout_accessed"] is False,
            "downstream_blocked": p6_1b["p6_2_allowed"] is False and p6_1b["p7_allowed"] is False,
        }
        if not all(source_checks.values()):
            raise ValueError(f"P6.1c source checks failed: {source_checks}")
        corpus = verify_frozen_corpus(args.data_dir)
        if corpus["p5_probe_corpus_frozen"] is not True:
            raise ValueError("P6.1c frozen corpus verification failed")
        device = torch.device("cuda")
        tokenizer = AutoTokenizer.from_pretrained(args.p5_checkpoint, trust_remote_code=False)
        prompts = load_prompts(args.prompts)
        train = PackedTokenDataset(args.data_dir / "packed", "train")
        dev = PackedTokenDataset(args.data_dir / "packed", "dev")
        model = load_densek3_model(args.p5_checkpoint, device=device, dtype=torch.bfloat16)
        a = canonical_observation(model, tokenizer)
        prior_a = p6_1["p6_1"]["cases"]["A_GQA_PARTIAL_ROPE"]
        if a["canonical_logits_sha256"] != prior_a["canonical_logits_sha256"]:
            raise ValueError("P6.1c frozen A canonical hash replay failed")
        donor = model.model.layers[3].self_attn
        if not isinstance(donor, DenseK3GQA) or donor.layer_idx != 3:
            raise ValueError("P6.1c layer 3 is not the frozen donor GQA")
        residual_batches, normalized_batches, calibration_manifest = collect_layer3_inputs(
            model,
            train,
            list(range(8)),
        )
        print("P6_1C_COVARIANCE=RUNNING", flush=True)
        covariance, covariance_report = covariance_from_batches(normalized_batches, device=device)
        print("P6_1C_COVARIANCE=PASS", flush=True)
        print("P6_1C_W0_WEIGHT_SVD=RUNNING", flush=True)
        weight_factor = deterministic_rank_factorization(
            donor.k_proj.weight,
            donor.v_proj.weight,
            num_heads=donor.num_heads,
            num_kv_heads=donor.num_key_value_heads,
            head_dim=donor.head_dim,
            rank=512,
        )
        print("P6_1C_W0_WEIGHT_SVD=PASS", flush=True)
        print("P6_1C_W1_ACTIVATION_WEIGHTED_SVD=RUNNING", flush=True)
        activation_factor = activation_weighted_rank_factorization(
            donor.k_proj.weight,
            donor.v_proj.weight,
            covariance,
            num_heads=donor.num_heads,
            num_kv_heads=donor.num_key_value_heads,
            head_dim=donor.head_dim,
            rank=512,
        )
        print("P6_1C_W1_ACTIVATION_WEIGHTED_SVD=PASS", flush=True)
        prior_factor = p6_1["p6_1"]["conversion_manifest"]["mapping"]["kv"]
        weight_factor_replay = {
            "kv_a": weight_factor.report["kv_a_sha256_fp32"] == prior_factor["kv_a_sha256_fp32"],
            "kv_b": weight_factor.report["kv_b_sha256_fp32"] == prior_factor["kv_b_sha256_fp32"],
        }
        if not all(weight_factor_replay.values()):
            raise ValueError(f"P6.1c C0 factor replay failed: {weight_factor_replay}")
        cases = {
            "C0": P6NoPEMLA(donor, weight_factor).eval(),
            "C1": P6NoPEMLA(donor, weight_factor).eval(),
            "C2": P6NoPEMLA(donor, activation_factor).eval(),
            "C3": P6NoPEMLA(donor, activation_factor).eval(),
        }
        print("P6_1C_N1_SCALE_MATCH=RUNNING", flush=True)
        scale_reports = {
            "C1": fit_case_scale(cases["C1"], normalized_batches),
            "C3": fit_case_scale(cases["C3"], normalized_batches),
        }
        print("P6_1C_N1_SCALE_MATCH=PASS", flush=True)
        b_module = P6NoPEFullAttention(donor, split_q_gate=False, apply_qk_norm=True).eval()
        selection_prefix = config["calibration"]["selection_token_prefix"]
        reference_hidden_batches, reference_calibration_manifest = calibration_reference_hidden(
            model,
            train,
            list(range(8)),
            prefix=selection_prefix,
        )
        frozen_c0 = p6_1["p6_1"]["cases"]["C_RANK512_MLA_NOPE"]
        case_reports = {}
        for name, module in cases.items():
            print(f"P6_1C_CASE_{name}=CALIBRATING", flush=True)
            local_metrics, local_hashes = calibration_local_metrics(
                b_module,
                module,
                residual_batches,
                normalized_batches,
                prefix=selection_prefix,
            )
            model.model.layers[3].self_attn = module
            model_metrics = calibration_model_metrics(
                model,
                train,
                list(range(8)),
                reference_hidden_batches,
                prefix=selection_prefix,
            )
            observation = canonical_observation(model, tokenizer)
            case_reports[name] = {
                "calibration": {
                    "local_vs_B": local_metrics,
                    "local_trace_hashes": local_hashes,
                    "model_vs_A": model_metrics,
                },
                "canonical_diagnostic_only": {
                    "model_vs_A": compare_models(a, observation, model.config.mixer_types),
                    "cross_entropy": observation["canonical_ce"],
                    "logits_sha256": observation["canonical_logits_sha256"],
                    "finite": observation["finite"],
                    "used_for_selection": False,
                },
                "finite": observation["finite"]
                and math.isfinite(model_metrics["cross_entropy"])
                and all(
                    value["reference_nan_count"] == 0
                    and value["actual_nan_count"] == 0
                    and value["reference_inf_count"] == 0
                    and value["actual_inf_count"] == 0
                    for value in (*local_metrics.values(), model_metrics["final_hidden"], model_metrics["logits"])
                ),
                "factorization": "weight_svd" if name in {"C0", "C1"} else "activation_weighted_svd",
                "latent_norm": "scale_matched" if name in {"C1", "C3"} else "ones",
            }
            print(
                f"P6_1C_CASE_{name}=CALIBRATED "
                f"MIXER_REL_L2={local_metrics['mixer_output']['relative_l2_error']:.9g}",
                flush=True,
            )
        c0_replay = {
            "canonical_logits_sha256": case_reports["C0"]["canonical_diagnostic_only"]["logits_sha256"]
            == frozen_c0["canonical_logits_sha256"],
            "canonical_ce": case_reports["C0"]["canonical_diagnostic_only"]["cross_entropy"]
            == frozen_c0["canonical_ce"],
        }
        if not all(c0_replay.values()):
            raise ValueError(f"P6.1c C0 anchor replay failed: {c0_replay}")
        selected = min(("C1", "C2", "C3"), key=lambda name: selection_key(case_reports[name]))
        selected_module = cases[selected]
        model.model.layers[3].self_attn = selected_module
        selected_full_dev = full_dev_ce(model, dev)
        selected_generation = generation_report(model, tokenizer, prompts)
        case_reports[selected]["full_dev_ce"] = selected_full_dev
        case_reports[selected]["generation"] = selected_generation
        b_full_dev = p6_1["p6_1"]["cases"]["B_FULL_RANK_NOPE"]["full_dev_ce"]
        thresholds = config["gate"]
        gate_checks = {
            "mixer": case_reports[selected]["calibration"]["local_vs_B"]["mixer_output"][
                "relative_l2_error"
            ]
            <= thresholds["mixer_relative_l2_max"],
            "full_dev": selected_full_dev - b_full_dev <= thresholds["b_to_candidate_full_dev_ce_delta_max"],
            "generation": selected_generation["not_collapsed"] is True,
            "finite": case_reports[selected]["finite"] is True and math.isfinite(selected_full_dev),
            "final_hidden_improved": case_reports[selected]["calibration"]["model_vs_A"]["final_hidden"][
                "cosine_similarity"
            ]
            > case_reports["C0"]["calibration"]["model_vs_A"]["final_hidden"]["cosine_similarity"],
            "logits_improved": case_reports[selected]["calibration"]["model_vs_A"]["logits"][
                "cosine_similarity"
            ]
            > case_reports["C0"]["calibration"]["model_vs_A"]["logits"]["cosine_similarity"],
        }
        passed = all(gate_checks.values())
        initialization_names = {
            "C1": "WEIGHT_SVD_PLUS_SCALE_MATCHED_LATENT_NORM",
            "C2": "ACTIVATION_WEIGHTED_RANK512_PLUS_ONES_LATENT_NORM",
            "C3": "ACTIVATION_WEIGHTED_RANK512_PLUS_SCALE_MATCHED_LATENT_NORM",
        }
        report = {
            "schema_version": 1,
            "stage": "P6.1c",
            "status": "PASS" if passed else "REDESIGN",
            "source_paths": {
                "p5_checkpoint": str(args.p5_checkpoint),
                "p6_1_report": str(args.p6_1_report),
                "p6_1b_report": str(args.p6_1b_report),
                "frozen_corpus": str(args.data_dir),
            },
            "config_checks": config_checks,
            "source_checks": source_checks,
            "calibration": calibration_manifest,
            "selection_calibration_reference": reference_calibration_manifest,
            "covariance": covariance_report,
            "factorizations": {"W0": weight_factor.report, "W1": activation_factor.report},
            "weight_factor_replay": weight_factor_replay,
            "scale_matching": scale_reports,
            "c0_replay": c0_replay,
            "cases": case_reports,
            "selection": {
                "selected_case": selected,
                "selected_initialization": initialization_names[selected],
                "selection_keys": {name: list(selection_key(case_reports[name])) for name in ("C1", "C2", "C3")},
                "canonical_used_for_selection": False,
                "full_dev_candidates_evaluated": [selected],
            },
            "gate": {
                "checks": gate_checks,
                "passed": passed,
                "thresholds": thresholds,
                "b_full_dev_ce": b_full_dev,
                "selected_full_dev_ce": selected_full_dev,
                "b_to_selected_full_dev_ce_delta": selected_full_dev - b_full_dev,
                "preferred_delta_met": selected_full_dev - b_full_dev
                <= thresholds["preferred_full_dev_ce_delta_max"],
            },
            "training_performed": False,
            "rank": 512,
            "rank_changed": False,
            "other_7_gqa_layers_converted": False,
            "true_latent_cache": False,
            "heldout_accessed": False,
            "p6_2_allowed": passed,
            "p7_allowed": False,
            "result_marker": "P6_1C=PASS" if passed else "P6_1C=REDESIGN",
            "completed_at": now(),
        }
        save_json(report_path, report)
        final = {
            "stage": "P6.1c",
            "status": report["status"],
            "selected_case": selected,
            "selected_initialization": initialization_names[selected],
            "heldout_accessed": False,
            "p6_2_allowed": passed,
            "p7_allowed": False,
            "result_marker": report["result_marker"],
            "completed_at": now(),
        }
        save_json(final_path, final)
        print(report["result_marker"], flush=True)
        print(f"P6_1C_SELECTED_CASE={selected}", flush=True)
        print(f"P6_1C_INITIALIZATION={initialization_names[selected]}", flush=True)
        print(f"P6_2_ALLOWED={str(passed).lower()}", flush=True)
        print("P7_ALLOWED=false", flush=True)
        return 0 if passed else 1
    except Exception as exc:  # noqa: BLE001 - persist every diagnostic failure
        failure = {
            "stage": "P6.1c",
            "status": "ERROR",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "heldout_accessed": False,
            "p6_2_allowed": False,
            "p7_allowed": False,
            "result_marker": "P6_1C=ERROR",
            "completed_at": now(),
        }
        save_json(report_path, failure)
        save_json(final_path, failure)
        print(failure["result_marker"], flush=True)
        print("P6_2_ALLOWED=false", flush=True)
        print("P7_ALLOWED=false", flush=True)
        print(f"P6_1C_EXCEPTION={type(exc).__name__}: {exc}", flush=True)
        return 1
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
