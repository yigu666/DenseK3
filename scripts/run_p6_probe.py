"""Run P6.0 frozen-P5 replay and the no-training layer-3 A/B/C MLA probe."""

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
from densek3_core.modeling.modeling_densek3 import DenseK3GQA, _apply_partial_rope
from densek3_core.recovery.p5_closeout import verify_native_checkpoint
from densek3_core.recovery.p5_corpus import PackedTokenDataset, verify_frozen_corpus
from densek3_core.recovery.p5_transition import hidden_drift, token_decision_drift
from densek3_core.recovery.p6_mla_probe import (
    P6NoPEFullAttention,
    P6NoPEMLA,
    _attention,
    conversion_manifest,
    deterministic_rank_factorization,
    tensor_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/architecture/p6-single-layer-mla-probe.yaml"
DEFAULT_P5 = ROOT / "outputs/densek3-core/checkpoints/p5/densek3-4b-k3-decay-silu-i1"
DEFAULT_P5_FINAL = ROOT / "outputs/densek3-core/manifests/p5/i1-closeout/p5-final-check.json"
DEFAULT_P5_INITIALIZATION = (
    ROOT / "outputs/densek3-core/manifests/p5/rethink-v3/p5-rethink-v3-initialization-probe.json"
)
DEFAULT_P5_FREEZE = ROOT / "outputs/densek3-core/manifests/p5/i1-closeout/p5-i1-freeze-report.json"
DEFAULT_DATA = ROOT / "data/p5-probe"
DEFAULT_PROMPTS = ROOT / "data/p4-smoke-prompts.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs/densek3-core/manifests/p6"
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
        "stage": config["stage"] == "P6.0-P6.1" and config["status"] == "FROZEN",
        "source": config["source"]["p5_final_required"] is True
        and config["source"]["architecture"] == "K3_DECAY_PLUS_SILU"
        and config["source"]["initialization"] == "RETENTION_MATCHED_ANALYTIC"
        and config["source"]["freeze_report"].endswith("p5-i1-freeze-report.json"),
        "single_layer": config["scope"]["probe_layer"] == 3
        and config["scope"]["converted_layers"] == 1
        and config["scope"]["other_layers_unchanged"] is True,
        "cases": config["cases"]
        == {"A": "GQA_PARTIAL_ROPE", "B": "FULL_RANK_ATTENTION_NOPE", "C": "RANK512_MLA_NOPE"},
        "mla": config["mla"]
        == {
            "heads": 16,
            "q_lora_rank": None,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 256,
            "qk_pos_head_dim": 0,
            "value_head_dim": 256,
            "position_encoding": "none",
            "output_gate": "full_rank_sigmoid",
            "output_gate_bias": False,
            "kv_a_layernorm_init": "ones",
            "q_norm": "excluded",
            "k_norm": "excluded",
            "cache_mode": "expanded_reference",
            "true_latent_cache": False,
        },
        "measurement_only": config["evaluation"]["automatic_scale_gate"] is False
        and config["scope"]["training"] is False
        and config["scope"]["scale_to_all_8_layers"] is False,
        "baseline_replay": config["evaluation"]["baseline_replay"]
        == {
            "canonical_ce_abs_tolerance": 0.0005,
            "full_dev_ce_abs_tolerance": 0.0005,
            "generation_tokens_exact": True,
        },
        "all_forbidden": all(config["forbidden"].values()),
    }
    if not all(checks.values()):
        raise ValueError(f"P6 probe config failed: {checks}")
    return checks


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("P6 probe requires the authoritative CUDA server")
    return torch.device("cuda")


def load_prompts(path: Path) -> list[dict[str, str]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if {item["id"] for item in records} != {"english", "chinese", "code", "factual", "longer"}:
        raise ValueError("P6 must reuse exactly the frozen five-category prompts")
    return records


@torch.no_grad()
def full_dev_ce(model: torch.nn.Module, dev: PackedTokenDataset, *, chunk_size: int = 64) -> float:
    if dev.split != "dev" or len(dev) != 97:
        raise ValueError("P6 requires exactly the frozen 97-sequence dev split")
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0
    for index in range(len(dev)):
        tokens = torch.from_numpy(dev[index].copy()).unsqueeze(0).to(device)
        hidden = model.model(input_ids=tokens, use_cache=False, return_dict=True).last_hidden_state[:, :-1]
        labels = tokens[:, 1:]
        for start in range(0, hidden.shape[1], chunk_size):
            end = min(start + chunk_size, hidden.shape[1])
            # Replay the frozen P5 initialization audit's authoritative FP32
            # readout exactly; this is a baseline identity check, not a new Gate.
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
            print(f"P6_FULL_DEV_PROGRESS={index + 1}/97", flush=True)
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
                "text": tokenizer.decode(generated[0], skip_special_tokens=False),
                "not_collapsed": bool(tokens)
                and max(counts.values()) / len(tokens) <= 0.75
                and len(counts) / len(tokens) >= 0.25,
            }
        )
    return {"records": records, "not_collapsed": all(item["not_collapsed"] for item in records)}


def generation_drift(reference: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    reference_records = {item["id"]: item["new_tokens"] for item in reference["records"]}
    actual_records = {item["id"]: item["new_tokens"] for item in actual["records"]}
    if reference_records.keys() != actual_records.keys():
        raise ValueError("P6 generation prompt IDs differ")
    records = []
    for prompt_id in sorted(reference_records):
        expected = reference_records[prompt_id]
        observed = actual_records[prompt_id]
        first = next(
            (
                index
                for index, (reference_token, actual_token) in enumerate(zip(expected, observed, strict=True))
                if reference_token != actual_token
            ),
            None,
        )
        records.append(
            {
                "id": prompt_id,
                "exact": first is None,
                "first_divergence_position": first,
                "reference_token": None if first is None else expected[first],
                "actual_token": None if first is None else observed[first],
            }
        )
    return {"exact": all(item["exact"] for item in records), "records": records}


@torch.no_grad()
def cache_smoke(model: torch.nn.Module, tokenizer: Any, *, expected_layer3_heads: int) -> dict[str, Any]:
    device = next(model.parameters()).device
    encoded = tokenizer(CANONICAL_PROMPT, return_tensors="pt", truncation=True, max_length=32)
    tokens = encoded["input_ids"].to(device)
    mask = encoded["attention_mask"].to(device)
    prefill = model(input_ids=tokens, attention_mask=mask, use_cache=True, return_dict=True)
    cache = prefill.past_key_values
    layer = cache.gqa_layer(3)
    prefill_seen = cache.seen_tokens
    key_shape = None if layer.key is None else list(layer.key.shape)
    value_shape = None if layer.value is None else list(layer.value.shape)
    next_token = prefill.logits[:, -1].argmax(dim=-1, keepdim=True)
    continuation = model(
        input_ids=next_token,
        attention_mask=torch.ones((1, tokens.shape[1] + 1), device=device, dtype=mask.dtype),
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )
    checks = {
        "prefill_seen_tokens": prefill_seen == tokens.shape[1],
        "layer_3_key_shape": key_shape == [1, expected_layer3_heads, tokens.shape[1], 256],
        "layer_3_value_shape": value_shape == [1, expected_layer3_heads, tokens.shape[1], 256],
        "continuation_seen_tokens": continuation.past_key_values.seen_tokens == tokens.shape[1] + 1,
        "continuation_shape": list(continuation.logits.shape) == [1, 1, model.config.vocab_size],
        "finite": bool(torch.isfinite(prefill.logits).all() and torch.isfinite(continuation.logits).all()),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "cache_mode": "expanded_reference",
        "true_latent_cache": False,
        "layer_3_key_shape": key_shape,
        "layer_3_value_shape": value_shape,
    }


@torch.no_grad()
def observe_model(
    model: torch.nn.Module,
    tokenizer: Any,
    dev: PackedTokenDataset,
    prompts: list[dict[str, str]],
) -> dict[str, Any]:
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
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), input_ids.cpu()[:, 1:].reshape(-1))
    dev_ce = full_dev_ce(model, dev)
    generation = generation_report(model, tokenizer, prompts)
    return {
        "input_ids": input_ids.cpu(),
        "logits": logits,
        "hidden_states": hidden,
        "public": {
            "canonical_ce": float(ce),
            "full_dev_ce": dev_ce,
            "canonical_logits_sha256": tensor_sha256(output.logits),
            "generation": generation,
            "generation_not_collapsed": generation["not_collapsed"],
            "finite": math.isfinite(float(ce))
            and math.isfinite(dev_ce)
            and bool(torch.isfinite(logits).all())
            and all(bool(torch.isfinite(value).all()) for value in hidden),
        },
    }


def compare(reference: dict[str, Any], actual: dict[str, Any], mixer_types: list[str]) -> dict[str, Any]:
    if not torch.equal(reference["input_ids"], actual["input_ids"]):
        raise ValueError("P6 canonical token IDs differ")
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
        "generation_drift": generation_drift(
            reference["public"]["generation"],
            actual["public"]["generation"],
        ),
    }


@torch.no_grad()
def single_layer_diagnostics(
    donor: DenseK3GQA,
    full_rank: P6NoPEFullAttention,
    mla: P6NoPEMLA,
    residual_hidden_states: torch.Tensor,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
) -> dict[str, Any]:
    batch, sequence, _ = hidden_states.shape
    mask = torch.ones((batch, sequence), device=hidden_states.device)
    output_a = donor(hidden_states, position_ids, mask, None, False)
    output_b = full_rank(hidden_states, position_ids, mask, None, False)
    output_c = mla(hidden_states, position_ids, mask, None, False)
    packed_a = donor.q_proj(hidden_states).view(batch, sequence, donor.num_heads, 2, donor.head_dim)
    q_raw_a, gate_a = packed_a.unbind(dim=3)
    q_a = donor.q_norm(q_raw_a).transpose(1, 2)
    k_raw_compact_a = donor.k_proj(hidden_states).view(
        batch,
        sequence,
        donor.num_key_value_heads,
        donor.head_dim,
    )
    k_a = donor.k_norm(k_raw_compact_a).transpose(1, 2)
    v_compact_a = donor.v_proj(hidden_states).view(
        batch,
        sequence,
        donor.num_key_value_heads,
        donor.head_dim,
    )
    q_a, k_a = _apply_partial_rope(q_a, k_a, position_ids, donor.rotary_dim, donor.rope_theta)
    k_a = k_a.repeat_interleave(donor.group_size, dim=1)
    v_a = v_compact_a.transpose(1, 2).repeat_interleave(donor.group_size, dim=1)
    core_a = _attention(q_a, k_a, v_a, mask, past_length=0, dropout=0.0, training=False)

    q_b, k_b, v_b, gate_b, k_raw_b = full_rank.project_qkvg(hidden_states)
    core_b = _attention(q_b, k_b, v_b, mask, past_length=0, dropout=0.0, training=False)
    k_compressed, v_compressed, latent = mla.project_kv(hidden_states, apply_latent_norm=False)
    k_normed, v_normed, _ = mla.project_kv(hidden_states, apply_latent_norm=True)
    q_c = mla.q_proj(hidden_states).view(batch, sequence, mla.num_heads, mla.head_dim)
    gate_c = mla.g_proj(hidden_states).view(batch, sequence, mla.num_heads, mla.head_dim)
    core_c = _attention(
        q_c.transpose(1, 2),
        k_normed.transpose(1, 2),
        v_normed.transpose(1, 2),
        mask,
        past_length=0,
        dropout=0.0,
        training=False,
    )
    t0_mask = torch.ones((batch, 1), device=hidden_states.device)
    t0_positions = torch.zeros((batch, 1), device=hidden_states.device, dtype=torch.long)
    donor_t0 = donor(hidden_states[:, :1], t0_positions, t0_mask, None, False)
    full_rank_t0 = full_rank(hidden_states[:, :1], t0_positions, t0_mask, None, False)
    return {
        "input_shape": list(hidden_states.shape),
        "case_b_t0_matches_donor_exact": torch.equal(donor_t0, full_rank_t0),
        "mixer_output": {
            "A_to_B": hidden_drift(output_a, output_b),
            "B_to_C": hidden_drift(output_b, output_c),
            "A_to_C": hidden_drift(output_a, output_c),
        },
        "post_attention_residual_hidden": {
            "A_to_B": hidden_drift(residual_hidden_states + output_a, residual_hidden_states + output_b),
            "B_to_C": hidden_drift(residual_hidden_states + output_b, residual_hidden_states + output_c),
            "A_to_C": hidden_drift(residual_hidden_states + output_a, residual_hidden_states + output_c),
        },
        "attention_core_output": {
            "A_to_B": hidden_drift(core_a, core_b),
            "B_to_C": hidden_drift(core_b, core_c),
            "A_to_C": hidden_drift(core_a, core_c),
        },
        "split_projection_numerics": {
            "q": hidden_drift(q_raw_a, q_c),
            "gate": hidden_drift(gate_a, gate_c),
        },
        "linear_compression_without_latent_norm": {
            "k_vs_unnormalized_full_rank": hidden_drift(k_raw_b, k_compressed),
            "v": hidden_drift(v_b.transpose(1, 2), v_compressed),
        },
        "qk_norm_exclusion": {
            "q": hidden_drift(q_b.transpose(1, 2), q_raw_a),
            "k": hidden_drift(k_b.transpose(1, 2), k_raw_b),
        },
        "latent_norm_effect": {
            "latent": {
                "shape": list(latent.shape),
                "rms_before": float(latent.float().square().mean().sqrt()),
            },
            "k": hidden_drift(k_compressed, k_normed),
            "v": hidden_drift(v_compressed, v_normed),
        },
        "attribution_boundaries": {
            "A_to_B_contains": ["ROPE_REMOVAL_ONLY"],
            "B_to_C_contains": [
                "Q_NORM_EXCLUSION",
                "K_NORM_EXCLUSION",
                "RANK512_COMPRESSION",
                "KV_A_RMSNORM_INSERTION",
                "PACKED_Q_GATE_TO_SPLIT_GEMM_NUMERICS",
            ],
            "pure_linear_compression_reported_separately": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--p5-checkpoint", type=Path, default=DEFAULT_P5)
    parser.add_argument("--p5-final", type=Path, default=DEFAULT_P5_FINAL)
    parser.add_argument("--p5-initialization-report", type=Path, default=DEFAULT_P5_INITIALIZATION)
    parser.add_argument("--p5-freeze-report", type=Path, default=DEFAULT_P5_FREEZE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in (
        "config",
        "p5_checkpoint",
        "p5_final",
        "p5_initialization_report",
        "p5_freeze_report",
        "data_dir",
        "prompts",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "p6-single-layer-probe.json"
    final_path = args.output_dir / "p6-final-check.json"
    print("P6_0_P5_BASELINE=RUNNING", flush=True)
    print("P6_1_SINGLE_LAYER_MLA_PROBE=BLOCKED", flush=True)
    print("P6_2_ALLOWED=false", flush=True)
    try:
        from transformers import AutoTokenizer

        config_checks = validate_config(load_yaml(args.config))
        p5_final = load_json(args.p5_final)
        p5_initialization = load_json(args.p5_initialization_report)
        p5_freeze = load_json(args.p5_freeze_report)
        final_checks = {
            "stage": p5_final["stage"] == "P5" and p5_final["status"] == "PASS",
            "frozen": p5_final["p5_frozen"] is True and p5_final["p6_allowed"] is True,
            "architecture": p5_final["architecture"] == "K3_DECAY_PLUS_SILU",
            "initialization": p5_final["initialization"] == "RETENTION_MATCHED_ANALYTIC",
            "checkpoint": Path(p5_final["checkpoint_directory"]).resolve() == args.p5_checkpoint,
            "freeze_report": p5_freeze["status"] == "PASS" and p5_freeze["passed"] is True,
        }
        if not all(final_checks.values()):
            raise ValueError(f"P6 frozen P5 final-check failed: {final_checks}")
        checkpoint = verify_native_checkpoint(args.p5_checkpoint, verify_tensors=True)
        if checkpoint["status"] != "PASS":
            raise ValueError(f"P6 P5 checkpoint verification failed: {checkpoint['checks']}")
        corpus = verify_frozen_corpus(args.data_dir)
        if corpus["p5_probe_corpus_frozen"] is not True:
            raise ValueError("P6 frozen dev corpus verification failed")
        device = require_cuda()
        tokenizer = AutoTokenizer.from_pretrained(args.p5_checkpoint, trust_remote_code=False)
        prompts = load_prompts(args.prompts)
        dev = PackedTokenDataset(args.data_dir / "packed", "dev")

        baseline_model = load_densek3_model(args.p5_checkpoint, device=device, dtype=torch.bfloat16)
        baseline = observe_model(baseline_model, tokenizer, dev, prompts)
        baseline_public = baseline["public"]
        frozen_i1 = p5_initialization["initializations"]["I1"]
        frozen_generations = {
            item["id"]: item["new_tokens"] for item in frozen_i1["generations"]
        }
        replay_generations = {
            item["id"]: item["new_tokens"] for item in baseline_public["generation"]["records"]
        }
        baseline_cache = cache_smoke(baseline_model, tokenizer, expected_layer3_heads=4)
        baseline_gate = {
            "checkpoint_hashes": checkpoint["status"] == "PASS" and checkpoint["tensor_hashes_verified"] is True,
            "config": baseline_model.config.kda_parameterization == "p5_k3_decay_silu",
            "tensor_count": checkpoint["checks"]["tensor_count"],
            "topology": len(baseline_model.config.kda_indices) == 24 and len(baseline_model.config.gqa_indices) == 8,
            "ce": baseline_public["finite"],
            "canonical_ce_replay": abs(
                baseline_public["canonical_ce"] - frozen_i1["canonical_ce"]
            )
            <= 0.0005,
            "full_dev_ce_replay": abs(
                baseline_public["full_dev_ce"] - frozen_i1["full_dev_ce"]
            )
            <= 0.0005,
            "generation": baseline_public["generation_not_collapsed"],
            "generation_tokens_exact": replay_generations == frozen_generations,
            "cache": baseline_cache["passed"],
            "frozen_forward_hash": baseline_public["canonical_logits_sha256"]
            == p5_freeze["bf16_forward"]["reloaded"]["logits_hash"],
            "frozen_cache_gate": p5_freeze["cache"]["passed"] is True,
            "frozen_generation_gate": p5_freeze["generation"]["not_collapsed"] is True,
        }
        if not all(baseline_gate.values()):
            raise ValueError(f"P6.0 baseline Gate failed: {baseline_gate}")
        print("P6_0_P5_BASELINE=PASS", flush=True)
        print("P6_1_SINGLE_LAYER_MLA_PROBE=RUNNING", flush=True)

        donor = baseline_model.model.layers[3].self_attn
        if not isinstance(donor, DenseK3GQA) or donor.layer_idx != 3:
            raise ValueError("P6.1 layer 3 is not the frozen first GQA layer")
        factorization = deterministic_rank_factorization(
            donor.k_proj.weight,
            donor.v_proj.weight,
            num_heads=donor.num_heads,
            num_kv_heads=donor.num_key_value_heads,
            head_dim=donor.head_dim,
            rank=512,
        )
        manifest = conversion_manifest(donor, factorization)
        full_rank = P6NoPEFullAttention(donor).eval()
        mla = P6NoPEMLA(donor, factorization).eval()
        residual_hidden = baseline["hidden_states"][3].to(device)
        hidden_input = baseline_model.model.layers[3].input_layernorm(residual_hidden)
        position_ids = torch.arange(hidden_input.shape[1], device=device).unsqueeze(0)
        local = single_layer_diagnostics(
            donor,
            full_rank,
            mla,
            residual_hidden,
            hidden_input,
            position_ids,
        )
        if not local["case_b_t0_matches_donor_exact"]:
            raise ValueError("P6 Case B failed its position-zero pure-NoPE isolation")

        baseline_model.model.layers[3].self_attn = full_rank
        case_b = observe_model(baseline_model, tokenizer, dev, prompts)
        case_b_cache = cache_smoke(baseline_model, tokenizer, expected_layer3_heads=16)
        case_b_public = {**case_b["public"], **compare(baseline, case_b, baseline_model.config.mixer_types)}
        case_b_public["cache"] = case_b_cache
        baseline_model.model.layers[3].self_attn = mla
        case_c = observe_model(baseline_model, tokenizer, dev, prompts)
        case_c_cache = cache_smoke(baseline_model, tokenizer, expected_layer3_heads=16)
        case_c_public = {**case_c["public"], **compare(baseline, case_c, baseline_model.config.mixer_types)}
        case_c_public["cache"] = case_c_cache
        b_to_c = compare(case_b, case_c, baseline_model.config.mixer_types)
        finite = (
            baseline_public["finite"]
            and case_b_public["finite"]
            and case_c_public["finite"]
            and case_b_cache["passed"]
            and case_c_cache["passed"]
        )
        report = {
            "schema_version": 1,
            "stage": "P6.0-P6.1",
            "status": "PASS" if finite else "FAIL",
            "source_paths": {
                "p5_checkpoint": str(args.p5_checkpoint),
                "p5_final": str(args.p5_final),
                "p5_initialization_report": str(args.p5_initialization_report),
                "p5_freeze_report": str(args.p5_freeze_report),
                "frozen_dev_corpus": str(args.data_dir),
            },
            "config_checks": config_checks,
            "p5_final_checks": final_checks,
            "p5_checkpoint_verification": checkpoint,
            "p6_0": {
                "status": "PASS",
                "gate": baseline_gate,
                "baseline": baseline_public,
                "cache": baseline_cache,
            },
            "p6_1": {
                "status": "PASS_MEASUREMENT_COMPLETE" if finite else "FAIL_NONFINITE",
                "probe_layer": 3,
                "training_performed": False,
                "conversion_manifest": manifest,
                "single_layer_diagnostics": local,
                "cases": {
                    "A_GQA_PARTIAL_ROPE": baseline_public,
                    "B_FULL_RANK_NOPE": case_b_public,
                    "C_RANK512_MLA_NOPE": case_c_public,
                },
                "attribution": {
                    "A_to_B": case_b_public,
                    "B_to_C": b_to_c,
                    "A_to_C": case_c_public,
                    "interpretation_requires_review": True,
                },
                "cache_mode": "expanded_reference",
                "true_latent_cache": False,
                "other_7_gqa_layers_converted": False,
            },
            "decision": "REVIEW_REQUIRED",
            "heldout_accessed": False,
            "p6_2_allowed": False,
            "p7_allowed": False,
            "result_marker": "P6_1_SINGLE_LAYER_MLA_PROBE=PASS_MEASUREMENT_COMPLETE",
            "completed_at": now(),
        }
        save_json(report_path, report)
        final = {
            "stage": "P6",
            "status": "AWAITING_P6_1_REVIEW",
            "p6_0_baseline": True,
            "p6_1_measurement_complete": finite,
            "decision": "REVIEW_REQUIRED",
            "heldout_accessed": False,
            "p6_2_allowed": False,
            "p7_allowed": False,
            "result_marker": report["result_marker"],
            "completed_at": now(),
        }
        save_json(final_path, final)
        print(report["result_marker"], flush=True)
        print("P6_1_DECISION=REVIEW_REQUIRED", flush=True)
        print("P6_2_ALLOWED=false", flush=True)
        print("P7_ALLOWED=false", flush=True)
        return 0 if finite else 1
    except Exception as exc:  # noqa: BLE001 - every P6 failure must persist
        failure = {
            "stage": "P6.0-P6.1",
            "status": "ERROR",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "p6_2_allowed": False,
            "p7_allowed": False,
            "result_marker": "P6_PROBE=ERROR",
            "completed_at": now(),
        }
        save_json(report_path, failure)
        save_json(final_path, failure)
        print("P6_PROBE=ERROR", flush=True)
        print("P6_2_ALLOWED=false", flush=True)
        print("P7_ALLOWED=false", flush=True)
        print(f"P6_EXCEPTION={type(exc).__name__}: {exc}", flush=True)
        return 1
    finally:
        clear_cuda()


if __name__ == "__main__":
    raise SystemExit(main())
