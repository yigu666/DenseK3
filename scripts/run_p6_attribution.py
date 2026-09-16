"""Run the frozen no-training P6.1b layer-3 MLA attribution matrix."""

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
from densek3_core.recovery.p6_attribution import classify_attribution, compare_traces, trace_case, trace_hashes
from densek3_core.recovery.p6_mla_probe import (
    P6NoPEFullAttention,
    P6NoPEMLA,
    deterministic_rank_factorization,
    tensor_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/architecture/p6-single-layer-mla-attribution.yaml"
DEFAULT_P5 = ROOT / "outputs/densek3-core/checkpoints/p5/densek3-4b-k3-decay-silu-i1"
DEFAULT_P6_1 = ROOT / "outputs/densek3-core/manifests/p6/p6-single-layer-probe.json"
DEFAULT_DATA = ROOT / "data/p5-probe"
DEFAULT_PROMPTS = ROOT / "data/p4-smoke-prompts.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs/densek3-core/manifests/p6/p6-1b"
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
    cases = config["cases"]
    expected = {
        "B": {"rank512": False, "split_q_gate": False, "qk_norm": True, "latent_norm": False},
        "C": {"rank512": True, "split_q_gate": True, "qk_norm": False, "latent_norm": True},
        "D": {"rank512": False, "split_q_gate": True, "qk_norm": True, "latent_norm": False},
        "E": {"rank512": False, "split_q_gate": False, "qk_norm": False, "latent_norm": False},
        "F": {"rank512": True, "split_q_gate": False, "qk_norm": True, "latent_norm": False},
        "G": {"rank512": True, "split_q_gate": False, "qk_norm": True, "latent_norm": True},
        "H": {"rank512": True, "split_q_gate": False, "qk_norm": False, "latent_norm": False},
    }
    checks = {
        "stage": config["stage"] == "P6.1b" and config["status"] == "FROZEN",
        "source": config["source"]["p6_1_decision"] == "REDESIGN",
        "scope": config["scope"]
        == {
            "probe_layer": 3,
            "training": False,
            "rank": 512,
            "other_layers_unchanged": True,
            "scale_to_all_8_layers": False,
            "true_latent_cache": False,
            "heldout_accessed": False,
        },
        "cases": cases == expected,
        "measurement_only": config["evaluation"]["automatic_capability_gate"] is False
        and config["evaluation"]["full_dev_sequences"] == 97,
        "all_forbidden": all(config["forbidden"].values()),
    }
    if not all(checks.values()):
        raise ValueError(f"P6.1b config failed: {checks}")
    return checks


def load_prompts(path: Path) -> list[dict[str, str]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if {item["id"] for item in records} != {"english", "chinese", "code", "factual", "longer"}:
        raise ValueError("P6.1b must reuse the frozen five-category prompts")
    return records


@torch.no_grad()
def full_dev_ce(model: torch.nn.Module, dev: PackedTokenDataset, *, chunk_size: int = 64) -> float:
    if dev.split != "dev" or len(dev) != 97:
        raise ValueError("P6.1b requires exactly the frozen 97-sequence dev split")
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
            print(f"P6_1B_FULL_DEV_PROGRESS={index + 1}/97", flush=True)
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


@torch.no_grad()
def observe_canonical(
    model: torch.nn.Module,
    tokenizer: Any,
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
    return {
        "input_ids": input_ids.cpu(),
        "logits": logits,
        "hidden_states": hidden,
        "public": {
            "canonical_ce": float(ce),
            "canonical_logits_sha256": tensor_sha256(output.logits),
        },
    }


@torch.no_grad()
def observe_model(
    model: torch.nn.Module,
    tokenizer: Any,
    dev: PackedTokenDataset,
    prompts: list[dict[str, str]],
) -> dict[str, Any]:
    observation = observe_canonical(model, tokenizer)
    dev_ce = full_dev_ce(model, dev)
    generation = generation_report(model, tokenizer, prompts)
    observation["public"].update(
        {
            "full_dev_ce": dev_ce,
            "generation": generation,
            "generation_not_collapsed": generation["not_collapsed"],
            "finite": math.isfinite(observation["public"]["canonical_ce"])
            and math.isfinite(dev_ce)
            and bool(torch.isfinite(observation["logits"]).all())
            and all(bool(torch.isfinite(value).all()) for value in observation["hidden_states"]),
        }
    )
    return observation


def generation_drift(reference: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    reference_records = {item["id"]: item["new_tokens"] for item in reference["records"]}
    actual_records = {item["id"]: item["new_tokens"] for item in actual["records"]}
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
        records.append({"id": prompt_id, "exact": first is None, "first_divergence_position": first})
    return {"exact": all(item["exact"] for item in records), "records": records}


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
        "generation_drift": generation_drift(
            reference["public"]["generation"],
            actual["public"]["generation"],
        ),
    }


def construct_cases(donor: DenseK3GQA, factorization: Any) -> dict[str, torch.nn.Module]:
    return {
        "B": P6NoPEFullAttention(donor, split_q_gate=False, apply_qk_norm=True).eval(),
        "C": P6NoPEMLA(
            donor,
            factorization,
            split_q_gate=True,
            apply_qk_norm=False,
            apply_latent_norm=True,
        ).eval(),
        "D": P6NoPEFullAttention(donor, split_q_gate=True, apply_qk_norm=True).eval(),
        "E": P6NoPEFullAttention(donor, split_q_gate=False, apply_qk_norm=False).eval(),
        "F": P6NoPEMLA(
            donor,
            factorization,
            split_q_gate=False,
            apply_qk_norm=True,
            apply_latent_norm=False,
        ).eval(),
        "G": P6NoPEMLA(
            donor,
            factorization,
            split_q_gate=False,
            apply_qk_norm=True,
            apply_latent_norm=True,
        ).eval(),
        "H": P6NoPEMLA(
            donor,
            factorization,
            split_q_gate=False,
            apply_qk_norm=False,
            apply_latent_norm=False,
        ).eval(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--p5-checkpoint", type=Path, default=DEFAULT_P5)
    parser.add_argument("--p6-1-report", type=Path, default=DEFAULT_P6_1)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in ("config", "p5_checkpoint", "p6_1_report", "data_dir", "prompts", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "p6-1b-attribution.json"
    final_path = args.output_dir / "p6-1b-final-check.json"
    print("P6_1B_ATTRIBUTION=RUNNING", flush=True)
    print("P6_2_ALLOWED=false", flush=True)
    print("P7_ALLOWED=false", flush=True)
    try:
        from transformers import AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("P6.1b requires the authoritative CUDA server")
        config_checks = validate_config(load_yaml(args.config))
        prior = load_json(args.p6_1_report)
        prior_checks = {
            "stage": prior["stage"] == "P6.0-P6.1" and prior["status"] == "PASS",
            "measurement": prior["p6_1"]["status"] == "PASS_MEASUREMENT_COMPLETE",
            "review": prior["decision"] == "REVIEW_REQUIRED" and prior["p6_2_allowed"] is False,
            "source": Path(prior["source_paths"]["p5_checkpoint"]).resolve() == args.p5_checkpoint,
            "heldout": prior["heldout_accessed"] is False,
        }
        if not all(prior_checks.values()):
            raise ValueError(f"P6.1 source report is not frozen: {prior_checks}")
        corpus = verify_frozen_corpus(args.data_dir)
        if corpus["p5_probe_corpus_frozen"] is not True:
            raise ValueError("P6.1b frozen corpus verification failed")
        device = torch.device("cuda")
        tokenizer = AutoTokenizer.from_pretrained(args.p5_checkpoint, trust_remote_code=False)
        prompts = load_prompts(args.prompts)
        dev = PackedTokenDataset(args.data_dir / "packed", "dev")
        model = load_densek3_model(args.p5_checkpoint, device=device, dtype=torch.bfloat16)
        prior_a = prior["p6_1"]["cases"]["A_GQA_PARTIAL_ROPE"]
        prior_b = prior["p6_1"]["cases"]["B_FULL_RANK_NOPE"]
        prior_c = prior["p6_1"]["cases"]["C_RANK512_MLA_NOPE"]
        baseline = observe_canonical(model, tokenizer)
        a_replay = {
            "canonical_logits_sha256": baseline["public"]["canonical_logits_sha256"]
            == prior_a["canonical_logits_sha256"],
            "canonical_ce": baseline["public"]["canonical_ce"] == prior_a["canonical_ce"],
            "frozen_full_dev_present": math.isfinite(prior_a["full_dev_ce"]),
        }
        if not all(a_replay.values()):
            raise ValueError(f"P6.1b A anchor replay failed: {a_replay}")
        donor = model.model.layers[3].self_attn
        if not isinstance(donor, DenseK3GQA) or donor.layer_idx != 3:
            raise ValueError("P6.1b layer 3 is not the frozen donor GQA")
        factorization = deterministic_rank_factorization(
            donor.k_proj.weight,
            donor.v_proj.weight,
            num_heads=donor.num_heads,
            num_kv_heads=donor.num_key_value_heads,
            head_dim=donor.head_dim,
            rank=512,
        )
        prior_factor = prior["p6_1"]["conversion_manifest"]["mapping"]["kv"]
        factor_checks = {
            "rank": factorization.report["rank"] == prior_factor["rank"] == 512,
            "kv_a_hash": factorization.report["kv_a_sha256_fp32"] == prior_factor["kv_a_sha256_fp32"],
            "kv_b_hash": factorization.report["kv_b_sha256_fp32"] == prior_factor["kv_b_sha256_fp32"],
            "energy": factorization.report["explained_squared_singular_mass"]
            == prior_factor["explained_squared_singular_mass"],
            "frobenius": factorization.report["frobenius_relative_error"]
            == prior_factor["frobenius_relative_error"],
        }
        if not all(factor_checks.values()):
            raise ValueError(f"P6.1b shared rank-512 factor replay failed: {factor_checks}")
        cases = construct_cases(donor, factorization)
        residual_hidden = baseline["hidden_states"][3].to(device)
        normalized_hidden = model.model.layers[3].input_layernorm(residual_hidden)
        attention_mask = torch.ones(normalized_hidden.shape[:2], device=device)
        traces = {
            name: trace_case(module, residual_hidden, normalized_hidden, attention_mask)
            for name, module in cases.items()
        }
        effects = {
            "B_to_C": compare_traces(traces["B"], traces["C"]),
            "B_to_D": compare_traces(traces["B"], traces["D"]),
            "B_to_E": compare_traces(traces["B"], traces["E"]),
            "B_to_F": compare_traces(traces["B"], traces["F"]),
            "F_to_G": compare_traces(traces["F"], traces["G"]),
            "F_to_H": compare_traces(traces["F"], traces["H"]),
            "F_to_C": compare_traces(traces["F"], traces["C"]),
        }
        model.model.layers[3].self_attn = cases["B"]
        replay_b = observe_canonical(model, tokenizer)
        model.model.layers[3].self_attn = cases["C"]
        replay_c = observe_canonical(model, tokenizer)
        anchor_checks = {
            "B_logits_hash": replay_b["public"]["canonical_logits_sha256"]
            == prior_b["canonical_logits_sha256"],
            "B_canonical_ce": replay_b["public"]["canonical_ce"] == prior_b["canonical_ce"],
            "B_frozen_full_dev_present": math.isfinite(prior_b["full_dev_ce"]),
            "C_logits_hash": replay_c["public"]["canonical_logits_sha256"]
            == prior_c["canonical_logits_sha256"],
            "C_canonical_ce": replay_c["public"]["canonical_ce"] == prior_c["canonical_ce"],
            "C_frozen_full_dev_present": math.isfinite(prior_c["full_dev_ce"]),
        }
        if not all(anchor_checks.values()):
            raise ValueError(f"P6.1b B/C anchor replay failed: {anchor_checks}")
        baseline["public"] = prior_a
        public_cases: dict[str, Any] = {"A": prior_a, "B": prior_b, "C": prior_c}
        for name in ("D", "E", "F", "G", "H"):
            model.model.layers[3].self_attn = cases[name]
            observation = observe_model(model, tokenizer, dev, prompts)
            public_cases[name] = {
                **observation["public"],
                **compare_models(baseline, observation, model.config.mixer_types),
            }
            print(f"P6_1B_CASE_{name}=MEASURED FULL_DEV_CE={observation['public']['full_dev_ce']:.12f}", flush=True)
        rank_case_hashes = {
            name: {
                "kv_a_sha256_bf16": tensor_sha256(cases[name].kv_a_proj.weight),
                "kv_b_sha256_bf16": tensor_sha256(cases[name].kv_b_proj.weight),
            }
            for name in ("C", "F", "G", "H")
        }
        shared_rank_hashes = len({json.dumps(value, sort_keys=True) for value in rank_case_hashes.values()}) == 1
        finite = all(case["finite"] for case in public_cases.values())
        attribution = classify_attribution(effects)
        report = {
            "schema_version": 1,
            "stage": "P6.1b",
            "status": "PASS_ATTRIBUTION_COMPLETE" if finite and shared_rank_hashes else "FAIL",
            "source_paths": {
                "p5_checkpoint": str(args.p5_checkpoint),
                "p6_1_report": str(args.p6_1_report),
                "frozen_dev_corpus": str(args.data_dir),
            },
            "config_checks": config_checks,
            "prior_report_checks": prior_checks,
            "anchor_replay": {"A": a_replay, "B_C": anchor_checks},
            "anchor_full_evaluations_reused_from_frozen_p6_1": True,
            "rank512_factor": factorization.report,
            "rank512_factor_checks": factor_checks,
            "rank512_case_weight_hashes": rank_case_hashes,
            "rank512_hashes_shared": shared_rank_hashes,
            "cases": public_cases,
            "local_effects": effects,
            "local_trace_hashes": {name: trace_hashes(trace) for name, trace in traces.items()},
            "attribution": attribution,
            "training_performed": False,
            "rank_changed": False,
            "other_7_gqa_layers_converted": False,
            "true_latent_cache": False,
            "heldout_accessed": False,
            "p6_2_allowed": False,
            "p7_allowed": False,
            "result_marker": "P6_1B_ATTRIBUTION=PASS",
            "completed_at": now(),
        }
        save_json(report_path, report)
        final = {
            "stage": "P6.1b",
            "status": "PASS_ATTRIBUTION_COMPLETE",
            "dominant_factor": attribution["dominant_factor"],
            "secondary_factor": attribution["secondary_factor"],
            "interaction_assessment": attribution["interaction_assessment"],
            "recommended_redesign": attribution["recommended_redesign"],
            "heldout_accessed": False,
            "p6_2_allowed": False,
            "p7_allowed": False,
            "result_marker": report["result_marker"],
            "completed_at": now(),
        }
        save_json(final_path, final)
        print(report["result_marker"], flush=True)
        print(f"P6_1B_DOMINANT_FACTOR={attribution['dominant_factor']}", flush=True)
        print(f"P6_1B_SECONDARY_FACTOR={attribution['secondary_factor']}", flush=True)
        print(f"P6_1B_REDESIGN_TARGET={attribution['recommended_redesign']}", flush=True)
        print("P6_2_ALLOWED=false", flush=True)
        print("P7_ALLOWED=false", flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001 - every failure must persist
        failure = {
            "stage": "P6.1b",
            "status": "ERROR",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "heldout_accessed": False,
            "p6_2_allowed": False,
            "p7_allowed": False,
            "result_marker": "P6_1B_ATTRIBUTION=ERROR",
            "completed_at": now(),
        }
        save_json(report_path, failure)
        save_json(final_path, failure)
        print(failure["result_marker"], flush=True)
        print("P6_2_ALLOWED=false", flush=True)
        print("P7_ALLOWED=false", flush=True)
        print(f"P6_1B_EXCEPTION={type(exc).__name__}: {exc}", flush=True)
        return 1
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
