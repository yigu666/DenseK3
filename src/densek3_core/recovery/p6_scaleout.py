"""Frozen-donor calibration, drift, and candidate helpers for P6.2 scale-out."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from densek3_core.modeling.modeling_densek3 import DenseK3GQA
from densek3_core.recovery.p5_corpus import PackedTokenDataset
from densek3_core.recovery.p6_attribution import TRACE_FIELDS, trace_case, trace_hashes
from densek3_core.recovery.p6_initialization import DriftAccumulator
from densek3_core.recovery.p6_mla_probe import P6NoPEFullAttention, P6NoPEMLA, tensor_sha256
from densek3_core.transplant.full_model import DeterministicShardWriter


@torch.no_grad()
def collect_frozen_donor_calibration(
    model: torch.nn.Module,
    train: PackedTokenDataset,
    layer_indices: list[int],
    sequence_indices: list[int],
    *,
    selection_prefix: int,
) -> dict[str, Any]:
    """Capture every target layer from the untouched P5 donor before conversion."""
    if train.split != "train":
        raise ValueError("P6.2 calibration must use the frozen train split")
    if layer_indices != [3, 7, 11, 15, 19, 23, 27, 31]:
        raise ValueError("P6.2 target layers differ from the locked eight GQA positions")
    device = next(model.parameters()).device
    normalized = {index: [] for index in layer_indices}
    residual_prefix = {index: [] for index in layer_indices}
    baseline_curves: list[list[torch.Tensor]] = []
    baseline_final: list[torch.Tensor] = []
    token_hashes = []
    for ordinal, sequence_index in enumerate(sequence_indices, start=1):
        tokens = torch.from_numpy(train[sequence_index].copy()).unsqueeze(0).to(device)
        target_inputs: dict[int, torch.Tensor] = {}
        layer_outputs: dict[int, torch.Tensor] = {}
        handles = []
        for layer_index, layer in enumerate(model.model.layers):
            handles.append(
                layer.register_forward_hook(
                    lambda _module, _inputs, output, index=layer_index, values=layer_outputs: values.__setitem__(
                        index,
                        output.detach(),
                    )
                )
            )
            if layer_index in normalized:
                handles.append(
                    layer.input_layernorm.register_forward_pre_hook(
                        lambda _module, inputs, index=layer_index, values=target_inputs: values.__setitem__(
                            index,
                            inputs[0].detach(),
                        )
                    )
                )
        try:
            output = model.model(input_ids=tokens, use_cache=False, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        if set(target_inputs) != set(layer_indices) or set(layer_outputs) != set(range(32)):
            raise RuntimeError("P6.2 did not capture every frozen donor layer exactly once")
        for layer_index in layer_indices:
            residual = target_inputs[layer_index]
            residual_prefix[layer_index].append(residual[:, :selection_prefix].cpu())
            normalized[layer_index].append(
                model.model.layers[layer_index].input_layernorm(residual).cpu()
            )
        baseline_curves.append(
            [layer_outputs[index][:, :selection_prefix].cpu() for index in range(32)]
        )
        baseline_final.append(output.last_hidden_state[:, :selection_prefix].cpu())
        token_hashes.append(tensor_sha256(tokens))
        print(
            f"P6_2_FROZEN_DONOR_CAPTURE={ordinal}/{len(sequence_indices)}",
            flush=True,
        )
    return {
        "normalized": normalized,
        "residual_prefix": residual_prefix,
        "baseline_curves": baseline_curves,
        "baseline_final": baseline_final,
        "manifest": {
            "split": "train",
            "sequence_indices": sequence_indices,
            "sequence_length": train.sequence_length,
            "selection_prefix": selection_prefix,
            "token_sha256": token_hashes,
            "per_layer_normalized_input_sha256": {
                str(index): [tensor_sha256(value) for value in normalized[index]]
                for index in layer_indices
            },
            "all_statistics_from_untouched_p5_donor": True,
            "sequential_self_conditioning": False,
        },
    }


@torch.no_grad()
def local_layer_metrics(
    donor: DenseK3GQA,
    target: P6NoPEMLA,
    residual_batches: list[torch.Tensor],
    normalized_batches: list[torch.Tensor],
    *,
    prefix: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reference = P6NoPEFullAttention(donor, split_q_gate=False, apply_qk_norm=True).eval()
    device = next(target.parameters()).device
    fields = (*TRACE_FIELDS, "post_attention_residual_hidden")
    accumulators = {name: DriftAccumulator() for name in fields}
    hashes = {"reference": [], "target": []}
    for residual_cpu, normalized_cpu in zip(residual_batches, normalized_batches, strict=True):
        residual = residual_cpu[:, :prefix].to(device)
        normalized_hidden = normalized_cpu[:, :prefix].to(device)
        mask = torch.ones(normalized_hidden.shape[:2], device=device)
        reference_trace = trace_case(reference, residual, normalized_hidden, mask)
        target_trace = trace_case(target, residual, normalized_hidden, mask)
        for name in fields:
            accumulators[name].update(reference_trace[name], target_trace[name])
        hashes["reference"].append(trace_hashes(reference_trace))
        hashes["target"].append(trace_hashes(target_trace))
    return {name: value.finalize() for name, value in accumulators.items()}, hashes


class DecisionAccumulator:
    def __init__(self) -> None:
        self.positions = 0
        self.argmax_matches = 0
        self.top10_overlap_sum = 0
        self.top10_min_overlap = 10
        self.top10_histogram: Counter[int] = Counter()

    @torch.no_grad()
    def update(self, reference_logits: torch.Tensor, actual_logits: torch.Tensor) -> None:
        reference_argmax = reference_logits.argmax(dim=-1)
        actual_argmax = actual_logits.argmax(dim=-1)
        matches = reference_argmax == actual_argmax
        reference_top10 = reference_logits.topk(10, dim=-1).indices
        actual_top10 = actual_logits.topk(10, dim=-1).indices
        overlaps = (
            reference_top10.unsqueeze(-1)
            .eq(actual_top10.unsqueeze(-2))
            .any(dim=-1)
            .sum(dim=-1)
            .cpu()
            .reshape(-1)
            .tolist()
        )
        self.positions += matches.numel()
        self.argmax_matches += int(matches.sum())
        self.top10_overlap_sum += sum(overlaps)
        self.top10_min_overlap = min(self.top10_min_overlap, min(overlaps, default=10))
        self.top10_histogram.update(overlaps)

    def finalize(self) -> dict[str, Any]:
        return {
            "positions": self.positions,
            "argmax_agreement_fraction": self.argmax_matches / self.positions,
            "argmax_disagreements": self.positions - self.argmax_matches,
            "top10_min_overlap": self.top10_min_overlap,
            "top10_mean_overlap": self.top10_overlap_sum / self.positions,
            "top10_histogram": {str(key): value for key, value in sorted(self.top10_histogram.items())},
        }


def local_scaleout_safety(
    per_layer: dict[int, dict[str, Any]],
    *,
    mixer_relative_l2_max: float,
) -> dict[str, Any]:
    """Apply P6.2d per-layer safety without reusing the L23 pilot Gate."""
    expected = {3, 7, 11, 15, 19, 23, 27, 31}
    if set(per_layer) != expected:
        raise ValueError("P6.2d local safety requires exactly the eight MLA layers")
    results = {}
    for layer_index in sorted(expected):
        item = per_layer[layer_index]
        n1_mixer = float(item["n1"]["mixer_output"]["relative_l2_error"])
        best_mixer = float(item["best"]["mixer_output"]["relative_l2_error"])
        checks = {
            "finite": bool(item["finite"])
            and math.isfinite(n1_mixer)
            and math.isfinite(best_mixer),
            "fresh_n1_not_extreme": n1_mixer <= mixer_relative_l2_max,
            "calibration_non_destructive": best_mixer <= n1_mixer,
            "calibrated_not_extreme": best_mixer <= mixer_relative_l2_max,
        }
        results[str(layer_index)] = {
            "n1_mixer_relative_l2": n1_mixer,
            "best_mixer_relative_l2": best_mixer,
            "n1_to_best_improvement_fraction": (n1_mixer - best_mixer)
            / max(n1_mixer, torch.finfo(torch.float64).eps),
            "checks": checks,
            "passed": all(checks.values()),
        }
    passed = all(item["passed"] for item in results.values())
    return {
        "status": "PASS" if passed else "STOP_LOCAL_SAFETY",
        "passed": passed,
        "per_layer": results,
        "mixer_relative_l2_max": mixer_relative_l2_max,
        "shared_layer23_absolute_gate_applied": False,
    }


def architecture_shock_recovery(
    *,
    p5_ce: float,
    old_p6_ce: float,
    new_p6_ce: float,
) -> dict[str, float]:
    """Quantify how much of the old eight-layer architecture shock was removed."""
    old_shock = old_p6_ce - p5_ce
    if old_shock <= 0:
        raise ValueError("P6.2d requires a positive frozen old-P6 architecture shock")
    absolute_recovered = old_p6_ce - new_p6_ce
    return {
        "p5_ce": p5_ce,
        "old_p6_ce": old_p6_ce,
        "new_p6_ce": new_p6_ce,
        "old_architecture_shock": old_shock,
        "absolute_ce_recovered": absolute_recovered,
        "old_architecture_shock_recovery_fraction": absolute_recovered / old_shock,
    }


@torch.no_grad()
def full_model_shock_metrics(
    model: torch.nn.Module,
    train: PackedTokenDataset,
    sequence_indices: list[int],
    baseline_curves: list[list[torch.Tensor]],
    baseline_final: list[torch.Tensor],
    *,
    prefix: int,
    logits_chunk_size: int = 32,
) -> dict[str, Any]:
    """Measure the 32-layer error trajectory on calibration, never canonical."""
    device = next(model.parameters()).device
    layer_accumulators = [DriftAccumulator() for _ in range(32)]
    final_accumulator = DriftAccumulator()
    logits_accumulator = DriftAccumulator()
    decisions = DecisionAccumulator()
    total_loss = 0.0
    total_tokens = 0
    for sequence_index, reference_curve, reference_final_cpu in zip(
        sequence_indices,
        baseline_curves,
        baseline_final,
        strict=True,
    ):
        tokens = torch.from_numpy(train[sequence_index][:prefix].copy()).unsqueeze(0).to(device)
        layer_outputs: dict[int, torch.Tensor] = {}
        handles = [
            layer.register_forward_hook(
                lambda _module, _inputs, output, index=index, values=layer_outputs: values.__setitem__(
                    index,
                    output.detach(),
                )
            )
            for index, layer in enumerate(model.model.layers)
        ]
        try:
            actual_final = model.model(input_ids=tokens, use_cache=False, return_dict=True).last_hidden_state
        finally:
            for handle in handles:
                handle.remove()
        if set(layer_outputs) != set(range(32)):
            raise RuntimeError("P6.2 candidate drift capture missed a decoder layer")
        for index in range(32):
            layer_accumulators[index].update(reference_curve[index].to(device), layer_outputs[index])
        reference_final = reference_final_cpu.to(device)
        final_accumulator.update(reference_final, actual_final)
        for start in range(0, prefix, logits_chunk_size):
            end = min(start + logits_chunk_size, prefix)
            reference_logits = F.linear(reference_final[:, start:end], model.lm_head.weight).float()
            actual_logits = F.linear(actual_final[:, start:end], model.lm_head.weight).float()
            logits_accumulator.update(reference_logits, actual_logits)
            decisions.update(reference_logits, actual_logits)
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
    layer_curve = []
    mla_indices = set(model.config.mla_indices)
    for index, accumulator in enumerate(layer_accumulators):
        layer_curve.append(
            {
                **accumulator.finalize(),
                "layer_index": index,
                "layer_type": "mla" if index in mla_indices else "kda",
                "is_mla_boundary": index in mla_indices,
            }
        )
    return {
        "per_layer_hidden_drift": layer_curve,
        "mla_boundary_drift": [item for item in layer_curve if item["is_mla_boundary"]],
        "final_hidden": final_accumulator.finalize(),
        "logits": logits_accumulator.finalize(),
        "token_decisions": decisions.finalize(),
        "calibration_cross_entropy": total_loss / total_tokens,
        "calibration_tokens_scored": total_tokens,
    }


def write_candidate_checkpoint(
    model: torch.nn.Module,
    output_dir: Path,
    tokenizer_source: Path,
    provenance: dict[str, Any],
    *,
    max_shard_size: int = 4_000_000_000,
    stage: str = "P6.2-CANDIDATE",
) -> dict[str, Any]:
    """Write a native reloadable P6 candidate; formal freeze remains P6.4."""
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"P6.2 candidate directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = DeterministicShardWriter(output_dir, max_shard_size=max_shard_size, stage=stage)
    tensor_count = 0
    parameter_count = 0
    mla_hashes: dict[str, str] = {}
    for name, value in sorted(model.named_parameters()):
        if name == "lm_head.weight":
            continue
        writer.add(name, value)
        tensor_count += 1
        parameter_count += value.numel()
        if ".self_attn." in name and any(
            name.startswith(f"model.layers.{index}.") for index in model.config.mla_indices
        ):
            mla_hashes[name] = tensor_sha256(value)
    index = writer.finish()
    model.config.save_pretrained(output_dir)
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "merges.txt",
        "vocab.json",
        "chat_template.json",
        "LICENSE",
        "README.md",
    ):
        source = tokenizer_source / name
        if source.is_file():
            shutil.copy2(source, output_dir / name)
    core = {
        "schema_version": 1,
        "stage": stage,
        "status": "CANDIDATE_NOT_FROZEN",
        "architecture": "24_KDA_PLUS_8_STRICT_NOPE_MLA_QK_NORM_PRESERVED"
        if model.config.mla_qk_norm
        else "24_KDA_PLUS_8_STRICT_NOPE_MLA",
        "initialization": "ACTIVATION_WEIGHTED_RANK512_PLUS_SCALE_MATCHED_LATENT_NORM",
        "qk_norm_semantics": "PRESERVE_DONOR_QK_RMSNORM"
        if model.config.mla_qk_norm
        else "REMOVED",
        "cache_mode": "expanded_reference",
        "true_latent_cache": False,
        "tensor_count": tensor_count,
        "unique_parameter_count": parameter_count,
        "weight_index": index,
        "mla_parameter_hashes": mla_hashes,
        "provenance": provenance,
        "formal_save_reload_pending_p6_4": True,
        "p6_frozen": False,
    }
    serialized = json.dumps(core, indent=2, sort_keys=True, allow_nan=False) + "\n"
    (output_dir / "p6-candidate-manifest.json").write_text(serialized, encoding="utf-8")
    return {
        **core,
        "manifest_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "directory": str(output_dir),
    }


__all__ = [
    "DecisionAccumulator",
    "architecture_shock_recovery",
    "collect_frozen_donor_calibration",
    "full_model_shock_metrics",
    "local_layer_metrics",
    "local_scaleout_safety",
    "write_candidate_checkpoint",
]
