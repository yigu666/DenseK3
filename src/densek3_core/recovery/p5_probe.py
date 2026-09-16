"""Training objectives, scopes, schedules, and Gates for P5.3-Probe."""

from __future__ import annotations

import fnmatch
import math
from collections import Counter
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from densek3_core.modeling import DenseK3ForCausalLM

P5_PROBE_HIDDEN_ANCHORS = (2, 6, 10, 14, 18, 22, 26, 30)


def probe_a_trainable_names(model: DenseK3ForCausalLM) -> set[str]:
    """Return exactly the four decay-branch tensors at each of 24 KDA layers."""
    suffixes = ("A_log", "dt_bias", "f_a_proj.weight", "f_b_proj.weight")
    return {
        f"model.layers.{layer_index}.self_attn.{suffix}"
        for layer_index in model.config.kda_indices
        for suffix in suffixes
    }


def configure_probe_a_trainable_scope(model: DenseK3ForCausalLM) -> dict[str, Any]:
    """Freeze the complete model, then enable only P5 Probe-A decay branches."""
    if len(model.config.kda_indices) != 24:
        raise ValueError("Probe A requires the frozen production topology with exactly 24 KDA layers")
    expected = probe_a_trainable_names(model)
    parameters = dict(model.named_parameters())
    missing = sorted(expected - parameters.keys())
    if missing:
        raise ValueError(f"Probe A expected parameters are missing: {missing}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name in expected:
        parameters[name].requires_grad_(True)
    observed = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if observed != expected:
        raise ValueError(f"Probe A trainable scope mismatch: {sorted(observed ^ expected)}")
    suffixes = ("A_log", "dt_bias", "f_a_proj.weight", "f_b_proj.weight")
    counts = Counter(next(suffix for suffix in suffixes if name.endswith(suffix)) for name in observed)
    return {
        "trainable_tensor_count": len(observed),
        "trainable_numel": sum(parameters[name].numel() for name in observed),
        "trainable_names": sorted(observed),
        "suffix_counts": dict(sorted(counts.items())),
        "frozen_numel": sum(parameter.numel() for parameter in model.parameters())
        - sum(parameters[name].numel() for name in observed),
        "scope_exact": True,
    }


def probe_b_trainable_names(model: DenseK3ForCausalLM) -> set[str]:
    """Return the approved Probe-B compensation tensors at all 24 KDA layers."""
    suffixes = (
        "A_log",
        "dt_bias",
        "f_a_proj.weight",
        "f_b_proj.weight",
        "g_proj.weight",
        "o_norm.weight",
        "o_proj.weight",
    )
    return {
        f"model.layers.{layer_index}.self_attn.{suffix}"
        for layer_index in model.config.kda_indices
        for suffix in suffixes
    }


def configure_probe_b_trainable_scope(model: DenseK3ForCausalLM) -> dict[str, Any]:
    """Freeze the model, then enable the approved decay/gate/output Probe-B scope."""
    if len(model.config.kda_indices) != 24:
        raise ValueError("Probe B requires the frozen production topology with exactly 24 KDA layers")
    expected = probe_b_trainable_names(model)
    parameters = dict(model.named_parameters())
    missing = sorted(expected - parameters.keys())
    if missing:
        raise ValueError(f"Probe B expected parameters are missing: {missing}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name in expected:
        parameters[name].requires_grad_(True)
    observed = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if observed != expected:
        raise ValueError(f"Probe B trainable scope mismatch: {sorted(observed ^ expected)}")
    suffixes = (
        "A_log",
        "dt_bias",
        "f_a_proj.weight",
        "f_b_proj.weight",
        "g_proj.weight",
        "o_norm.weight",
        "o_proj.weight",
    )
    counts = Counter(next(suffix for suffix in suffixes if name.endswith(suffix)) for name in observed)
    trainable_numel = sum(parameters[name].numel() for name in observed)
    return {
        "trainable_tensor_count": len(observed),
        "trainable_numel": trainable_numel,
        "trainable_names": sorted(observed),
        "suffix_counts": dict(sorted(counts.items())),
        "frozen_numel": sum(parameter.numel() for parameter in model.parameters()) - trainable_numel,
        "scope_exact": True,
    }


def apply_probe_homotopy(model: DenseK3ForCausalLM, alpha: float | None) -> dict[str, Any]:
    """Select K3 decay plus SiLU/sigmoid homotopy without changing parameter tensors."""
    from densek3_core.modeling.modeling_densek3 import DenseK3KDA

    if alpha is not None and not 0.0 <= alpha <= 1.0:
        raise ValueError("Probe output-gate homotopy alpha must be in [0, 1]")
    identities = {
        name: (id(parameter), parameter.data_ptr(), parameter._version)
        for name, parameter in model.named_parameters()
    }
    model.config.kda_gate_mode = "kimi_k3"
    model.config.kda_gate_lower_bound = -5.0
    model.config.kda_output_gate_activation = "silu"
    model.config.kda_output_gate_alpha = alpha
    selected = 0
    for layer in model.model.layers:
        if isinstance(layer.self_attn, DenseK3KDA):
            layer.self_attn.o_norm.activation = "silu"
            layer.self_attn.o_norm.alpha = alpha
            selected += 1
    after = {
        name: (id(parameter), parameter.data_ptr(), parameter._version)
        for name, parameter in model.named_parameters()
    }
    checks = {
        "parameter_identity_unchanged": identities == after,
        "all_kda_layers_selected": selected == len(model.config.kda_indices) == 24,
        "k3_decay": model.config.kda_gate_mode == "kimi_k3"
        and model.config.kda_gate_lower_bound == -5.0,
        "alpha": model.config.kda_output_gate_alpha == alpha,
        "module_alpha": all(
            layer.self_attn.o_norm.activation == "silu" and layer.self_attn.o_norm.alpha == alpha
            for layer in model.model.layers
            if isinstance(layer.self_attn, DenseK3KDA)
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"Probe homotopy application failed: {checks}")
    return {"alpha": alpha, "kda_layer_count": selected, "checks": checks}


def make_probe_a_optimizer(
    model: DenseK3ForCausalLM,
    config: dict[str, Any],
) -> torch.optim.Optimizer:
    """Create a fresh fused AdamW with no decay on A_log/dt_bias."""
    matrix = []
    gate_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith(("A_log", "dt_bias")):
            gate_parameters.append(parameter)
        else:
            matrix.append(parameter)
    if not matrix or not gate_parameters:
        raise ValueError("Probe A optimizer groups are incomplete")
    return torch.optim.AdamW(
        [
            {"params": matrix, "weight_decay": float(config["weight_decay"])},
            {"params": gate_parameters, "weight_decay": 0.0},
        ],
        lr=float(config["learning_rate"]),
        betas=tuple(float(value) for value in config["betas"]),
        eps=float(config["eps"]),
        fused=True,
    )


def make_probe_b_optimizer(
    model: DenseK3ForCausalLM,
    config: dict[str, Any],
) -> torch.optim.Optimizer:
    """Create a fresh Probe-B AdamW; no optimizer state is inherited from Probe A."""
    matrix = []
    vector = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith(("A_log", "dt_bias", "o_norm.weight")):
            vector.append(parameter)
        else:
            matrix.append(parameter)
    if not matrix or not vector:
        raise ValueError("Probe B optimizer groups are incomplete")
    optimizer = torch.optim.AdamW(
        [
            {"params": matrix, "weight_decay": float(config["weight_decay"])},
            {"params": vector, "weight_decay": 0.0},
        ],
        lr=float(config["learning_rate"]),
        betas=tuple(float(value) for value in config["betas"]),
        eps=float(config["eps"]),
        fused=True,
    )
    if optimizer.state:
        raise ValueError("A fresh Probe-B optimizer unexpectedly contains state")
    return optimizer


def learning_rate_at_step(
    step: int,
    total_steps: int,
    *,
    base_learning_rate: float,
    warmup_ratio: float,
    minimum_ratio: float,
) -> float:
    """Return a deterministic warmup plus cosine-decay learning rate."""
    if step < 0 or step >= total_steps or total_steps <= 0:
        raise ValueError("Learning-rate step must be inside a positive schedule")
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    if step < warmup_steps:
        return base_learning_rate * (step + 1) / warmup_steps
    denominator = max(total_steps - warmup_steps - 1, 1)
    progress = min(max((step - warmup_steps) / denominator, 0.0), 1.0)
    multiplier = minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_learning_rate * multiplier


def normalized_hidden_recovery_loss(
    student_hidden: tuple[torch.Tensor, ...],
    teacher_hidden: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Mean tokenwise cosine distance over all selected and final anchors."""
    if len(student_hidden) != len(teacher_hidden) or not student_hidden:
        raise ValueError("Teacher/student hidden anchor counts differ or are empty")
    losses = []
    for student, teacher in zip(student_hidden, teacher_hidden, strict=True):
        if student.shape != teacher.shape:
            raise ValueError("Teacher/student hidden anchor shapes differ")

        def anchor_loss(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return (1.0 - F.cosine_similarity(value.float(), target.float(), dim=-1)).mean()

        losses.append(checkpoint(anchor_loss, student, teacher, use_reentrant=False))
    return torch.stack(losses).mean()


def chunked_causal_lm_loss(
    hidden_states: torch.Tensor,
    labels: torch.LongTensor,
    lm_head_weight: torch.Tensor,
    *,
    chunk_size: int,
    checkpoint_chunks: bool,
) -> torch.Tensor:
    """Compute exact causal CE without materializing all 2048xV FP32 logits at once."""
    if hidden_states.shape[:2] != labels.shape or chunk_size <= 0:
        raise ValueError("Invalid hidden/label shape or LM-logit chunk size")
    shifted_hidden = hidden_states[:, :-1]
    shifted_labels = labels[:, 1:]
    total = shifted_labels.numel()
    if total == 0:
        raise ValueError("Causal LM loss requires at least two tokens")
    losses = []
    for start in range(0, shifted_hidden.shape[1], chunk_size):
        end = min(start + chunk_size, shifted_hidden.shape[1])
        hidden_chunk = shifted_hidden[:, start:end]
        label_chunk = shifted_labels[:, start:end]

        def loss_chunk(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            logits = F.linear(value, lm_head_weight).float()
            return F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target.reshape(-1),
                reduction="sum",
            )

        if checkpoint_chunks and torch.is_grad_enabled():
            losses.append(checkpoint(loss_chunk, hidden_chunk, label_chunk, use_reentrant=False))
        else:
            losses.append(loss_chunk(hidden_chunk, label_chunk))
    return torch.stack(losses).sum() / total


def gradient_report(parameters: list[torch.nn.Parameter], clip_norm: float) -> dict[str, Any]:
    """Check every trainable gradient before clipping and return norm/finiteness."""
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    all_present = len(gradients) == len(parameters)
    all_finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    norm = torch.nn.utils.clip_grad_norm_(parameters, clip_norm) if all_present and all_finite else torch.tensor(float("nan"))
    return {
        "gradient_tensor_count": len(gradients),
        "trainable_tensor_count": len(parameters),
        "all_gradients_present": all_present,
        "all_gradients_finite": all_finite,
        "pre_clip_global_norm": float(norm),
        "clip_norm": clip_norm,
        "passed": all_present and all_finite and math.isfinite(float(norm)),
    }


def evaluate_probe_a_gate(
    evaluations: list[dict[str, Any]],
    gate: dict[str, Any],
    *,
    completed_budget: bool,
) -> dict[str, Any]:
    """Select the best safe GO within budget; the final checkpoint is not privileged."""
    if len(evaluations) < 2:
        return {"decision": "CONTINUE", "reason": "insufficient_evaluations"}
    assessments = []
    for index, item in enumerate(evaluations):
        ce = float(item["canonical"]["cross_entropy"])
        finite = bool(item["finite"])
        generation_ok = bool(item["generation"]["not_collapsed"])
        cosine = float(item["canonical"]["logits_cosine"])
        improvement = None
        continuing = False
        if index:
            improvement = float(evaluations[index - 1]["canonical"]["cross_entropy"]) - ce
            continuing = improvement >= float(gate["continuing_ce_improvement_min"])
        common = finite and generation_ok and cosine >= float(gate["logits_cosine_min"])
        strong = index > 0 and common and ce <= float(gate["strong_canonical_ce_max"])
        go = index > 0 and common and ce <= float(gate["go_canonical_ce_max"]) and continuing
        assessments.append(
            {
                "evaluation_index": index,
                "step": item.get("step"),
                "effective_tokens": item.get("effective_tokens"),
                "canonical_ce": ce,
                "logits_cosine": cosine,
                "finite": finite,
                "generation_not_collapsed": generation_ok,
                "latest_interval_ce_improvement": improvement,
                "recovery_curve_continuing": continuing,
                "strong": strong,
                "go": go,
            }
        )

    strong_candidates = [item for item in assessments if item["strong"]]
    go_candidates = [item for item in assessments if item["go"]]
    qualified = strong_candidates or go_candidates
    best = min(qualified, key=lambda item: (item["canonical_ce"], item["effective_tokens"] or 0)) if qualified else None
    qualification = "STRONG_GO" if strong_candidates else "GO" if go_candidates else None
    if strong_candidates:
        best = min(strong_candidates, key=lambda item: (item["canonical_ce"], item["effective_tokens"] or 0))

    latest = assessments[-1]
    latest_improvement = latest["latest_interval_ce_improvement"]
    plateau = latest_improvement is not None and latest_improvement < float(
        gate["plateau_ce_improvement_max_exclusive"]
    )
    explicit_no_go = (
        completed_budget
        and latest["canonical_ce"] > float(gate["no_go_canonical_ce_min_exclusive"])
        and plateau
    )
    qualified_index = best["evaluation_index"] if best is not None else None
    later_safety_failures = [
        item
        for item in assessments
        if qualified_index is not None
        and item["evaluation_index"] > qualified_index
        and (not item["finite"] or not item["generation_not_collapsed"])
    ]
    if later_safety_failures:
        decision = "NO_GO"
        pass_mode = "SAFETY_FAILURE_AFTER_QUALIFIED_CHECKPOINT"
    elif qualification is not None:
        decision = qualification
        pass_mode = "BEST_CHECKPOINT_WITHIN_BUDGET"
    elif completed_budget and explicit_no_go:
        decision = "NO_GO"
        pass_mode = "EXPLICIT_NO_GO"
    elif completed_budget:
        decision = "REVIEW_REQUIRED"
        pass_mode = "NO_APPROVED_AUTOMATIC_DECISION"
    else:
        decision = "CONTINUE"
        pass_mode = "AWAIT_MORE_BUDGET"
    return {
        "decision": decision,
        "pass_mode": pass_mode,
        "canonical_ce": latest["canonical_ce"],
        "latest_interval_ce_improvement": latest_improvement,
        "recovery_curve_continuing": latest["recovery_curve_continuing"],
        "plateau": plateau,
        "explicit_no_go_condition": explicit_no_go,
        "logits_cosine": latest["logits_cosine"],
        "generation_not_collapsed": latest["generation_not_collapsed"],
        "finite": latest["finite"],
        "completed_budget": completed_budget,
        "best_qualified_evaluation": best,
        "checkpoint_assessments": assessments,
        "later_safety_failures": later_safety_failures,
        "checks": {
            "strong": bool(strong_candidates),
            "go": bool(go_candidates),
            "explicit_no_go": explicit_no_go,
            "safety_failure_after_qualified": bool(later_safety_failures),
        },
    }


def evaluate_probe_b_gate(
    evaluations: list[dict[str, Any]],
    *,
    p4_canonical_ce: float,
    post_switch_canonical_ce: float,
    gate: dict[str, Any],
    completed_budget: bool,
) -> dict[str, Any]:
    """Evaluate Probe B using switch-shock recovery while preserving safety checks."""
    shock = float(post_switch_canonical_ce) - float(p4_canonical_ce)
    if shock <= 0.0 or not math.isfinite(shock):
        raise ValueError("Probe B requires a positive finite architecture switch shock")
    if not evaluations:
        return {"decision": "CONTINUE", "reason": "insufficient_evaluations"}
    assessments = []
    for index, item in enumerate(evaluations):
        ce = float(item["canonical"]["cross_entropy"])
        recovery = (float(post_switch_canonical_ce) - ce) / shock
        improvement = None if index == 0 else recovery - assessments[-1]["recovery_fraction"]
        safe = bool(item["finite"]) and bool(item["generation"]["not_collapsed"])
        assessments.append(
            {
                "evaluation_index": index,
                "step": item.get("step"),
                "effective_tokens": item.get("effective_tokens"),
                "canonical_ce": ce,
                "recovery_fraction": recovery,
                "latest_interval_recovery_improvement": improvement,
                "safe": safe,
                "go": index > 0 and safe and recovery >= float(gate["go_recovery_fraction_min"]),
            }
        )
    candidates = [item for item in assessments if item["go"]]
    best = (
        max(
            candidates,
            key=lambda item: (item["recovery_fraction"], -(item["effective_tokens"] or 0)),
        )
        if candidates
        else None
    )
    safety_failures = [item for item in assessments if not item["safe"]]
    latest = assessments[-1]
    plateau = (
        latest["latest_interval_recovery_improvement"] is not None
        and latest["latest_interval_recovery_improvement"]
        < float(gate["plateau_recovery_improvement_max_exclusive"])
    )
    explicit_no_go = (
        completed_budget
        and latest["safe"]
        and latest["recovery_fraction"] < float(gate["no_go_recovery_fraction_max_exclusive"])
        and plateau
    )
    if safety_failures:
        decision, mode = "NO_GO", "SAFETY_FAILURE"
    elif best is not None:
        decision, mode = "GO", "BEST_CHECKPOINT_WITHIN_BUDGET"
    elif explicit_no_go:
        decision, mode = "NO_GO", "EXPLICIT_LOW_RECOVERY_PLATEAU"
    elif completed_budget:
        decision, mode = "REVIEW_REQUIRED", "NO_APPROVED_AUTOMATIC_DECISION"
    else:
        decision, mode = "CONTINUE", "AWAIT_MORE_BUDGET"
    return {
        "decision": decision,
        "pass_mode": mode,
        "p4_canonical_ce": float(p4_canonical_ce),
        "post_switch_canonical_ce": float(post_switch_canonical_ce),
        "switch_shock_ce": shock,
        "best_qualified_evaluation": best,
        "latest_evaluation": latest,
        "checkpoint_assessments": assessments,
        "explicit_no_go_condition": explicit_no_go,
        "safety_failures": safety_failures,
        "completed_budget": completed_budget,
    }


def matches_trainable_pattern(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


__all__ = [
    "P5_PROBE_HIDDEN_ANCHORS",
    "chunked_causal_lm_loss",
    "configure_probe_a_trainable_scope",
    "configure_probe_b_trainable_scope",
    "apply_probe_homotopy",
    "evaluate_probe_a_gate",
    "evaluate_probe_b_gate",
    "gradient_report",
    "learning_rate_at_step",
    "make_probe_a_optimizer",
    "make_probe_b_optimizer",
    "normalized_hidden_recovery_loss",
    "probe_a_trainable_names",
    "probe_b_trainable_names",
]
