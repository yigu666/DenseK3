"""Validation contract for the recovery plan that must be frozen after P5.2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

P5_RECOVERY_PLAN_REQUIRED_DECISIONS = (
    "training_data_manifest",
    "validation_data_manifest",
    "contamination_policy",
    "teacher_objective_weights",
    "causal_lm_objective_weight",
    "hidden_distillation_layers",
    "trainable_parameter_scope",
    "sequence_length_schedule",
    "training_token_budget",
    "gpu_hour_budget",
    "optimizer",
    "learning_rate_schedule",
    "checkpoint_schedule",
    "validation_schedule",
    "recovery_success_thresholds",
)


def recovery_plan_readiness(plan: dict[str, Any]) -> dict[str, Any]:
    """Report whether all post-P5.2 choices have been explicitly frozen."""
    decisions = plan.get("decisions")
    if not isinstance(decisions, dict):
        decisions = {}
    missing = [name for name in P5_RECOVERY_PLAN_REQUIRED_DECISIONS if decisions.get(name) is None]
    checks = {
        "schema_version": plan.get("schema_version") == 1,
        "stage": plan.get("stage") == "P5.3-P5.5",
        "status_frozen": plan.get("status") == "FROZEN",
        "p5_2_review_recorded": isinstance(plan.get("p5_2_review"), dict)
        and plan["p5_2_review"].get("decision") in {"PROCEED", "PROCEED_WITH_REVISED_PLAN"},
        "all_decisions_present": not missing,
        "forbidden_architectures_absent": plan.get("forbidden_architecture_changes") == ["MLA", "AttnRes", "SiTU-GLU"],
    }
    return {
        "ready": all(checks.values()),
        "checks": checks,
        "missing_decisions": missing,
    }


def load_recovery_plan(path: str | Path) -> dict[str, Any]:
    """Load a YAML plan without treating an unresolved template as executable."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("P5 recovery-plan validation requires PyYAML") from exc
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("P5 recovery plan must be a YAML mapping")
    return value


def require_frozen_recovery_plan(path: str | Path) -> dict[str, Any]:
    """Return a frozen plan or fail before any optimizer/training state is created."""
    plan = load_recovery_plan(path)
    readiness = recovery_plan_readiness(plan)
    if not readiness["ready"]:
        raise ValueError(f"P5 recovery plan is not frozen: {readiness}")
    return plan


__all__ = [
    "P5_RECOVERY_PLAN_REQUIRED_DECISIONS",
    "load_recovery_plan",
    "recovery_plan_readiness",
    "require_frozen_recovery_plan",
]
