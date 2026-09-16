"""Frozen contracts for the no-training P5 I1 analytic closeout."""

from __future__ import annotations

import math
from typing import Any

import torch

from densek3_core.recovery.p5_closeout import tensor_sha256
from densek3_core.recovery.p5_rethink_v3 import P5_RETHINK_V3_FORMULA_HASH

P5_I1_P4_HELDOUT_CE = 2.256015
P5_I1_HELDOUT_MAX_DELTA = 0.15
P5_I1_HELDOUT_MAX_CE = P5_I1_P4_HELDOUT_CE + P5_I1_HELDOUT_MAX_DELTA


def validate_i1_closeout_config(config: dict[str, Any]) -> dict[str, bool]:
    candidate = config["candidate"]
    architecture = config["architecture"]
    heldout = config["heldout"]
    artifact = config["artifact"]
    checks = {
        "stage": config["stage"] == "P5-RETHINK-V3-CLOSEOUT",
        "frozen": config["status"] == "FROZEN",
        "unique_i1": candidate["initialization"]
        == "I1_RETENTION_MATCHED_ANALYTIC_LOCAL_VALUE_AND_SLOPE"
        and candidate["unique"] is True,
        "no_training_or_i2": candidate["training_performed"] is False
        and candidate["calibration_used"] is False
        and candidate["i2_used"] is False,
        "architecture": architecture
        == {
            "topology": "24_kda_8_gqa",
            "decay_formula": "kimi_k3_lower_bounded_sigmoid",
            "lower_bound": -5.0,
            "output_gate": "silu",
        },
        "heldout_gate": heldout["split"] == "heldout"
        and heldout["sequences"] == 97
        and math.isclose(heldout["p4_ce"], P5_I1_P4_HELDOUT_CE, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(heldout["maximum_delta"], P5_I1_HELDOUT_MAX_DELTA, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(heldout["maximum_p5_ce"], P5_I1_HELDOUT_MAX_CE, rel_tol=0.0, abs_tol=1e-12)
        and heldout["candidate_count"] == 1,
        "artifact_accounting": artifact["tensor_count"] == 546
        and artifact["unique_parameters"] == 4_224_329_984
        and artifact["mapping_tensor_count"] == 48
        and artifact["native_checkpoint_required"] is True,
        "all_forbidden": all(config["forbidden"].values()),
    }
    if not all(checks.values()):
        raise ValueError(f"P5 I1 closeout config failed: {checks}")
    return checks


def validate_i1_probe(report: dict[str, Any]) -> dict[str, bool]:
    i1 = report["initializations"]["I1"]
    initialization = i1["initialization_manifest"]
    aggregate = initialization["aggregate"]
    reference_ce = float(report["reference_p4"]["full_dev_ce"])
    i1_ce = float(i1["full_dev_ce"])
    checks = {
        "probe_pass": report["stage"] == "P5-RETHINK-V3-INITIALIZATION-PROBE"
        and report["status"] == "PASS",
        "selected_i1": report["comparison"]["selected_zero_shot_initialization"] == "I1",
        "zero_shot_dev_gate": math.isfinite(reference_ce)
        and math.isfinite(i1_ce)
        and i1_ce <= reference_ce + 0.15,
        "i1_safe": i1["finite"] is True and i1["generation_not_collapsed"] is True,
        "no_training": report["training_performed"] is False
        and report["optimizer_created"] is False,
        "heldout_unopened": report["heldout_accessed"] is False
        and report["payload_splits_opened"] == ["dev"],
        "i2_not_run": report["i2_allowed"] is False
        and report["initializations"]["I2"]["status"] == "BLOCKED_NOT_RUN",
        "formula": report["formula_hash"]
        == initialization["formula_hash"]
        == P5_RETHINK_V3_FORMULA_HASH,
        "mapping_48": len(initialization["changed_tensor_names"]) == 48
        and initialization["checks"]["changed_tensor_count"] is True,
        "head_accounting": aggregate["total_heads"] == 768
        and aggregate["representable_heads"] == 765
        and aggregate["lower_clipped_heads"] == 0
        and aggregate["upper_clipped_heads"] == 3,
    }
    if not all(checks.values()):
        raise ValueError(f"P5 I1 probe prerequisite failed: {checks}")
    return checks


def i1_heldout_gate(p5_ce: float) -> dict[str, Any]:
    if not math.isfinite(float(p5_ce)):
        raise ValueError("P5 I1 heldout CE must be finite")
    delta = float(p5_ce) - P5_I1_P4_HELDOUT_CE
    return {
        "p4_ce": P5_I1_P4_HELDOUT_CE,
        "p5_ce": float(p5_ce),
        "signed_delta": delta,
        "maximum_delta": P5_I1_HELDOUT_MAX_DELTA,
        "maximum_p5_ce": P5_I1_HELDOUT_MAX_CE,
        "passed": float(p5_ce) <= P5_I1_HELDOUT_MAX_CE,
    }


def i1_checkpoint_provenance(
    initialization: dict[str, Any],
    *,
    initialization_evidence_sha256: str,
) -> dict[str, Any]:
    aggregate = initialization["aggregate"]
    return {
        "type": "RETENTION_MATCHED_ANALYTIC",
        "name": initialization["initialization"],
        "formula_hash": initialization["formula_hash"],
        "source": "FROZEN_P4_FULL_CHECKPOINT",
        "changed_tensor_count": len(initialization["changed_tensor_names"]),
        "changed_tensor_names": initialization["changed_tensor_names"],
        "changed_tensor_hash": initialization["changed_tensor_hash"],
        "decay_heads_total": aggregate["total_heads"],
        "analytic_representable": aggregate["representable_heads"],
        "lower_clipped": aggregate["lower_clipped_heads"],
        "upper_clipped": aggregate["upper_clipped_heads"],
        "clipped_heads": aggregate["clipped_heads"],
        "calibration_used": False,
        "training_used": False,
        "initialization_evidence_sha256": initialization_evidence_sha256,
    }


def verify_i1_replay_binding(
    checkpoint_manifest: dict[str, Any],
    replay_tensors: dict[str, torch.Tensor],
    replay_manifest: dict[str, Any],
    *,
    expected_p4_parameter_hash: str,
) -> dict[str, Any]:
    expected = {item["name"]: item["sha256"] for item in checkpoint_manifest["parameters"]}
    provenance = checkpoint_manifest["initialization_provenance"]
    tensor_checks = {
        name: expected.get(name) == tensor_sha256(value)
        for name, value in sorted(replay_tensors.items())
    }
    checks = {
        "p4_source_hash": checkpoint_manifest["source_p4_parameter_hash"]
        == expected_p4_parameter_hash,
        "analytic_type": provenance["type"] == "RETENTION_MATCHED_ANALYTIC",
        "formula_hash": provenance["formula_hash"]
        == replay_manifest["formula_hash"]
        == P5_RETHINK_V3_FORMULA_HASH,
        "changed_tensor_count": provenance["changed_tensor_count"]
        == len(replay_tensors)
        == 48,
        "changed_tensor_names": provenance["changed_tensor_names"]
        == replay_manifest["changed_tensor_names"]
        == sorted(replay_tensors),
        "changed_tensor_hash": provenance["changed_tensor_hash"]
        == replay_manifest["changed_tensor_hash"],
        "formal_runtime_dtype": replay_manifest.get("replay_source", {}).get(
            "formal_runtime_dtype"
        )
        == "float32",
        "head_accounting": provenance["decay_heads_total"] == 768
        and provenance["analytic_representable"] == 765
        and provenance["lower_clipped"] == 0
        and provenance["upper_clipped"] == 3,
        "all_48_tensor_hashes_exact": len(tensor_checks) == 48 and all(tensor_checks.values()),
        "no_training_or_calibration": provenance["training_used"] is False
        and provenance["calibration_used"] is False,
    }
    return {"passed": all(checks.values()), "checks": checks, "tensor_checks": tensor_checks}


def evaluate_i1_closeout_gate(
    heldout: dict[str, Any],
    artifact: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "unique_i1_heldout": heldout["candidate"] == "I1" and heldout["candidate_count"] == 1,
        "heldout": heldout["gate"]["passed"] is True,
        "heldout_finite": heldout["finite"] is True,
        "checkpoint_verification": artifact["verification"]["status"] == "PASS",
        "source_p4_tensor_hashes": artifact["source_p4_verification"]["status"] == "PASS"
        and artifact["source_p4_verification"]["tensor_hashes_verified"] == 546,
        "mapping_replay": artifact["mapping_replay"]["passed"] is True,
        "native_architecture": artifact["architecture"]["passed"] is True,
        "native_semantics": artifact["native_semantics"]["passed"] is True,
        "save_reload": artifact["save_reload"]["passed"] is True,
        "bf16_forward": artifact["bf16_forward"]["passed"] is True,
        "cache": artifact["cache"]["passed"] is True,
        "generation": artifact["generation"]["not_collapsed"] is True,
        "no_training": artifact["training_performed"] is False,
    }
    return {"passed": all(checks.values()), "checks": checks}


__all__ = [
    "P5_I1_HELDOUT_MAX_CE",
    "P5_I1_HELDOUT_MAX_DELTA",
    "P5_I1_P4_HELDOUT_CE",
    "evaluate_i1_closeout_gate",
    "i1_checkpoint_provenance",
    "i1_heldout_gate",
    "validate_i1_closeout_config",
    "validate_i1_probe",
    "verify_i1_replay_binding",
]
