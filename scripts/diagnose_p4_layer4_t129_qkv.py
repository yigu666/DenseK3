"""Measure the final P4.4 layer-4 T=129 BF16 QKV/output distributions."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from densek3_core.evaluation.p4_decay_diagnostics import compare, error_distribution, parameter_embedding_audit
from densek3_core.kda.metrics import OPTIMIZED_BF16_ATOL, OPTIMIZED_BF16_RTOL
from densek3_core.modeling import DenseK3Config
from densek3_core.transplant.full_model import P4_MANIFEST_NAME, verify_p4_artifact
from densek3_core.transplant.gdn_to_kda import convert_gdn_to_kda, load_qwen_gdn_layer
from densek3_core.transplant.single_layer import DenseK3QwenCompatMixer, QwenGDNReferenceMixer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models/Qwen3.5-4B-Base"
DEFAULT_ARTIFACT_DIR = ROOT / "outputs/densek3-core/checkpoints/p4/densek3-4b-qwen-compat"
DEFAULT_PRIOR_DIAGNOSTIC = ROOT / "outputs/densek3-core/manifests/p4/p4-layer-4-decay-diagnostics.json"
DEFAULT_OUTPUT = ROOT / "outputs/densek3-core/manifests/p4/p4-layer-4-t129-qkv-distribution.json"

LAYER_INDEX = 4
BATCH_SIZE = 1
SEQUENCE_LENGTH = 129
HIDDEN_SEED = 404129
INITIAL_STATE_SEED = 404130
INITIAL_STATE_SCALE = 0.02
P3_NUMERICS_AMENDMENT = "P3-NUM-2026-08-10-01"
P4_CANDIDATE_AMENDMENT = "P4-NUM-2026-08-11-01"
P3_OUTPUT_DISTRIBUTION_LIMITS = {
    "relative_l2_error_max": 7.0e-3,
    "cosine_similarity_min": 0.99997,
    "mean_abs_error_max": 1.0e-3,
    "p99_9_abs_error_max": 4.0e-3,
    "failed_allclose_fraction_max": 1.0e-4,
    "max_abs_error_max": 1.25e-1,
    "nan_count_max": 0,
    "inf_count_max": 0,
}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sanitize_json(item: Any) -> Any:
    if isinstance(item, float) and not math.isfinite(item):
        return None
    if isinstance(item, dict):
        return {key: _sanitize_json(child) for key, child in item.items()}
    if isinstance(item, (list, tuple)):
        return [_sanitize_json(child) for child in item]
    return item


def save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_sanitize_json(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def seeded_hidden(hidden_size: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(HIDDEN_SEED)
    value = torch.randn((BATCH_SIZE, SEQUENCE_LENGTH, hidden_size), generator=generator)
    return value.to(device=device, dtype=torch.bfloat16)


def seeded_initial_state(config: Any, device: torch.device) -> torch.Tensor:
    shape = (
        BATCH_SIZE,
        config.value_num_heads,
        config.value_head_dim,
        config.qk_head_dim,
    )
    generator = torch.Generator(device="cpu").manual_seed(INITIAL_STATE_SEED)
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(device) * INITIAL_STATE_SCALE


def evaluate_p3_output_distribution(distribution: dict[str, Any]) -> dict[str, Any]:
    limits = P3_OUTPUT_DISTRIBUTION_LIMITS
    checks = {
        "relative_l2_error": distribution["relative_l2_error"] <= limits["relative_l2_error_max"],
        "cosine_similarity": distribution["cosine_similarity"] >= limits["cosine_similarity_min"],
        "mean_abs_error": distribution["mean_abs_error"] <= limits["mean_abs_error_max"],
        "p99_9_abs_error": distribution["p99_9_abs_error"] <= limits["p99_9_abs_error_max"],
        "failed_allclose_fraction": (
            distribution["failed_allclose_fraction"] <= limits["failed_allclose_fraction_max"]
        ),
        "max_abs_error": distribution["max_abs_error"] <= limits["max_abs_error_max"],
        "nan_count": (
            distribution["reference_nan_count"] + distribution["actual_nan_count"]
            <= limits["nan_count_max"]
        ),
        "inf_count": (
            distribution["reference_inf_count"] + distribution["actual_inf_count"]
            <= limits["inf_count_max"]
        ),
    }
    return {
        "amendment_source": P3_NUMERICS_AMENDMENT,
        "thresholds_only": True,
        "limits": limits,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _safe_ratio(numerator: float, denominator: float) -> dict[str, Any]:
    if denominator == 0.0:
        return {
            "value": 1.0 if numerator == 0.0 else None,
            "defined": numerator == 0.0,
            "zero_projection_error": True,
        }
    return {
        "value": numerator / denominator,
        "defined": True,
        "zero_projection_error": False,
    }


def amplification_summary(
    projection: dict[str, Any],
    post_conv: dict[str, Any],
) -> dict[str, Any]:
    return {
        "max_abs_error_ratio": _safe_ratio(post_conv["max_abs_error"], projection["max_abs_error"]),
        "mean_abs_error_ratio": _safe_ratio(post_conv["mean_abs_error"], projection["mean_abs_error"]),
        "relative_l2_error_ratio": _safe_ratio(
            post_conv["relative_l2_error"],
            projection["relative_l2_error"],
        ),
        "automatic_acceptance_threshold": None,
        "assessment_mode": "measurement_only",
    }


def _load_prior_evidence(path: Path) -> tuple[dict[str, Any], dict[str, bool]]:
    prior = json.loads(path.read_text(encoding="utf-8"))
    causal = prior.get("causal_checks", {})
    checks = {
        "diagnostics_completed": prior.get("diagnostics_completed") is True,
        "formal_status_unchanged": prior.get("p4_4_formal_status") == "FAIL_UNCHANGED",
        "artifact_manifest_verified": causal.get("artifact_manifest_matches_recomputed_conversion") is True,
        "parameter_embedding_exact": causal.get("parameter_embedding_exact") is True,
        "fp32_formal_strict_parity": causal.get("fp32_formal_strict_parity") is True,
        "forced_fp32_strict_parity": causal.get("forced_fp32_strict_parity") is True,
        "t7_both_injected_exact": causal.get("t7_both_injected_output_and_state_exact") is True,
        "t129_both_injected_exact": causal.get("t129_both_injected_output_and_state_exact") is True,
        "original_bf16_failure_reproduced": causal.get("original_bf16_failure_reproduced") is True,
    }
    return prior, checks


def _artifact_layer_check(
    artifact_dir: Path,
    source_hash: str,
    target_hash: str,
) -> dict[str, Any]:
    verification = verify_p4_artifact(artifact_dir)
    manifest_path = artifact_dir / P4_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    layer = next(item for item in manifest["layer_conversions"] if item["layer_index"] == LAYER_INDEX)
    checks = {
        "manifest_verified": verification["status"] == "PASS",
        "source_hash_match": layer["source_hash"] == source_hash,
        "target_hash_match": layer["target_hash"] == target_hash,
    }
    return {
        "manifest": str(manifest_path),
        "verification": verification,
        "layer_conversion": layer,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _distribution_pair(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    return error_distribution(
        reference,
        actual,
        torch.bfloat16,
        axis_names=("batch", "token", "feature"),
        top_k=16,
    )


def _initial_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": "P4.4-L4-T129-QKV-DISTRIBUTION",
        "status": "STARTING",
        "diagnostic_only": True,
        "measurement_completed": False,
        "formal_gate_modified": False,
        "converter_modified": False,
        "formal_backend_modified": False,
        "new_injection_run": False,
        "numerics_amendment_approved": False,
        "p4_4_formal_status": "FAIL_UNCHANGED",
        "p5_allowed": False,
        "layer_index": LAYER_INDEX,
        "started_at": args.started_at,
        "result_marker": "P4_04_T129_QKV_DISTRIBUTION=RUNNING",
    }


def run(args: argparse.Namespace, report: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("The authoritative P4 T=129 distribution diagnostic requires CUDA")
    device = torch.device("cuda")
    config = DenseK3Config.from_pretrained(args.artifact_dir)
    if LAYER_INDEX not in config.kda_indices:
        raise ValueError(f"Layer {LAYER_INDEX} is not a KDA layer in the P4 artifact")
    _, prior_checks = _load_prior_evidence(args.prior_diagnostic)
    report.update(
        {
            "status": "RUNNING",
            "last_completed_phase": "environment_and_prior_evidence",
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device),
                "transformers": package_version("transformers"),
            },
            "fixed_case": {
                "batch_size": BATCH_SIZE,
                "sequence_length": SEQUENCE_LENGTH,
                "dtype": "bfloat16",
                "hidden_seed": HIDDEN_SEED,
                "initial_state_mode": "random",
                "initial_state_seed": INITIAL_STATE_SEED,
                "initial_state_scale": INITIAL_STATE_SCALE,
            },
            "prior_diagnostic": str(args.prior_diagnostic),
            "prior_evidence_checks": prior_checks,
            "prior_evidence_valid": all(prior_checks.values()),
        }
    )
    save_json(args.output, report)

    bridge, source_state, source_audit = load_qwen_gdn_layer(
        args.model_dir,
        layer_index=LAYER_INDEX,
        device="cpu",
    )
    conversion = convert_gdn_to_kda(source_state, bridge)
    artifact_check = _artifact_layer_check(
        args.artifact_dir,
        conversion.manifest["source_hash"],
        conversion.manifest["target_hash"],
    )
    donor_cpu = QwenGDNReferenceMixer(bridge, source_state)
    target_cpu = DenseK3QwenCompatMixer(bridge, conversion.target_state)
    parameter_audit = parameter_embedding_audit(bridge, donor_cpu, target_cpu)
    report.update(
        {
            "last_completed_phase": "source_artifact_and_parameters",
            "source_audit": source_audit,
            "artifact_consistency": artifact_check,
            "parameter_embedding_audit": parameter_audit,
        }
    )
    save_json(args.output, report)

    donor = donor_cpu.to(device, torch.bfloat16)
    target = target_cpu.to(device, torch.bfloat16)
    hidden = seeded_hidden(bridge.hidden_size, device)
    initial_state = seeded_initial_state(bridge, device)
    report["fixed_case"].update(
        {
            "hidden_sha256": tensor_sha256(hidden),
            "initial_state_sha256": tensor_sha256(initial_state),
        }
    )
    with torch.inference_mode():
        donor_trace = donor.trace(hidden, initial_state)
        target_trace = target.trace(hidden, initial_state)

    output_distribution = error_distribution(
        donor_trace.output,
        target_trace.output,
        torch.bfloat16,
        axis_names=("batch", "token", "hidden_channel"),
        top_k=32,
    )
    functional = {
        "output": compare(donor_trace.output, target_trace.output, torch.bfloat16),
        "final_state": compare(donor_trace.final_state, target_trace.final_state, torch.bfloat16),
        "decay": compare(donor_trace.decay, target_trace.decay, torch.bfloat16),
        "beta": compare(donor_trace.beta, target_trace.beta, torch.bfloat16),
    }
    projection_fields = {
        "q": (donor_trace.q_projection, target_trace.q_projection),
        "k": (donor_trace.k_projection, target_trace.k_projection),
        "v": (donor_trace.v_projection, target_trace.v_projection),
    }
    post_conv_fields = {
        "q": (donor_trace.q_conv, target_trace.q_conv),
        "k": (donor_trace.k_conv, target_trace.k_conv),
        "v": (donor_trace.v_conv, target_trace.v_conv),
    }
    projection = {
        name: _distribution_pair(reference, actual)
        for name, (reference, actual) in projection_fields.items()
    }
    post_conv = {
        name: _distribution_pair(reference, actual)
        for name, (reference, actual) in post_conv_fields.items()
    }
    amplification = {
        name: amplification_summary(projection[name], post_conv[name])
        for name in ("q", "k", "v")
    }
    output_thresholds = evaluate_p3_output_distribution(output_distribution)
    original_p3_preconditions = {
        "fp32_strict_parity": prior_checks["fp32_formal_strict_parity"],
        "final_state_existing_gate": functional["final_state"]["passed"],
        "decay_existing_gate": functional["decay"]["passed"],
        "beta_existing_gate": functional["beta"]["passed"],
        "both_injected_exact": prior_checks["t129_both_injected_exact"],
        "all_finite": all(
            item["reference_nan_count"] == 0
            and item["actual_nan_count"] == 0
            and item["reference_inf_count"] == 0
            and item["actual_inf_count"] == 0
            for item in functional.values()
        ),
    }
    original_p3_eligible = all(original_p3_preconditions.values())
    measurement_valid = (
        all(prior_checks.values())
        and artifact_check["passed"]
        and parameter_audit["passed"]
        and original_p3_preconditions["all_finite"]
    )
    composite_candidate = (
        measurement_valid
        and output_thresholds["passed"]
        and functional["final_state"]["passed"]
        and functional["beta"]["passed"]
        and prior_checks["t129_both_injected_exact"]
    )
    report.update(
        {
            "status": "PASS" if measurement_valid else "INCONCLUSIVE",
            "measurement_completed": True,
            "last_completed_phase": "t129_output_qkv_and_post_conv_distributions",
            "completed_at": now(),
            "standard_allclose": {
                "rtol": OPTIMIZED_BF16_RTOL,
                "atol": OPTIMIZED_BF16_ATOL,
            },
            "functional_comparisons": functional,
            "final_output_distribution": output_distribution,
            "projection_distributions": projection,
            "post_conv_distributions": post_conv,
            "post_conv_amplification": amplification,
            "p3_output_distribution_threshold_evaluation": output_thresholds,
            "p3_original_fallback": {
                "amendment": P3_NUMERICS_AMENDMENT,
                "preconditions": original_p3_preconditions,
                "eligible": original_p3_eligible,
                "accepted": False,
                "reason": (
                    "P3 fallback is not accepted directly because its original decay-existing-Gate "
                    "precondition must remain authoritative"
                ),
            },
            "p4_composite_gate_candidate": {
                "candidate_amendment": P4_CANDIDATE_AMENDMENT,
                "eligible_for_human_review": composite_candidate,
                "candidate_output_rule": "p3_bf16_distribution_thresholds",
                "candidate_layer_rule": "p4_auxiliary_intermediate_hierarchy",
                "approved": False,
                "pass_mode": None,
            },
            "decision": (
                "ELIGIBLE_FOR_P4_AMENDMENT_REVIEW"
                if composite_candidate
                else "KEEP_P4_4_FROZEN"
            ),
            "result_marker": (
                "P4_04_T129_QKV_DISTRIBUTION=PASS"
                if measurement_valid
                else "P4_04_T129_QKV_DISTRIBUTION=INCONCLUSIVE"
            ),
        }
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--prior-diagnostic", type=Path, default=DEFAULT_PRIOR_DIAGNOSTIC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.started_at = now()
    report = _initial_report(args)
    save_json(args.output, report)
    try:
        report = run(args, report)
    except Exception as exc:  # noqa: BLE001 - always preserve the machine-readable evidence envelope
        report.update(
            {
                "status": "EXCEPTION",
                "completed_at": now(),
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
                "result_marker": "P4_04_T129_QKV_DISTRIBUTION=EXCEPTION",
            }
        )
    save_json(args.output, report)
    print(report["result_marker"])
    if "p3_output_distribution_threshold_evaluation" in report:
        passed = report["p3_output_distribution_threshold_evaluation"]["passed"]
        print(f"P3_OUTPUT_DISTRIBUTION_THRESHOLDS={'PASS' if passed else 'FAIL'}")
    print(f"P4_4_FORMAL_STATUS={report['p4_4_formal_status']}")
    print("P4_NUMERICS_AMENDMENT_APPROVED=false")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
