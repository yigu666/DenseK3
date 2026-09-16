"""Collect layer-4 causal evidence without changing the P4 converter or Gate."""

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
import torch.nn.functional as F

from densek3_core.evaluation.p4_decay_diagnostics import (
    P4_DIAGNOSTIC_LAYER,
    P4_LOCAL_SEED,
    P4_LOCAL_SEQUENCE,
    compare,
    decay_path_diagnostics,
    error_distribution,
    forced_fp32_diagnostics,
    injection_diagnostics,
    parameter_embedding_audit,
    replay_p4_hidden,
    seeded_hidden,
    summarize_case,
    trace_comparisons,
)
from densek3_core.modeling import DenseK3Config
from densek3_core.transplant.full_model import P4_MANIFEST_NAME, verify_p4_artifact
from densek3_core.transplant.gdn_to_kda import convert_gdn_to_kda, load_qwen_gdn_layer
from densek3_core.transplant.single_layer import DenseK3QwenCompatMixer, QwenGDNReferenceMixer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models/Qwen3.5-4B-Base"
DEFAULT_ARTIFACT_DIR = ROOT / "outputs/densek3-core/checkpoints/p4/densek3-4b-qwen-compat"
DEFAULT_OUTPUT = ROOT / "outputs/densek3-core/manifests/p4/p4-layer-4-decay-diagnostics.json"
SWEEP_SEQUENCES = (1, 7, 65, 129)
SWEEP_SEEDS = (20260810, 20260811, 20260812)
STATE_SEED = 404130
STRESS_INPUT_SEED = 404129


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def save_json(path: Path, value: dict[str, Any]) -> None:
    def sanitize(item: Any) -> Any:
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, dict):
            return {key: sanitize(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [sanitize(child) for child in item]
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(sanitize(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _artifact_consistency(
    artifact_dir: Path,
    layer_index: int,
    source_hash: str,
    target_hash: str,
) -> dict[str, Any]:
    path = artifact_dir / P4_MANIFEST_NAME
    verified = verify_p4_artifact(artifact_dir)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    matching = [item for item in manifest["layer_conversions"] if item["layer_index"] == layer_index]
    if len(matching) != 1:
        raise ValueError(f"Expected exactly one layer-{layer_index} conversion in {path}")
    recorded = matching[0]
    checks = {
        "source_hash_match": recorded["source_hash"] == source_hash,
        "target_hash_match": recorded["target_hash"] == target_hash,
        "p3_frozen": manifest.get("p3_frozen") is True,
        "p4_manifest_status_pass": manifest.get("status") == "PASS",
        "manifest_self_verification_pass": verified.get("status") == "PASS",
    }
    return {
        "manifest": str(path),
        "manifest_hash": manifest.get("manifest_hash_without_self"),
        "verification": verified,
        "recorded_layer_conversion": recorded,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _gate_only_case(
    donor: QwenGDNReferenceMixer,
    target: DenseK3QwenCompatMixer,
    hidden: torch.Tensor,
    *,
    seed: int,
    sequence: int,
) -> dict[str, Any]:
    config = donor.config
    with torch.inference_mode():
        donor_raw = F.linear(hidden, donor.state["in_proj_a.weight"])
        donor_channel = donor_raw.unsqueeze(-1).expand(
            -1,
            -1,
            -1,
            config.qk_head_dim,
        )
        target_f_a = F.linear(hidden, target.state["f_a_proj.weight"])
        target_channel = F.linear(target_f_a, target.state["f_b_proj.weight"]).view(
            *hidden.shape[:2],
            config.value_num_heads,
            config.qk_head_dim,
        )
        donor_decay, donor_beta = donor.gates(hidden)
        target_decay, target_beta = target.gates(hidden)
    target_repeat = target_f_a[..., : config.value_num_heads].unsqueeze(-1).expand_as(target_channel)
    tensors = (
        donor_raw,
        donor_channel,
        target_f_a,
        target_channel,
        donor_decay,
        target_decay,
        donor_beta,
        target_beta,
    )
    return {
        "seed": seed,
        "sequence_length": sequence,
        "input_sha256": tensor_sha256(hidden),
        "raw_a": error_distribution(donor_raw, target_f_a[..., : config.value_num_heads], torch.bfloat16, top_k=2),
        "f_b_replication": error_distribution(target_repeat, target_channel, torch.bfloat16, top_k=2),
        "channel_a": error_distribution(
            donor_channel,
            target_channel,
            torch.bfloat16,
            axis_names=("batch", "token", "value_head", "key_channel"),
            top_k=2,
        ),
        "decay": error_distribution(
            donor_decay,
            target_decay,
            torch.bfloat16,
            axis_names=("batch", "token", "value_head", "key_channel"),
            top_k=2,
        ),
        "beta": compare(donor_beta, target_beta, torch.bfloat16),
        "all_finite": all(bool(torch.isfinite(tensor).all()) for tensor in tensors),
    }


def _seed_length_sweep(
    donor: QwenGDNReferenceMixer,
    target: DenseK3QwenCompatMixer,
    device: torch.device,
) -> dict[str, Any]:
    cases = []
    for sequence in SWEEP_SEQUENCES:
        for seed in SWEEP_SEEDS:
            hidden = seeded_hidden(donor.config.hidden_size, sequence, seed).to(
                device=device,
                dtype=torch.bfloat16,
            )
            cases.append(_gate_only_case(donor, target, hidden, seed=seed, sequence=sequence))
    return {
        "scope": "projection/decay only; recurrence intentionally excluded from this sweep",
        "sequences": list(SWEEP_SEQUENCES),
        "seeds": list(SWEEP_SEEDS),
        "cases_total": len(cases),
        "cases": cases,
        "all_completed": len(cases) == len(SWEEP_SEQUENCES) * len(SWEEP_SEEDS),
        "all_finite": all(case["all_finite"] for case in cases),
        "f_b_replication_exact_all_cases": all(
            case["f_b_replication"]["num_exact"] == case["f_b_replication"]["num_elements"]
            for case in cases
        ),
    }


def _make_initial_states(config: Any, device: torch.device) -> dict[str, torch.Tensor | None]:
    shape = (1, config.value_num_heads, config.value_head_dim, config.qk_head_dim)
    generator = torch.Generator(device="cpu").manual_seed(STATE_SEED)
    random_state = torch.randn(shape, generator=generator, dtype=torch.float32).to(device) * 0.02
    return {
        "none": None,
        "zeros": torch.zeros(shape, dtype=torch.float32, device=device),
        "random": random_state,
    }


def _state_isolation(
    donor: QwenGDNReferenceMixer,
    target: DenseK3QwenCompatMixer,
    hidden: torch.Tensor,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    states = _make_initial_states(donor.config, hidden.device)
    cases = {}
    random_traces = None
    for mode, state in states.items():
        with torch.inference_mode():
            donor_trace = donor.trace(hidden, state)
            target_trace = target.trace(hidden, state)
        if mode == "random":
            random_traces = (donor_trace, target_trace)
        cases[mode] = {
            "initial_state_sha256": None if state is None else tensor_sha256(state),
            "donor_decay_sha256": tensor_sha256(donor_trace.decay),
            "target_decay_sha256": tensor_sha256(target_trace.decay),
            "functional": summarize_case(
                donor_trace,
                target_trace,
                torch.bfloat16,
                include_distribution=label == "exact_p4_t7",
            ),
        }
    donor_decay_hashes = {case["donor_decay_sha256"] for case in cases.values()}
    target_decay_hashes = {case["target_decay_sha256"] for case in cases.values()}
    result = {
        "label": label,
        "sequence_length": hidden.shape[1],
        "state_seed": STATE_SEED,
        "cases": cases,
        "decay_independent_of_initial_state": len(donor_decay_hashes) == 1 and len(target_decay_hashes) == 1,
        "all_finite": all(case["functional"]["all_finite"] for case in cases.values()),
        "completed": set(cases) == {"none", "zeros", "random"},
    }
    if random_traces is None:
        raise RuntimeError("Random-state isolation trace was not produced")
    return result, random_traces


def _state_functional_gate_passed(isolation: dict[str, Any]) -> bool:
    required = ("output", "final_state", "beta")
    return all(
        case["functional"]["comparisons"][name]["passed"]
        for case in isolation["cases"].values()
        for name in required
    )


def _causal_checks(report: dict[str, Any]) -> dict[str, bool]:
    fp32 = report["exact_p4_replay"]["fp32"]["functional"]["comparisons"]
    bf16 = report["exact_p4_replay"]["bf16"]["functional"]["comparisons"]
    path = report["exact_p4_replay"]["bf16"]["decay_path"]
    forced = report["forced_fp32_decay_projection"]
    injection_t7 = report["exact_injection_isolation"]["exact_p4_t7"]
    injection_t129 = report["exact_injection_isolation"]["stress_t129_random_state"]
    states = report["initial_state_isolation"]
    sweep = report["seed_length_sweep"]
    raw = path["raw_a_logits"]
    replication = path["target_f_b_vs_exact_repeat"]
    forced_required = ("raw_a_logits", "channel_a_logits", "decay")
    return {
        "artifact_manifest_matches_recomputed_conversion": report["artifact_consistency"]["passed"],
        "parameter_embedding_exact": report["parameter_embedding_audit"]["passed"],
        "fp32_formal_strict_parity": all(fp32[name]["passed"] for name in ("output", "final_state", "decay", "beta")),
        "original_bf16_failure_reproduced": (
            bf16["output"]["passed"]
            and bf16["final_state"]["passed"]
            and bf16["beta"]["passed"]
            and not bf16["decay"]["passed"]
        ),
        "decay_path_mismatch_present_at_raw_a": raw["num_exact"] < raw["num_elements"],
        "f_b_is_exact_replication": replication["num_exact"] == replication["num_elements"],
        "recorded_logits_recompute_trace_decay_strictly": all(
            item["passed"] for item in path["decay_transform_replay"].values()
        ),
        "forced_fp32_strict_parity": all(
            forced["fp32_donor_vs_target"][name]["passed"] for name in forced_required
        ),
        "forced_fp32_cast_decay_strict_parity": forced["decay_strict_parity_after_bf16_cast"],
        "t7_both_injected_output_and_state_exact": injection_t7["both_injected_exact"],
        "t129_both_injected_output_and_state_exact": injection_t129["both_injected_exact"],
        "t7_initial_state_isolation_complete": (
            states["exact_p4_t7"]["completed"]
            and states["exact_p4_t7"]["decay_independent_of_initial_state"]
            and states["exact_p4_t7"]["all_finite"]
            and _state_functional_gate_passed(states["exact_p4_t7"])
        ),
        "t129_initial_state_isolation_complete": (
            states["stress_t129"]["completed"]
            and states["stress_t129"]["decay_independent_of_initial_state"]
            and states["stress_t129"]["all_finite"]
            and _state_functional_gate_passed(states["stress_t129"])
        ),
        "seed_length_sweep_complete_and_finite": (
            sweep["all_completed"] and sweep["all_finite"] and sweep["f_b_replication_exact_all_cases"]
        ),
    }


def _initial_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": "P4.4-L4-DIAG",
        "status": "STARTING",
        "diagnostic_only": True,
        "diagnostics_completed": False,
        "formal_gate_modified": False,
        "converter_modified": False,
        "formal_backend_modified": False,
        "numerics_amendment_approved": False,
        "p4_4_formal_status": "FAIL_UNCHANGED",
        "p5_allowed": False,
        "layer_index": args.layer_index,
        "started_at": args.started_at,
        "result_marker": "P4_04_LAYER4_CAUSAL_DIAGNOSTICS=RUNNING",
    }


def run(args: argparse.Namespace, report: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("The authoritative P4 layer-4 diagnostic requires CUDA")
    device = torch.device("cuda")
    report.update(
        {
            "status": "RUNNING",
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device),
                "transformers": package_version("transformers"),
            },
            "last_completed_phase": "environment",
        }
    )
    save_json(args.output, report)
    config = DenseK3Config.from_pretrained(args.artifact_dir)
    if args.layer_index != P4_DIAGNOSTIC_LAYER:
        raise ValueError("This frozen diagnostic is intentionally scoped to P4 layer 4")
    bridge, source_state, source_audit = load_qwen_gdn_layer(
        args.model_dir,
        layer_index=args.layer_index,
        device="cpu",
    )
    conversion = convert_gdn_to_kda(source_state, bridge)
    artifact = _artifact_consistency(
        args.artifact_dir,
        args.layer_index,
        conversion.manifest["source_hash"],
        conversion.manifest["target_hash"],
    )
    donor_cpu = QwenGDNReferenceMixer(bridge, source_state)
    target_cpu = DenseK3QwenCompatMixer(bridge, conversion.target_state)
    parameter_audit = parameter_embedding_audit(bridge, donor_cpu, target_cpu)
    cpu_hidden, draw_one_based = replay_p4_hidden(
        bridge.hidden_size,
        config.kda_indices,
        args.layer_index,
    )
    report.update(
        {
            "source_audit": source_audit,
            "artifact_consistency": artifact,
            "parameter_embedding_audit": parameter_audit,
            "exact_p4_replay": {
                "seed": P4_LOCAL_SEED,
                "sequence_length": P4_LOCAL_SEQUENCE,
                "draw_ordinal_zero_based": draw_one_based - 1,
                "draw_ordinal_one_based": draw_one_based,
                "kda_indices_before_or_at_layer": config.kda_indices[:draw_one_based],
                "cpu_input_sha256": tensor_sha256(cpu_hidden),
            },
            "last_completed_phase": "source_artifact_and_input_replay",
        }
    )
    save_json(args.output, report)

    donor_fp32 = donor_cpu.to(device, torch.float32)
    target_fp32 = target_cpu.to(device, torch.float32)
    hidden_fp32 = cpu_hidden.to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        donor_fp32_trace = donor_fp32.trace(hidden_fp32)
        target_fp32_trace = target_fp32.trace(hidden_fp32)
    fp32_case = {
        "functional": summarize_case(donor_fp32_trace, target_fp32_trace, torch.float32, include_distribution=True),
        "trace": trace_comparisons(donor_fp32_trace, target_fp32_trace, torch.float32),
    }
    report["exact_p4_replay"]["fp32"] = fp32_case
    report["last_completed_phase"] = "fp32_formal_replay"
    save_json(args.output, report)
    del donor_fp32, target_fp32, hidden_fp32, donor_fp32_trace, target_fp32_trace
    torch.cuda.empty_cache()

    donor = donor_cpu.to(device, torch.bfloat16)
    target = target_cpu.to(device, torch.bfloat16)
    hidden = cpu_hidden.to(device=device, dtype=torch.bfloat16)
    with torch.inference_mode():
        donor_trace = donor.trace(hidden)
        target_trace = target.trace(hidden)
        forced = forced_fp32_diagnostics(donor, target, hidden, donor_trace, target_trace)
        injection_t7 = injection_diagnostics(donor, target, hidden, donor_trace, target_trace)
    bf16_case = {
        "functional": summarize_case(donor_trace, target_trace, torch.bfloat16, include_distribution=True),
        "trace": trace_comparisons(donor_trace, target_trace, torch.bfloat16),
        "decay_path": decay_path_diagnostics(
            donor_trace,
            target_trace,
            torch.bfloat16,
            donor_mixer=donor,
            target_mixer=target,
        ),
    }
    report["exact_p4_replay"]["bf16"] = bf16_case
    report["forced_fp32_decay_projection"] = forced
    report["exact_injection_isolation"] = {"exact_p4_t7": injection_t7}
    report["last_completed_phase"] = "bf16_exact_replay_and_t7_injection"
    save_json(args.output, report)

    state_t7, _ = _state_isolation(donor, target, hidden, label="exact_p4_t7")
    stress_hidden = seeded_hidden(bridge.hidden_size, 129, STRESS_INPUT_SEED).to(
        device=device,
        dtype=torch.bfloat16,
    )
    state_t129, stress_random_traces = _state_isolation(donor, target, stress_hidden, label="stress_t129")
    stress_state = _make_initial_states(bridge, device)["random"]
    with torch.inference_mode():
        injection_t129 = injection_diagnostics(
            donor,
            target,
            stress_hidden,
            stress_random_traces[0],
            stress_random_traces[1],
            initial_state=stress_state,
        )
    report["exact_injection_isolation"]["stress_t129_random_state"] = injection_t129
    report["initial_state_isolation"] = {
        "exact_p4_t7": state_t7,
        "stress_t129": state_t129,
    }
    report["last_completed_phase"] = "initial_state_and_t129_injection"
    save_json(args.output, report)
    sweep = _seed_length_sweep(donor, target, device)
    report["seed_length_sweep"] = sweep
    report["last_completed_phase"] = "seed_length_sweep"
    report["completed_at"] = now()
    save_json(args.output, report)
    checks = _causal_checks(report)
    supported = all(checks.values())
    report.update(
        {
            "status": "PASS" if supported else "INCONCLUSIVE",
            "diagnostics_completed": True,
            "original_failure_reproduced": checks["original_bf16_failure_reproduced"],
            "causal_checks": checks,
            "causal_conclusion": {
                "status": "SUPPORTED" if supported else "NOT_FULLY_SUPPORTED",
                "hypothesis": (
                    "BF16 32-row donor in_proj_a GEMM versus 128-row target f_a GEMM shape/kernel path; "
                    "f_b replication, decay transform, recurrence, gate, norm, and output projection remain equivalent"
                ),
                "bf16_decay_path_first_observed_mismatch": (
                    "raw_a_logits" if checks["decay_path_mismatch_present_at_raw_a"] else None
                ),
                "f_b_replication_exact": checks["f_b_is_exact_replication"],
                "forced_fp32_removes_mismatch": (
                    checks["forced_fp32_strict_parity"]
                    and checks["forced_fp32_cast_decay_strict_parity"]
                ),
                "recurrence_and_postprocessing_restored_under_identical_inputs": (
                    checks["t7_both_injected_output_and_state_exact"]
                    and checks["t129_both_injected_output_and_state_exact"]
                ),
            },
            "next_decision": (
                "Review this evidence before proposing any P4-scoped auxiliary-decay numerics amendment; "
                "this report does not authorize one"
            ),
            "result_marker": f"P4_04_LAYER4_CAUSAL_DIAGNOSTICS={'PASS' if supported else 'INCONCLUSIVE'}",
        }
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer-index", type=int, default=P4_DIAGNOSTIC_LAYER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.started_at = now()
    report = _initial_report(args)
    save_json(args.output, report)
    try:
        report = run(args, report)
    except Exception as exc:  # noqa: BLE001 - diagnostics must always persist their failure envelope
        report.update(
            {
                "status": "EXCEPTION",
                "completed_at": now(),
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
                "result_marker": "P4_04_LAYER4_CAUSAL_DIAGNOSTICS=EXCEPTION",
            }
        )
    save_json(args.output, report)
    print(report["result_marker"])
    print(f"P4_4_FORMAL_STATUS={report['p4_4_formal_status']}")
    print("P4_NUMERICS_AMENDMENT_APPROVED=false")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
