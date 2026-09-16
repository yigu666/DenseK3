"""Collect deterministic P2 parity and backward metrics on the server GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from densek3_core.kda.fla_backend import fla_chunk_kda, fla_fused_recurrent_kda
from densek3_core.kda.metrics import (
    GATE_GRADIENT_ATOL,
    GATE_GRADIENT_RTOL,
    INPUT_GRADIENT_ATOL,
    INPUT_GRADIENT_RTOL,
    error_metrics,
    optimized_output_tolerance,
    optimized_state_tolerance,
)
from densek3_core.kda.reference import kda_reference_from_raw

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs/densek3-core/manifests/p2"


def make_inputs(sequence: int, dtype: torch.dtype, *, key_dim: int = 128, value_dim: int = 128):
    torch.manual_seed(42)
    device = torch.device("cuda")
    q = torch.randn(1, sequence, 16, key_dim, dtype=dtype, device=device)
    k = torch.randn_like(q)
    v = torch.randn(1, sequence, 32, value_dim, dtype=dtype, device=device)
    raw_decay = torch.randn(1, sequence, 32, key_dim, dtype=dtype, device=device)
    beta_logits = torch.randn(1, sequence, 32, dtype=dtype, device=device)
    a_log = torch.zeros(32, dtype=torch.float32, device=device)
    dt_bias = torch.zeros(32 * key_dim, dtype=torch.float32, device=device)
    return q, k, v, raw_decay, beta_logits, a_log, dt_bias


def tolerance(
    dtype: torch.dtype,
    gradient: bool = False,
    parameter_gate: bool = False,
    final_state: bool = False,
) -> tuple[float, float]:
    if not gradient:
        return optimized_state_tolerance(dtype) if final_state else optimized_output_tolerance(dtype)
    if parameter_gate:
        return GATE_GRADIENT_RTOL, GATE_GRADIENT_ATOL
    return INPUT_GRADIENT_RTOL, INPUT_GRADIENT_ATOL


def compared(
    reference: torch.Tensor,
    actual: torch.Tensor,
    dtype: torch.dtype,
    *,
    gradient: bool = False,
    parameter_gate: bool = False,
    final_state: bool = False,
) -> dict:
    rtol, atol = tolerance(dtype, gradient, parameter_gate, final_state)
    metrics = error_metrics(reference, actual)
    metrics.update({"rtol": rtol, "atol": atol, "passed": torch.allclose(reference, actual, rtol=rtol, atol=atol)})
    return metrics


def chunk_forward_cases() -> list[dict]:
    cases = []
    for dtype, sequence, initial_mode in [
        (torch.float32, 7, "none"),
        (torch.float32, 65, "random"),
        (torch.bfloat16, 7, "none"),
        (torch.bfloat16, 129, "random"),
    ]:
        tensors = make_inputs(sequence, dtype)
        q, k, v, raw_decay, beta_logits, a_log, dt_bias = tensors
        state = None
        if initial_mode == "random":
            state = torch.randn(1, 32, 128, 128, dtype=torch.float32, device="cuda") * 0.02
        reference_output, reference_state = kda_reference_from_raw(
            *tensors, initial_state=state, output_final_state=True
        )
        actual_output, actual_state = fla_chunk_kda(*tensors, initial_state=state)
        cases.append(
            {
                "dtype": str(dtype).removeprefix("torch."),
                "B": 1,
                "T": sequence,
                "H": 16,
                "HV": 32,
                "K": 128,
                "V": 128,
                "initial_state": initial_mode,
                "output": compared(reference_output, actual_output, dtype),
                "final_state": compared(reference_state, actual_state, dtype, final_state=True),
            }
        )
    return cases


def recurrent_cases() -> list[dict]:
    cases = []
    for dtype in (torch.float32, torch.bfloat16):
        tensors = make_inputs(7, dtype)
        reference_output, reference_state = kda_reference_from_raw(*tensors, output_final_state=True)
        state = None
        outputs = []
        for index in range(7):
            token_tensors = tuple(tensor[:, index : index + 1] for tensor in tensors[:5]) + tensors[5:]
            output, state = fla_fused_recurrent_kda(*token_tensors, initial_state=state)
            outputs.append(output)
        actual_output = torch.cat(outputs, dim=1)
        cases.append(
            {
                "dtype": str(dtype).removeprefix("torch."),
                "B": 1,
                "T": 7,
                "H": 16,
                "HV": 32,
                "K": 128,
                "V": 128,
                "output": compared(reference_output, actual_output, dtype),
                "final_state": compared(reference_state, state, dtype, final_state=True),
            }
        )
    return cases


def continuation_cases() -> list[dict]:
    tensors = make_inputs(129, torch.bfloat16)
    one_shot_output, one_shot_state = fla_chunk_kda(*tensors)
    cases = []
    for split in (1, 17, 47, 128):
        prefix_tensors = tuple(tensor[:, :split] for tensor in tensors[:5]) + tensors[5:]
        prefix_output, state = fla_chunk_kda(*prefix_tensors)
        outputs = [prefix_output]
        for index in range(split, 129):
            token_tensors = tuple(tensor[:, index : index + 1] for tensor in tensors[:5]) + tensors[5:]
            output, state = fla_fused_recurrent_kda(*token_tensors, initial_state=state)
            outputs.append(output)
        resumed_output = torch.cat(outputs, dim=1)
        cases.append(
            {
                "split": split,
                "output": compared(one_shot_output, resumed_output, torch.bfloat16),
                "final_state": compared(one_shot_state, state, torch.bfloat16, final_state=True),
            }
        )
    return cases


def backward_run(backend: str):
    tensors = list(make_inputs(7, torch.bfloat16, key_dim=32, value_dim=32))
    for index in range(len(tensors)):
        tensors[index] = tensors[index].detach().requires_grad_(True)
    state = (torch.randn(1, 32, 32, 32, dtype=torch.float32, device="cuda") * 0.02).requires_grad_(True)
    if backend == "reference":
        output, final_state = kda_reference_from_raw(*tensors, initial_state=state, output_final_state=True)
    else:
        output, final_state = fla_chunk_kda(*tensors, initial_state=state)
    torch.manual_seed(42)
    output_weight = torch.randn_like(output)
    state_weight = torch.randn_like(final_state) * 0.1
    loss = (output.float() * output_weight.float()).sum() + (final_state * state_weight).sum()
    return torch.autograd.grad(loss, (*tensors, state))


def backward_metrics() -> dict:
    reference = backward_run("reference")
    actual = backward_run("fla")
    names = ("dq", "dk", "dv", "ddecay", "dbeta", "dA_log", "ddt_bias", "d_initial_state")
    gradients = {}
    for name, expected, observed in zip(names, reference, actual, strict=True):
        gradients[name] = compared(
            expected,
            observed,
            torch.bfloat16,
            gradient=True,
            parameter_gate=name in {"dA_log", "ddt_bias"},
        )
        gradients[name]["finite_gradient_ratio"] = float(torch.isfinite(observed).float().mean())
    return {
        "status": "PASS" if all(item["passed"] for item in gradients.values()) else "FAIL",
        "seed": 42,
        "shape": {"B": 1, "T": 7, "H": 16, "HV": 32, "K": 32, "V": 32},
        "dtype": "bfloat16",
        "gradients": gradients,
    }


def all_cases_pass(cases: list[dict]) -> bool:
    return all(case["output"]["passed"] and case["final_state"]["passed"] for case in cases)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P2 parity collection requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunk = chunk_forward_cases()
    recurrent = recurrent_cases()
    continuation = continuation_cases()
    parity_status = all_cases_pass(chunk) and all_cases_pass(recurrent) and all_cases_pass(continuation)
    parity = {
        "stage": "P2",
        "status": "PASS" if parity_status else "FAIL",
        "seed": 42,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "reference_vs_fla_chunk": chunk,
        "reference_vs_fla_fused_recurrent": recurrent,
        "continuation_parity": continuation,
    }
    backward = backward_metrics()
    (args.output_dir / "p2-parity-results.json").write_text(json.dumps(parity, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "p2-backward-results.json").write_text(json.dumps(backward, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"parity": parity["status"], "backward": backward["status"]}, indent=2))
    return 0 if parity_status and backward["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
