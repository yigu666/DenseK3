"""Reproducible CUDA baseline for the three P2 KDA paths."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from densek3_core.kda.fla_backend import fla_chunk_kda, fla_fused_recurrent_kda
from densek3_core.kda.reference import kda_reference_from_raw

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/densek3-core/manifests/p2/p2-performance.json"


def synchronize() -> None:
    torch.cuda.synchronize()


def measure(name: str, function, tokens: int, warmup: int, iterations: int) -> dict:
    for _ in range(warmup):
        function()
    synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(iterations):
        function()
    synchronize()
    latency = (time.perf_counter() - started) / iterations
    return {
        "backend": name,
        "latency_ms": latency * 1000,
        "tokens_per_second": tokens / latency,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "warmup": warmup,
        "iterations": iterations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequence", type=int, default=129)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P2 performance baseline requires CUDA")
    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch, sequence = args.batch, args.sequence
    q = torch.randn(batch, sequence, 16, 128, dtype=dtype, device=device)
    k = torch.randn_like(q)
    v = torch.randn(batch, sequence, 32, 128, dtype=dtype, device=device)
    raw_decay = torch.randn(batch, sequence, 32, 128, dtype=dtype, device=device)
    beta_logits = torch.randn(batch, sequence, 32, dtype=dtype, device=device)
    a_log = torch.zeros(32, dtype=torch.float32, device=device)
    dt_bias = torch.zeros(32 * 128, dtype=torch.float32, device=device)

    def reference():
        return kda_reference_from_raw(q, k, v, raw_decay, beta_logits, a_log, dt_bias)

    def chunk():
        return fla_chunk_kda(q, k, v, raw_decay, beta_logits, a_log, dt_bias, output_final_state=False)

    def recurrent():
        state = None
        for index in range(sequence):
            _, state = fla_fused_recurrent_kda(
                q[:, index : index + 1],
                k[:, index : index + 1],
                v[:, index : index + 1],
                raw_decay[:, index : index + 1],
                beta_logits[:, index : index + 1],
                a_log,
                dt_bias,
                initial_state=state,
            )
        return state

    results = {
        "status": "PASS",
        "seed": 42,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "dtype": str(dtype),
        "shape": {"B": batch, "T": sequence, "H": 16, "HV": 32, "K": 128, "V": 128},
        "measurements": [
            measure("pytorch_reference", reference, batch * sequence, args.warmup, args.iterations),
            measure("fla_chunk", chunk, batch * sequence, args.warmup, args.iterations),
            measure("fla_fused_recurrent_trajectory", recurrent, batch * sequence, args.warmup, args.iterations),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

