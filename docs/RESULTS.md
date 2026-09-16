# Verified Results

DenseK3-4B is compared directly with its Qwen3.5-4B-Base donor. All capability
values below come from PASS artifacts produced on 2026-09-14/15. The released
checkpoint used the qualified Titan-compatible FP16 path; Qwen used BF16 SDPA.
Shared-GPU throughput is diagnostic only, so no speed claim is reported.

## General capability — measured

| Benchmark | Qwen3.5-4B-Base | DenseK3-4B |
|---|---:|---:|
| MMLU official dev, 5-shot | 73.10% (10,265 / 14,042) | 67.67% (9,502 / 14,042) |
| WikiText-103 raw test CE | 1.9334 | 2.0821 |
| WikiText-103 raw test PPL | 6.9130 | 8.0211 |
| LongBench-v2 controlled MC-LL Accuracy, `context ≤128K` (298 cases) | 37.25% (111 / 298) | 30.20% (90 / 298) |

LongBench-v2 is the controlled MC-LL Accuracy subset, not the official generation
leaderboard metric.

## Long-context quality — measured

RULER is an explicitly reduced fixed-sample suite: four examples for each of 13
tasks at each length (52 cases per model and length), not the official full
500-sample-per-task setting.

| Context | Qwen3.5-4B-Base | DenseK3-4B |
|---:|---:|---:|
| 4K | 88.46 | 84.49 |
| 8K | 88.27 | 76.15 |
| 16K | 86.35 | 65.96 |
| 32K | 85.03 | 60.19 |
| 64K | 84.36 | 53.97 |
| 128K | 78.37 | 49.87 |

The aggregate public record is `results/verified-results.json`; it binds the raw
PASS reports by SHA-256. These results show capability retention after migration,
not superiority to the donor.

## Memory — derived and estimated from architecture

For the eight attention layers in FP16/BF16, Qwen-equivalent expanded GQA K/V grows
by approximately 32 KiB/token, while DenseK3's rank-512 latent cache grows by 8
KiB/token. The resulting approximately 75% reduction applies only to persistent
sequence-growing attention state.

| Context | Qwen-equivalent cache | DenseK3 latent cache |
|---:|---:|---:|
| 128K | ~4 GiB | ~1 GiB |
| 256K | ~8 GiB | ~2 GiB |
| 512K | ~16 GiB | ~4 GiB |

## 512K runtime — measured

An exact-semantics P10-T research runtime probe completed full prefill at 524,288
tokens and continued autoregressive decoding, with no sliding window or approximate
attention. Peak GPU allocated memory was exactly 15,762,103,296 bytes (14.68 GiB);
persistent latent state was exactly 4,294,967,296 bytes, with zero persistent
expanded K and V. This is runtime/cache evidence, not a DenseK3-4B 512K quality
benchmark. Standard quality evidence stops at 128K.

## Recovery evidence

P10 joint recovery recorded full-development CE 2.389680047457615 and passed its
reduced-scope Titan gate. Its formal status remains
`TITAN_FP16_PROVISIONAL_PENDING_NATIVE_BF16_REPLAY`; it is not evidence of completed
native RTX 5090 BF16/full-scope retraining.

For evidence boundaries and the later research disclosure, see
[RELEASE_SCOPE.md](RELEASE_SCOPE.md).
