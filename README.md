# DenseK3

**[English](README.md) | [中文](README_zh.md)**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE) [![Model: DenseK3-4B](https://img.shields.io/badge/Hugging%20Face-DenseK3--4B-yellow.svg)](https://huggingface.co/yigu666/DenseK3-4B)

🤗 **Model release:** [qinfu19/DenseK3-4B]((https://huggingface.co/qinfu19/DenseK3))

DenseK3 is an independent research project exploring whether a dense language
model can inherit Kimi K3-style recurrent/attention ideas while retaining the
compatibility and initialization benefits of Qwen3.5-4B-Base. The public
checkpoint is **DenseK3-4B**. It is not an official Moonshot AI or Kimi release,
and it does not claim to reproduce the complete Kimi K3 system.

## ✨ Highlights

- Open P0–P11 mainline: architecture migration, component and joint recovery,
  Qwen exact-token OPD, Kimi text-space corrective distillation, and evaluation
  contracts are included in the public source tree.
- A 3:1 hybrid backbone with 24 Kimi Delta Attention (KDA) layers and 8
  strict-NoPE Multi-head Latent Attention (MLA) layers.
- True latent MLA cache: each MLA layer persists only a normalized
  `[B,T,512]` history; expanded K/V tensors are temporary and never persistent.
- Eight four-layer Block AttnRes depth blocks (65 routing sites) and dense
  SiTU-GLU channel mixing (`beta=4`, `linear_beta=25`).
- Qwen tokenizer/vocabulary, embedding, tied LM head, hidden size, and FFN width
  are preserved wherever the migration contract permits.
- A standalone Hugging Face export is clean-room parity validated and can be
  loaded without the donor checkpoint.

## 💡 Why DenseK3?

Long-context models often trade recurrent efficiency, attention quality, and
compatibility with existing pretrained checkpoints. DenseK3 makes that trade-off
explicit: it tests a white-box migration from a Qwen donor to a dense K3-style
hybrid, then measures what is retained and what is lost. The project is intended
as an auditable research artifact rather than a claim that a single architecture
choice is universally better.

## 🏗️ Architecture

### Overview

| Component | Qwen3.5-4B-Base donor | DenseK3-4B |
|---|---:|---:|
| Decoder layers | 32 | 32 |
| Hidden / FFN size | 2,560 / 9,216 | 2,560 / 9,216 |
| Vocabulary | 248,320 | 248,320 (Qwen tokenizer ids are covered) |
| Token mixers | 24 Gated DeltaNet + 8 Gated Attention | **24 KDA + 8 strict-NoPE MLA** |
| Mixer pattern | 3:1 | `(KDA,KDA,KDA,MLA) × 8` |
| MLA layers (zero based) | — | `3, 7, 11, 15, 19, 23, 27, 31` |
| Channel mixer | SwiGLU | Dense SiTU-GLU |
| Depth mixer | Standard residual | Block AttnRes |
| Sequence-growing attention state | Expanded GQA K/V | 512-dimensional latent cache |
| Embedding / LM head | Qwen pretrained, tied | Preserved, tied |

The complete equations, tensor mappings, normalization rules, and cache
invariants live in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

### Kimi Delta Attention (KDA)

The 24 recurrent mixers use Q/K/V head counts `16/16/32`, head dimension 128,
depthwise causal convolution kernel 4, and decay projection rank 128. Their
recurrent state is FP32, V-first, and shaped `[B,32,128,128]`. The canonical
parameterization uses K3-style lower-bounded decay and a SiLU output gate.

### strict-NoPE MLA

The eight MLA layers use 16 heads, KV latent rank 512, NoPE query/key dimension
256, value dimension 256, Q/K normalization, and a full-rank sigmoid output gate.
RoPE is disabled on this path by contract. Expanded K/V is reconstructed only
for the current attention operation.

### True Latent Cache

For each MLA layer, the persistent sequence state is exactly `[B,T,512]`.
Persistent expanded K/V is zero. KDA recurrent states and AttnRes depth sources
remain separate from the MLA sequence cache.

### Block AttnRes

The 32 decoder layers form eight blocks of four layers. Each layer has a pre-mixer
and pre-FFN routing site, followed by one output site: `32×2+1=65`
content-dependent sites.

### Dense SiTU-GLU

DenseK3 keeps a dense FFN rather than Kimi K3's LatentMoE. The released SiTU
activation uses `beta=4` and `linear_beta=25` around the Qwen-sized 9,216-wide
FFN.

### Preserved Qwen components

The donor tokenizer/vocabulary, token embedding, tied LM head, hidden and FFN
dimensions, RMSNorm parameterization, and contracted Q/K normalization semantics
are retained. The migration changes token mixing, depth mixing, FFN activation,
and the long-context state representation. See
[`docs/MIGRATION.md`](docs/MIGRATION.md) for the deterministic mapping and gates.

## 🔄 From Qwen3.5 to DenseK3

The public research path is staged so that each architectural variable has an
independent contract and regression gate:

```text
Qwen3.5-4B-Base
  → KDA primitive parity and Gated DeltaNet-to-KDA transplant
  → 24 KDA + 8 attention hybrid conversion and KDA recovery
  → strict-NoPE MLA migration and calibration
  → true latent cache
  → Block AttnRes
  → dense SiTU-GLU
  → joint recovery
  → Qwen exact-token OPD + Kimi text-space corrective distillation
  → DenseK3-4B
```

The P0–P11 stage contracts, checkpoint rules, recovery gates, and teacher roles
are public in [`docs/MIGRATION.md`](docs/MIGRATION.md),
[`docs/TRAINING.md`](docs/TRAINING.md), and
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md). The internal stage label
`P11.6` is retained only in provenance documents; it is not a second public
model name.

## 📊 Evaluation

The tables below reproduce the frozen dataset, case IDs, prompts, and scoring
contracts in [`docs/RESULTS.md`](docs/RESULTS.md). They are existing release
evidence, not newly run benchmarks for this documentation pass.

### General capability

| Benchmark | Qwen donor | DenseK3-4B |
|---|---:|---:|
| MMLU 5-shot (14,042 questions) | **73.10%** (10,265/14,042) | 67.67% (9,502/14,042) |
| WikiText-103 raw test CE | **1.9334** | 2.0821 |
| WikiText-103 raw test PPL | **6.9130** | 8.0211 |

DenseK3-4B retains approximately `67.67/73.10 = 92.6%` of the donor MMLU
accuracy on this contract, while trailing the donor on the reported aggregates.

### Long-context quality

LongBench-v2 is the documented **controlled MC-LL Accuracy** subset with
`context ≤128K` and 298 eligible cases; it is not the official generation
leaderboard metric:

| Model | Accuracy |
|---|---:|
| Qwen donor | **37.25%** (111/298) |
| DenseK3-4B | 30.20% (90/298) |

Reduced RULER uses 13 tasks × 4 cases per context length (52 cases per length):

| Context | Qwen donor | DenseK3-4B |
|---:|---:|---:|
| 4K | **88.46%** | 84.49% |
| 8K | **88.27%** | 76.15% |
| 16K | **86.35%** | 65.96% |
| 32K | **85.03%** | 60.19% |
| 64K | **84.36%** | 53.97% |
| 128K | **78.37%** | 49.87% |

DenseK3-4B does not claim long-context quality superiority.

## 💾 Memory Efficiency

For the sequence-growing attention state, donor-equivalent 8×GQA stores about
32 KiB/token, while the eight MLA latent histories store 8 KiB/token. This is a
derived **75% reduction in persistent sequence-growing attention-state payload
only**; it is not a claim about total VRAM, throughput, or end-to-end memory.

| Context | Donor-equivalent GQA state | DenseK3 latent state |
|---:|---:|---:|
| 128K | ~4 GiB | ~1 GiB |
| 256K | ~8 GiB | ~2 GiB |
| 512K | ~16 GiB | ~4 GiB |

## 🚀 Long-Context Runtime

A separate exact-semantics P10-T runtime probe processed **524,288 tokens**:

| Measurement | Recorded value |
|---|---:|
| Full prefill | PASS |
| Continued autoregressive decode | PASS |
| Peak allocated | `15,762,103,296` bytes (≈14.68 GiB) |
| Peak reserved | `15,994,978,304` bytes |
| Persistent latent cache | `4,294,967,296` bytes (4 GiB) |
| Persistent expanded K/V | 0 |

This is exact-semantics runtime evidence from a separate probe, not 512K task
quality, not a throughput result, and not a total-VRAM claim. Standard quality
evidence stops at 128K. Details are in [`docs/LONG_CONTEXT.md`](docs/LONG_CONTEXT.md).

## ⚡ Quick Start

### Use DenseK3-4B

Download the standalone model to a project/data disk and run deterministic greedy
decoding. No donor checkpoint is required:

```bash
hf download yigu666/DenseK3-4B \
  --local-dir models/DenseK3-4B \
  --repo-type model
```

The exported model uses custom Transformers code and Flash Linear Attention
(FLA). Review the bundled Python files before enabling remote code:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "yigu666/DenseK3-4B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="cuda",
).eval()
inputs = tokenizer(
    "DenseK3 preserves the following invariant:", return_tensors="pt"
).to(model.device)
output = model.generate(**inputs, max_new_tokens=32, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### Development / Reproduction

The verified environment snapshot is [`requirements.txt`](requirements.txt):

```bash
git clone https://github.com/yigu666/DenseK3.git
cd DenseK3
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q tests titan/tests --disable-warnings
```

The staged-tree validation result is `281 passed, 2 skipped in 14.31s`; the two
skips are donor-metadata checks. Read the relevant frozen contract before
starting any P0–P11 runner.

## 🧪 Reproduction

Datasets and donor weights are intentionally not bundled. Download the pinned
evaluation assets with the documented commands, then prepare reduced RULER data:

```bash
python scripts/download_evaluation_assets.py --root "$PWD"
python evaluation/prepare_ruler_data.py
```

The full environment, upstream revisions, P0–P11 entry points, checkpoint
assembly, and inference-only evaluation commands are in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md). Public data remains under
its upstream license; do not upload private teacher responses, keys, caches, or
local server paths.

## 📁 Repository Structure

```text
DenseK3/
├── src/                         # public model and migration implementation
├── titan/                       # recovery/distillation contracts and runners
├── evaluation/                 # frozen evaluation utilities
├── tests/                       # public regression tests
├── docs/                        # architecture, training, results, reproduction
└── release/                     # export tooling, validation, and security evidence
```

The standalone model is published separately as
[`yigu666/DenseK3-4B`](https://huggingface.co/yigu666/DenseK3-4B). The separately
prepared delta payload is an audit artifact over its reproduced parent, not a
second public model product. Donor weights, datasets, optimizer state,
credentials, private caches, and noncanonical checkpoints are not committed to
this GitHub source tree.

## 📚 Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — equations, tensor shapes, and cache invariants.
- [`docs/MIGRATION.md`](docs/MIGRATION.md) — donor-to-DenseK3 mapping and staged gates.
- [`docs/TRAINING.md`](docs/TRAINING.md) — joint recovery and dual-teacher distillation contracts.
- [`docs/LONG_CONTEXT.md`](docs/LONG_CONTEXT.md) — latent-cache accounting and runtime evidence.
- [`docs/RESULTS.md`](docs/RESULTS.md) — frozen evaluation protocol and exact aggregates.
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — environment, data commands, and P0–P11 runners.
- [`docs/RELEASE_SCOPE.md`](docs/RELEASE_SCOPE.md) — public/omitted scope and failure boundaries.
- [`release/RELEASE_VALIDATION.md`](release/RELEASE_VALIDATION.md) — release gates and clean-room HF parity.
- [`release/SECURITY_AUDIT.md`](release/SECURITY_AUDIT.md) — credential, path, cache, and provenance hygiene.
- [`MODEL_CARD.md`](MODEL_CARD.md) — GitHub pointer to the concise bilingual Hub cards.
- [`DOCS_RESTRUCTURE_AUDIT.md`](DOCS_RESTRUCTURE_AUDIT.md) — pre-change documentation audit.
- [`DOCS_RESTRUCTURE_FINAL.md`](DOCS_RESTRUCTURE_FINAL.md) — final responsibilities, wording, and blockers.

## ⚠️ Scope and Limitations

- The public identity is **DenseK3-4B**. `P11.6` is internal provenance only.
- This is a text-only research base model and it trails its Qwen donor on the
  reported capability and long-context aggregates. It is not instruction-tuned.
- The 512K number is runtime feasibility only; standard quality evidence stops
  at 128K. No speed advantage or total-VRAM reduction is claimed.
- A later long-context architecture line was attempted and is disclosed as an
  exploration, but it was not promoted. Its implementation, checkpoints,
  failure logs, and detailed experimental artifacts are intentionally omitted;
  see [`docs/RELEASE_SCOPE.md`](docs/RELEASE_SCOPE.md).
- Custom code and FLA are required. The model is not safety-aligned for
  unrestricted or safety-critical production use; consult the model card for
  safety, bias, and evaluation disclosures.

## 🙏 Acknowledgements

DenseK3 uses and cites Qwen/Qwen3.5, Hugging Face Transformers, Flash Linear
Attention, Kimi K3/Kimi Linear, and Attention Residuals work. See
[`NOTICE`](NOTICE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for
upstream terms and provenance.

## 📖 Citation

```bibtex
@software{densek3_2026,
  title  = {DenseK3: A Qwen3.5-initialized Dense K3-style Language Model},
  author = {yigu666 and DenseK3 contributors},
  year   = {2026},
  url    = {https://github.com/yigu666/DenseK3}
}
```

## 📄 License

Original DenseK3 code is released under the Apache License 2.0; see
[`LICENSE`](LICENSE). Qwen, Kimi, FLA, Transformers, datasets, and all other
third-party materials retain their upstream licenses and terms. Redistribution
of model weights must preserve [`NOTICE`](NOTICE) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
