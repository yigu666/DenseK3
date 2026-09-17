# DenseK3

**[English](README.md) | [中文](README_zh.md)**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)
[![Model: DenseK3-4B](https://img.shields.io/badge/Hugging%20Face-DenseK3--4B-yellow.svg)](https://huggingface.co/qinfu19/DenseK3-4B)

🤗 **Model:** [qinfu19/DenseK3-4B](https://huggingface.co/qinfu19/DenseK3-4B)

**DenseK3 explores whether pretrained knowledge can survive a large white-box architecture migration.**

Starting from **Qwen3.5-4B-Base**, we migrate its

```text
24 × Gated DeltaNet
 8 × Gated Attention
```

hybrid backbone into

```text
24 × Kimi Delta Attention (KDA)
 8 × strict-NoPE Multi-head Latent Attention (MLA)
```

and further introduce **True Latent Cache, Block AttnRes, and Dense SiTU-GLU**.

The resulting public model is **DenseK3-4B**, a 4.23B-parameter text-only base language model.

DenseK3 is an independent research project. It is **not an official Moonshot AI / Kimi release**, and it does not claim to reproduce the complete Kimi K3 system.

---

## ✨ Highlights

* **Qwen3.5 → Dense K3 white-box migration**
  Migrates the donor's `24 Gated DeltaNet + 8 Gated Attention` backbone into **24 KDA + 8 strict-NoPE MLA**, while retaining the Qwen tokenizer, vocabulary, embedding/LM-head semantics, hidden size, and FFN width.

* **Stage-wise auditable migration**
  KDA, MLA, True Latent Cache, Block AttnRes, Dense SiTU-GLU, joint recovery, and capability distillation are introduced and validated stage by stage rather than collapsed into one opaque retraining run.

* **92.6% MMLU accuracy retention**
  DenseK3-4B reaches **67.67%** on MMLU 5-shot versus **73.10%** for Qwen3.5-4B-Base.

* **True Latent Cache**
  Each MLA layer persistently stores only a `[B, T, 512]` latent history. Expanded K/V tensors are temporary and are not persisted.

* **75% lower sequence-growing attention-state payload**
  Donor-equivalent GQA state scales at approximately **32 KiB/token**, versus **8 KiB/token** for the eight DenseK3 MLA latent histories.

* **Standalone Hugging Face release**
  DenseK3-4B is exported as a self-contained custom Transformers model and passed clean-room structural, logit, cache, deterministic-generation, and save/load parity validation.

---

## 💡 Why DenseK3?

Modern pretrained language models contain substantial knowledge acquired through expensive large-scale pretraining. New model architectures, however, are often introduced through training from scratch.

DenseK3 studies a different question:

> **Can an already pretrained model be migrated into a substantially different architecture while preserving a meaningful fraction of its learned capability?**

The project therefore treats model architecture and pretrained knowledge as partially separable objects.

Instead of reproducing Kimi K3 from scratch, DenseK3 starts from Qwen3.5-4B-Base and progressively replaces its token-mixing, depth-mixing, FFN activation, and long-context state representation.

The goal is not to claim that DenseK3 is universally better than Qwen. The project investigates three concrete questions:

1. How much pretrained capability survives a large architecture migration?
2. Which migration techniques are required to keep the converted model functional?
3. What memory characteristics emerge from the resulting KDA/MLA hybrid?

---

## 🏗️ Architecture

### Overview

| Component                |                       Qwen3.5-4B-Base |                     DenseK3-4B |
| ------------------------ | ------------------------------------: | -----------------------------: |
| Decoder layers           |                                    32 |                             32 |
| Hidden size              |                                 2,560 |                          2,560 |
| FFN intermediate size    |                                 9,216 |                          9,216 |
| Vocabulary               |                               248,320 |                        248,320 |
| Token mixers             | 24 Gated DeltaNet + 8 Gated Attention | **24 KDA + 8 strict-NoPE MLA** |
| Mixer pattern            |                                   3:1 |     `(KDA, KDA, KDA, MLA) × 8` |
| MLA layers (0-based)     |                                     — | `3, 7, 11, 15, 19, 23, 27, 31` |
| Channel mixer            |                                SwiGLU |             **Dense SiTU-GLU** |
| Depth mixing             |                Standard residual path |              **Block AttnRes** |
| Attention sequence state |                      Expanded GQA K/V |         **512-d latent cache** |
| Embedding / LM head      |                Qwen pretrained / tied |           **Preserved / tied** |

The 32-layer backbone follows a fixed 3:1 pattern:

```text
KDA → KDA → KDA → MLA
          × 8
```

Conceptually, DenseK3 combines four architectural axes:

```text
Token Mixing
    KDA + strict-NoPE MLA

Depth Mixing
    Block AttnRes

Channel Mixing
    Dense SiTU-GLU

Long-Context State
    KDA recurrent state
    + MLA True Latent Cache
```

Detailed equations, tensor shapes, normalization rules, and parameter mappings are documented in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

### Kimi Delta Attention (KDA)

DenseK3 replaces the donor's 24 recurrent Gated DeltaNet mixers with Kimi Delta Attention.

The released KDA configuration uses:

```text
Q heads       = 16
K heads       = 16
V heads       = 32

head dim      = 128
conv kernel   = 4
decay rank    = 128

recurrent state:
[B, 32, 128, 128]
```

The canonical implementation uses **K3-style lower-bounded decay** and a **SiLU output gate**, while preserving the numerical semantics required by the migrated model.

KDA provides the primary recurrent sequence-modeling path without accumulating a token-wise K/V cache for the 24 KDA layers.

---

### strict-NoPE MLA

DenseK3 inserts one MLA layer after every three KDA layers:

```text
3, 7, 11, 15, 19, 23, 27, 31
```

The released MLA configuration uses:

```text
attention heads      = 16
KV latent rank       = 512
QK NoPE head dim     = 256
value head dim       = 256
RoPE component       = disabled
```

The MLA path is **strict-NoPE**: no RoPE component is used inside MLA attention.

The migration is not treated as a weight-space low-rank approximation alone. DenseK3 preserves donor Q/K normalization semantics and uses function-aware calibration to reduce the functional disruption caused by the attention conversion.

---

### True Latent Cache

For each of the eight MLA layers, the persistent sequence state is:

```text
[B, T, 512]
```

with:

```text
persistent latent      = yes
persistent expanded K  = no
persistent expanded V  = no
```

Expanded K/V tensors are reconstructed only when needed for the current attention computation.

KDA recurrent states and AttnRes depth sources are separate from this MLA sequence cache.

---

### Block AttnRes

The 32 decoder layers are organized into eight four-layer depth blocks.

Each decoder layer contains two content-dependent routing sites:

```text
pre-token-mixer
pre-FFN
```

followed by one output routing site:

```text
32 × 2 + 1 = 65 AttnRes sites
```

DenseK3 therefore combines:

```text
Token mixing:
KDA / MLA

Depth mixing:
Block AttnRes
```

---

### Dense SiTU-GLU

DenseK3 intentionally keeps a dense FFN rather than adopting the full Kimi K3 LatentMoE system.

The released FFN uses:

```text
hidden size        = 2560
intermediate size  = 9216

beta               = 4
linear_beta        = 25
```

The Qwen SwiGLU path is migrated to **Dense SiTU-GLU**, which constrains activation outliers around the GLU product.

This is a major reason the model is described as a **Dense K3-style model**, rather than a full Kimi K3 reproduction.

---

### Preserved Qwen Components

DenseK3 does not reinitialize the complete donor network.

The migration preserves the Qwen donor's:

```text
Tokenizer
Vocabulary
Token embedding
Tied LM head
Hidden size
FFN width
RMSNorm parameterization
Contracted Q/K normalization semantics
```

while changing the primary:

```text
Token mixing
Depth mixing
FFN activation structure
Long-context state representation
```

The tensor-level donor → DenseK3 mapping is documented in [`docs/MIGRATION.md`](docs/MIGRATION.md).

---

## 🔄 From Qwen3.5 to DenseK3

DenseK3 uses a staged migration strategy so that major architectural changes can be independently validated.

```text
Qwen3.5-4B-Base
        │
        ▼
KDA primitive parity
        │
        ▼
Gated DeltaNet → KDA transplant
        │
        ▼
24 KDA + 8 Gated Attention
        │
        ▼
KDA dynamics calibration
        │
        ▼
Gated Attention → strict-NoPE MLA
        │
        ▼
True Latent Cache
        │
        ▼
Block AttnRes
        │
        ▼
Dense SiTU-GLU
        │
        ▼
Joint Recovery
        │
        ▼
Capability Distillation
        │
        ▼
DenseK3-4B
```

The final recovery path includes:

* **Qwen exact-token online policy distillation (OPD)** using the shared tokenizer space;
* **Kimi text-space corrective distillation** for additional corrective teacher supervision.

The internal P0–P11 stage identifiers remain available for provenance and exact reproduction, but they are not required to understand or use the public model.

See:

* [`docs/MIGRATION.md`](docs/MIGRATION.md)
* [`docs/TRAINING.md`](docs/TRAINING.md)
* [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)

---

## 📊 Evaluation

The results below use the frozen datasets, case IDs, prompts, and scoring contracts documented in [`docs/RESULTS.md`](docs/RESULTS.md).

### General Capability

| Benchmark                   |            Qwen3.5-4B-Base |            DenseK3-4B |
| --------------------------- | -------------------------: | --------------------: |
| MMLU 5-shot, 57 subjects    | **73.10%** (10,265/14,042) | 67.67% (9,502/14,042) |
| WikiText-103 raw-test CE ↓  |                 **1.9334** |                2.0821 |
| WikiText-103 raw-test PPL ↓ |                 **6.9130** |                8.0211 |

DenseK3-4B retains approximately:

```text
67.67 / 73.10 ≈ 92.6%
```

of the donor's MMLU 5-shot accuracy after the architecture migration.

---

### Long-Context Quality

#### LongBench-v2

DenseK3 uses a controlled **MC-LL Accuracy** metric over examples whose tokenized context length is no greater than 131,072 tokens.

This is a comparative conditional-likelihood metric and **not the official LongBench-v2 generation leaderboard metric**.

| Model           |       MC-LL Accuracy |
| --------------- | -------------------: |
| Qwen3.5-4B-Base | **37.25%** (111/298) |
| DenseK3-4B      |      30.20% (90/298) |

#### Reduced RULER

The released RULER evaluation is a fixed reduced suite:

```text
13 tasks × 4 cases/task
= 52 cases per context length
```

| Context | Qwen3.5-4B-Base | DenseK3-4B |
| ------: | --------------: | ---------: |
|      4K |      **88.46%** |     84.49% |
|      8K |      **88.27%** |     76.15% |
|     16K |      **86.35%** |     65.96% |
|     32K |      **85.03%** |     60.19% |
|     64K |      **84.36%** |     53.97% |
|    128K |      **78.37%** |     49.87% |

This is not the full official RULER leaderboard configuration.

DenseK3-4B does **not** claim general-capability or long-context-quality superiority over its Qwen donor.

---

## 💾 Memory Efficiency

For eight attention layers under a two-byte cache representation:

```text
Qwen-equivalent GQA

8 layers
× 2 (K,V)
× 4 KV heads
× 256 dim
× 2 bytes
= 32,768 bytes/token
= 32 KiB/token
```

DenseK3 MLA:

```text
8 layers
× 512 latent dim
× 2 bytes
= 8,192 bytes/token
= 8 KiB/token
```

This is a derived:

```text
75% reduction
```

in **sequence-growing persistent attention-state payload**.

It is not a claim about total VRAM or end-to-end throughput.

| Context | Qwen-equivalent GQA | DenseK3 latent state |
| ------: | ------------------: | -------------------: |
|    128K |              ~4 GiB |           **~1 GiB** |
|    256K |              ~8 GiB |           **~2 GiB** |
|    512K |             ~16 GiB |           **~4 GiB** |

---

## 🚀 Long-Context Runtime

A separate exact-semantics runtime probe on the pre-distillation DenseK3 parent architecture processed a complete **524,288-token** context and continued autoregressive decoding.

| Measurement                     |                         Recorded value |
| ------------------------------- | -------------------------------------: |
| Full prefill                    |                                   PASS |
| Continued autoregressive decode |                                   PASS |
| Peak allocated GPU memory       | `15,762,103,296` bytes ≈ **14.68 GiB** |
| Peak reserved GPU memory        |                 `15,994,978,304` bytes |
| Persistent latent state         |      `4,294,967,296` bytes = **4 GiB** |
| Persistent expanded K           |                                      0 |
| Persistent expanded V           |                                      0 |

The probe used True Latent Cache with no sliding window and no approximate attention.

This validates:

> **512K runtime feasibility and cache semantics**

It does **not** establish:

> **512K effective-context retrieval or reasoning quality**

Standard public task-quality evaluation in this release stops at **128K**.

See [`docs/LONG_CONTEXT.md`](docs/LONG_CONTEXT.md).

---

## ⚡ Quick Start

### Use DenseK3-4B

DenseK3-4B is released as a standalone Hugging Face checkpoint. The donor model is not required for inference.

Install a CUDA-compatible PyTorch build, followed by the minimal runtime dependencies:

```bash
python -m pip install \
  "transformers==5.12.1" \
  "accelerate>=1.14" \
  "safetensors>=0.8" \
  "flash-linear-attention==0.5.2"
```

Then load the model:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "qinfu19/DenseK3-4B"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="cuda",
).eval()

inputs = tokenizer(
    "DenseK3 preserves the following invariant:",
    return_tensors="pt",
).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=32,
    do_sample=False,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

DenseK3-4B is a **base causal language model**, not an instruction-tuned chat model.

The validated release path is **Hugging Face Transformers with the bundled custom model code**. vLLM, SGLang, Docker Model Runner, and other serving backends are not part of the current validation claim unless explicitly documented otherwise.

Because the model uses custom Hub code, review the bundled implementation before enabling `trust_remote_code=True`. Pin a specific Hugging Face revision when immutable code provenance is required.

---

### Development / Reproduction

For migration research, evaluation, or exact environment reproduction:

```bash
git clone https://github.com/yigu666/DenseK3.git
cd DenseK3

python -m pip install -r requirements.txt
python -m pip install -e .
```

Run the public regression suite:

```bash
python -m pytest -q tests titan/tests --disable-warnings
```

The staged public tree currently records:

```text
281 passed
2 skipped
```

with the two skips corresponding to donor-metadata checks.

---

## 🧪 Reproduction

The GitHub repository does not bundle donor weights or datasets.

Public evaluation assets can be prepared with the documented utilities:

```bash
python scripts/download_evaluation_assets.py --root "$PWD"
python evaluation/prepare_ruler_data.py
```

Exact environment setup, upstream revisions, migration-stage entry points, training contracts, checkpoint assembly, and inference-only benchmark reproduction are documented in:

[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)

Research-stage delta artifacts and assembly provenance are retained for migration reproducibility, but they are **not separate public model products**.

---

## 📁 Repository Structure

```text
DenseK3/
├── src/                  # model and architecture-migration implementation
├── titan/                # recovery and distillation implementation
├── evaluation/           # frozen evaluation utilities
├── tests/                # unit, parity, and regression tests
├── docs/                 # architecture, migration, training, results, reproduction
└── release/              # standalone export and release-validation evidence
```

The recommended user-facing checkpoint is:

🤗 **[qinfu19/DenseK3-4B](https://huggingface.co/qinfu19/DenseK3-4B)**

No donor weights, datasets, optimizer states, credentials, private caches, or noncanonical research checkpoints are committed to the GitHub source tree.

---

## 📚 Documentation

| Document                                                         | Description                                                 |
| ---------------------------------------------------------------- | ----------------------------------------------------------- |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)                   | Architecture equations, tensor shapes, and cache invariants |
| [`docs/MIGRATION.md`](docs/MIGRATION.md)                         | Qwen3.5 → DenseK3 parameter mapping and staged migration    |
| [`docs/TRAINING.md`](docs/TRAINING.md)                           | Joint recovery and capability-distillation contracts        |
| [`docs/LONG_CONTEXT.md`](docs/LONG_CONTEXT.md)                   | Latent-cache accounting and long-context runtime evidence   |
| [`docs/RESULTS.md`](docs/RESULTS.md)                             | Frozen evaluation protocols and aggregate results           |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)             | Environment, data, runners, and checkpoint reproduction     |
| [`docs/RELEASE_SCOPE.md`](docs/RELEASE_SCOPE.md)                 | Public release scope and research-artifact boundaries       |
| [`release/RELEASE_VALIDATION.md`](release/RELEASE_VALIDATION.md) | Standalone export and clean-room parity validation          |
| [`release/SECURITY_AUDIT.md`](release/SECURITY_AUDIT.md)         | Credential, path, cache, and provenance audit               |

---

## ⚠️ Scope and Limitations

* **DenseK3-4B** is the canonical public model. Internal stage identifiers such as `P11.6` are retained only for research provenance.
* DenseK3-4B is a **text-only base causal language model**, not an instruction-tuned conversational assistant.
* DenseK3-4B trails Qwen3.5-4B-Base on the reported aggregate capability and long-context benchmarks.
* The project does not claim a throughput advantage over Qwen.
* The 512K result is runtime feasibility evidence; standardized task-quality evidence currently stops at 128K.
* The derived 75% memory reduction applies only to sequence-growing persistent attention-state payload, not total VRAM.
* The model uses custom Transformers code and requires the FLA runtime dependency.
* DenseK3-4B has not undergone dedicated safety alignment or dedicated bias/toxicity/red-team evaluation.

Post-canonical experimental work is documented separately where appropriate and is not part of the DenseK3-4B release.

---

## 🙏 Acknowledgements

DenseK3 builds on ideas, models, and open-source work from:

* Qwen / Qwen3.5
* Kimi / Kimi K3 / Kimi Linear
* Attention Residuals
* Flash Linear Attention
* Hugging Face Transformers

DenseK3-4B is an independent research release and is not affiliated with or endorsed by Moonshot AI.

See [`NOTICE`](NOTICE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for upstream licenses, attribution, and provenance.

---

## 📖 Citation

If DenseK3 is useful in your research, please cite:

```bibtex
@software{densek3_2026,
  author = {yigu666},
  title  = {DenseK3: A Qwen3.5-Initialized Dense K3-Style Language Model},
  year   = {2026},
  url    = {https://github.com/yigu666/DenseK3}
}
```

---

## 📄 License

Original DenseK3 code is released under the **Apache License 2.0**.

Qwen, Kimi, Flash Linear Attention, Hugging Face Transformers, datasets, and other third-party materials retain their respective upstream licenses and terms.

See:

* [`LICENSE`](LICENSE)
* [`NOTICE`](NOTICE)
* [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)
