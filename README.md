# DenseK3

**[English](README_en.md) | [中文](README.md)**

[![许可证：Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)
[![模型：DenseK3-4B](https://img.shields.io/badge/Hugging%20Face-DenseK3--4B-yellow.svg)](https://huggingface.co/qinfu19/DenseK3-4B)

🤗 **模型：** [qinfu19/DenseK3-4B](https://huggingface.co/qinfu19/DenseK3-4B)

**DenseK3 探索预训练知识能否在大规模白盒架构迁移后继续保留。**

我们从 **Qwen3.5-4B-Base** 出发，将其

```text
24 × Gated DeltaNet
 8 × Gated Attention
```

混合骨干迁移为

```text
24 × Kimi Delta Attention (KDA)
 8 × strict-NoPE Multi-head Latent Attention (MLA)
```

并进一步引入 **True Latent Cache、Block AttnRes 和 Dense SiTU-GLU**。

最终公开模型为 **DenseK3-4B**，这是一个拥有 4.23B 参数的纯文本 base language model。

DenseK3 是独立研究项目。它**不是 Moonshot AI / Kimi 的官方发布**

---

## ✨ Highlights

* **Qwen3.5 → Dense K3 白盒迁移**
  将 donor 的 `24 Gated DeltaNet + 8 Gated Attention` 骨干迁移为 **24 KDA + 8 strict-NoPE MLA**，同时保留 Qwen tokenizer、词表、embedding/LM-head 语义、hidden size 和 FFN 宽度。

* **按阶段可审计的迁移**
  KDA、MLA、True Latent Cache、Block AttnRes、Dense SiTU-GLU、联合恢复和能力蒸馏均按阶段引入并验证，而不是合并为一次不透明的重新训练。

* **保留 92.6% 的 MMLU 准确率**
  DenseK3-4B 在 MMLU 5-shot 上达到 **67.67%**，Qwen3.5-4B-Base 为 **73.10%**。

* **True Latent Cache**
  每个 MLA 层只持久化 `[B, T, 512]` latent history。展开后的 K/V 仅临时生成，不会持久化。

* **随序列增长的注意力状态负载降低 75%**
  donor 等价的 GQA 状态约为 **32 KiB/token**，而 DenseK3 的 8 个 MLA latent history 约为 **8 KiB/token**。

* **独立 Hugging Face 发布**
  DenseK3-4B 被导出为自包含的自定义 Transformers 模型，并通过了 clean-room 结构、logit、cache、确定性生成以及保存/加载 parity 验证。

---

## 💡 Why DenseK3?

现代预训练语言模型通过昂贵的大规模预训练获得了大量知识。然而，新架构通常需要从头训练。

DenseK3 研究一个不同的问题：

> **一个已经预训练的模型，能否迁移到显著不同的架构，同时保留其已学习能力中有意义的一部分？**

因此，本项目将模型架构与预训练知识视为部分可分离的对象。

DenseK3从 Qwen3.5-4B-Base 出发，逐步替换其 token mixing、depth mixing、FFN activation 和长上下文状态表示。

研究三个具体问题：

1. 大规模架构迁移后，多少预训练能力能够保留？
2. 为保持转换后模型可用，需要哪些迁移技术？
3. 得到的 KDA/MLA 混合架构会呈现怎样的内存特征？

---

## 🏗️ Architecture

### Overview

| 组件                     |                       Qwen3.5-4B-Base |                     DenseK3-4B |
| ------------------------ | ------------------------------------: | -----------------------------: |
| Decoder 层数             |                                    32 |                             32 |
| Hidden size              |                                 2,560 |                          2,560 |
| FFN intermediate size    |                                 9,216 |                          9,216 |
| Vocabulary               |                               248,320 |                        248,320 |
| Token mixers             | 24 Gated DeltaNet + 8 Gated Attention | **24 KDA + 8 strict-NoPE MLA** |
| Mixer pattern            |                                   3:1 |     `(KDA, KDA, KDA, MLA) × 8` |
| MLA layers（从 0 开始）  |                                     — | `3, 7, 11, 15, 19, 23, 27, 31` |
| Channel mixer            |                                SwiGLU |             **Dense SiTU-GLU** |
| Depth mixing             |                Standard residual path |              **Block AttnRes** |
| Attention sequence state |                      Expanded GQA K/V |         **512-d latent cache** |
| Embedding / LM head      |                Qwen pretrained / tied |           **Preserved / tied** |

32 层骨干遵循固定的 3:1 模式：

```text
KDA → KDA → KDA → MLA
          × 8
```

从概念上看，DenseK3 结合了四个架构维度：

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

详细公式、张量形状、归一化规则和参数映射见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

---

### Kimi Delta Attention (KDA)

DenseK3 将 donor 的 24 个循环 Gated DeltaNet mixer 替换为 Kimi Delta Attention。

发布的 KDA 配置为：

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

规范实现采用 **K3 风格的下界约束 decay** 和 **SiLU output gate**，同时保持迁移模型所需的数值语义。

KDA 提供主要的循环序列建模路径，24 个 KDA 层不会累积逐 token 的 K/V cache。

---

### strict-NoPE MLA

DenseK3 每三个 KDA 层后插入一个 MLA 层：

```text
3, 7, 11, 15, 19, 23, 27, 31
```

发布的 MLA 配置为：

```text
attention heads      = 16
KV latent rank       = 512
QK NoPE head dim     = 256
value head dim       = 256
RoPE component       = disabled
```

MLA 路径是 **strict-NoPE**：MLA attention 内不使用 RoPE 分量。

该迁移不只是权重空间中的低秩近似。DenseK3 保留 donor 的 Q/K 归一化语义，并使用 function-aware calibration，降低 attention 转换造成的功能扰动。

---

### True Latent Cache

对于 8 个 MLA 层中的每一个，持久化序列状态为：

```text
[B, T, 512]
```

其中：

```text
persistent latent      = yes
persistent expanded K  = no
persistent expanded V  = no
```

展开后的 K/V 仅在当前 attention 计算需要时重建。

KDA 循环状态与 AttnRes depth sources 独立于该 MLA 序列 cache。

---

### Block AttnRes

32 个 decoder 层被组织为 8 个四层 depth block。

每个 decoder 层包含两个内容依赖的路由站点：

```text
pre-token-mixer
pre-FFN
```

随后是一个 output routing site：

```text
32 × 2 + 1 = 65 AttnRes sites
```

因此 DenseK3 结合了：

```text
Token mixing:
KDA / MLA

Depth mixing:
Block AttnRes
```

---

### Dense SiTU-GLU

DenseK3 有意保留 dense FFN，而不是采用完整 Kimi K3 LatentMoE 系统。

发布的 FFN 使用：

```text
hidden size        = 2560
intermediate size  = 9216

beta               = 4
linear_beta        = 25
```

Qwen SwiGLU 路径迁移为 **Dense SiTU-GLU**，用于约束 GLU 乘积周围的 activation outlier。

因此这是该模型被称为 **Dense K3-style model**、不是完整 Kimi K3 复现的重要原因。

---

### Preserved Qwen Components

DenseK3 不会重新初始化整个 donor 网络。

迁移保留 Qwen donor 的：

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

同时改变主要的：

```text
Token mixing
Depth mixing
FFN activation structure
Long-context state representation
```

张量级 donor → DenseK3 映射见 [`docs/MIGRATION.md`](docs/MIGRATION.md)。

---

## 🔄 From Qwen3.5 to DenseK3

DenseK3 使用分阶段迁移策略，使主要架构变化能够独立验证。

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

最终恢复路径包括：

* **Qwen exact-token online policy distillation (OPD)**：使用共享 tokenizer 空间；
* **Kimi text-space corrective distillation**：获得额外的纠偏 teacher supervision。

内部 P0–P11 阶段标识仍用于 provenance 与精确复现，但理解和使用公开模型并不需要这些标识。

参见：

* [`docs/MIGRATION.md`](docs/MIGRATION.md)
* [`docs/TRAINING.md`](docs/TRAINING.md)
* [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)

---

## 📊 Evaluation

以下结果使用 [`docs/RESULTS.md`](docs/RESULTS.md) 中记录的冻结数据集、case ID、prompt 和评分契约。

### General Capability

| Benchmark                   |            Qwen3.5-4B-Base |            DenseK3-4B |
| --------------------------- | -------------------------: | --------------------: |
| MMLU 5-shot，57 个 subject  | **73.10%** (10,265/14,042) | 67.67% (9,502/14,042) |
| WikiText-103 raw-test CE ↓  |                 **1.9334** |                2.0821 |
| WikiText-103 raw-test PPL ↓ |                 **6.9130** |                8.0211 |

架构迁移后，DenseK3-4B 保留 donor MMLU 5-shot 准确率的约：

```text
67.67 / 73.10 ≈ 92.6%
```

---

### Long-Context Quality

#### LongBench-v2

DenseK3 在 tokenized context length 不超过 131,072 tokens 的样本上，使用受控的 **MC-LL Accuracy** 指标。

这是一个比较性的 conditional-likelihood 指标，**不是官方 LongBench-v2 generation leaderboard 指标**。

| Model           |       MC-LL Accuracy |
| --------------- | -------------------: |
| Qwen3.5-4B-Base | **37.25%** (111/298) |
| DenseK3-4B      |      30.20% (90/298) |

#### Reduced RULER

发布的 RULER 评测是固定的 reduced suite：

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

---

## 💾 Memory Efficiency

在 two-byte cache representation 下，8 个 attention 层的记账如下：

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

DenseK3 MLA：

```text
8 layers
× 512 latent dim
× 2 bytes
= 8,192 bytes/token
= 8 KiB/token
```

推导得到：

```text
75% reduction
```

即 **随序列增长的持久化 attention-state payload** 降低 75%。

这不表示总 VRAM 或端到端吞吐量的提升。

| Context | Qwen-equivalent GQA | DenseK3 latent state |
| ------: | ------------------: | -------------------: |
|    128K |              ~4 GiB |           **~1 GiB** |
|    256K |              ~8 GiB |           **~2 GiB** |
|    512K |             ~16 GiB |           **~4 GiB** |

---

## 🚀 Long-Context Runtime

一次独立的 exact-semantics runtime probe 在预蒸馏 DenseK3 parent architecture 上处理了完整的 **524,288-token** 上下文，并继续进行自回归解码。

| 测量                            |                         记录值 |
| ------------------------------- | ----------------------------: |
| Full prefill                    |                         PASS |
| Continued autoregressive decode |                         PASS |
| Peak allocated GPU memory       | `15,762,103,296` bytes ≈ **14.68 GiB** |
| Peak reserved GPU memory        |             `15,994,978,304` bytes |
| Persistent latent state         |  `4,294,967,296` bytes = **4 GiB** |
| Persistent expanded K           |                            0 |
| Persistent expanded V           |                            0 |

该 probe 使用 True Latent Cache，无 sliding window，也没有 approximate attention。

它验证了：

> **512K runtime 可行性与 cache 语义**

但没有证明：

> **512K 有效上下文检索或推理质量**

本次发布的标准任务质量评测截至 **128K**。

详见 [`docs/LONG_CONTEXT.md`](docs/LONG_CONTEXT.md)。

---

## ⚡ Quick Start

### 使用 DenseK3-4B

DenseK3-4B 以 standalone Hugging Face checkpoint 发布。推理时不需要 donor 模型。

先安装兼容 CUDA 的 PyTorch，再安装最小运行时依赖：

```bash
python -m pip install \
  "transformers==5.12.1" \
  "accelerate>=1.14" \
  "safetensors>=0.8" \
  "flash-linear-attention==0.5.2"
```

然后加载模型：

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

DenseK3-4B 是 **base causal language model**

由于模型使用自定义 Hub 代码，启用 `trust_remote_code=True` 前请审阅随附实现。需要不可变代码 provenance 时，请固定具体的 Hugging Face revision。

---

### Development / Reproduction

如需进行迁移研究、评测或精确环境复现：

```bash
git clone https://github.com/yigu666/DenseK3.git
cd DenseK3

python -m pip install -r requirements.txt
python -m pip install -e .
```

运行公开回归测试套件：

```bash
python -m pytest -q tests titan/tests --disable-warnings
```

当前 staged public tree 记录：

```text
281 passed
2 skipped
```

其中两个 skip 对应 donor-metadata 检查。

---

## 🧪 Reproduction

GitHub 仓库不包含 donor 权重或数据集。

可使用文档中的工具准备公开评测资源：

```bash
python scripts/download_evaluation_assets.py --root "$PWD"
python evaluation/prepare_ruler_data.py
```

完整环境设置、上游 revision、迁移阶段入口、训练契约、checkpoint 组装和仅推理 benchmark 复现说明见：

[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)

研究阶段的 delta artifact 与组装 provenance 为迁移复现而保留，但它们**不是独立的公开模型产品**。

---

## 📁 Repository Structure

```text
DenseK3/
├── src/                  # 模型与架构迁移实现
├── titan/                # 恢复与蒸馏实现
├── evaluation/           # 冻结评测工具
├── tests/                # 单元、parity 与回归测试
├── docs/                 # 架构、迁移、训练、结果与复现文档
└── release/              # standalone 导出与发布验证证据
```

推荐用户使用的 checkpoint 是：

🤗 **[qinfu19/DenseK3-4B](https://huggingface.co/qinfu19/DenseK3-4B)**

GitHub 源码树不提交 donor 权重、数据集、optimizer state、凭据、私有 cache 或非 canonical research checkpoint。

---

## 📚 Documentation

| 文档                                                               | 说明                                             |
| ------------------------------------------------------------------ | ------------------------------------------------ |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)                     | 架构公式、张量形状与 cache 不变量                |
| [`docs/MIGRATION.md`](docs/MIGRATION.md)                           | Qwen3.5 → DenseK3 参数映射与分阶段迁移           |
| [`docs/TRAINING.md`](docs/TRAINING.md)                             | 联合恢复与能力蒸馏契约                           |
| [`docs/LONG_CONTEXT.md`](docs/LONG_CONTEXT.md)                     | latent-cache 记账与长上下文 runtime 证据          |
| [`docs/RESULTS.md`](docs/RESULTS.md)                               | 冻结评测协议与聚合结果                           |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)               | 环境、数据、runner 与 checkpoint 复现             |
| [`docs/RELEASE_SCOPE.md`](docs/RELEASE_SCOPE.md)                   | 公开范围与研究 artifact 边界                     |
| [`release/RELEASE_VALIDATION.md`](release/RELEASE_VALIDATION.md)   | standalone 导出与 clean-room parity 验证          |
| [`release/SECURITY_AUDIT.md`](release/SECURITY_AUDIT.md)           | 凭据、路径、cache 与 provenance 审计              |

---

## ⚠️ Scope and Limitations

* **DenseK3-4B** 是规范公开模型。`P11.6` 等内部阶段标识仅用于研究 provenance。
* DenseK3-4B 是**纯文本 base causal language model**，不是 instruction-tuned conversational assistant。
* 在报告的聚合能力与长上下文 benchmark 上，DenseK3-4B 略低于 Qwen3.5-4B-Base。
* 512K 结果是 runtime 可行性证据；标准化任务质量证据截至 128K。
* 推导的 75% 内存降低仅适用于随序列增长的持久化 attention-state payload，不代表总 VRAM 降低。
* 模型使用自定义 Transformers 代码，并需要 FLA runtime 依赖。
* DenseK3-4B 尚未进行专门的安全对齐或专门的 bias/toxicity/red-team 评测。

规范版本之后的实验工作将在适当位置单独记录，不属于 DenseK3-4B 发布内容。

---

## 🙏 Acknowledgements

DenseK3 基于以下思想、模型和开源工作构建：

* Qwen / Qwen3.5
* Kimi / Kimi K3 / Kimi Linear
* Attention Residuals
* Flash Linear Attention
* Hugging Face Transformers

DenseK3-4B 是独立研究发布，与 Moonshot AI 无隶属关系，也未得到其认可或背书。

上游许可证、署名和 provenance 见 [`NOTICE`](NOTICE) 与 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

---

## 📖 Citation

如果 DenseK3 对你的研究有帮助，请引用：

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

DenseK3 原创代码以 **Apache License 2.0** 发布。

Qwen、Kimi、Flash Linear Attention、Hugging Face Transformers、datasets 及其他第三方材料继续遵循各自上游许可证与条款。

详见：

* [`LICENSE`](LICENSE)
* [`NOTICE`](NOTICE)
* [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)
