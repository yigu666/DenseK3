# DenseK3

**[English](README.md) | [中文](README_zh.md)**

[![许可证：Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE) [![模型：DenseK3-4B](https://img.shields.io/badge/Hugging%20Face-DenseK3--4B-yellow.svg)](https://huggingface.co/yigu666/DenseK3-4B)

🤗 **模型发布：** [yigu666/DenseK3-4B](https://huggingface.co/yigu666/DenseK3-4B)

DenseK3 是一个独立研究项目，探索在保留 Qwen3.5-4B-Base 兼容性与初始化
优势的同时，能否引入 Kimi K3 风格的循环/注意力思想。公开检查点名称为
**DenseK3-4B**。本项目不是 Moonshot AI 或 Kimi 的官方发布，也不声称复现
完整的 Kimi K3 系统。

## ✨ Highlights

- 开源 P0–P11 主线：架构迁移、组件与联合恢复、Qwen 精确 token OPD、
  Kimi 文本空间纠偏蒸馏，以及评测契约均包含在公开源码中。
- 3:1 混合骨干：24 层 Kimi Delta Attention（KDA）与 8 层严格 NoPE
  Multi-head Latent Attention（MLA）。
- 真正的 latent MLA cache：每个 MLA 层只持久化归一化的 `[B,T,512]`
  历史；展开后的 K/V 只在当前操作中临时生成，从不持久化。
- 8 个四层 Block AttnRes 深度块（65 个路由站点），以及 dense
  SiTU-GLU 通道混合（`beta=4`、`linear_beta=25`）。
- 在迁移契约允许的范围内保留 Qwen tokenizer/词表、embedding、绑定的
  LM head、hidden size 与 FFN 宽度。
- 独立 Hugging Face 导出已经通过 clean-room parity 验证，无需 donor
  checkpoint 即可加载。

## 💡 Why DenseK3?

长上下文模型往往需要在循环效率、注意力质量和既有预训练检查点兼容性之间
取舍。DenseK3 将这个取舍显式化：我们对 Qwen donor 做白盒迁移，构建 dense
K3 风格混合架构，再测量哪些能力被保留、哪些能力发生损失。本项目定位为
可审计的研究材料，而不是宣称某个架构选择在所有场景都更优。

## 🏗️ Architecture

### Overview

| 组件 | Qwen3.5-4B-Base donor | DenseK3-4B |
|---|---:|---:|
| Decoder 层数 | 32 | 32 |
| Hidden / FFN 尺寸 | 2,560 / 9,216 | 2,560 / 9,216 |
| 词表 | 248,320 | 248,320（覆盖 Qwen tokenizer ids） |
| Token mixer | 24 Gated DeltaNet + 8 Gated Attention | **24 KDA + 8 strict-NoPE MLA** |
| Mixer 模式 | 3:1 | `(KDA,KDA,KDA,MLA) × 8` |
| MLA 层（从 0 开始） | — | `3, 7, 11, 15, 19, 23, 27, 31` |
| Channel mixer | SwiGLU | Dense SiTU-GLU |
| Depth mixer | Standard residual | Block AttnRes |
| 随序列增长的注意力状态 | 展开的 GQA K/V | 512 维 latent cache |
| Embedding / LM head | Qwen 预训练、绑定 | 保留、绑定 |

完整公式、张量映射、归一化规则和 cache 不变量见
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

### Kimi Delta Attention (KDA)

24 个循环 mixer 使用 Q/K/V head 数 `16/16/32`、head dimension 128、
depthwise causal convolution kernel 4 和 decay projection rank 128。循环
状态为 FP32、V-first，形状为 `[B,32,128,128]`。规范参数化使用 K3 风格的
下界约束 decay 与 SiLU 输出 gate。

### strict-NoPE MLA

8 个 MLA 层使用 16 heads、KV latent rank 512、NoPE query/key dimension
256、value dimension 256、Q/K normalization 和 full-rank sigmoid output
gate。该路径按契约禁用 RoPE；展开后的 K/V 只为当前注意力操作重建。

### True Latent Cache

每个 MLA 层的持久序列状态严格为 `[B,T,512]`，持久化的展开 K/V 为 0。
KDA 循环状态与 AttnRes 深度源独立于 MLA 序列 cache。

### Block AttnRes

32 个 decoder 层组成 8 个四层 block。每层包含一个 pre-mixer 和一个
pre-FFN 路由站点，随后是一个 output 站点，共 `32×2+1=65` 个内容依赖
站点。

### Dense SiTU-GLU

DenseK3 使用 dense FFN，而不是 Kimi K3 的 LatentMoE。发布的 SiTU 激活在
Qwen 规格的 9,216 宽 FFN 周围使用 `beta=4` 与 `linear_beta=25`。

### Preserved Qwen components

保留 donor tokenizer/词表、token embedding、绑定的 LM head、hidden/FFN
尺寸、RMSNorm 参数化和约定的 Q/K normalization 语义。迁移改变了 token
mixing、depth mixing、FFN 激活与长上下文状态表示。确定性映射与门禁见
[`docs/MIGRATION.md`](docs/MIGRATION.md)。

## 🔄 From Qwen3.5 to DenseK3

公开研究路径按阶段组织，使每个架构变量都有独立契约和回归门禁：

```text
Qwen3.5-4B-Base
  → KDA primitive parity 与 Gated DeltaNet-to-KDA transplant
  → 24 KDA + 8 attention hybrid conversion 与 KDA recovery
  → strict-NoPE MLA migration 与 calibration
  → true latent cache
  → Block AttnRes
  → dense SiTU-GLU
  → joint recovery
  → Qwen exact-token OPD + Kimi text-space corrective distillation
  → DenseK3-4B
```

P0–P11 阶段契约、检查点规则、恢复门禁和 teacher 角色公开于
[`docs/MIGRATION.md`](docs/MIGRATION.md)、[`docs/TRAINING.md`](docs/TRAINING.md)
和 [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)。内部阶段标签
`P11.6` 只在 provenance 文档中保留，不是第二个公开模型名称。

## 📊 Evaluation

下表遵循 [`docs/RESULTS.md`](docs/RESULTS.md) 中冻结的数据集、case ID、
prompt 与评分契约，是已有发布证据，并非本次文档整理中新跑的 benchmark。

### General capability

| Benchmark | Qwen donor | DenseK3-4B |
|---|---:|---:|
| MMLU 5-shot（14,042 题） | **73.10%**（10,265/14,042） | 67.67%（9,502/14,042） |
| WikiText-103 raw test CE | **1.9334** | 2.0821 |
| WikiText-103 raw test PPL | **6.9130** | 8.0211 |

在该契约下，DenseK3-4B 的 MMLU 准确率约为 donor 的
`67.67/73.10 = 92.6%`，但报告中的聚合指标仍低于 donor。

### Long-context quality

LongBench-v2 使用文档定义的 **controlled MC-LL Accuracy** 子集，
`context ≤128K`、298 个符合条件的 case；它不是官方 generation leaderboard
指标：

| Model | Accuracy |
|---|---:|
| Qwen donor | **37.25%**（111/298） |
| DenseK3-4B | 30.20%（90/298） |

Reduced RULER 为每个上下文长度 13 个任务 × 4 个 case（每个长度 52 个 case）：

| Context | Qwen donor | DenseK3-4B |
|---:|---:|---:|
| 4K | **88.46%** | 84.49% |
| 8K | **88.27%** | 76.15% |
| 16K | **86.35%** | 65.96% |
| 32K | **85.03%** | 60.19% |
| 64K | **84.36%** | 53.97% |
| 128K | **78.37%** | 49.87% |

DenseK3-4B 不声称长上下文质量优于 donor。

## 💾 Memory Efficiency

对于随序列增长的注意力状态，donor 等价的 8×GQA 约为 32 KiB/token，8
个 MLA latent history 为 8 KiB/token。这只表示持久化、随序列增长的注意力
状态负载减少 **75%（推导值）**，不表示总 VRAM、吞吐量或端到端内存减少。

| Context | Donor-equivalent GQA state | DenseK3 latent state |
|---:|---:|---:|
| 128K | ~4 GiB | ~1 GiB |
| 256K | ~8 GiB | ~2 GiB |
| 512K | ~16 GiB | ~4 GiB |

## 🚀 Long-Context Runtime

一次独立的 exact-semantics P10-T runtime probe 处理了 **524,288 tokens**：

| 测量 | 记录值 |
|---|---:|
| Full prefill | PASS |
| Continued autoregressive decode | PASS |
| Peak allocated | `15,762,103,296` bytes（≈14.68 GiB） |
| Peak reserved | `15,994,978,304` bytes |
| Persistent latent cache | `4,294,967,296` bytes（4 GiB） |
| Persistent expanded K/V | 0 |

这是独立 probe 的 exact-semantics runtime 证据，不是 512K 任务质量、吞吐量
结果或总 VRAM 声明；标准质量证据截至 128K。详见
[`docs/LONG_CONTEXT.md`](docs/LONG_CONTEXT.md)。

## ⚡ Quick Start

### Use DenseK3-4B

将 standalone 模型下载到项目盘/数据盘后进行确定性的 greedy decoding，
不需要 donor checkpoint：

```bash
hf download yigu666/DenseK3-4B \
  --local-dir models/DenseK3-4B \
  --repo-type model
```

导出模型使用自定义 Transformers 代码和 Flash Linear Attention（FLA）。
启用 remote code 前请先审阅随模型提供的 Python 文件：

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

已验证的环境快照为 [`requirements.txt`](requirements.txt)：

```bash
git clone https://github.com/yigu666/DenseK3.git
cd DenseK3
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q tests titan/tests --disable-warnings
```

staged tree 的验证结果为 `281 passed, 2 skipped in 14.31s`；两个 skip 均为
donor metadata 检查。启动任意 P0–P11 runner 前，请先阅读对应的冻结契约。

## 🧪 Reproduction

数据集和 donor 权重不会随仓库打包。请按文档命令下载固定版本的评测资源，
再准备 reduced RULER 数据：

```bash
python scripts/download_evaluation_assets.py --root "$PWD"
python evaluation/prepare_ruler_data.py
```

完整环境、上游 revision、P0–P11 入口、checkpoint 组装和仅推理评测命令见
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)。公开数据遵循上游许可；
不要上传私有 teacher responses、keys、cache 或本地服务器路径。

## 📁 Repository Structure

```text
DenseK3/
├── src/                         # 公开模型与迁移实现
├── titan/                       # 恢复/蒸馏契约与 runner
├── evaluation/                 # 冻结评测工具
├── tests/                       # 公开回归测试
├── docs/                        # 架构、训练、结果与复现文档
└── release/                     # 导出工具、发布验证与安全证据
```

standalone 模型单独发布为
[`yigu666/DenseK3-4B`](https://huggingface.co/yigu666/DenseK3-4B)。另行准备的
delta payload 是相对于复现 parent 的审计材料，不是第二个公开模型产品。本
GitHub 源码树不提交 donor 权重、数据集、optimizer state、凭据、私有 cache 或
非 canonical 检查点。

## 📚 Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — 公式、张量形状与 cache 不变量。
- [`docs/MIGRATION.md`](docs/MIGRATION.md) — donor 到 DenseK3 的映射与阶段门禁。
- [`docs/TRAINING.md`](docs/TRAINING.md) — 联合恢复与双 teacher 蒸馏契约。
- [`docs/LONG_CONTEXT.md`](docs/LONG_CONTEXT.md) — latent-cache 记账与 runtime 证据。
- [`docs/RESULTS.md`](docs/RESULTS.md) — 冻结评测协议与精确聚合结果。
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — 环境、数据命令与 P0–P11 runner。
- [`docs/RELEASE_SCOPE.md`](docs/RELEASE_SCOPE.md) — 公开/省略范围与失败边界。
- [`release/RELEASE_VALIDATION.md`](release/RELEASE_VALIDATION.md) — 发布门禁与 clean-room HF parity。
- [`release/SECURITY_AUDIT.md`](release/SECURITY_AUDIT.md) — 凭据、路径、cache 与 provenance 审计。
- [`MODEL_CARD.md`](MODEL_CARD.md) — 指向双语 Hub model card 的 GitHub 页面。
- [`DOCS_RESTRUCTURE_AUDIT.md`](DOCS_RESTRUCTURE_AUDIT.md) — 文档重构前审计。
- [`DOCS_RESTRUCTURE_FINAL.md`](DOCS_RESTRUCTURE_FINAL.md) — 最终职责、统一措辞与剩余 blocker。

## ⚠️ Scope and Limitations

- 公开身份是 **DenseK3-4B**；`P11.6` 仅表示内部 provenance。
- 这是一个 text-only research base model，在报告的能力和长上下文聚合指标上
  低于 Qwen donor，且不是 instruction-tuned。
- 512K 数字只表示 runtime 可行性；标准质量证据截至 128K。不声称速度优势或
  总 VRAM 减少。
- 我们曾尝试另一条长上下文架构线，并公开披露这一探索，但它没有被提升为发布
  结果。其实现、检查点、失败日志和详细实验材料有意不纳入；详见
  [`docs/RELEASE_SCOPE.md`](docs/RELEASE_SCOPE.md)。
- 运行需要自定义代码和 FLA。本模型没有针对 unrestricted 或 safety-critical
  production 做安全对齐；安全、偏差与评测披露见 model card。

## 🙏 Acknowledgements

DenseK3 使用并引用 Qwen/Qwen3.5、Hugging Face Transformers、Flash Linear
Attention、Kimi K3/Kimi Linear 和 Attention Residuals 工作。上游条款与来源见
[`NOTICE`](NOTICE) 和 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

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

DenseK3 原创代码采用 Apache License 2.0，见 [`LICENSE`](LICENSE)。Qwen、Kimi、
FLA、Transformers、datasets 及其他第三方材料继续遵循各自上游许可。重新分发
模型权重时必须保留 [`NOTICE`](NOTICE) 与
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
