"""Export canonical DenseK3 P11.6 as a standalone Hugging Face directory.

This is an inference-only release utility. It materializes the already-selected
effective P11.6 state, adds the separately registered AttnRes parameters, emits
self-contained custom Hub code, and writes safetensors shards. It never constructs
an optimizer, calls backward, trains, or changes weights.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import save_torch_state_dict
from transformers import GenerationConfig


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _clean_chunk(text: str) -> str:
    lines = []
    skip_parenthesized_import = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("from __future__ import"):
            continue
        if skip_parenthesized_import:
            if ")" in line:
                skip_parenthesized_import = False
            continue
        if stripped.startswith("from densek3_core.") or stripped.startswith("from ."):
            if "(" in line and ")" not in line:
                skip_parenthesized_import = True
            continue
        if stripped in {
            "import densek3_core.modeling.modeling_densek3 as modeling_densek3",
            "from densek3_core.modeling import modeling_densek3",
        }:
            continue
        lines.append(line)
    return "\n".join(lines).strip() + "\n"


def _build_modeling_source() -> str:
    chunks: list[str] = [
        '"""Standalone DenseK3 P11.6 custom Transformers runtime."""\n\n'
        "from __future__ import annotations\n\n"
        "from .configuration_densek3 import DenseK3Config\n"
        "P6_MLA_INDICES = (3, 7, 11, 15, 19, 23, 27, 31)\n",
        _clean_chunk(_read("src/densek3_core/kda/contracts.py")),
        _clean_chunk(_read("src/densek3_core/kda/state.py")),
        _clean_chunk(_read("src/densek3_core/kda/reference.py")),
        _clean_chunk(_read("src/densek3_core/kda/fla_backend.py")),
        _clean_chunk(_read("src/densek3_core/modeling/hybrid_cache.py")),
    ]

    latent = _read("titan/scripts/p7_latent_cache.py")
    latent = latent[: latent.index("_ORIGINAL_CACHE_CLASS:")]
    latent = latent.replace(
        "    from densek3_core.modeling.modeling_densek3 import _expanded_nope_attention\n\n",
        "",
    )
    latent = latent.replace("keys, values = zip(*reconstructed, strict=True)", "keys, values = zip(*reconstructed)")
    chunks.append(_clean_chunk(latent))
    chunks.append(
        "\nTitanP7HybridCache.latent_layer_indices = ()\n"
        "DenseK3HybridCache = TitanP7HybridCache\n"
        "_ORIGINAL_MLA_FORWARD: Any = None\n"
    )

    chunks.extend(
        [
            _clean_chunk(_read("titan/attnres/state.py")),
            _clean_chunk(_read("titan/attnres/contracts.py")),
            _clean_chunk(_read("titan/attnres/metrics.py")),
        ]
    )
    attnres = _read("titan/attnres/block_reference.py")
    attnres = attnres.replace(
        "    from densek3_core.modeling.modeling_densek3 import DenseK3ModelOutput\n\n",
        "",
    ).replace(
        "        from densek3_core.modeling import modeling_densek3\n\n"
        "        past_key_values = modeling_densek3.DenseK3HybridCache(backbone.config)",
        "        past_key_values = DenseK3HybridCache(backbone.config)",
    )
    chunks.append(_clean_chunk(attnres))
    chunks.append(_clean_chunk(_read("titan/situ/activation.py")))

    core = _clean_chunk(_read("src/densek3_core/modeling/modeling_densek3.py"))
    chunks.append(core)
    chunks.append(
        r'''

# Canonical P11.6 uses the exact P7 latent-cache and P9 SiTU runtime semantics.
_ORIGINAL_MLA_FORWARD = DenseK3MLA.forward
DenseK3MLA.forward = latent_mla_forward


def _p11_situ_mlp_forward(self: DenseK3MLP, hidden_states: torch.Tensor) -> torch.Tensor:
    gate = self.gate_proj(hidden_states)
    up = self.up_proj(hidden_states)
    return self.down_proj(situ_glu(gate, up, beta=4.0, linear_beta=25.0))


DenseK3MLP.forward = _p11_situ_mlp_forward
_DenseK3ForCausalLMBase = DenseK3ForCausalLM


class DenseK3ForCausalLM(_DenseK3ForCausalLMBase):
    """Canonical P11.6 model with registered AttnRes and exact runtime overlays."""

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: DenseK3Config):
        super().__init__(config)
        self.attnres = BlockAttnResReference(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            block_size_residuals=(
                FORMAL_BLOCK_SIZE_RESIDUALS
                if config.num_hidden_layers == 32
                else max(2, 2 * max(1, config.num_hidden_layers // 4))
            ),
            transformer_layers=config.num_hidden_layers,
            total_routing_sites=config.num_hidden_layers * 2 + 1,
            final_site=config.num_hidden_layers * 2,
            formal_block_count=(
                FORMAL_BLOCK_COUNT if config.num_hidden_layers == 32 else max(1, config.num_hidden_layers // 4)
            ),
        )
        enable_block_attnres_runtime(self, self.attnres)

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Any = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache", True),
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        *,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 16,
        do_sample: bool = False,
        eos_token_id: int | list[int] | None = None,
        **kwargs: Any,
    ) -> torch.LongTensor:
        if do_sample:
            raise ValueError("The standalone release generate() supports deterministic greedy decoding only")
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise ValueError(f"Unsupported generation options: {unsupported}")
        return self.greedy_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )


__all__ = ["DenseK3Config", "DenseK3ForCausalLM", "DenseK3HybridCache"]
'''.strip()
        + "\n"
    )
    return "\n\n".join(chunks)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_callable(spec: str) -> Any:
    module_name, function_name = spec.split(":", 1)
    return getattr(importlib.import_module(module_name), function_name)


def _model_card(namespace: str) -> str:
    model_id = f"{namespace}/DenseK3-4B"
    return f"""---
model_name: DenseK3-4B
library_name: transformers
pipeline_tag: text-generation
license: apache-2.0
base_model: Qwen/Qwen3.5-4B-Base
language:
  - en
  - zh
tags:
  - densek3
  - qwen
  - kda
  - mla
  - long-context
  - custom-code
---

# DenseK3-4B

[GitHub project](https://github.com/yigu666/DenseK3) · [中文说明](README_zh.md)

DenseK3-4B is an independent 4.23B-parameter, text-only causal language model
initialized from `Qwen/Qwen3.5-4B-Base` and migrated to a dense K3-style hybrid
of recurrent and latent-attention layers. It is a research checkpoint, not an
official Moonshot AI or Kimi release, and it is not instruction-tuned.

## 🚀 Quick Start

The export contains custom Transformers code and does not require the donor
checkpoint. Install the tested snapshot (including `flash-linear-attention==0.5.2`)
from the source repository:

```bash
python -m pip install -r https://raw.githubusercontent.com/yigu666/DenseK3/main/requirements.txt
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{model_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto"
).eval()
inputs = tokenizer("DenseK3 preserves the following invariant:", return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=32, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

FLA (`flash-linear-attention`) is required. Review bundled custom code before
enabling `trust_remote_code=True`; the release path is deterministic greedy
decoding.

## 🏗️ Model Details

32 decoder layers retain hidden/FFN sizes 2,560/9,216 and use 24 KDA plus eight
strict-NoPE MLA layers in `(KDA,KDA,KDA,MLA) × 8`. MLA is at zero-based layers
3, 7, 11, 15, 19, 23, 27, and 31. KDA uses Q/K/V heads 16/16/32 with dimension
128; MLA uses 16 heads, KV latent rank 512, and NoPE q/k/value dimensions
256/256. Eight four-layer Block AttnRes blocks and dense SiTU-GLU are retained;
each MLA layer persists `[B,T,512]` and no expanded K/V.

## 🎯 Intended Uses

Architecture-migration, recovery, distillation, cache, and long-context runtime
research. This base checkpoint is not a conversational or production safety model.

## 🔄 Training and Migration

The Qwen donor was migrated through KDA, strict-NoPE MLA, latent cache, Block
AttnRes, and dense SiTU-GLU changes, followed by component/joint recovery and
dual-teacher distillation. Qwen supplied same-tokenizer exact-token OPD targets;
Kimi supplied reviewed text-space corrective trajectories. Export performs no training.

## 📚 Training Data

Verified release artifacts include a 1,024-record general prompt pool across four
domains (416/256/96/256), a train-only FineWeb-Edu probe with 1,998,848 effective
tokens (976×2,048), and provenance references to MMLU, ARC, Belebele, and IFEval.
Qwen teacher outputs and Kimi API corrective text are not redistributed.

**NOT VERIFIED FROM RELEASE ARTIFACTS:** the exact final post-selection training
record count and total token count for the complete distillation mixture.
Evaluation sets are separate and are downloaded from their upstream sources.

## 📊 Evaluation

| Benchmark | Qwen donor | DenseK3-4B |
|---|---:|---:|
| MMLU 5-shot (14,042 questions) | 73.10% (10,265/14,042) | 67.67% (9,502/14,042) |
| WikiText-103 raw test CE / PPL | 1.9334 / 6.9130 | 2.0821 / 8.0211 |
| LongBench-v2 controlled MC-LL, `context ≤128K` | 37.25% (111/298) | 30.20% (90/298) |

Reduced RULER (13 tasks × 4 cases per length) for Qwen/DenseK3-4B is
88.46/84.49 (4K), 88.27/76.15 (8K), 86.35/65.96 (16K), 85.03/60.19 (32K),
84.36/53.97 (64K), and 78.37/49.87 (128K). LongBench is controlled MC-LL, not
the official generation leaderboard metric.

## 💾 Memory and Context

The latent attention-state accounting is about 8 KiB/token versus 32 KiB/token
for donor-equivalent 8×GQA, a derived 75% reduction in that persistent payload
only. A separate exact-semantics runtime probe completed 524,288-token prefill
and continued decode: peak allocated `15,762,103,296` bytes (≈14.68 GiB), peak
reserved `15,994,978,304` bytes, persistent latent `4,294,967,296` bytes (4 GiB),
and persistent expanded K/V 0. This is not 512K task-quality evidence; quality
evidence stops at 128K.

## ✅ Standalone Export Validation

Clean-room standalone parity is **PASS**: structural names/counts, numeric logits,
top-1/top-5 sets, latent-cache behavior, greedy generation, and Safetensors
save/load round-trip all pass. Maximum absolute last-token logit difference is
`0.017578125`. See `PARITY_SUMMARY.json` and the GitHub release validation report.

## ⚠️ Limitations, Safety, and Bias

This is a base, text-only model that trails its donor on reported aggregates. No
dedicated safety, bias, toxicity, or red-team evaluation is available; outputs may
be inaccurate, unsafe, offensive, or biased. Do not use it for safety-critical,
medical, legal, financial, unrestricted production, or 512K quality decisions.
A later long-context architecture line was attempted but not promoted; its
 implementation, checkpoints, failure logs, and detailed artifacts are omitted.
Custom code and FLA are required, and throughput superiority is not claimed.

## 🧬 Provenance and Reproducibility

The public identity is **DenseK3-4B**. The donor is `Qwen/Qwen3.5-4B-Base` at
revision `1001bb4d826a52d1f399e183466143f4da7b741b`. Frozen internal research
provenance for this export is historically tracked as **P11.6**; this is metadata
only, not a second public name. Contracts, protocols, environment, and omitted
scope are maintained at https://github.com/yigu666/DenseK3.

Donor weights, datasets, optimizer state, teacher responses, credentials, and
private server paths are not included.

## 🙏 Acknowledgements

See the GitHub notices for Qwen/Qwen3.5, Transformers, FLA, Kimi K3/Kimi Linear,
Attention Residuals, and benchmark upstream terms.

## 📖 Citation

```bibtex
@software{{densek3_2026,
  author = {{yigu666 and DenseK3 contributors}},
  title = {{DenseK3: A Qwen3.5-initialized Dense K3-style Language Model}},
  year = {{2026}},
  url = {{https://github.com/yigu666/DenseK3}}
}}
```

## 📄 License

Released under Apache-2.0. Qwen, Kimi, FLA, Transformers, datasets, and all other
third-party materials retain their upstream terms; preserve the GitHub notices.
"""


def _model_card_zh(namespace: str) -> str:
    model_id = f"{namespace}/DenseK3-4B"
    return f"""---
model_name: DenseK3-4B
library_name: transformers
pipeline_tag: text-generation
license: apache-2.0
base_model: Qwen/Qwen3.5-4B-Base
language:
  - en
  - zh
tags:
  - densek3
  - qwen
  - kda
  - mla
  - long-context
  - custom-code
---

# DenseK3-4B

[GitHub 项目](https://github.com/yigu666/DenseK3) · [English](README.md)

DenseK3-4B 是一个独立的 4.23B 参数、纯文本 causal language model，从
`Qwen/Qwen3.5-4B-Base` 初始化并迁移为 dense K3 风格的循环与 latent
attention 混合架构。这是研究检查点，不是 Moonshot AI 或 Kimi 官方发布，
也不是 instruction-tuned 模型。

## 🚀 Quick Start

导出目录包含自定义 Transformers 实现，不需要 donor checkpoint。已验证的依赖
快照（包含 `flash-linear-attention==0.5.2`）可直接从源码仓库安装：

```bash
python -m pip install -r https://raw.githubusercontent.com/yigu666/DenseK3/main/requirements.txt
```

启用 custom Hub code 前请审阅 `configuration_densek3.py` 和
`modeling_densek3.py`：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{model_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto"
).eval()
inputs = tokenizer("DenseK3 preserves the following invariant:", return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=32, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

发布路径是确定性的 greedy decoding；采样、微调和未经审阅的自定义代码修改不
属于本发布契约。

## 🏗️ Model Details

32 层 decoder 保留 hidden/FFN 尺寸 2,560/9,216，使用 24 KDA + 8 strict-NoPE
MLA，模式为 `(KDA,KDA,KDA,MLA) × 8`。MLA 位于从 0 开始的
`3, 7, 11, 15, 19, 23, 27, 31` 层。KDA 的 Q/K/V heads 为 `16/16/32`、
head dimension 128；MLA 为 16 heads、KV latent rank 512、NoPE q/k/value
dimensions 256/256。8 个四层 Block AttnRes block 与 dense SiTU-GLU 保留；每个
MLA 层持久化 `[B,T,512]`，不持久化展开 K/V。

## 🎯 Intended Uses

用于架构迁移、恢复、蒸馏、cache 和长上下文 runtime 的可复现研究。本 base
checkpoint 不是对话助手或生产安全模型。

## 🔄 Training and Migration

模型从 Qwen donor 初始化，完成 KDA、strict-NoPE MLA、latent cache、Block
AttnRes 和 dense SiTU-GLU 迁移，随后进行组件/联合恢复与双 teacher 蒸馏。Qwen
提供同 tokenizer 的 exact-token OPD target，Kimi 提供经过审阅的文本空间纠偏轨迹。
导出本身不执行训练。

## 📚 Training Data

发布材料核实了 1,024 条 general prompt pool（四个 domain 计数 416/256/96/256）、
train-only FineWeb-Edu probe（1,998,848 effective tokens，976×2,048），以及
MMLU、ARC、Belebele、IFEval 的来源引用。Qwen teacher 输出和 Kimi API 纠偏文本
不重新分发。

**NOT VERIFIED FROM RELEASE ARTIFACTS：**完整 distillation mixture 的最终
post-selection 训练记录数与总 token 数。评测集与训练数据分开，需从上游下载。

## 📊 Evaluation

| Benchmark | Qwen donor | DenseK3-4B |
|---|---:|---:|
| MMLU 5-shot（14,042 题） | 73.10%（10,265/14,042） | 67.67%（9,502/14,042） |
| WikiText-103 raw test CE / PPL | 1.9334 / 6.9130 | 2.0821 / 8.0211 |
| LongBench-v2 controlled MC-LL，`context ≤128K` | 37.25%（111/298） | 30.20%（90/298） |

LongBench 是 controlled MC-LL Accuracy 子集，不是官方 generation leaderboard。
Reduced RULER（每个长度 13 tasks × 4 cases）的 Qwen/DenseK3-4B 为：4K
88.46/84.49、8K 88.27/76.15、16K 86.35/65.96、32K 85.03/60.19、64K
84.36/53.97、128K 78.37/49.87。

## 💾 Memory and Context

donor 等价 8×GQA 约 32 KiB/token，8 个 MLA latent history 约 8 KiB/token，
这是持久随序列增长 attention-state payload 的推导性 75% 减少，不是总 VRAM 或
吞吐量声明。独立 exact-semantics runtime probe 完成 524,288 tokens prefill 与
continued decode，peak allocated `15,762,103,296` bytes（≈14.68 GiB），peak
reserved `15,994,978,304` bytes，持久 latent `4,294,967,296` bytes（4 GiB），
持久展开 K/V 为 0；这不等于 512K 任务质量验证，标准质量证据截至 128K。

## ✅ Standalone Export Validation

clean-room standalone parity 为 **PASS**：结构名称/数量、数值 logits、top-1/top-5
集合、latent-cache、greedy generation 和 Safetensors save/load round-trip 均通过；
最后 token logits 最大绝对差为 `0.017578125`。详见 `PARITY_SUMMARY.json` 与
GitHub release validation report。

## ⚠️ Limitations, Safety, and Bias

这是 base、纯文本模型，不是 instruction-tuned 或安全对齐模型。报告指标低于
Qwen donor，且没有专门的 safety、bias、toxicity 或 red-team evaluation；输出可能
不准确、不安全、冒犯或带有偏差。512K 结果只表示 runtime 可行性，不是任务质量。
另一条长上下文架构线曾被尝试但未提升为发布结果，其实现、检查点和失败日志均省略。
不要用于安全关键、医疗、法律、金融或 unrestricted production 决策。需要 custom
code 和 FLA，并且不声称吞吐量优势。

## 🧬 Provenance and Reproducibility

公开身份是 **DenseK3-4B**；donor 为 `Qwen/Qwen3.5-4B-Base`，revision
`1001bb4d826a52d1f399e183466143f4da7b741b`。本冻结导出的内部研究 provenance
历史上记为 **P11.6**，该标签只用于 metadata，不是第二个公开名称。契约、协议、
环境与省略范围见 https://github.com/yigu666/DenseK3。

本模型仓库不包含 donor 权重、数据集、optimizer state、teacher 回复、凭据或私有
服务器路径。

## 🙏 Acknowledgements

感谢 Qwen/Qwen3.5、Transformers、FLA、Kimi K3/Kimi Linear、Attention Residuals
和上游 benchmark 作者；请参阅 GitHub notices。

## 📖 Citation

```bibtex
@software{{densek3_2026,
  author = {{yigu666 and DenseK3 contributors}},
  title  = {{DenseK3: A Qwen3.5-initialized Dense K3-style Language Model}},
  year   = {{2026}},
  url    = {{https://github.com/yigu666/DenseK3}}
}}
```

## 📄 License

DenseK3 原创材料采用 Apache-2.0；Qwen、Kimi、FLA、Transformers、datasets 及其他
第三方材料继续遵循上游许可，重新分发权重时请保留 GitHub notices。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--loader", default="evaluation.model_loader:load_p11")
    parser.add_argument("--hf-namespace", default="yigu666")
    parser.add_argument("--max-shard-size", default="4GB")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty export directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    loader = _load_callable(args.loader)
    model, attnres, tokenizer, metadata, _ = loader(args.checkpoint)
    if model.training or attnres.training:
        raise RuntimeError("Canonical loader must return eval-mode modules")
    if any(parameter.requires_grad for parameter in model.parameters()) or any(
        parameter.requires_grad for parameter in attnres.parameters()
    ):
        raise RuntimeError("Canonical model parameters must be frozen")

    config = model.config.to_dict()
    config.update(
        {
            "architectures": ["DenseK3ForCausalLM"],
            "auto_map": {
                "AutoConfig": "configuration_densek3.DenseK3Config",
                "AutoModelForCausalLM": "modeling_densek3.DenseK3ForCausalLM",
            },
            "model_type": "densek3",
            "densek3_public_model_name": "DenseK3-4B",
            "canonical_internal_stage": "P11.6",
            "densek3_runtime": "KDA_MLA_LATENT_CACHE_ATTNRES_SITU",
            "mla_cache_mode": "true_latent",
        }
    )
    for key in ("_name_or_path", "transformers_version"):
        config.pop(key, None)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    generation_config = GenerationConfig.from_model_config(model.config)
    generation_config.do_sample = False
    generation_config.save_pretrained(output)
    tokenizer.save_pretrained(output)

    config_source = _read("src/densek3_core/modeling/configuration_densek3.py")
    config_source = config_source.replace(
        '"AutoConfig": "densek3_core.modeling.configuration_densek3.DenseK3Config",',
        '"AutoConfig": "configuration_densek3.DenseK3Config",',
    ).replace(
        '"AutoModel": "densek3_core.modeling.modeling_densek3.DenseK3Model",',
        "",
    ).replace(
        '"AutoModelForCausalLM": "densek3_core.modeling.modeling_densek3.DenseK3ForCausalLM",',
        '"AutoModelForCausalLM": "modeling_densek3.DenseK3ForCausalLM",',
    )
    (output / "configuration_densek3.py").write_text(config_source, encoding="utf-8")
    (output / "modeling_densek3.py").write_text(_build_modeling_source(), encoding="utf-8")

    state = dict(model.state_dict())
    state.update({f"attnres.{name}": value for name, value in attnres.state_dict().items()})
    save_torch_state_dict(
        state,
        output,
        max_shard_size=args.max_shard_size,
        safe_serialization=True,
        shared_tensors_to_discard=["lm_head.weight"],
        metadata={"format": "pt", "canonical_release": "DenseK3-4B (P11.6)"},
    )

    export_map = {
        name: {
            "source": f"canonical_loader.effective_state_dict::{name}",
            "source_artifacts": [
                "P10-T parent (materialized by frozen loader; identity in canonical_reference.json)",
                "P11.6 model-dense-overrides.safetensors",
                "P11.6 attnres.safetensors",
            ],
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for name, value in sorted(state.items())
    }
    (output / "WEIGHT_EXPORT_MAP.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "public_model_name": "DenseK3-4B",
                "canonical_internal_stage": "P11.6",
                "mapping": export_map,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    shutil.copy2(ROOT / "LICENSE", output / "LICENSE")
    shutil.copy2(ROOT / "NOTICE", output / "NOTICE")
    shutil.copy2(ROOT / "THIRD_PARTY_NOTICES.md", output / "THIRD_PARTY_NOTICES.md")
    (output / "README.md").write_text(_model_card(args.hf_namespace), encoding="utf-8")
    (output / "README_zh.md").write_text(_model_card_zh(args.hf_namespace), encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "model_name": "DenseK3-4B",
        "canonical_internal_stage": "P11.6",
        "base_model": "Qwen/Qwen3.5-4B-Base",
        "base_model_revision": "1001bb4d826a52d1f399e183466143f4da7b741b",
        "format": "standalone_huggingface_safetensors",
        "source_loader": args.loader,
        "source_metadata": metadata,
        "training_performed": False,
        "backward_performed": False,
        "optimizer_constructed": False,
        "weights_modified": False,
        "parity_status": "PENDING_SEPARATE_VALIDATION",
        "export_type": "standalone_huggingface",
        "architecture": "DenseK3ForCausalLM",
    }
    # Remove machine-local paths while retaining non-sensitive provenance fields.
    manifest["source_metadata"].pop("checkpoint", None)
    manifest["source_metadata"].pop("parent", None)
    (output / "RELEASE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    files = sorted(path for path in output.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    (output / "SHA256SUMS").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in files), encoding="utf-8"
    )
    print(json.dumps({"status": "EXPORTED_PENDING_PARITY", "output": str(output), "files": len(files) + 1}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
