---
model_name: DenseK3-4B
library_name: transformers
pipeline_tag: text-generation
license: apache-2.0
base_model: Qwen/Qwen3.5-4B-Base
tags:
  - densek3
  - kda
  - mla
  - long-context
---

# DenseK3-4B — GitHub model-card pointer

DenseK3-4B is the independent, text-only research model released with this
repository. It is initialized from `Qwen/Qwen3.5-4B-Base`, uses a 24-KDA/8-MLA
hybrid with true latent cache, Block AttnRes, and dense SiTU-GLU, and is not an
official Moonshot AI or Kimi release.

The concise, bilingual model cards for Hub users are:

- [Hugging Face model card — English](https://huggingface.co/qinfu19/DenseK3-4B/blob/main/README.md)
- [Hugging Face model card — 中文](https://huggingface.co/qinfu19/DenseK3-4B/blob/main/README_zh.md)

## Release evidence

- Architecture and deterministic donor mapping: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`docs/MIGRATION.md`](docs/MIGRATION.md).
- Joint recovery and dual-teacher distillation contracts: [`docs/TRAINING.md`](docs/TRAINING.md).
- Frozen benchmark protocol and exact aggregates: [`docs/RESULTS.md`](docs/RESULTS.md).
- Standalone export parity and file hashes: [`release/RELEASE_VALIDATION.md`](release/RELEASE_VALIDATION.md) and the [Hub parity summary](https://huggingface.co/qinfu19/DenseK3-4B/blob/main/PARITY_SUMMARY.json).
- Omitted branches, failed candidates, and disclosure-only long-context exploration: [`docs/RELEASE_SCOPE.md`](docs/RELEASE_SCOPE.md).

The internal provenance label `P11.6` is metadata only; it is not a second public
model name. A separately prepared delta payload is an auditable provenance
artifact over its reproduced parent, whereas the Hugging Face repository is the
standalone model product.

## Limitations

The checkpoint is a base model and trails its donor on the reported aggregate
capability and long-context quality results. It has no dedicated safety, bias,
toxicity, or red-team evaluation. The 512K result is an exact-semantics runtime
probe, not 512K task-quality evidence, a throughput claim, or a total-VRAM claim.
Custom code and Flash Linear Attention are required.

## License

Apache-2.0 for original DenseK3 material. Qwen, Kimi, FLA, Transformers,
datasets, and benchmark assets retain their upstream terms; preserve
[`NOTICE`](NOTICE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
