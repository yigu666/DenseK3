# Third-Party Notices

This file records dependencies and references identified during the public-release
provenance audit. It does not relicense third-party work. The DenseK3 original
source is Apache-2.0; each item below keeps its own upstream terms.

## Qwen3.5-4B-Base

- Upstream: `Qwen/Qwen3.5-4B-Base` at revision
  `1001bb4d826a52d1f399e183466143f4da7b741b`.
- License shown by the upstream model card: Apache License 2.0.
- Role: donor weights, tokenizer, vocabulary, and preserved embedding/LM-head
  initialization. Donor weights and source files are not vendored in this GitHub
  source tree; the standalone model export contains the effective DenseK3 weights
  and must retain the donor's redistribution terms.
- Attribution: retain the upstream model card and license when redistributing
  donor-derived artifacts.

## Transformers

- Upstream: https://github.com/huggingface/transformers
- License: Apache License 2.0 (upstream distribution).
- Role: `PretrainedConfig`, `PreTrainedModel`, tokenizer and AutoClass interfaces.
- Copied/modified files: none. DenseK3 custom Hub files are original release files
  and are not a copy of the Transformers repository.

## PyTorch

- Upstream: https://github.com/pytorch/pytorch
- License: BSD-style PyTorch license (see the installed distribution).
- Role: tensor, module, SDPA, and CUDA runtime.
- Copied/modified files: none.

## flash-linear-attention (FLA)

- Upstream: https://github.com/fla-org/flash-linear-attention
- License: MIT (verify the exact version's distribution before redistribution).
- Role: optional/pinned KDA backend dependency (`flash-linear-attention==0.5.2` in
  the release snapshot).
- Copied/modified files: none; FLA is imported as an installed dependency.

## Kimi Linear / Kimi K3

- Upstream research reference: https://github.com/MoonshotAI/Kimi-Linear
- Upstream terms: consult the repository and paper for the exact code/model terms
  applicable to any material obtained independently.
- Role: architectural and terminology reference for KDA/MLA-style design.
- Copied/modified files: no upstream repository snapshot is included. DenseK3 is
  not an official Moonshot AI or Kimi model.

## Attention Residuals

- Reference: the Attention Residuals research/code publication used by the project.
- License and exact upstream revision: `NEEDS_USER_METADATA`; legal review is
  required if any implementation was copied or adapted outside the audited tree.
- Role: architectural reference. The public Block AttnRes implementation is the
  DenseK3 project implementation; no upstream repository snapshot was found in the
  staged tree.

## Datasets and evaluation assets

MMLU (`cais/mmlu`), WikiText-103 (`Salesforce/wikitext`), LongBench-v2
(`THUDM/LongBench-v2`), FineWeb-Edu, and NVIDIA RULER are downloaded from their
upstream sources by user-run scripts. Dataset contents are not committed here.
Each dataset retains its own license, terms, and citation requirements.

## Legal review status

The technical audit found no wholesale third-party source copy in the staged tree.
Maintainer confirmation of any externally copied/adapted implementation and exact
license/revision is `NEEDS_USER_METADATA`. Resolve that item before making a public
release; this document is not legal advice.

