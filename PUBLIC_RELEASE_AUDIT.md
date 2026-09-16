# DenseK3-4B Public-Release Audit

Audit date: 2026-09-16  
Audited tree: `open_source_release/github`  
Release target: GitHub `yigu666/DenseK3` and Hugging Face `qinfu19/DenseK3-4B`

This audit covers the final staged source and standalone model package. It is a
release-readiness audit, not new training, checkpoint selection, or benchmarking.
`PASS` means that the item is already supported by the staged public tree; `ACTION`
means release work is required; `NEEDS_USER_METADATA` means the repository cannot
establish the fact without the maintainer.

## Executive finding

The source, frozen P0-P11 contracts, joint-recovery/distillation code, evaluation
runner, result records, tests, environment snapshot, and standalone model package
are prepared for public release. The public model is DenseK3-4B; P11.6 is internal
provenance. The later long-context architecture exploration is disclosure-only and
its implementation/checkpoints are excluded.

The auditable P11.6 payload remains a delta over a reproduced P10-T parent. The
separate DenseK3-4B Hugging Face directory materializes all effective tensors and
required custom code. It passed fresh-process AutoClass, prediction, greedy,
cache, and save/load parity; both formats are retained for different provenance
and usability purposes.

## 1. Current project tree — PASS

The staged public tree contains the P0-P11 source, contracts, tests, evaluation
code, model-release scripts, and release documents (the exact file list is the
tree itself; generated weights remain in the separate model staging directory):

```text
configs/                 frozen P0-P6 contracts
docs/                    architecture, implementation, results and reproduction docs
evaluation/              capability evaluation and P11.6 assembly loader
results/                 aggregate verified results
scripts/                 migration, recovery and data-materialization entry points
src/densek3_core/        installable Python package
tests/                   public core tests
titan/                   P7-P11 cache, AttnRes, SiTU, joint recovery and distillation
LICENSE, NOTICE          current legal files
MODEL_CARD.md            current delta-checkpoint card
pyproject.toml           package metadata
requirements.txt         observed release-verification environment snapshot
```

Excluded by the release staging rules and `.gitignore`: datasets, donor weights,
generated outputs, caches, private infrastructure, P12/Q4 source and artifacts,
failed/noncanonical checkpoints, logs, credentials, and runtime environments.

## 2. Python package name — PASS

- Distribution name: `densek3-core` (`pyproject.toml`).
- Import package: `densek3_core` (`src/densek3_core`).
- Packaging backend: `setuptools.build_meta`, `src` layout.

## 3. Model and configuration classes — PASS

- `DenseK3Config(PretrainedConfig)`, unique `model_type = "densek3"`.
- `DenseK3PreTrainedModel(PreTrainedModel)` with `config_class = DenseK3Config`.
- `DenseK3Model` and `DenseK3ForCausalLM` are implemented.
- Embedding/LM-head accessors and tied weights are implemented.
- The standalone `release/hf/DenseK3-4B/` package provides Hub-local
  `configuration_densek3.py` and `modeling_densek3.py` with a unique `model_type`
  and valid `auto_map`.
- The constructor registers AttnRes and implements SiTU and true-latent cache
  semantics without user-side runtime patch calls.
- Clean-room `AutoConfig` and `AutoModelForCausalLM` loading passed on the CUDA
  host; see `release/hf_export/validation/PARITY_SUMMARY.json`.

## 4. Checkpoint loader — PASS

`evaluation/model_loader.py::load_p11` validates the three required P11 files,
loads the reproduced P10-T parent, applies `model-dense-overrides.safetensors`,
loads `attnres.safetensors`, enables the P7 latent-cache runtime, freezes all
parameters, and switches both modules to evaluation mode. It constructs no
optimizer and performs no backward pass.

The auditable delta payload is:

```text
model-dense-overrides.safetensors
attnres.safetensors
p11-6-fast-candidate-manifest.json
p11-6-fast-runtime-config.json
```

The separate `release/hf/DenseK3-4B/` directory is a complete
Transformers `save_pretrained`-compatible safetensors export.

## 5. Tokenizer loading — PASS

The assembly loader calls `AutoTokenizer.from_pretrained(TOKENIZER,
local_files_only=True, trust_remote_code=False)`, where the tokenizer is inherited
from the pinned Qwen3.5-4B-Base donor. The standalone export copies the donor
tokenizer files into the model directory and the clean-room gate loads them
locally. The Qwen embedding matrix is padded beyond the tokenizer's highest
assigned id, so coverage rather than exact count equality is the intended
invariant.

## 6. Inference entry — PASS

The verified inference path is `evaluation/model_loader.py::load_p11` followed by
the model forward call. `evaluation/run_capability_benchmarks.py worker` is the
public benchmark worker. The README includes a direct-download standalone
example after the export gate.

## 7. Generation entry — PASS

`DenseK3ForCausalLM.greedy_generate` implements deterministic cached greedy
generation. Canonical P11 runtime scripts also perform greedy decoding. The
standalone class exposes deterministic `.generate(..., do_sample=False)` and the
clean-room gate matches the reference greedy token sequence.

## 8. Evaluation entry — PASS

`evaluation/run_capability_benchmarks.py` exposes verified `prepare`, `worker`,
`orchestrate`, and `summarize` subcommands. `evaluation/prepare_ruler_data.py`
materializes the reduced RULER suite after the pinned upstream repository has been
downloaded. Aggregate evidence is stored in `results/verified-results.json` and
described in `docs/RESULTS.md`.

## 9. Exact dependency requirements — PASS as environment snapshot

`requirements.txt` records the observed `zmm_2` release-verification environment:

```text
torch==2.8.0+cu128
transformers==5.12.1
accelerate==1.14.0
numpy==2.3.5
pyarrow==21.0.0
PyYAML==6.0.3
safetensors==0.8.0
flash-linear-attention==0.5.2
triton==3.4.0
datasets==4.5.0
huggingface-hub==1.29.0
pytest==9.0.2
ruff==0.14.10
```

The PyTorch CUDA 12.8 extra index is included. This is a snapshot, not a bundled
environment and not a portability guarantee for every GPU/platform.

## 10. `requirements.txt` / `pyproject.toml` — PASS

Both files exist. `pyproject.toml` uses compatible lower bounds and requires Python
3.10 or later; `requirements.txt` pins the observed Python 3.13.11 server packages,
including `flash-linear-attention` for the KDA backend.

## 11. Tests — PASS

The staged repository contains the core test suite under `tests/` and P7-P11 tests
under `titan/tests/`. The final staged-tree run on `zmm_2` recorded `281 passed,
2 skipped` for `pytest -q tests titan/tests`; both skips require donor metadata
downloaded separately. The clean-room parity result is recorded in
`release/RELEASE_VALIDATION.md`; these results do not substitute for the frozen
capability evidence in `docs/RESULTS.md`.

## 12. Current public documentation — PASS

Present: `README.md`, `README_zh.md`, `MODEL_CARD.md`,
`DOCS_RESTRUCTURE_AUDIT.md`, `DOCS_RESTRUCTURE_FINAL.md`, `docs/RESULTS.md`,
`docs/REPRODUCIBILITY.md`, `docs/RELEASE_SCOPE.md`,
`release/SECURITY_AUDIT.md`, P1 architecture documents, and P2-P5 implementation
documents. The standalone `DenseK3-4B` directory carries aligned English and
Chinese Hugging Face cards.

The bilingual README, architecture/migration/training/long-context/reproduction
documents, release scope, model card, and validation report are present.

## 13. Absolute local paths — PASS

A recursive text scan of the staged public tree found no Windows drive paths or
private `/home`, `/data`, `/dataB`, or `/mnt` paths. Documentation uses repository-
relative paths and user-defined environment variables. The same scan was repeated
for both GitHub and Hugging Face final trees; tokenizer vocabulary content was
excluded from credential-word matching because it is upstream data.

## 14. API keys / secrets — PASS with documentation-only placeholder

No credential value, private key, Hugging Face token, or API-key assignment was
found. Reproduction documentation contains only the explicit placeholder
`MOONSHOT_API_KEY="<your-key>"`. `titan/secrets/README.md` documents environment-only
secret handling; `titan/secrets/.gitignore` excludes all actual contents. Final
security scans must distinguish these explanatory words from credential values.

## 15. Private usernames / hosts — PASS

No private account name, supplied SSH address, password, or private hostname occurs
in the staged public tree. The release target's public GitHub handle `yigu666` is
not private metadata. The public Hugging Face handle is `qinfu19`.

## 16. Large generated artifacts — PASS

No staged GitHub source file is 1 MiB or larger. Model/data/archive extensions
are kept outside ordinary Git history. The standalone model has three
safetensors shards with a maximum size below 4 GB.

## 17. Third-party copied source — PASS subject to maintainer legal review

No vendored FLA, Kimi, Qwen, Transformers, or dataset repository snapshot is present.
The code imports FLA as an installed dependency and imports Qwen model classes from
Transformers. Kimi K3/Kimi Linear and Attention Residuals are architectural and
research references. The current audit found no upstream copyright/SPDX header in
project Python files and no evidence of a wholesale copied third-party file.

This is a technical provenance audit, not legal advice. If any implementation was
copied or closely adapted outside Git history, its exact source/revision and
modified-file notice are `NEEDS_USER_METADATA` and must be resolved before push.

## 18. Licenses — PASS with maintainer review

- Original repository code: Apache License 2.0 (`LICENSE`).
- Qwen3.5-4B-Base donor: Apache-2.0; weights are not vendored in GitHub.
- flash-linear-attention: MIT; installed dependency, not vendored.
- Transformers/PyTorch and datasets: external dependencies/assets under their own
  upstream terms.
- Kimi K3/Kimi Linear and Attention Residuals: reference attribution is present,
  but upstream license/revision details must be recorded without relicensing their
  work as original DenseK3 code.

`THIRD_PARTY_NOTICES.md`, `NOTICE`, and `CITATION.cff` are present. Model-weight
redistribution must still comply with the donor license and preserve required
notices; maintainer legal provenance review remains an explicit pre-push action.

## 19. Checkpoint format — PASS

The delta tensors and standalone model shards use safetensors. The standalone
directory includes `model.safetensors.index.json` and `SHA256SUMS`; its verified
clean-room export is the release format. The delta tensors retain these hashes:

```text
attnres.safetensors                 1,342,352 bytes
SHA-256 012c322c2bb0c394f10811ea2beceaa85dd8fd0b63f1c23918cd7dbb03780bc3

model-dense-overrides.safetensors  5,624,948,088 bytes
SHA-256 65c4f6d72a791c339785f3407f54f64c33639a01f7012f58ed07d601cab4dfe6
```

The manifest and runtime configuration are also hash-bound in the release staging
area. The delta excludes optimizer, GradScaler, training cache, logs, and datasets.

## 20. Standalone Hugging Face directory — PASS

It was generated without retraining by materializing the canonical DenseK3-4B
effective state (internal provenance P11.6) from the donor/P10-T chain and exporting self-contained custom Hub code plus
complete safetensors shards. All required gates passed:

1. include the canonical KDA/MLA backbone, SiTU, Block AttnRes, and true latent-cache
   runtime semantics without changing floating-point behavior;
2. copy the inherited Qwen tokenizer and special-token configuration;
3. use a Hub-local `auto_map` and unique `model_type`;
4. load in a fresh process and fresh directory with no project source on
   `PYTHONPATH`;
5. pass `AutoConfig`, `AutoModelForCausalLM`, tokenizer, forward, generation, and
   cache smoke tests;
6. match the assembled P11.6 reference on fixed prediction, greedy tokens, cache
   structure, and record the maximum numerical logit difference;
7. emit weight index, `SHA256SUMS`, and a release manifest.

The labels “standalone” and “download and run” are supported by the recorded
clean-room parity result in `release/RELEASE_VALIDATION.md`.

## Metadata still required from the maintainer

- Hugging Face namespace: `qinfu19` (published repository: `qinfu19/DenseK3-4B`).
- Preferred author name(s), ORCID(s), and contact for `CITATION.cff`:
  `NEEDS_USER_METADATA` (the GitHub project handle `yigu666` can be used provisionally).
- Confirmation of any non-obvious copied/adapted third-party implementation and its
  exact upstream revision: `NEEDS_USER_METADATA`.
