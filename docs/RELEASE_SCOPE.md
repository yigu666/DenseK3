# Release Scope

## Canonical release

DenseK3-4B is the only public canonical model. It is the outcome of the P0-P11
staged migration, joint recovery, and dual-teacher distillation pipeline; `P11.6`
is retained as internal provenance. No later branch, quantized derivative, or
failed candidate supersedes it.

## Public source and evidence

- P0-P11 architecture contracts and staged implementation;
- KDA reference/parity and deterministic Gated DeltaNet-to-KDA transplant;
- 24-KDA hybrid conversion, KDA recovery, and strict-NoPE MLA migration;
- true latent cache, Block AttnRes, and dense SiTU-GLU;
- P10 joint recovery and P11 Qwen OPD/Kimi corrective distillation code;
- environment package snapshot and dataset download/materialization commands;
- evaluation runner, aggregate results, raw-evidence hashes, and regression tests;
- the delta files associated with internal stage P11.6 and, only after parity
  validation, the standalone HF export.

## Disclosure-only research branch

After P11.6, a separate long-context architecture line was explored. A P10-T
runtime probe demonstrated exact-semantics 512K prefill/decode feasibility with true
latent cache, but that later line was not promoted. DenseK3-4B therefore remains
canonical.

This disclosure documents that long-context exploration occurred. It does not claim
512K effective-context quality for DenseK3-4B and does not expose the unpromoted
implementation or artifacts.

## Intentionally excluded

- unpromoted long-context source, configurations, detailed plans, checkpoints,
  intermediate outputs, and failure logs;
- historical failed implementations and any checkpoint whose final gate is FAIL,
  BLOCKED, PENDING, UNKNOWN, or NEEDS_SOURCE;
- datasets, generated caches, teacher/API content, credentials, W&B state, and
  private infrastructure details;
- conda/runtime environments, vendored dependency trees, and third-party repository
  snapshots;
- optimizer, GradScaler, and training-resume state;
- model checkpoints other than the canonical DenseK3-4B release payload.

These artifacts are excluded to keep the release scientifically unambiguous,
license-auditable, free of private data, and focused on the reproducible successful
path rather than internal bookkeeping.
