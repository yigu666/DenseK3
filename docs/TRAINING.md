# Recovery and Distillation

This repository exposes the recovery and distillation implementation used after
deterministic architecture migration. It does not contain datasets, teacher caches,
API responses, optimizer states, or failed checkpoints.

## Recovery objective

Architecture conversion changes the function even when most donor tensors can be
mapped deterministically. Recovery minimizes a weighted combination of frozen
signals such as teacher/student logits, selected hidden or mixer outputs, and
language-model cross-entropy. Exact weights, trainable parameter scopes, schedules,
precision modes, and gates are defined in the versioned YAML contracts and scripts;
the documentation does not override them.

The workflow is fail-closed:

1. materialize or verify the required corpus and donor artifacts;
2. validate the parent checkpoint and tensor inventory;
3. enable only the contract-listed trainable parameters;
4. run the bounded recovery stage with resumable state;
5. evaluate the frozen gate and write a hash-bound manifest;
6. promote only a passing candidate.

## Joint recovery

Joint recovery integrates KDA dynamics, strict-NoPE MLA, Block AttnRes, and dense
SiTU-GLU after their isolated gates. P10 code is public under `titan/scripts` and its
contracts under `titan/configs`. The retained Titan run is reduced-scope FP16
evidence. Its full-development CE was 2.389680047457615, but its formal provenance
remains provisional pending the native BF16/full-scope replay; no stronger claim is
made.

## Dual-teacher distillation

P11 combines:

- **Qwen exact-token OPD:** online probability distillation at exactly aligned
  student tokens, preserving tokenizer semantics;
- **Kimi text-space correction:** corrective responses expressed in text space,
  with explicit schemas, cost calibration, validation, and resumable ingestion.

The public code includes contracts, materialization entry points, calibration,
resume behavior, and launch scripts. Private API responses and keys are not part of
the release. Users provide `MOONSHOT_API_KEY` through their environment if they
choose to reproduce that path and remain responsible for provider and dataset terms.

## Data

No training dataset is committed. Public scripts describe how to materialize the
P5 probe corpus and P11 general corpus; FineWeb-Edu and every external dataset must
be obtained from its upstream source. Keep downloads outside Git history.

## Reproduction boundary

Running recovery is expensive and is not necessary to use the standalone model
release. The release task itself performs no training, backward pass, optimizer
construction, benchmark rerun, or checkpoint reselection. Commands are listed in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md); inspect the frozen contract before any
GPU run.

