# Canonical DenseK3-4B Reference

This record freezes the source used for standalone export. The public product name
is **DenseK3-4B**; `P11.6` is retained only as internal canonical provenance.

## Frozen source

- Internal canonical stage: `P11.6`.
- Donor: `Qwen/Qwen3.5-4B-Base`.
- Donor revision: `1001bb4d826a52d1f399e183466143f4da7b741b`.
- Parent: the selected P10-T joint-recovery checkpoint, identified by the parent
  report returned by the frozen loader (the server path is deliberately not
  published).
- Delta: `model-dense-overrides.safetensors`, `attnres.safetensors`, the P11.6
  candidate manifest, and the runtime configuration. File hashes are recorded in
  `canonical_reference.json` and the release model `SHA256SUMS`.

The loader used for this snapshot is
`release/hf_export/reference_loader.py::load_canonical_densek3_reference`. It
delegates to the pre-existing public `evaluation.model_loader.load_p11` without
editing that loader. It performs no optimizer construction, backward pass, or
weight update.

## Frozen architecture

```text
32 decoder layers, hidden 2560, intermediate 9216, vocabulary 248320
24 KDA + 8 strict-NoPE MLA, pattern (KDA,KDA,KDA,MLA) x 8
MLA indices: [3, 7, 11, 15, 19, 23, 27, 31]
KDA: Q=16, K=16, V=32, head_dim=128, conv=4, decay_rank=128
MLA: heads=16, kv_lora_rank=512, qk_nope=256, value=256, RoPE disabled
Block AttnRes: 8 depth blocks, 65 sites
Dense SiTU-GLU: beta=4, linear_beta=25
Embedding/LM head: Qwen initialized and tied
```

## Frozen runtime semantics

- Model weights are loaded in the verified Titan-compatible FP16 path.
- KDA recurrent state and reductions retain the canonical FP32 boundaries and
  V-first layout `[B,32,128,128]`.
- MLA cache stores normalized latents `[B,T,512]` at the eight MLA layers. Persistent
  expanded K and V are absent; temporary K/V are reconstructed for attention.
- AttnRes depth sources are forward-local and never enter the sequence cache.
- SiTU uses the uncapped gate for its sigmoid and `beta=4`, `linear_beta=25`.
- Generation parity is deterministic greedy (`do_sample=False`).

The machine-readable parameter names, shapes, dtypes, effective parameter count,
configuration snapshot, and artifact hashes are emitted by
`freeze_reference.py` into `canonical_reference.json`. Before that capture is run,
any field marked `PENDING_CAPTURE` is not a release claim.

