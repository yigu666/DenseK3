# Qwen3.5 to DenseK3 Migration

DenseK3-4B was produced by staged white-box architecture migration. Each step
froze its tensor mapping and numerical gates before recovery; later stages did not
silently change earlier contracts. The frozen public state is internally
provenance-tracked as P11.6; P labels below are navigation aids, not public model
names or assumed reader knowledge.

## Migration map

| Technical step | Stage index | Public outcome |
|---|---|---|
| Architecture and tensor contracts | P0-P1 | frozen 32-layer text scope and donor inventory |
| KDA primitive | P2 | recurrent reference semantics, continuation/backward checks, FLA parity |
| Single-layer transplant | P3 | deterministic Gated DeltaNet-to-KDA tensor mapping |
| 24-KDA hybrid | P4 | complete 24 KDA + 8 donor-attention backbone |
| KDA dynamics calibration | P5 | target K3-style decay/gate behavior and recovery |
| MLA migration | P6 | eight strict-NoPE rank-512 MLA layers |
| Latent cache | P7 | true `[B,T,512]` persistent MLA cache |
| Block AttnRes | P8 | eight depth blocks and 65 routing sites |
| Dense SiTU-GLU | P9 | dense SiTU activation with folded recovery scales |
| Joint recovery | P10 | shared recovery of the integrated core |
| Dual-teacher distillation | P11 | Qwen exact-token OPD plus Kimi text correction; P11.6 selected |

## KDA primitive and single-layer transplant

The project first established a PyTorch reference recurrence and compared forward,
state continuation, gradients, and cached decoding with the FLA KDA backend. The
single-layer transplant then mapped donor fused Q/K/V, depthwise convolution, decay,
beta, output-gate, normalization, and output-projection tensors using only explicit
copy/slice/repeat/construct operations. Missing shapes fail closed.

## Full hybrid and KDA dynamics calibration

The transplant was expanded to all 24 donor Gated DeltaNet positions while the eight
full-attention positions remained as controls. Recovery then moved the compatibility
parameterization to the intended K3-style lower-bounded decay and SiLU output gate.
Calibrations targeted function preservation and cached-generation behavior, not
weight similarity alone.

## MLA migration and latent cache

The eight attention positions were replaced one boundary at a time with rank-512
strict-NoPE MLA. Low-rank initialization was followed by output-aware calibration,
Q/K normalization bridging, and scale-out across layers 3, 7, 11, 15, 19, 23, 27,
and 31. The runtime then changed persistent storage from expanded keys/values to one
normalized 512-wide latent per token and MLA layer, with exact temporary expansion.

## AttnRes and SiTU integration

Block AttnRes introduced content-dependent depth reads at 65 sites while keeping
depth state local to a forward pass. SiTU replaced SwiGLU's activation in all dense
FFNs; selected per-channel recovery scales were deterministically folded into the
down projection. Neither step adds MoE experts.

## Joint recovery and distillation

P10 jointly recovered the integrated KDA/MLA/AttnRes/SiTU core. The public Titan
reduced-scope result passed its frozen gate but remains explicitly provisional
(`TITAN_FP16_PROVISIONAL_PENDING_NATIVE_BF16_REPLAY`); it is not relabeled as a
native RTX 5090 BF16/full-scope run.

P11 used two complementary teacher signals:

- Qwen3.5-4B-Base exact-token online probability distillation (OPD), aligned to the
  student's token positions;
- Kimi text-space corrective feedback, handled through explicit API/data contracts
  rather than redistributed private teacher content.

Resume-safe checkpoints and immutable manifests governed selection. P11.6 is the
only canonical release. See [TRAINING.md](TRAINING.md) for objectives and
[RELEASE_SCOPE.md](RELEASE_SCOPE.md) for the public boundary.
