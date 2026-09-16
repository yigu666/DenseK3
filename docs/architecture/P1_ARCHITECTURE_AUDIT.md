# DenseK3-Core P1 Architecture Audit

- Status: complete
- Audit date: 2026-08-09
- Donor checkpoint: `Qwen3.5-4B-Base`
- P1 scope: text backbone architecture, KDA/MLA/AttnRes interfaces, weight-transplant feasibility, cache contracts, and downstream architecture decisions
- Out of scope: model implementation, weight conversion execution, training, vision tower recovery, MTP recovery, and kernel optimization

## 1. Executive decisions

P1 locks the following decisions for subsequent phases.

1. **DenseK3-Core is text-only through P10.** The Qwen vision tower and MTP module remain source assets but are excluded from DenseK3 model construction and checkpoint conversion. The tokenizer, tied token embedding, 32-layer text backbone, final norm, and causal LM interface remain in scope.
2. **The P4 model keeps Qwen's 32-layer topology.** Layers `0-2, 4-6, ..., 28-30` become KDA; layers `3, 7, ..., 31` remain Qwen gated GQA. This is exactly 24 KDA layers and 8 global-attention layers in a repeated 3:1 pattern.
3. **DenseK3 layers are implemented independently.** `flash-linear-attention` is an operator backend. Its `KimiDeltaAttention`, `MultiheadLatentAttention`, model classes, configuration classes, and cache classes are references, not DenseK3 implementation dependencies.
4. **DenseK3 uses conventional multiplicative RMSNorm weights.** Qwen's backbone, final, Q-head, and K-head RMSNorm parameters are stored as zero-centered offsets and must be converted as `target_weight = 1 + source_weight`. Qwen's GDN gated output norm already stores a direct scale and must not receive this transform.
5. **P3/P4 use a mathematically compatible KDA bridge.** The scalar GDN decay is embedded into KDA's channel-wise decay, retaining unbounded negative-Softplus decay and the SiLU output gate. This permits an exact recurrence at initialization, subject to backend numerical error.
6. **P5 changes the KDA bridge to the Kimi K3 target.** The recovered target uses lower-bounded decay with `g_min = -5` and a full-rank Sigmoid output gate. These changes are not checkpoint-equivalent and require capability-recovery training.
7. **P6 uses donor-compatible strict-NoPE MLA.** The proposed DenseK3 MLA has 16 heads, `qk_nope_head_dim=256`, `qk_pos_head_dim=0`, `v_head_dim=256`, no Q LoRA, `kv_lora_rank=512`, and a full-rank Sigmoid output gate. This preserves the 4096-wide attention value/output path and targets exactly 75% less per-token global-attention cache than the donor GQA.
8. **P7 must implement a true latent cache.** The current FLA MLA layer caches reconstructed full K/V and explicitly marks compressed caching as a TODO. It also requires `flash-attn`, which is intentionally absent. DenseK3 therefore needs its own reference MLA/cache path before any optional optimized backend.
9. **P8 defines AttnRes block size in Transformer layers, not sublayers.** DenseK3 uses `attnres_block_layers=4`, producing 8 complete depth blocks over 32 Transformer layers plus the embedding source. When calling an API whose block size counts attention and MLP sublayers, the equivalent value is `8`.
10. **P9 uses SiTU-GLU with `beta=4` and `linear_beta=25`.** The Qwen FFN widths and weights are retained, while the activation change is treated as non-equivalent and recovered through training.

## 2. Evidence and provenance

### 2.1 Fixed versions

| Asset | Declared fixed revision | Local verification |
|---|---|---|
| Kimi-Linear | `8c1d85eb6b5f8fcefb15758691b0ce50b0827ce3` | Content audited; no `.git` metadata |
| Kimi-K3 | `3cb39dfd32e51c3328e2e4b4af21341247d06c43` | Content audited; no `.git` metadata |
| Attention-Residuals | `85e22310fe5ee860b4a023de312d791de8a5a5e6` | Content audited; no `.git` metadata |
| flash-linear-attention | `3c4c54ae7397d37130d7101edd0f4eb596af896d` | Content audited; no `.git` metadata |
| FlashKDA | `1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b` | Content present; not selected as a runtime backend |
| Qwen3.5-4B-Base | `1001bb4d826a52d1f399e183466143f4da7b741b` | Metadata/index audited; weights are server-only |
| Kimi-K3 HF reference | `9f62e4e9fffbd0a83ddd60e1c209d828994b3569` | Content audited; no `.git` metadata |
| Kimi-Linear HF reference | `3b171c17bfc4ee348599b6781a2ca8715c21c8dc` | Content audited; no `.git` metadata |

The supplied commit identifiers cannot be recomputed from the local copies because every reference directory lacks `.git`. P1 therefore records a deterministic SHA-256 tree digest for each local asset in `outputs/densek3-core/manifests/architecture/p1-reference-manifest.json`. These digests anchor the exact files used by this audit but do not prove equality to the declared upstream commits.

The local Qwen assets do not contain Transformers source. Qwen behavior was cross-checked against the official `transformers==5.12.1` source:

- <https://github.com/huggingface/transformers/blob/v5.12.1/src/transformers/models/qwen3_5/modeling_qwen3_5.py>
- <https://github.com/huggingface/transformers/blob/v5.12.1/src/transformers/cache_utils.py>

The architecture intent and equations were checked against fixed local reports.
Those reference PDFs are intentionally not bundled in the public tree; their
declared revisions and local tree digests are recorded in the audit manifest.

### 2.2 Sources of truth

When sources disagree, DenseK3 uses this precedence:

1. Mathematical recurrence and architectural intent in the fixed technical reports.
2. Fixed checkpoint configuration and weight index for actual tensor inventory.
3. Fixed HF reference source for checkpoint naming and public forward/cache behavior.
4. Fixed FLA naive operators for numerical reference behavior.
5. Fixed FLA optimized operators for backend calling conventions and supported modes.
6. README prose and diagrams for contextual explanation only.

### 2.3 Reference architecture comparison

| Field | Kimi Linear HF reference | Kimi K3 HF reference | DenseK3 implication |
|---|---:|---:|---|
| Hidden size | 2,304 | 7,168 | Preserve Qwen's 2,560 |
| Transformer layers | 27 | 93 | Preserve Qwen's 32 |
| KDA / global layers | 20 / 7 | 69 / 24 | Use Qwen's 24 / 8 topology |
| Hybrid pattern | 3 KDA + 1 MLA, final MLA | 3 KDA + 1 MLA, final MLA | Qwen already has the same 3:1 positions and a final global layer |
| KDA heads / head dimension | 32 / 128 | 96 / 128 | Preserve donor GVA as 16 QK heads and 32 value heads, all 128-d |
| KDA convolution | 4 | 4 | Preserve 4 |
| KDA decay | Unbounded negative-Softplus | Lower-bounded Sigmoid, `g_min=-5` | Bridge first, K3 gate in P5 |
| KDA output gate | Low-rank Sigmoid | Full-rank Sigmoid | Full-rank donor gate; SiLU bridge then Sigmoid |
| MLA heads | 32 | 96 | Preserve donor 16-head grouping for transplantability |
| MLA Q rank | none | 1,536 | None in DenseK3 P6 |
| MLA KV rank | 512 | 512 | 512 in DenseK3 P6/P7 |
| MLA QK content / auxiliary dims | 128 / 64 | 128 / 64 | DenseK3 uses strict-NoPE 256 / 0 to preserve donor width |
| MLA V head dimension | 128 | 128 | DenseK3 uses 256 to preserve donor output width |
| AttnRes | absent in released Kimi Linear config | 12 Transformer layers/block | 4 Transformer layers/block for 8 blocks |
| FFN activation | SiLU | SiTU with 4 / 25 caps | SiTU introduced in P9 |

DenseK3 is therefore a Qwen-width transplant architecture informed by Kimi K3, not a width-scaled copy of the Kimi K3 checkpoint.

## 3. Qwen3.5-4B-Base donor audit

### 3.1 Text configuration

| Field | Value | DenseK3 treatment |
|---|---:|---|
| Vocabulary size | 248,320 | Preserve |
| Hidden size | 2,560 | Preserve |
| FFN intermediate size | 9,216 | Preserve |
| Transformer layers | 32 | Preserve |
| Linear/global pattern | 3 GDN + 1 gated GQA | Replace GDN with KDA; preserve pattern |
| Linear-attention layers | 24 | KDA target |
| Full-attention layers | 8 | GQA through P5; MLA from P6 |
| RMSNorm epsilon | `1e-6` | Preserve |
| Hidden activation | SiLU | Preserve through P8; SiTU-GLU in P9 |
| Tied embeddings | true | Preserve |
| Max positions | 262,144 | Preserve as an API/config ceiling through P5 |
| Global Q heads | 16 | Preserve through P5 |
| Global KV heads | 4 | Preserve through P5 |
| Global head dimension | 256 | Preserve through P5 |
| Global output gate | true | Preserve; gate is packed in `q_proj` |
| Partial rotary factor | 0.25 | 64 rotated dimensions per 256-d head |
| RoPE theta | 10,000,000 | Preserve through P5 |
| Linear key heads | 16 | KDA Q/K heads |
| Linear value heads | 32 | KDA value/decay/write heads |
| Linear K/V head dimensions | 128 / 128 | Preserve |
| Short convolution kernel | 4 | Preserve |

The exact full-attention layer indices are:

```text
3, 7, 11, 15, 19, 23, 27, 31
```

All other layer indices from `0` through `31` are linear-attention layers.

### 3.2 Residual and FFN structure

Each Qwen decoder layer is PreNorm:

```text
x = x + token_mixer(input_norm(x))
x = x + mlp(post_attention_norm(x))
```

The FFN is a bias-free SwiGLU-style module:

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

All 32 layers use the same FFN width and two per-layer RMSNorms.

### 3.3 Qwen GDN tensor geometry

For batch `B`, sequence length `T`, hidden width `D=2560`, key heads `H=16`, value heads `HV=32`, key dimension `K=128`, and value dimension `V=128`:

| Tensor/module | Shape |
|---|---|
| `in_proj_qkv.weight` | `[8192, 2560]` |
| Projected Q | `[B, T, 16, 128]` |
| Projected K | `[B, T, 16, 128]` |
| Projected V | `[B, T, 32, 128]` |
| `conv1d.weight` | `[8192, 1, 4]` |
| `in_proj_a.weight` | `[32, 2560]` |
| `in_proj_b.weight` | `[32, 2560]` |
| `in_proj_z.weight` | `[4096, 2560]` |
| `A_log` | `[32]` |
| `dt_bias` | `[32]` |
| `norm.weight` | `[128]` |
| `out_proj.weight` | `[2560, 4096]` |
| Recurrent state | `[B, 32, 128, 128]` |
| Fused convolution state | `[B, 8192, 4]` |

The donor recurrence is the scalar-per-value-head Gated DeltaNet rule:

```text
g[h] = -exp(A_log[h]) * softplus(a[h] + dt_bias[h])
beta[h] = sigmoid(b[h])
S = exp(g)[..., None, None] * S
S = S + k[..., None] * ((v - k^T S) * beta)[..., None, :]
o = q^T S
```

Q and K are L2-normalized, and the 16 Q/K heads are repeated to the 32 value heads before the recurrence.

### 3.4 Qwen gated GQA geometry

| Tensor/module | Shape | Note |
|---|---|---|
| `q_proj.weight` | `[8192, 2560]` | Per head, the last dimension is packed as 256 Q + 256 output-gate channels |
| `k_proj.weight` | `[1024, 2560]` | 4 KV heads x 256 |
| `v_proj.weight` | `[1024, 2560]` | 4 KV heads x 256 |
| `o_proj.weight` | `[2560, 4096]` | 16 heads x 256 input |
| `q_norm.weight` | `[256]` | Stored as zero-centered offset |
| `k_norm.weight` | `[256]` | Stored as zero-centered offset |
| K cache per layer | `[B, 4, T, 256]` | BF16 in the validated donor path |
| V cache per layer | `[B, 4, T, 256]` | BF16 in the validated donor path |

The attention output is multiplied by `sigmoid(gate)` before `o_proj`.

### 3.5 Weight inventory

The checkpoint index contains 738 tensor entries:

| Category | Tensor entries | P4 classification |
|---|---:|---|
| Text embedding + final norm | 2 | In scope |
| Linear-attention tensors | 216 | In scope; converted to KDA |
| Full-attention tensors | 48 | In scope; retained through P5 |
| FFN tensors | 96 | In scope; retained |
| Per-layer backbone norms | 64 | In scope; offset-to-scale transform |
| Vision tower/projector | 297 | Out of scope through P10 |
| MTP | 15 | Out of scope through P10 |
| Unknown | 0 | None |

All 738 source entries therefore have an explicit disposition. The 426 text-backbone entries are retained or transformed; 312 vision/MTP entries are intentionally excluded.

### 3.6 Parameter counts inferred from configuration and source

| Component | Parameters |
|---|---:|
| One donor GDN mixer | 42,139,840 |
| One gated GQA mixer | 36,700,672 |
| One FFN | 70,778,880 |
| One pair of layer norms | 5,120 |
| Text backbone including tied embedding and final norm | 4,205,751,296 |
| One DenseK3 KDA bridge mixer | 42,913,952 |
| P4 text backbone | 4,224,329,984 |

The P4 increase is 774,112 parameters per converted layer, or 18,578,688 parameters across 24 KDA layers. It comes from expanding scalar decay logits and biases into a trainable low-rank channel-wise gate.

## 4. KDA audit

### 4.1 Mathematical contract

KDA replaces GDN's scalar retention with a per-key-channel retention vector. In logical `[K,V]` state layout:

```text
S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_(t-1) + beta_t k_t v_t^T
o_t = S_t^T q_t
```

Equivalently in the recurrence used by the naive FLA reference:

```text
S = exp(g)[..., None] * S
delta = (v - k^T S) * beta
S = S + k * delta^T
o = q^T S
```

The distinguishing KDA tensor is `g: [B,T,HV,K]`; GDN uses `g: [B,T,HV]`.

### 4.2 Kimi Linear versus Kimi K3 gate

| Property | Kimi Linear / P4 bridge | Kimi K3 / P5 target |
|---|---|---|
| Decay logits | Low-rank, per channel | Low-rank, per channel |
| Log-decay mapping | `-exp(A) * softplus(z + bias)` | `g_min * sigmoid(exp(A) * (z + bias))` |
| Range | `(-inf, 0)` | `(g_min, 0)`, with `g_min=-5` |
| `A_log` initialization | log-uniform equivalent to 1..16 | zero |
| Output gate | Low-rank Sigmoid in Kimi Linear | Full-rank Sigmoid in Kimi K3 |
| DenseK3 P4 compatibility | Full-rank SiLU copied from Qwen | Not used |

The P5 gate change is architectural rather than a kernel-only optimization. The lower-bounded formula must be visible in configuration, checkpoint metadata, and reference tests.

### 4.3 DenseK3 KDA dimensions

DenseK3 uses FLA's supported Grouped Value Attention geometry:

```text
H  = 16 Q/K heads
HV = 32 value/decay/write heads
K  = 128
V  = 128
G  = HV / H = 2
```

The recurrent state remains fixed-size at `[B,32,128,128]`. Q/K are repeated by `G=2` inside the mathematical operator, not materialized as checkpoint weights.

### 4.4 Why the FLA layer cannot be reused wholesale

The fixed FLA `KimiDeltaAttention` is valuable as an interface reference and already supports GVA, padding removal, chunk/recurrent dispatch, and KDA cache state. It is not the DenseK3 layer for three reasons:

1. Its output gate is a low-rank two-layer projection with a bias; DenseK3 needs a full-rank bias-free projection to copy Qwen and match Kimi K3.
2. Its default output-gate activation is Sigmoid; the exact transplant bridge needs SiLU until P5.
3. DenseK3 needs an explicit two-stage decay parameterization (`unbounded_softplus` then `lower_bounded_sigmoid`) and its own stable checkpoint names.

DenseK3 will call `chunk_kda` and `fused_recurrent_kda` through a small backend adapter.

## 5. MLA audit

### 5.1 Reference behavior

The Kimi HF reference MLA:

- optionally compresses Q through `q_lora_rank`;
- compresses the K/V source through `kv_lora_rank`;
- reconstructs per-head content K and V through an up-projection;
- applies no explicit positional encoding when `mla_use_nope=true`;
- supports a full-rank output gate in Kimi K3;
- currently caches reconstructed full K/V in the provided HF reference.

The fixed FLA MLA similarly reconstructs full K/V and contains an explicit TODO to cache only compressed K/V plus the positional component. It also raises at construction when `flash-attn` is unavailable. It is therefore a useful formula/reference implementation but not a viable P6/P7 runtime layer in the current environment.

### 5.2 DenseK3 P6 target

| Field | Value | Reason |
|---|---:|---|
| Heads | 16 | Preserves donor query-head partition |
| Q/K NoPE dimension | 256 | Preserves donor Q projection width |
| Positional Q/K dimension | 0 | Strict NoPE and no positional cache |
| V head dimension | 256 | Preserves 4096-wide value/output path |
| Q LoRA rank | none | Direct donor Q initialization |
| KV LoRA rank | 512 | Kimi-compatible latent size and 75% cache reduction target |
| Output gate | Full-rank Sigmoid | Kimi K3 target; initialized from donor packed gate |
| Output projection | `[2560,4096]` | Direct donor copy |

The K/V low-rank factorization cannot be exact at rank 512 in general. P6 conversion uses a deterministic truncated SVD of the repeated donor K/V operator and records reconstruction error. P6 is therefore a recovery phase, not a parity-only rewrite.

### 5.3 Cache implication

The donor's 8 GQA layers cache 4096 BF16 bytes per token per layer, or 32 KiB per token across the model. At one million tokens and batch size 1 this is approximately 29.8 GiB.

Strict-NoPE MLA with `kv_lora_rank=512` caches 1024 BF16 bytes per token per layer, or 8 KiB across 8 layers. At one million tokens this is approximately 7.63 GiB, exactly 75% less than the donor global-attention cache. This reduction is only realized after P7 implements latent caching; reconstruct-then-cache MLA does not meet the requirement.

## 6. AttnRes audit

### 6.1 Operator contract

For residual sources `v_i: [...,D]`, a learned per-sublayer pseudo-query `w: [D]`, and an RMSNorm scale `r: [D]`:

```text
k_i = RMSNorm(v_i; r)
p_i = softmax_i(w^T k_i)
output = sum_i p_i * v_i
```

The fixed FLA operator performs the scoring and mixture in FP32, returns the residual dtype, supports an optionally fused output RMSNorm, and tests both forward and backward against a naive reference.

### 6.2 Block semantics

The Attention Residuals report defines each attention mixer and each MLP as a residual sublayer. Block AttnRes sums outputs within a block and applies depth attention over completed block representations, the embedding, and the current partial block.

There is a naming mismatch in the references:

- Kimi K3 checkpoint/config uses `attn_res_block_size=12` to mean 12 Transformer layers per block.
- FLA's current model config uses `attnres_block_size` to count attention and MLP sublayers, so `12` means 6 Transformer layers.

DenseK3 avoids this ambiguity by exposing `attnres_block_layers`. Its value is 4 for the 32-layer model, yielding 8 complete blocks and mirroring Kimi K3's target of roughly eight depth blocks. Backend/model code that counts sublayers receives `2 * attnres_block_layers = 8`.

### 6.3 Initialization

Pseudo-query projections are zero-initialized, making depth-attention logits equal and the initial mixture uniform over available sources. AttnRes RMSNorm scales use standard multiplicative-one initialization. These are new P8 parameters and have no Qwen donor tensors.

Uniform depth mixing is not equivalent to Qwen's standard additive residual stream. P8 therefore requires its own recovery stage and must not claim zero-shot parity merely because the pseudo-queries start at zero.

## 7. SiTU-GLU audit

Kimi K3 replaces the unbounded SwiGLU response with:

```text
gate = beta * tanh(gate_input / beta) * sigmoid(gate_input)
up   = linear_beta * tanh(up_input / linear_beta)
out  = gate * up
```

DenseK3 uses `beta=4` and `linear_beta=25`, matching the fixed Kimi K3 configuration. The three Qwen FFN matrices retain their shapes and are copied directly, but the activation change is not functionally equivalent and must be recovered in P9.

## 8. Milestone architecture states

| Milestone | Linear layers | Global layers | Residual | FFN activation | Position handling |
|---|---|---|---|---|---|
| Donor baseline | 24 GDN | 8 gated GQA | Standard PreNorm | SiLU GLU | Partial MRoPE on GQA |
| P4 bridge | 24 compatible KDA | 8 gated GQA | Standard PreNorm | SiLU GLU | Partial MRoPE on GQA |
| P5 recovered KDA | 24 lower-bounded KDA | 8 gated GQA | Standard PreNorm | SiLU GLU | Partial MRoPE on GQA |
| P6 | 24 lower-bounded KDA | 8 NoPE MLA | Standard PreNorm | SiLU GLU | No explicit position in MLA |
| P7 | Same as P6 | True latent cache | Standard PreNorm | SiLU GLU | Same as P6 |
| P8 | Same as P7 | Same as P7 | Block AttnRes, 4 Transformer layers/block | SiLU GLU | Same as P6 |
| P9/P10 | Same as P8 | Same as P8 | Block AttnRes | SiTU-GLU | Same as P6 |

## 9. Risks and explicit non-blockers

| Risk | Impact | Required response |
|---|---|---|
| Local references lack Git metadata | Cannot prove local/upstream commit identity | Preserve tree hashes; restore Git provenance before publication if required |
| Exact Transformers 5.12.1 Qwen source is not stored locally | Offline reproducibility gap | Vendor only the necessary interface observations into tests/docs; do not copy source into `src/` |
| Server host alias `zmm_2` is not resolvable from this workstation | P1 cannot inspect server files directly | Execute P2 GPU tests through the user's established server channel |
| Local Qwen weights are intentionally absent | P1 cannot calculate empirical tensor statistics or logits | Use server-only conversion/audit scripts in P3; keep local tests synthetic |
| FLA MLA requires absent `flash-attn` and caches full K/V | P6/P7 cannot reuse it as-is | Implement PyTorch/SDPA reference and latent-cache contract independently |
| KDA bridge and K3 gate differ | P5 loss spike is possible | Introduce changes under explicit config flags and run staged recovery |
| MLA rank-512 factorization is approximate | P6 loss spike is expected | Record SVD reconstruction error and recover with training |
| Training/evaluation datasets and budgets are not yet fixed | Capability thresholds cannot be numerically finalized | Must be decided before P5; does not block P2 implementation |

The absence of `flash-attn`, `causal-conv1d`, FlashKDA, DeepSpeed, xFormers, and wandb remains a non-blocker for P2-P5. Qwen's PyTorch convolution fallback and FLA KDA kernels already cover correctness work.

## 10. P1 exit gate

- [x] Qwen layer topology and dimensions are fixed.
- [x] All 738 donor tensor entries have an explicit disposition.
- [x] KDA mathematical, tensor, padding, and cache contracts are identified.
- [x] Exact-versus-approximate transplant boundaries are identified.
- [x] P4, P5, P6, P7, P8, and P9 architecture transitions are explicit.
- [x] AttnRes block-size ambiguity is resolved.
- [x] FLA whole-layer reuse limitations are documented.
- [x] Reference assets are anchored with local tree hashes.
- [x] Numerical and integration gates are specified in `P1_CONTRACTS_AND_GATES.md`.

P1 is complete. P2 may begin with an independent PyTorch KDA recurrence, followed by FLA parity and cache/padding coverage.
