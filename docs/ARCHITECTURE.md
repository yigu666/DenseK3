# DenseK3-4B Architecture

This document describes the canonical DenseK3-4B text model without assuming
knowledge of the internal P0-P11 project stages. The frozen state is internally
provenance-tracked as P11.6; that marker is not a public model name. DenseK3 is a
white-box migration of Qwen3.5-4B-Base into a dense K3-style architecture; it is
not a from-scratch K3 pretraining run and does not reproduce Kimi K3's LatentMoE.

## End-to-end data path

For token ids `x in N^[B,T]`, the model applies:

```text
token embedding [248320, 2560]
  -> 32 decoder layers in the fixed (KDA, KDA, KDA, MLA) x 8 topology
       -> pre-mixer RMSNorm -> KDA or strict-NoPE MLA
       -> Block AttnRes depth write/read
       -> pre-FFN RMSNorm -> dense SiTU-GLU
       -> Block AttnRes depth write/read
  -> final Block AttnRes read
  -> final RMSNorm
  -> tied LM head [248320, 2560]
  -> logits [B,T,248320]
```

`B` is batch size, `T` sequence length, `D=2560` hidden width, and `I=9216`
feed-forward intermediate width. The embedding and language-model head share the
same Qwen-initialized parameter tensor.

## Fixed hybrid topology

The 32 layers contain 24 Kimi Delta Attention (KDA) mixers and eight Multi-head
Latent Attention (MLA) mixers:

```text
KDA -> KDA -> KDA -> MLA, repeated eight times
MLA layer indices: 3, 7, 11, 15, 19, 23, 27, 31 (zero based)
```

Each layer remains pre-normalized. Block AttnRes changes how residual sources are
read across depth, while KDA/MLA mix information across token positions.

## Kimi Delta Attention

KDA is a recurrent linear-attention mixer. It has 16 query heads, 16 key heads,
32 value heads, 128 dimensions per head, causal depthwise convolution kernel four,
and a rank-128 decay projection. For normalized input `X in R^[B,T,2560]`:

```text
Q = SiLU(DWConv4(X Wq))       [B,T,16,128]
K = SiLU(DWConv4(X Wk))       [B,T,16,128]
V = SiLU(DWConv4(X Wv))       [B,T,32,128]
F = (X Wfa) Wfb               [B,T,32,128]
beta = sigmoid(X Wbeta)       [B,T,32]
```

The K3-style target parameterization converts `F`, `A_log`, and `dt_bias` into a
lower-bounded forget/decay factor. The exact recurrence is implemented by the
reference operator and checked against the installed FLA backend. Conceptually,
for each token and value head, a V-first state `S` is updated by a gated delta rule:

```text
S_t = decay_t * S_(t-1) + beta_t * correction(K_t, V_t, S_(t-1))
Y_t = Q_t @ S_t
```

The persistent recurrent state is FP32 with shape `[B,32,128,128]`; three causal
convolution tails are also cached, each with `kernel_size - 1 = 3` positions.
`V-first` means the state axes are value-head/value-channel before key-channel,
matching the frozen operator contract.

Important parameter shapes (`out_features x in_features`) are:

| Parameter | Qwen donor Gated DeltaNet | DenseK3 KDA |
|---|---:|---:|
| fused/projected Q/K/V source | fused donor tensors | split `Wq,Wk [2048,2560]`, `Wv [4096,2560]` |
| depthwise Q/K convolution | donor fused convolution slices | `[2048,1,4]` each |
| depthwise V convolution | donor fused convolution slice | `[4096,1,4]` |
| decay projection A | donor low-rank factors | `Wfa [128,2560]` |
| decay projection B | donor low-rank factors | `Wfb [4096,128]` |
| beta projection | donor beta path | `[32,2560]` |
| output gate | donor output gate | `[4096,2560]` |
| output projection | donor output projection | `[2560,4096]` |

The deterministic transplant admits only copy, slice, repeat, and explicitly
recorded construction operations. It does not silently initialize missing donor
weights.

## Strict-NoPE Multi-head Latent Attention

MLA compresses key/value history into a rank-512 latent representation. `NoPE`
means that no rotary or other positional component is applied to query or key in
these eight layers; causality still comes from the attention mask.

For `X in R^[B,T,2560]` and 16 heads:

```text
Q = RMSNorm(X Wq)                    [B,T,16,256]
C = RMSNorm(X Wa)                    [B,T,512]
[K,V] = reshape(C Wb)                [B,T,16,256] each
G = sigmoid(X Wg)                    [B,T,4096]
O = Wo(Attention(Q,K,V) * G)         [B,T,2560]
```

Parameter shapes are:

| Parameter | Qwen gated-attention donor | DenseK3 strict-NoPE MLA |
|---|---:|---:|
| query projection | gated Q projection, donor-specific packing | `Wq [4096,2560]` |
| query output gate | donor query gate | `Wg [4096,2560]` |
| key/value source | full-rank K/V projections | `Wa [512,2560]` |
| latent expansion | not present | `Wb [8192,512]` |
| output projection | `[2560,4096]` | `[2560,4096]` |

The migration uses function-aware calibration rather than treating low-rank weight
reconstruction error as a sufficient proxy for model behavior. Q/K RMSNorm is kept
in the selected production contract. `q_lora_rank` is absent, positional head width
is zero, and `qk_nope_head_dim = value_head_dim = 256`.

## True latent cache

During cached inference, every MLA layer persistently stores only normalized latent
states:

```text
C_cache: [B,T,512]
persistent expanded K: zero
persistent expanded V: zero
```

K and V are reconstructed temporarily from `C_cache` for the attention computation;
they are not appended to persistent cache objects. At FP16/BF16 element width, the
eight MLA layers therefore grow by `8 * 512 * 2 = 8192` bytes per token (8 KiB),
versus about 32 KiB/token for the eight donor GQA key/value caches. This is a derived
75% reduction in sequence-growing attention state, not total process memory.

KDA state is recurrent and does not grow with `T`; only MLA latent history grows
linearly. Cache reordering and batch repeat/select are implemented for deterministic
generation. Cache cropping is intentionally unsupported because recurrent KDA state
cannot be truncated like a conventional KV cache.

## Block Attention Residuals

Block Attention Residuals (Block AttnRes) route among representations from different
depths. The 32 decoder layers are divided into eight four-layer depth blocks. There
are two read/write sites per layer (mixer and FFN) plus one final read:

```text
32 * 2 + 1 = 65 routing sites
```

At site `s`, each available source `H_i in R^[B,T,2560]` is normalized, scored by a
learned query `q_s in R^[2560]`, and combined:

```text
score_i = <q_s, RMSNorm_s(H_i)>
alpha_i = softmax_i(score_i)
read_s = sum_i alpha_i H_i
```

Every site contains a query `[2560]` and RMSNorm scale `[2560]`. Depth sources are
ephemeral within one forward pass and are never stored in the sequence cache.

## Dense SiTU-GLU

The model keeps a dense 9216-wide feed-forward network instead of LatentMoE. SiTU
(a smooth thresholded activation used in the migrated gated linear unit) replaces
the donor SwiGLU activation while retaining the donor projection shapes:

```text
gate = X W_gate        [B,T,9216]
up   = X W_up          [B,T,9216]
Y    = W_down SiTU(gate, up; beta=4, linear_beta=25)
```

`W_gate` and `W_up` are `[9216,2560]`; `W_down` is `[2560,9216]`. The selected
activation is dense: every decoder layer owns one complete FFN and no expert router
or sparse-expert parameters exist.

## Normalization and preserved Qwen components

RMSNorm computes `x / sqrt(mean(x^2) + eps)` and applies a learned per-channel
scale. The model preserves the Qwen pre-norm decoder organization, final RMSNorm,
tokenizer/vocabulary, token embedding, tied LM head, and their pretrained
initialization. Mixer-specific Q/K normalization is explicitly retained where the
frozen migration contract requires it. Strict-NoPE disables positional rotation;
it does not remove causal masking or normalization.

The public implementation and frozen contracts are the source of truth for exact
casting, decay, gating, and state-layout semantics.
