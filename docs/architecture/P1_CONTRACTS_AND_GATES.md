# P1 Tensor, Cache, and Acceptance Contracts

This document defines the interfaces and correctness gates that P2 and later phases must satisfy.

## 1. Notation

```text
B  batch size
T  padded or dense sequence length
N  number of variable-length sequences
D  hidden size = 2560
H  KDA Q/K heads = 16
HV KDA value/decay heads = 32
K  KDA key head dimension = 128
V  KDA value head dimension = 128
C  short-convolution kernel size = 4
```

Dense tensors use sequence-major-inside-batch layout unless stated otherwise:

```text
hidden_states: [B,T,D]
q, k:          [B,T,H,K]
v:             [B,T,HV,V]
g:             [B,T,HV,K]
beta:          [B,T,HV]
output:        [B,T,HV,V]
```

## 2. KDA operator contract

### 2.1 Reference recurrence

The independent PyTorch reference is the source of truth. It must:

1. convert recurrence arithmetic to FP32;
2. optionally L2-normalize Q and K over `K`;
3. repeat Q/K heads by `HV/H` without changing checkpoint tensors;
4. apply per-channel log decay before the delta update;
5. return the output in the input V dtype;
6. return the final recurrent state in FP32 when requested.

For each token:

```text
q = l2norm(q) * K**-0.5
k = l2norm(k)
S = S * exp(g)[..., :, None]
prediction = sum_k(k[..., k, None] * S[..., k, :])
delta = beta[..., None] * (v - prediction)
S = S + k[..., :, None] * delta[..., None, :]
o = sum_k(q[..., k, None] * S[..., k, :])
```

The reference must accept an optional initial state and must not mutate it.

### 2.2 Gate modes

The layer exposes named modes rather than loosely coupled booleans:

```text
qwen_compat:
  g = -exp(A_log[h]) * softplus(raw_g[h,k] + dt_bias[h,k])

kimi_k3:
  g = -5 * sigmoid(exp(A_log[h]) * (raw_g[h,k] + dt_bias[h,k]))
```

Both return log-space decay in `[B,T,HV,K]`.

Beta is `sigmoid(beta_logits)` in all DenseK3 stages through P10. Negative-eigenvalue mode is disabled.

### 2.3 Backend adapter

The FLA adapter must explicitly pass:

```text
use_qk_l2norm_in_kernel = true
use_gate_in_kernel = true
use_beta_sigmoid_in_kernel = true
allow_neg_eigval = false
output_final_state = use_cache
cu_seqlens = optional int32 cumulative lengths
```

For P4 compatibility, `safe_gate=false` and `lower_bound=null`. For the P5 Kimi K3 target, the adapter selects the lower-bounded mode with `lower_bound=-5`.

The adapter must not depend on FLA model/config/cache classes.

### 2.4 Dense and variable-length inputs

- Dense/padded API: `hidden_states [B,T,D]`, `attention_mask [B,T]`, with 1 for valid tokens and 0 for padding.
- Packed API: `hidden_states [1,total_tokens,D]`, `cu_seqlens [N+1]`, beginning at 0 and ending at `total_tokens`.
- Explicit `cu_seqlens` takes precedence over `attention_mask`.
- Packed sequences must not share recurrent or convolution state across boundaries.
- Only left padding is required for generation compatibility; padding outputs are zero before the residual add.

## 3. KDA cache contract

Each KDA layer owns:

```text
KDACacheLayer:
  q_conv_state: [B,2048,C]
  k_conv_state: [B,2048,C]
  v_conv_state: [B,4096,C]
  recurrent_state: logical [B,HV,K,V], FP32
  recurrent_state_layout: "k_first" or "v_first"
  seen_tokens: int
  initialized: bool
```

The initial DenseK3 optimized layout is `v_first`, matching the Kimi HF/FLA layer path. The public cache metadata must carry the layout explicitly because `K=V=128` makes both layouts have the same shape.

Required behavior:

- prefill with `T>=1` creates all four states and sets `seen_tokens=T`;
- a cached single token uses the recurrent kernel and updates states in place where supported;
- a cached multi-token continuation uses chunk KDA with the incoming recurrent state and correct convolution left context;
- `reorder_cache(beam_idx)` reorders all four states along batch dimension;
- `reset()` zeroes or releases states and resets `seen_tokens`;
- `crop()` is unsupported for the recurrent state and must either be a documented no-op or raise a clear error;
- no cache object is serialized as model weights.

Approximate fixed cache footprint per batch item for 24 KDA layers:

```text
FP32 recurrent state: 24 * 32 * 128 * 128 * 4 bytes = 48 MiB
BF16 convolution states: 24 * 8192 * 4 * 2 bytes = 1.5 MiB
```

## 4. P4 GQA cache contract

Each retained GQA layer owns:

```text
key:   [B,4,T,256]
value: [B,4,T,256]
seen_tokens: shared model position
```

The cache grows along `T`, supports beam reorder, and is used as the model-wide sequence-length authority because KDA recurrent layers do not encode a recoverable token count in tensor shape.

## 5. P6/P7 MLA cache contract

### 5.1 P6 reference mode

P6 may reconstruct full K/V for correctness, but this mode must be named `expanded` and must not be reported as a latent cache.

### 5.2 P7 true latent mode

Each MLA layer stores only:

```text
compressed_kv: [B,T,512]  # BF16 by default
seen_tokens: int
```

The strict-NoPE target has no positional K cache. Full per-head K/V are reconstructed only for the current computation tile or token and are not appended to persistent cache storage.

Required parity:

- expanded-cache and latent-cache modes produce matching outputs under the numerical threshold;
- prefill plus token decode matches a single dense causal forward at every compared suffix position;
- latent cache storage is exactly `B*T*512` elements per MLA layer, excluding allocator metadata;
- beam reorder acts on the latent tensor;
- no hidden full-K/V tensor persists after the layer returns.

## 6. Model forward contract

The text causal LM must accept:

```text
input_ids: [B,T] xor inputs_embeds: [B,T,D]
attention_mask: optional [B,total_context]
past_key_values: optional DenseK3Cache
use_cache: bool
labels: optional [B,T]
```

It returns a Transformers-compatible causal LM output containing:

```text
logits: [B,T,248320]
loss: scalar when labels are supplied
past_key_values: DenseK3Cache when use_cache=true
hidden_states/attentions: optional only when implemented and explicitly requested
```

Behavioral rules:

- exactly one of `input_ids` and `inputs_embeds` is supplied;
- `use_cache` is disabled during gradient-checkpointed training;
- embedding and LM-head weights remain tied after initialization, load, save, device movement, and reload;
- logits are computed from conventionally scaled final-RMSNorm output;
- causal masks are supplied only to GQA/MLA layers; KDA receives a 2D padding mask or packed lengths;
- the model tracks a single logical token position even though KDA states are fixed-size.

## 7. P2 numerical test matrix

### 7.1 Shapes and dtypes

Minimum dense cases:

```text
B: 1, 2, 4
T: 1, 2, 3, 4, 7, 31, 63, 64, 65, 127, 128, 129, 1024
H/HV: 2/2, 2/4, 16/32
K/V: 32/32, 64/64, 128/128, 64/128
dtype: float32 reference; bfloat16 optimized; float16 smoke where supported
```

The non-square `K=64,V=128` cases are mandatory for catching state-layout errors even though the final model uses `128/128`.

Minimum packed cases:

```text
cu_seqlens = [0,1,4,11]
cu_seqlens = [0,63,128]
cu_seqlens = [0,101,303,1205]
```

Padding cases must cover left-padded batches with unequal lengths and an all-valid mask.

### 7.2 Compared values

Every differentiable P2 case compares:

```text
output
final recurrent state
dq, dk, dv, dg, dbeta
dA_log and ddt_bias when gate fusion is active
d_initial_state when initial state requires grad
```

Also test:

- chunk versus naive recurrence;
- fused recurrent versus naive recurrence;
- one-shot chunk versus two consecutive cached chunks;
- dense padded versus packed representation;
- `qwen_compat` scalar-expanded KDA versus the donor GDN recurrence;
- in-kernel gate/beta activation versus precomputed activation;
- cache reorder/reset and single-token decode.

### 7.3 Initial thresholds

Use deterministic seed 42. FP32 reference-to-reference comparisons use:

```text
rtol = 1e-5
atol = 1e-6
```

Approved numerics amendment `P1-NUM-2026-08-10-01` adds an explicit FP32 reference-to-optimized-FLA output threshold:

```text
output: rtol = 1e-4, atol = 1e-4
```

Approved numerics amendment `P1-NUM-2026-08-10-02` separately adds an FP32 reference-to-optimized-FLA final-state threshold:

```text
final state: rtol = 2e-3, atol = 2e-3
```

Neither amendment changes the FP32 reference-to-reference threshold. The optimized thresholds account for FLA KDA's explicit TF32/TensorCore path on supported NVIDIA GPUs. The first RTX 5090 P2.6 run measured maximum absolute output errors from `1.85e-5` through `1.00e-4`. After the output threshold was approved, every FP32 output case passed and only final-state comparisons failed, with maximum absolute errors from `5.268e-4` through `1.149e-3`. All BF16 cases passed their original threshold in both runs.

BF16 optimized comparisons start with:

```text
output and final state: rtol = 6e-3, atol = 6e-3
input gradients:         rtol = 8e-3, atol = 8e-3
gate parameter grads:    rtol = 1e-2, atol = 1e-2
```

In addition to elementwise checks, record maximum absolute error, mean absolute error, and relative L2 error. Thresholds may only be relaxed with a documented numerical cause and a separately approved architecture/numerics decision.

The previously observed FLA KDA comparison maximum absolute error of `0.000977` is a baseline observation, not a universal acceptance threshold.

## 8. P3/P4 checkpoint and integration gates

P3 conversion passes only if:

- all 738 donor entries are classified;
- all 426 text entries are consumed exactly once unless a one-to-many transform is declared;
- no vision/MTP entry appears in the target;
- every target tensor has a declared initialization source;
- save/reload yields no missing or unexpected target keys;
- tied weights share storage after reload;
- the scalar-expanded KDA projection identity is verified on random FP32 hidden states;
- the split QKV/convolution path matches the donor fused path in FP32;
- a JSON conversion report is emitted.

P4 model integration passes only if:

- layer types and indices exactly match the P1 topology;
- forward and backward run in BF16 on the RTX 5090;
- prefill, cached single-token decode, and cached multi-token continuation run;
- batch beam reorder works;
- left-padding and packed variable-length KDA paths work;
- no NaN/Inf occurs in activations, loss, gradients, or cache;
- peak memory and tokens/s are recorded against the unmodified Qwen baseline on identical inputs.

## 9. Baseline evaluation protocol

P1 fixes the measurement protocol; dataset-specific capability thresholds remain pending because `data/` is empty and no training/evaluation budget has been selected.

### 9.1 Deterministic structural baseline

Use fixed token IDs rather than text tokenization for model-regression tests:

```text
seed = 42
B = 1 and 2
T = 1, 4, 64, 512, 2048
token IDs sampled uniformly from [0, vocab_size), excluding special multimodal IDs
```

Record:

- last-token and all-token logits summaries;
- cross-entropy on next-token-shifted random labels;
- prefill/decode consistency;
- per-layer activation RMS;
- peak allocated/reserved GPU memory;
- prefill tokens/s and decode tokens/s.

### 9.2 Fixed natural-language smoke set

Before P4 completion, add a small version-controlled JSONL smoke set covering English, Chinese, code, arithmetic, long repetition/retrieval, and left-padded batching. It is for regression and generation sanity only, not capability claims.

### 9.3 Capability baseline required before P5

Before recovery training begins, freeze:

- a held-out language-model loss/perplexity corpus;
- short-context general evaluation;
- code evaluation;
- long-context retrieval and repository-context evaluation;
- maximum context lengths for each stage;
- contamination policy;
- exact tokenizer and prompt formatting;
- training-token, GPU-hour, and checkpoint budgets.

P5, P6, P8, and P9 must each report deltas from both the original Qwen donor and the immediately preceding milestone.

## 10. Failure reporting

Every failed gate report must include:

```text
stage and git/worktree identifier
environment manifest
seed
input shape/dtype/mode
backend
gate parameterization
cache state/layout
max/mean/relative error or traceback
whether the failure reproduces in the FP32 reference
```

No performance result is valid unless all correctness gates for the same code, shape family, and dtype are green.
