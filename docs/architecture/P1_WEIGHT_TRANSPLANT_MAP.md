# P1 Weight Transplant Map

This document classifies every Qwen3.5-4B-Base source tensor and defines the planned transforms for P3, P6, P8, and P9. It is a conversion specification, not an implementation.

## 1. Naming conventions

Source paths use the checkpoint prefix:

```text
model.language_model.layers.{i}
```

The DenseK3 implementation should use a stable text-only prefix such as:

```text
model.layers.{i}
```

The converter must define the final prefix in one place and must not infer tensor roles from shape alone.

Conversion classes:

- **copy**: same values and semantics, possibly with a path rename;
- **transform**: deterministic lossless value transformation;
- **construct**: new tensor deterministically constructed from donor tensors;
- **approximate**: deterministic initialization with measurable reconstruction error;
- **new**: initialized without a donor tensor;
- **exclude**: intentionally absent from the target checkpoint.

## 2. P4 source inventory disposition

| Source category | Entries | Disposition |
|---|---:|---|
| Text embedding | 1 | copy |
| Final text RMSNorm | 1 | transform: add 1 |
| Per-layer input/post-attention RMSNorm | 64 | transform: add 1 |
| FFN projections | 96 | copy |
| Gated GQA linear projections | 32 | copy |
| Gated GQA Q/K head norms | 16 | transform: add 1 |
| GDN tensors copied without value change | 120 | copy |
| GDN tensors expanded/split into KDA | 96 | transform/construct |
| Vision | 297 | exclude |
| MTP | 15 | exclude |
| Unknown | 0 | fail if encountered |
| **Total** | **738** | Fully classified |

The 426 text tensors consist of 249 copy entries and 177 transform/construct entries. The target has additional tensors because each GDN decay projection becomes a two-matrix KDA projection.

## 3. Global and shared tensors

| Source | Target | Class | Rule |
|---|---|---|---|
| `model.language_model.embed_tokens.weight` | `model.embed_tokens.weight` | copy | Preserve and tie to LM head |
| `model.language_model.norm.weight` | `model.norm.weight` | transform | `target = 1 + source` |
| `model.language_model.layers.{i}.input_layernorm.weight` | `model.layers.{i}.input_layernorm.weight` | transform | `target = 1 + source` |
| `model.language_model.layers.{i}.post_attention_layernorm.weight` | `model.layers.{i}.post_attention_layernorm.weight` | transform | `target = 1 + source` |
| `model.language_model.layers.{i}.mlp.gate_proj.weight` | same logical path | copy | Preserve |
| `model.language_model.layers.{i}.mlp.up_proj.weight` | same logical path | copy | Preserve |
| `model.language_model.layers.{i}.mlp.down_proj.weight` | same logical path | copy | Preserve |

Why add 1: Transformers 5.12.1 implements Qwen backbone RMSNorm as `normalized_x * (1 + weight)`, while DenseK3's canonical RMSNorm scale is directly multiplicative.

## 4. P3/P4 GDN to KDA bridge

For every Qwen linear-attention layer, define:

```text
D  = 2560
H  = 16
HV = 32
K  = 128
V  = 128
QK = H*K  = 2048
VV = HV*V = 4096
```

### 4.1 Direct tensors

| Qwen source | DenseK3 KDA target | Shape | Rule |
|---|---|---:|---|
| `linear_attn.A_log` | `self_attn.A_log` | `[32]` | copy in P4 compatibility mode |
| `linear_attn.in_proj_b.weight` | `self_attn.b_proj.weight` | `[32,2560]` | copy |
| `linear_attn.in_proj_z.weight` | `self_attn.g_proj.weight` | `[4096,2560]` | copy; full-rank output gate |
| `linear_attn.norm.weight` | `self_attn.o_norm.weight` | `[128]` | copy; this is already a direct scale |
| `linear_attn.out_proj.weight` | `self_attn.o_proj.weight` | `[2560,4096]` | copy |

The P4 bridge applies SiLU to `g_proj(x)`, exactly matching the donor. P5 retains the copied weights but changes the activation to Sigmoid.

### 4.2 Fused QKV split

Source:

```text
W_qkv = linear_attn.in_proj_qkv.weight  # [8192,2560]
```

The source output order is all Q channels, then all K channels, then all V channels. Construct:

```text
q_proj.weight = W_qkv[0:2048]
k_proj.weight = W_qkv[2048:4096]
v_proj.weight = W_qkv[4096:8192]
```

This is lossless.

### 4.3 Depthwise convolution split

Source:

```text
W_conv = linear_attn.conv1d.weight  # [8192,1,4]
```

Construct:

```text
q_conv1d.weight = W_conv[0:2048]
k_conv1d.weight = W_conv[2048:4096]
v_conv1d.weight = W_conv[4096:8192]
```

Because the donor convolution is depthwise, splitting the channels into three independent depthwise convolutions is lossless.

### 4.4 Scalar-to-channel decay embedding

Source:

```text
W_a = linear_attn.in_proj_a.weight  # [32,2560]
```

Target:

```text
f_a_proj.weight  # [128,2560]
f_b_proj.weight  # [4096,128]
```

Construct both tensors as zero, then:

```text
f_a_proj.weight[0:32, :] = W_a
f_b_proj.weight.reshape(32,128,128)[h,k,h] = 1
    for h in 0..31 and k in 0..127
```

This guarantees:

```text
f_b_proj(f_a_proj(x))[h,k] == in_proj_a(x)[h]
```

for every input `x`. The remaining 96 latent channels start at zero but are trainable, allowing the gate to become genuinely channel-wise during recovery.

### 4.5 Decay bias expansion

Source:

```text
dt_bias  # [32]
```

Target:

```text
dt_bias  # [4096]
```

Construct:

```text
target_dt_bias = source_dt_bias.repeat_interleave(128)
```

Together with the projection construction above and copied `A_log`, this reproduces the donor scalar decay in every KDA key channel when P4 uses the unbounded negative-Softplus parameterization.

### 4.6 Bridge equivalence boundary

The following settings are required for recurrence equivalence:

```text
decay_parameterization = unbounded_softplus
lower_bound = null
output_gate_activation = silu
beta_activation = sigmoid
qk_l2norm = true
allow_negative_eigenvalues = false
```

P5 deliberately breaks equivalence by changing decay to lower-bounded Sigmoid and the output gate to Sigmoid.

### 4.7 State layout

No cache tensor is converted between models. The target starts with an empty cache. The logical recurrent state is `[B,HV,K,V]`; the optimized backend may store it physically as V-first when explicitly declared by the cache layer. K and V are both 128 here, so shape checks alone cannot detect an accidental transpose; P2 must use non-symmetric values and explicit semantic tests.

## 5. P4 gated GQA preservation

For layer indices `3,7,11,15,19,23,27,31`:

| Source | P4 target | Class |
|---|---|---|
| `self_attn.q_proj.weight` | same logical tensor | copy |
| `self_attn.k_proj.weight` | same logical tensor | copy |
| `self_attn.v_proj.weight` | same logical tensor | copy |
| `self_attn.o_proj.weight` | same logical tensor | copy |
| `self_attn.q_norm.weight` | same logical tensor | transform: add 1 |
| `self_attn.k_norm.weight` | same logical tensor | transform: add 1 |

The packed Q/output-gate layout must remain exactly as Transformers 5.12.1 interprets it: reshape to `[... ,16,512]`, then split the last dimension into Q `[... ,16,256]` and gate `[... ,16,256]`.

## 6. P5 KDA architecture recovery initialization

P5 begins from the P4 checkpoint and changes:

```text
decay_parameterization: unbounded_softplus -> lower_bounded_sigmoid
lower_bound: null -> -5
output_gate_activation: silu -> sigmoid
```

No tensor is shape-converted at this boundary. `A_log`, channel biases, and output-gate weights are retained, but the function they parameterize changes. A fresh optimizer state is recommended for the affected KDA gate parameters.

## 7. P6 GQA to strict-NoPE MLA

For each of the 8 full-attention layers, the target uses:

```text
num_heads = 16
qk_nope_head_dim = 256
qk_pos_head_dim = 0
v_head_dim = 256
q_lora_rank = null
kv_lora_rank = 512
```

### 7.1 Unpack donor Q and output gate

The donor `q_proj.weight` is not a simple `[Q; gate]` concatenation over its first dimension. Its output is grouped per head as `[q_head, gate_head]` after reshaping. The converter must do:

```text
W = source_q_proj.reshape(16, 512, 2560)
W_q = W[:, 0:256, :].reshape(4096, 2560)
W_g = W[:, 256:512, :].reshape(4096, 2560)
```

Then:

| Constructed tensor | MLA target | Class |
|---|---|---|
| `W_q` | `q_proj.weight` | transform, lossless |
| `W_g` | `g_proj.weight` | transform, lossless values; activation changes to Sigmoid |

### 7.2 Build the full donor K/V operator

1. Reshape donor K and V from `[1024,2560]` to `[4,256,2560]`.
2. Repeat each KV head four times to align with 16 query heads.
3. For every target head, concatenate its 256 K rows and 256 V rows.
4. Flatten to a desired reconstruction matrix `W_kv_full: [8192,2560]` in the exact row order expected by MLA's K/V up-projection.

### 7.3 Rank-512 factorization

Compute a deterministic truncated SVD in FP32:

```text
W_kv_full ~= U_512 @ diag(S_512) @ Vh_512
kv_b_proj.weight = U_512 @ diag(sqrt(S_512))  # [8192,512]
kv_a_proj.weight = diag(sqrt(S_512)) @ Vh_512 # [512,2560]
```

Alternative redistribution of singular values is allowed only if the reconstructed product is identical. Record at minimum:

- Frobenius relative error;
- spectral relative error;
- maximum absolute error;
- explained squared singular-value mass.

This mapping is approximate and must never be reported as parity-preserving.

### 7.4 Other MLA tensors

| Donor | MLA target | Class | Rule |
|---|---|---|---|
| `o_proj.weight` | `o_proj.weight` | copy | Both are `[2560,4096]` |
| `q_norm.weight` | none | exclude at P6 | Per-head data-dependent norm cannot be folded into Q projection |
| `k_norm.weight` | none | exclude at P6 | Per-head data-dependent norm cannot be folded into KV factorization |
| none | `kv_a_layernorm.weight` | new | Initialize to ones |

P6 must report the excluded Q/K norm parameters explicitly rather than silently ignoring them.

## 8. P8 AttnRes parameters

New per Transformer layer:

```text
attn_res_proj.weight  [1,2560]  = 0
attn_res_norm.weight  [2560]    = 1
mlp_res_proj.weight   [1,2560]  = 0
mlp_res_norm.weight   [2560]    = 1
```

New final aggregation:

```text
output_res_proj.weight [1,2560] = 0
output_res_norm.weight [2560]   = 1
```

No Qwen tensor is consumed for these parameters.

Zero pseudo-queries produce uniform depth attention, not the donor's additive residual computation. P8 insertion is intentionally non-equivalent and requires recovery training.

## 9. P9 SiTU-GLU mapping

The FFN tensors remain direct copies:

```text
gate_proj.weight [9216,2560]
up_proj.weight   [9216,2560]
down_proj.weight [2560,9216]
```

Only the activation configuration changes from SiLU GLU to SiTU-GLU with `(beta, linear_beta) = (4,25)`. This is a semantic change and requires recovery training.

## 10. Converter invariants

The future P3/P6 converters must:

1. open source checkpoints read-only and write to a distinct output directory;
2. verify the donor revision/manifests before conversion;
3. enumerate all source keys and fail on any unclassified key;
4. validate every target shape before allocating the final shard;
5. perform norm additions, factorization, and reconstruction metrics in FP32;
6. preserve BF16 checkpoint dtype for copied/projected model weights unless the target schema explicitly requires FP32;
7. emit a JSON conversion report with source key, target key(s), class, shapes, dtypes, and transform name;
8. verify tied embedding identity after save/reload;
9. start with an empty runtime cache and never serialize donor cache state;
10. run a full target checkpoint save/reload before declaring success.
