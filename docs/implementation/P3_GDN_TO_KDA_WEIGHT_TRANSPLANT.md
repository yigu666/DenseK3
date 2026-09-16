# P3 GDN to KDA Weight Transplant

## 1. Scope

P3 answers one question: can one real Qwen3.5 GDN layer be mapped deterministically into a DenseK3 `qwen_compat` KDA layer with parameter, shape, recurrence, and save/reload parity?

P3 does not build or replace 24 linear-attention layers. It does not construct a complete model, load vision/MTP components, recover K3 gate semantics, train, distill, or optimize kernels. Those activities remain in later stages.

The authoritative donor is layer 0 from:

```text
models/Qwen3.5-4B-Base
revision: 1001bb4d826a52d1f399e183466143f4da7b741b
```

The converter opens only the shard containing the nine layer-0 `linear_attn` tensors. It never loads the full model and never writes into the source checkpoint directory.

## 2. Source contract

The source audit requires exactly:

```text
A_log                 [32]
conv1d.weight         [8192,1,4]
dt_bias               [32]
in_proj_a.weight      [32,2560]
in_proj_b.weight      [32,2560]
in_proj_qkv.weight    [8192,2560]
in_proj_z.weight      [4096,2560]
norm.weight           [128]
out_proj.weight       [2560,4096]
```

Any missing, additional, or shape-mismatched tensor fails P3.0.

## 3. Deterministic bridge

The target single-layer artifact contains fourteen tensors:

```text
A_log                  COPY       [32]
dt_bias                REPEAT     [4096]
q_proj.weight          SLICE      [2048,2560]
k_proj.weight          SLICE      [2048,2560]
v_proj.weight          SLICE      [4096,2560]
q_conv1d.weight        SLICE      [2048,1,4]
k_conv1d.weight        SLICE      [2048,1,4]
v_conv1d.weight        SLICE      [4096,1,4]
f_a_proj.weight        CONSTRUCT  [128,2560]
f_b_proj.weight        CONSTRUCT  [4096,128]
b_proj.weight          COPY       [32,2560]
g_proj.weight          COPY       [4096,2560]
o_norm.weight          COPY       [128]
o_proj.weight          COPY       [2560,4096]
```

No operation uses random initialization. `A_log` remains one parameter per value head and is broadcast over the 128 key channels at runtime. `dt_bias` is physically repeated over those channels. This is the P1-locked parameterization.

```text
source layer parameters = 42,139,840
target layer parameters = 42,913,952
```

The deterministic increase comes from the trainable rank-128 `f_a/f_b` channel-decay embedding; it is not silent initialization.

The scalar-decay construction is:

```text
f_a[0:32] = donor in_proj_a
f_a[32:128] = 0
f_b.reshape(32,128,128)[h,k,h] = 1
```

Therefore every channel `k` in value head `h` initially receives the same donor scalar decay logit.

## 4. Compatibility semantics

P3 uses only the P4 compatibility behavior:

```text
decay = -exp(A_log) * softplus(channel_logit + dt_bias)
beta = sigmoid(b_proj(x))
output gate = silu(g_proj(x))
Q/K L2 normalization = enabled
lower_bound = null
```

P3 must not use the P5 lower-bounded sigmoid decay or sigmoid output gate. Functional parity is evaluated before any P5 architecture recovery.

## 5. Gates

| Gate | Required result |
|---|---|
| P3.0 | real layer-0 source names and shapes pass |
| P3.1 | repeated conversion produces identical parameter and manifest hashes |
| P3.2 | every source/target tensor is classified; no UNKNOWN, UNMAPPED, or SILENT_INIT |
| P3.3 | scalar donor decay/beta equals the channel-embedded target decay/beta |
| P3.4 | mixer output and canonical V-first final state pass at T=1,7,65,129 under the ordered standard Gate and approved P3-only BF16 fallback |
| P3.5 | save/reload preserves parameter hash, manifest hash, output, and final state |

P3.4 uses a pure PyTorch recurrence on both sides so it isolates the weight bridge from FLA optimized-kernel numerics. FP32 TensorFloat-32 is disabled for this reference comparison. The donor K-first recurrence state is represented in the P2 canonical V-first layout `[B,HV,V,K]` before comparison.

P3.4 also emits four diagnostic sub-gates without changing the formal P3 tolerance or backend:

| Diagnostic | Evidence |
|---|---|
| P3.4a | donor/target tensors at every projection, convolution, gate, recurrence, norm, and output boundary; includes BF16 data with projection GEMMs forced to FP32 as a diagnostic-only control |
| P3.4b | native, donor-decay, donor-QKV (pre- and post-convolution), and both-injected recurrence cases |
| P3.4c | the same T=129 BF16 input with `None`, zero, and random initial states |
| P3.4d | exact counts, allclose failures, percentiles, histogram buckets, relative L2, and cosine for output/state/decay errors |

`P3_04A_PROJECTION_DIAGNOSTICS=PASS` means that every requested boundary was measured with matching layout and finite values; it does not require bit-exact projection values. The forced-FP32 projection branch is evidence only and cannot become the saved model implementation. `P3_04B_EXACT_INJECTION_ISOLATION=PASS` requires the both-injected case to recover donor output and state under the existing BF16 Gate.

The runner writes `p3-functional-parity.json` before it records P3.4 failure. The file is created for `PASS`, ordinary `FAIL`, and `EXCEPTION`, and includes case counts, the first failure, and diagnostic completion status. A thrown exception is therefore never the only P3.4 evidence.

### P3-only BF16 functional fallback

`P3-NUM-2026-08-10-01` approves a narrow fallback for Qwen GDN to DenseK3 `qwen_compat` transplantation. It does not change the standard allclose Gate, P1, P2, or the runtime backend. Standard allclose is always evaluated first. The fallback can be used only for a BF16 output-only failure after FP32 strict parity, BF16 state/decay/beta, P3.4a-d, exact both-injection, forced-FP32 projections, and all finiteness checks pass.

The conjunctive output limits are relative L2 `7e-3`, cosine `0.99997`, mean absolute error `1e-3`, p99.9 absolute error `4e-3`, failed-allclose fraction `1e-4`, maximum absolute error `0.125`, and zero NaN/Inf. The full decision and evidence are recorded in `P3_NUMERICS_DECISION_20260810.md`.

A successful fallback records `status=PASS`, `pass_mode=bf16_transplant_distribution_fallback`, `standard_allclose=false`, and `numerics_amendment=P3-NUM-2026-08-10-01`. This is not represented as ordinary BF16 allclose parity.

P3 is dependency-ordered. The first failed Gate blocks the remaining stages and creates `p3-failure.json`. P4 is allowed only when all six stages pass.

## 6. Server execution

```bash
cd "$DENSEK3_ROOT"
conda run -n zmm_2 python -m pip install -e . --no-deps
conda run -n zmm_2 ruff check pyproject.toml src tests scripts
conda run -n zmm_2 --no-capture-output python scripts/run_p3.py
```

Canonical results are written to:

```text
outputs/densek3-core/manifests/p3/
```

The converted single-layer artifact is written separately to:

```text
outputs/densek3-core/checkpoints/p3/layer-0-qwen-compat-kda/
```

The final markers are:

```text
P3_FINAL_CHECK=PASS
P4_ALLOWED=true
```
