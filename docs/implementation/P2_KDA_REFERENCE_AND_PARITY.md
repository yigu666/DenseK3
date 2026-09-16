# P2 KDA Reference and FLA Parity

- Authoritative execution date: 2026-08-10
- Final status: `PASS`
- First failed stage: none
- P1 contract modified: no
- P3 allowed: yes

## 1. Final outcome

The independent DenseK3 KDA reference, V-first FP32 recurrent state, FLA chunk and fused-recurrent adapters, GVA 16/32 mapping, continuation invariant, backward parity, numerical matrix, and RTX 5090 performance baseline all passed.

```text
P2_FINAL_CHECK=PASS
P4_ALLOWED is not evaluated by P2
P3_ALLOWED=true
```

The canonical machine-readable evidence is under `outputs/densek3-core/manifests/p2`. The earlier local connectivity failure is retained only under `outputs/densek3-core/manifests/history/p2-pre-final-20260810` and `docs/implementation/history`.

## 2. Stage results

| Stage | Name | Status |
|---|---|---|
| P2.0 | package/config contract | PASS |
| P2.1 | recurrent state contract | PASS |
| P2.2 | reference recurrent | PASS |
| P2.3 | reference full | PASS |
| P2.4 | reference self-parity | PASS |
| P2.5 | FLA chunk adapter | PASS |
| P2.6 | FLA chunk forward parity | PASS |
| P2.7 | FLA recurrent adapter | PASS |
| P2.8 | FLA recurrent parity | PASS |
| P2.9 | continuation parity | PASS |
| P2.10 | backward parity | PASS |
| P2.11 | numerical matrix | PASS |
| P2.12 | performance baseline | PASS |

## 3. Locked implementation contract

```text
q, k:      [B,T,H,K]
v:         [B,T,HV,V]
decay:     [B,T,HV,K]
beta:      [B,T,HV]
state:     [B,HV,V,K] FP32
output:    [B,T,HV,V]
H/HV:      16/32
K/V:       128/128
```

The recurrent state is physically value-first. The reference performs recurrence arithmetic in FP32, returns outputs in the V activation dtype, clones the initial state, and never mutates caller-owned state.

P4 compatibility remains `safe_gate=False, lower_bound=None`. P2 does not authorize P5's lower-bounded sigmoid decay or sigmoid output-gate semantics.

## 4. Approved optimized numerics

The following separately approved decisions are part of the frozen P2 Gate:

```text
P1-NUM-2026-08-10-01
P1-NUM-2026-08-10-02
```

```text
FP32 reference-to-reference:        rtol=1e-5, atol=1e-6
FP32 optimized token output:        rtol=1e-4, atol=1e-4
FP32 optimized final state:         rtol=2e-3, atol=2e-3
BF16 optimized output/final state:  rtol=6e-3, atol=6e-3
BF16 input gradients:               rtol=8e-3, atol=8e-3
BF16 gate-parameter gradients:      rtol=1e-2, atol=1e-2
```

The state-only adjustment accounts for FLA's TF32/TensorCore accumulation. It does not change the architecture or reference recurrence.

## 5. Numerical evidence

```text
FLA chunk FP32 output max abs:       8.3349645e-5
FLA chunk FP32 final-state max abs:  1.1486411e-3
FLA recurrent FP32 output max abs:   9.3132257e-9
FLA recurrent FP32 state max abs:    1.7881393e-7
FLA chunk BF16 output max abs:       4.8828125e-4
FLA chunk BF16 state max abs:        3.4162402e-3
Continuation BF16 output max abs:    4.8828125e-4
Continuation BF16 state max abs:     2.6424825e-3
Backward maximum absolute error:     1.953125e-3
```

All recorded tensors and gradients have zero NaN/Inf counts. All eight checked gradients have a finite ratio of 1.0.

## 6. Performance baseline

RTX 5090, BF16, `B=1,T=129,H=16,HV=32,K=128,V=128`, five warmups and twenty measured iterations:

| Backend | Latency | Tokens/s | Peak allocated |
|---|---:|---:|---:|
| PyTorch reference | 46.818 ms | 2,755 | 23,390,208 B |
| FLA chunk | 0.927 ms | 139,142 | 25,258,496 B |
| FLA recurrent trajectory | 16.288 ms | 7,920 | 15,926,272 B |

This is a single-operator baseline, not a full-model throughput claim.

## 7. Freeze and handoff

P2 is frozen. P3 may depend on the reference recurrence, state layout, FLA adapters, and approved tolerances, but must not redesign them. P3 is restricted to deterministic transplantation of one real Qwen3.5 GDN layer into DenseK3 `qwen_compat` KDA and its parameter/shape/functional/reproducibility gates.
