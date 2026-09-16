# P5 KDA Architecture Recovery

P5 starts from the frozen P4 artifact and deliberately changes KDA behavior. It is not a parity-preserving transplant.

## Locked transition

Only the following configuration triple changes at P5-init:

```text
P4: qwen_compat / lower_bound=None / SiLU output gate
P5: kimi_k3     / lower_bound=-5.0 / Sigmoid output gate
```

The decay formulas are:

```text
P4: g = -exp(A_log[h]) * softplus(raw_g[h,k] + dt_bias[h,k])
P5: g = -5 * sigmoid(exp(A_log[h]) * (raw_g[h,k] + dt_bias[h,k]))
```

All parameter objects and bytes are retained. The 24/8 topology, GQA, MLP, residual path, hybrid-cache layout, FP32 V-first recurrent state, beta Sigmoid, Q/K normalization, and FLA implementation remain unchanged. A fresh optimizer is required for recovery.

## Bounded pre-recovery runner

`scripts/run_p5.py` runs only P5.0-P5.2:

1. **P5.0** verifies every P4 tensor hash, revalidates `P4_FINAL_CHECK=PASS`, and exactly replays the frozen P4 probe, loss, and five greedy generations.
2. **P5.1** applies the configuration-only transition, verifies that no parameter object/version/storage changed, checks all 24 KDA layers, and runs BF16 full-forward plus cached continuation with the unchanged cache contract.
3. **P5.2** records the architecture shock: 32-layer hidden drift, final hidden/logits/CE/token decisions, five generations, 24 recurrent-state drifts, and per-layer decay/beta/output-gate distributions.

P5.2 is a measurement-completion Gate. It intentionally has no capability tolerance. A PASS means the shock was measured completely and all values were finite; it does not mean that the untrained P5-init model is recovered.

Authoritative outputs:

```text
outputs/densek3-core/manifests/p5/p5-p4-baseline.json
outputs/densek3-core/manifests/p5/p5-transition-manifest.json
outputs/densek3-core/manifests/p5/p5-k3-parameterization.json
outputs/densek3-core/manifests/p5/p5-transition-impact.json
outputs/densek3-core/manifests/p5/p5-pre-recovery-check.json
```

The pre-recovery runner never writes `P5_FINAL_CHECK=PASS`, always keeps `P6_ALLOWED=false`, and leaves P5.3 blocked pending review.

## P5.3-P5.5 boundary

`configs/training/p5-recovery-plan.yaml` lists the decisions that must be frozen after reviewing P5.2: datasets, contamination policy, teacher/objective weights, trainable scope, length schedule, token/GPU budgets, optimizer schedule, checkpoint/validation cadence, and recovery thresholds.

No training process may create optimizer state until that plan reports `status: FROZEN` and passes `require_frozen_recovery_plan`. MLA, AttnRes, and SiTU-GLU are outside P5.

## Factorial attribution after an extreme shock

If P5.2 shows an extreme architecture shock, `scripts/diagnose_p5_factorial.py` isolates the two changes without creating another training stage:

| Case | Decay | Output gate | Checkpoint-valid |
|---|---|---|---|
| A | P4 qwen_compat | SiLU | yes |
| B | K3 lower-bounded | SiLU | no, diagnostic only |
| C | P4 qwen_compat | Sigmoid | no, diagnostic only |
| D | K3 lower-bounded | Sigmoid | yes |

The formal configuration validator remains unchanged and continues to reject B/C. The runner changes no tensor, converter, FLA operator, recurrence, cache contract, or formal numerical Gate. It replays A and D against the frozen P5.2 endpoints, supports completed-case resume, and writes `p5-factorial-attribution.json` after every case.

For each scalar metric it records:

```text
decay effect = B - A
gate effect = C - A
joint effect = D - A
interaction = D - B - C + A
```

The same descriptive scalar-metric decomposition is recorded for CE, Layer-0/final/logit relative L2, cosine distance, and all 32 hidden layers. B/C must never be serialized as model configurations or used as P5 completion artifacts.
