# P4 — 24-KDA/8-GQA Qwen-compatible Hybrid Base

## Scope

P4 mechanically applies the frozen P3 GDN-to-KDA `qwen_compat` converter at all 24 Qwen linear-attention layers and constructs the complete text-only 32-layer model. It does not change the P2 KDA operator contract or the P3 transplant mathematics.

The locked topology is:

- KDA: layers `0,1,2,4,5,6,...,28,29,30` (24 total)
- original Qwen-compatible GQA: layers `3,7,11,15,19,23,27,31` (8 total)
- original Qwen SwiGLU, PreNorm residual path, token embedding and final norm
- LM head tied to the token embedding
- vision and MTP parameters excluded

P5 decay/gate changes, MLA, NoPE, AttnRes, SiTU-GLU, recovery training, LoRA/QLoRA and FlashKDA are outside P4.

## Implementation map

- `src/densek3_core/modeling/configuration_densek3.py`: locked Hugging Face configuration
- `src/densek3_core/modeling/modeling_densek3.py`: text-only hybrid model and explicit GQA wrapper
- `src/densek3_core/modeling/hybrid_cache.py`: 24 recurrent/conv caches plus 8 GQA KV caches
- `src/densek3_core/transplant/full_model.py`: streaming, deterministic whole-checkpoint converter and manifest
- `src/densek3_core/evaluation/p4_runtime.py`: low-memory indexed server loaders
- `scripts/run_p4.py`: eight dependency-ordered gates
- `scripts/p4_reload_probe.py`: P4.7 fresh-process reload probe
- `scripts/diagnose_p4_cache.py`: read-only P4.6 full/segmented cache causal isolation
- `data/p4-smoke-prompts.jsonl`: five fixed generation prompts

## Static contracts

The converter must observe exactly 738 indexed donor tensors: 426 text, 297 vision and 15 MTP. The output contains exactly 546 unique saved tensors and 4,224,329,984 unique parameters. `lm_head.weight` is a declared alias of `model.embed_tokens.weight` and is therefore not saved twice.

Every donor tensor is classified exactly once as `COPY`, `SLICE`, `REPEAT`, `RESHAPE`, `CONSTRUCT` or `EXCLUDE`. `UNKNOWN`, `UNMAPPED` and `SILENT_INIT` are forbidden.

The whole-model checkpoint is sharded deterministically. The manifest records a SHA-256 for every output tensor, a canonical whole-parameter hash, all 24 P3 conversion hashes and a self-verifiable manifest hash.

## Gates and evidence

| Gate | Evidence file |
|---|---|
| P4.0 source inventory | `p4-source-inventory.json` |
| P4.1 24 converter invocations | `p4-layer-conversion.json` |
| P4.2 exact parameter accounting | `p4-parameter-accounting.json` |
| P4.3 model construction | `p4-model-construction.json` |
| P4.4 all 24 layers, FP32/BF16 T=7 | `p4-all-layer-local-parity.json` |
| P4.5 donor/target error propagation | `p4-error-propagation.json` |
| P4.6 prefill/decode/hybrid cache | `p4-hybrid-cache.json` |
| P4.7 fresh-process reload/generation | `p4-save-load-generate.json` |

P4.5 is deliberately a diagnostic-completion gate: it records the 32-layer hidden-state relative-L2, cosine and RMS-ratio curve, final hidden/logit metrics, CE difference and top-token overlap. It does not invent a relaxed whole-model tolerance. This evidence determines whether P5 needs a compatibility-recovery substage.

P4.4 uses the approved, scope-limited composite hierarchy `P4-NUM-2026-08-11-01`. Final output and recurrent state remain functional hard Gates; beta retains its existing Gate. The scalar-to-channel decay intermediates may take a causally verified auxiliary exception only after the formal runner revalidates the frozen Layer-4 FP32, parameter-embedding, forced-FP32, exact-injection, finite-value, and T=129 output-distribution evidence. The original P3 fallback is not directly consumed because its decay-existing-Gate prerequisite remains unchanged.

This hierarchy is never applied to P4.5 whole-model hidden states, logits, loss or generation. P4.5 remains pure measurement without a pre-approved full-model tolerance.

P4.6 uses the separately approved `P4-CACHE-NUM-2026-08-11-01` composite hierarchy. The frozen direct cache-semantics report is a hard prerequisite. Integrated full-vs-incremental logits, cache tensors, generation-relevant argmax/Top-10 decisions, and B=2-vs-B=1 controls must each satisfy their scope-limited checks. Unused prefill readouts are diagnostic-only. Global allclose remains unchanged. See `P4_CACHE_NUMERICS_DECISION_20260811.md`.

The P4.6 causal investigation is closed. A formal replay uses `python scripts/run_p4.py --replay-p4-6`; it revalidates saved P4.0-P4.5 evidence and unlocks P4.7 only after `PASS_WITH_COMPOSITE_CACHE_EQUIVALENCE_GATE`.

## Authoritative server run

Run from the repository root (`$DENSEK3_ROOT` below):

```bash
conda run -n zmm_2 python -m pip install -e . --no-deps
conda run -n zmm_2 ruff check pyproject.toml src tests scripts
conda run -n zmm_2 pytest tests/p4 tests/p3 tests/contract -q

mkdir -p outputs/densek3-core/manifests/p4
set -o pipefail
conda run -n zmm_2 --no-capture-output \
  python scripts/run_p4.py --reconvert \
  2>&1 | tee outputs/densek3-core/manifests/p4/p4-run.log
P4_RC=${PIPESTATUS[0]}
echo "P4_RUN_EXIT_CODE=${P4_RC}"
cat outputs/densek3-core/manifests/p4/p4-final-check.json
```

The first run should use `--reconvert`. If a later GPU gate fails after P4.1 has completed, rerun without `--reconvert`; the runner verifies and reuses the saved 8.4 GB-class checkpoint instead of rebuilding it. The amendment rerun also requires the existing `p4-layer-4-decay-diagnostics.json` and `p4-layer-4-t129-qkv-distribution.json` in the P4 manifest directory.

For the approved P4.6-only replay, use `--replay-p4-6` without `--reconvert`. It additionally requires `p4-cache-causal-diagnostics.json` in the P4 manifest directory and revalidates its artifact identity before running any formal case.

Long-running gates persist partial progress before raising. The final gate is authoritative only when it contains `P4_FINAL_CHECK=PASS` and `p5_allowed: true`.
