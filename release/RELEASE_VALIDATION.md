# DenseK3-4B release validation

Validation date: 2026-09-16  
Public product: `DenseK3-4B`  
Internal provenance: `P11.6`

This report records inference/export gates only. No training, backward pass,
optimizer construction, checkpoint reselection, or new capability benchmark was
performed for the standalone export.

## Frozen reference

The reference was assembled by
`release/hf_export/reference_loader.py::load_canonical_densek3_reference`, which
delegates to the public P0-P11 loader and applies the frozen P11.6 delta. The
reference capture is stored in `release/hf_export/canonical_reference.json`.

| Check | Result |
|---|---:|
| Unique effective parameters (model + AttnRes) | 4,226,764,032 |
| Parameter entries | 692 |
| Model tensor dtype | FP16 |
| Recurrent-state boundary | FP32, V-first `[B,32,128,128]` |
| MLA cache layers | 8 (`3,7,11,15,19,23,27,31`) |

## Standalone export

`release/scripts/export_hf_standalone.py` materializes the effective state into
three safetensors shards (maximum shard size 4 GB), copies the pinned Qwen
tokenizer, emits Hub-local `AutoConfig`/`AutoModelForCausalLM` code, and writes a
weight map, manifest, and `SHA256SUMS`. The generated code contains no Q4 or
unpromoted long-context implementation.

## Clean-room parity gate

The gate was run on the authorized CUDA host with the `zmm_2` snapshot. The
standalone side ran in a fresh `python -I` process with `PYTHONPATH` cleared and
only the model directory supplied. The machine-readable report is
`release/hf_export/validation/PARITY_SUMMARY.json` and is copied into the model
directory as `PARITY_SUMMARY.json`.

| Gate | Result |
|---|---:|
| Hub-local `AutoConfig` and `AutoModelForCausalLM` load | PASS |
| 692 parameter names / 4,226,764,032 effective parameters | PASS |
| Fixed English, Chinese, code, and mixed prompts | PASS |
| Maximum absolute last-token logit difference | 0.017578125 (≤ 0.05) |
| Top-1 equality for all compared rows | PASS |
| Top-5 set equality for all compared rows | PASS |
| Cached prefill and two-token continuation | PASS |
| Eight latent caches `[B,T,512]`; persistent expanded K/V = 0 | PASS |
| Deterministic greedy generation (`max_new_tokens=4`) | PASS |
| `save_pretrained` safetensors round trip and reload | PASS |

Equal-valued logits can swap the order of a non-top-1 tie across kernels; the
report retains the diagnostic top-10 ordering while the release gate uses the
numeric threshold, top-1 identity, and top-5 set.

## Public-tree gates

- `python -m py_compile` passes for the public package and generated Hub code.
- The final staged public tree passed `python -m pytest -q tests titan/tests`
  on the authorized `zmm_2` environment: `281 passed, 2 skipped` in 14.31s.
  The skips require separately downloaded donor metadata and are not
  model-quality claims.
- A final security scan found no credentials, private hosts, absolute server
  paths, donor weights, datasets, or generated caches in the GitHub tree. The
  only secret-like strings are environment-variable names and explanatory test
  placeholders.
- The detailed release-hygiene record is `release/SECURITY_AUDIT.md`; it also
  records the clean standalone bundle boundary and SHA-256 verification.
- `THIRD_PARTY_NOTICES.md` preserves upstream attribution and license boundaries;
  maintainers must complete the noted legal provenance review before publication.

## Upload gates

The maintainer publication targets are GitHub
`https://github.com/yigu666/DenseK3` and Hugging Face
`https://huggingface.co/qinfu19/DenseK3-4B`. The GitHub tree and standalone model
directory remain available locally under the release staging area for audit and
future synchronization. Re-check live links and Hub file hashes after any
additional publication change. Review `RELEASE_SCOPE.md`, `MODEL_CARD.md`,
`NOTICE`, and `THIRD_PARTY_NOTICES.md` before distributing further updates.
