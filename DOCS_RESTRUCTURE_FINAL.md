# DenseK3 documentation restructure — final report

Date: 2026-09-16  
Scope: public GitHub documentation, standalone `DenseK3-4B` model cards, and
the model-card generator template only.

This report follows [`DOCS_RESTRUCTURE_AUDIT.md`](DOCS_RESTRUCTURE_AUDIT.md).
No model implementation, checkpoint, weight shard, tokenizer, evaluation code,
metric value, or scientific result was changed.

## GitHub README

`README.md` and `README_zh.md` now have the same navigation and section order:

1. project identity and language switch;
2. `✨ Highlights`;
3. the research motivation in `💡 Why DenseK3?`;
4. architecture overview and the six requested component subsections;
5. the Qwen-to-DenseK3 migration story, including joint recovery and both
   distillation roles;
6. exact evaluation tables;
7. memory accounting and a separate exact-semantics long-context runtime section;
8. standalone-model-first quick start, then development/reproduction;
9. repository structure, documentation map, scope, acknowledgements, citation,
   and license.

The front page now uses `DenseK3` as the project name and `DenseK3-4B` as the
model name. The internal label `P11.6` appears only as provenance context. The
full derivations, tensor mapping, frozen contracts, and release evidence remain
in their technical homes under `docs/` and `release/`; the front page links to
them instead of duplicating every equation or gate.

The former separate “model weights” narrative was reduced to a repository-tree
note: in the release staging area, `release/hf/DenseK3-4B/` is the standalone
product, while `model/densek3-p11.6/` is an auditable provenance-only delta
payload. Later
long-context architecture work is disclosed as an attempted, unpromoted
exploration. Its implementation, checkpoints, failure logs, and detailed
artifacts remain omitted; the boundary is documented in
[`docs/RELEASE_SCOPE.md`](docs/RELEASE_SCOPE.md).

The linked technical result pages were also normalized where a front-page reader
could otherwise mistake provenance for a model name: `docs/RESULTS.md` now labels
the released column `DenseK3-4B`, and `docs/LONG_CONTEXT.md` uses the same
runtime/quality boundary wording. Historical architecture audit pages no longer
link to reference PDFs that are intentionally excluded from the public tree.

## Hugging Face model cards

`release/hf/DenseK3-4B/README.md` and `README_zh.md` are concise model cards with
the same YAML fields, section responsibilities, and order:

- valid Hub metadata (`model_name`, `library_name`, `pipeline_tag`, `license`,
  `base_model`, `language`, and the six release tags);
- a short independent-model definition and non-official Moonshot/Kimi statement;
- model-first loading with `AutoTokenizer`, `AutoModelForCausalLM`,
  `trust_remote_code=True`, and an explicit FLA dependency;
- architecture facts, intended use, migration/recovery/distillation summary;
- a source-grounded training-data disclosure;
- exact frozen benchmark aggregates and context/memory boundaries;
- a dedicated standalone parity section;
- base-model, safety, bias, toxicity, red-team, and production limitations;
- provenance, reproducibility, acknowledgements, citation, and license.

Neither card puts an internal stage label in the opening paragraph. `P11.6` is
mentioned once, in `🧬 Provenance and Reproducibility`, and nowhere else in the
cards. The cards intentionally omit full equations, SVD derivations, and the
65-site derivation; those details are linked from GitHub.

The `_model_card()` and `_model_card_zh()` templates in
[`release/scripts/export_hf_standalone.py`](release/scripts/export_hf_standalone.py)
were updated together with the materialized cards so a future export does not
silently restore the old sparse card or its one-paragraph Chinese stub.

## Training-data disclosure

The cards disclose only values supported by release artifacts:

- Qwen donor initialization;
- the 1,024-record general prompt pool and its four verified domain counts;
- the train-only FineWeb-Edu probe accounting of 1,998,848 effective tokens
  (976 × 2,048);
- public-source provenance references to MMLU, ARC, Belebele, and IFEval;
- same-tokenizer Qwen OPD and API-only reviewed Kimi corrective text, without
  teacher responses, credentials, or private caches.

The exact final post-selection record count and total token count are explicitly
marked `NOT VERIFIED FROM RELEASE ARTIFACTS`. No value was inferred from a
benchmark size, model name, or internal path.

## Naming and identity

| Context | Canonical wording |
|---|---|
| Project | `DenseK3` |
| Public model | `DenseK3-4B` |
| Internal provenance | `P11.6` (provenance only) |
| Donor | `Qwen/Qwen3.5-4B-Base` |
| Official-status statement | Independent research model; not an official Moonshot AI/Kimi release |

No document presents the delta payload as a second public model or claims to
reproduce complete Kimi K3.

## Unified long-context wording

All public entry points now distinguish three claims:

1. **State accounting:** approximately 32 KiB/token donor-equivalent GQA versus
   8 KiB/token DenseK3 latent state, a derived 75% reduction in the persistent
   sequence-growing attention-state payload only.
2. **Runtime probe:** separate exact-semantics P10-T probe at 524,288 tokens,
   with full prefill and continued decode PASS, allocated
   `15,762,103,296` bytes (≈14.68 GiB), reserved `15,994,978,304` bytes,
   persistent latent `4,294,967,296` bytes (4 GiB), and expanded K/V 0.
3. **Quality:** reported quality evidence stops at 128K; the runtime probe is not
   512K task-quality or throughput evidence and is not a total-VRAM claim.

LongBench is consistently named the controlled MC-LL Accuracy subset (not the
official generation leaderboard), and reduced RULER is consistently described
as 13 tasks × 4 cases per context length (52 cases per length).

## Model loading verification

The public snippets use the standalone Hub ID `qinfu19/DenseK3`,
`AutoTokenizer`, `AutoModelForCausalLM`, `trust_remote_code=True`, and
`torch_dtype=torch.float16`. The cards identify custom code and
`flash-linear-attention==0.5.2`; the GitHub quick start points to the pinned
`requirements.txt` snapshot. No snippet requires a donor checkpoint or writes to
a system drive. The code is presented as deterministic greedy decoding, matching
the release contract.

## Link and command audit

- All repository-relative Markdown links in the two GitHub READMEs and both
  model cards resolve within the staged tree or to the intentional public GitHub/
  Hugging Face destinations.
- The GitHub test command is the final staged-tree command:
  `python -m pytest -q tests titan/tests --disable-warnings`.
- Dataset commands are download/prepare commands only; datasets and donor
  weights remain excluded from the release bundle.
- No private server address, credential, API key, teacher response, local cache,
  or Windows-specific path is present in the rewritten public entry points.

## Remaining blockers

1. The final post-selection training-record and total-token counts remain
   `NOT VERIFIED FROM RELEASE ARTIFACTS`; this is disclosed rather than guessed.
2. The maintainer has now published GitHub `yigu666/DenseK3` and Hugging Face
   `qinfu19/DenseK3`; live link and Hub-hash checks should be re-run after these
   URL edits, because this workspace does not re-fetch remote state.
3. Native hardware and dependency availability remain environment-specific; the
   release does not promote the provisional reduced-scope recovery evidence to a
   native full retrain claim.

These blockers do not require changing the model or its reported results. They
are retained as provenance and publication checks for maintainers.
