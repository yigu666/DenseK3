# DenseK3 public-documentation restructure audit

Audit date: 2026-09-16  
Scope: the staged GitHub tree and the standalone `DenseK3-4B` model directory.  
Method: read the current Markdown/YAML/code blocks and compare relative links
against the staged tree. No model, checkpoint, evaluation result, or scientific
claim was changed during this audit.

## 1. Responsibilities currently carried by the GitHub README

`README.md` currently mixes a useful project overview with model-card and release
material. It contains architecture facts, all benchmark tables, memory/runtime
measurements, installation and inference, dataset commands, repository layout,
scope limitations, model-weight formats, acknowledgements, citation, and license.
The first paragraph nevertheless starts from the internal P11.6 provenance and
the migration outcome, so the project motivation is implicit rather than explicit.

`README_zh.md` is substantially longer than the English page and repeats many of
the same architecture/evaluation facts. It also carries installation, model-weight
provenance, release-scope, and validation details that belong in linked documents.
Its headings and emoji navigation do not match the English structure, and its
first-screen language uses P11.6 as a visible model qualifier.

## 2. Responsibilities currently carried by the Hugging Face cards

The English card provides basic YAML, a short model description, architecture,
donor, intended/out-of-scope use, loading code, evaluation, memory, limitations,
reproducibility, citation, and license. It is missing a distinct training-data
disclosure, dedicated safety/bias section, explicit standalone-validation section,
and a clear GitHub/English/Chinese navigation block. It also repeats technical
details that are more useful in `docs/` and mentions P11.6 in multiple sections.

The Chinese card is only a short translation-style note with architecture, loading,
results, a long-context disclosure, and license. It does not mirror the English
card's responsibilities and has no YAML, intended-use/safety section, training-data
section, standalone-validation section, or provenance/reproducibility section.

## 3. Repeated material

The following is repeated across README, both HF cards, and technical docs:

- 32-layer topology, 24 KDA + 8 strict-NoPE MLA, latent cache, AttnRes, and SiTU;
- MMLU, WikiText, LongBench MC-LL, and reduced-RULER aggregates;
- 32-to-8 KiB/token cache accounting and the 512K runtime probe;
- donor identity and the P0-P11 migration narrative;
- custom-code loading and Apache/upstream-license language.

The headline facts may remain concise in both front pages. Equations, tensor
mapping, contracts, complete benchmark protocol, and release evidence should have
one technical home under `docs/` or `release/`.

## 4. Content to move or demote

- Move complete equations, tensor shapes, SVD/scale details, and normalization
  derivations to `docs/ARCHITECTURE.md` and `docs/MIGRATION.md`.
- Keep the staged migration story and recovery/distillation summary in GitHub;
  link detailed contracts and commands to `docs/MIGRATION.md`, `docs/TRAINING.md`,
  and `docs/REPRODUCIBILITY.md`.
- Keep release gate numbers only as a compact link/summary in the project README;
  retain the full clean-room evidence in `release/RELEASE_VALIDATION.md` and
  `PARITY_SUMMARY.json`.
- Move security/provenance mechanics out of the front-page narrative and point to
  `release/SECURITY_AUDIT.md`.
- Make `release/hf/DenseK3-4B/README*.md` model-use documents: model facts,
  loading, training/data disclosure, evaluation, limitations, safety, and
  provenance. Do not copy the full technical report there.

## 5. Conflicting or stale wording found

- Public identity is inconsistently framed as “DenseK3-4B (P11.6)” or “P11.6
  model”. The public name must be `DenseK3-4B`; P11.6 is provenance only.
- The English README is model-first and release-first, while the required project
  story needs an explicit “Why DenseK3?” motivation before architecture.
- The Chinese README reports the old split test counts (`223 + 56`) while the
  final staged-tree validation is `281 passed, 2 skipped`; front-page tests should
  link the validation report instead of duplicating stale counts.
- 512K byte values are authoritative only when written as allocated
  `15,762,103,296` bytes (`14.68 GiB`), reserved `15,994,978,304` bytes, and
  persistent latent `4,294,967,296` bytes. No page may imply 512K task quality or
  total-VRAM reduction.
- LongBench is a controlled comparative MC-LL metric, not the official generation
  leaderboard; RULER is a reduced `13 × 4` suite. Both labels must be consistent.
- The English HF card contains a YAML block but omits `model_name`, `language`, and
  a stable GitHub link/navigation. The base model is represented as a list in one
  card and should be normalized to the validated metadata form.
- The HF cards do not state whether the base model is instruction-tuned, whether
  dedicated safety/bias/red-team evaluations exist, or what training data was and
  was not redistributed.

## 6. Link audit

All currently used repository-relative links in the targeted public entry points
(`README*.md`, `MODEL_CARD.md`, and the standalone HF cards) resolve to files that
exist, including `docs/*`, `release/*`, `LICENSE`, `NOTICE`, and
`THIRD_PARTY_NOTICES.md`. The current publication targets are GitHub
`yigu666/DenseK3` and Hugging Face `qinfu19/DenseK3`; live availability is an
external state and is not re-fetched by this documentation audit. Historical
architecture audit pages retain links to
local reference PDFs that are intentionally excluded from the public tree; those
pages are not linked as release entry points and are outside this link claim.

## 7. Commands that need correction or clearer context

- The GitHub README currently presents environment installation before the
  user-facing standalone model path. Reorder it so `Use DenseK3-4B` comes first,
  followed by `Development / Reproduction`.
- The old English page does not provide a release-level distinction between
  direct HF inference and source reconstruction. The new pages must show both
  paths explicitly.
- Dataset commands are valid documented entry points, but the pages must say that
  they download/materialize data only when the reader chooses to reproduce an
  evaluation; no dataset is bundled.
- The export/parity commands in `docs/REPRODUCIBILITY.md` are the authoritative
  release commands and should be linked rather than duplicated in full.
- HF loading must keep `trust_remote_code=True`, install the pinned dependencies,
  and use the standalone `qinfu19/DenseK3` repository; it must not require a
  GitHub clone or a donor/P10/P11 checkpoint.

## 8. Metadata corrections required

The HF cards should use valid front matter with `model_name: DenseK3-4B`,
`library_name: transformers`, `pipeline_tag: text-generation`,
`license: apache-2.0`, `base_model: Qwen/Qwen3.5-4B-Base`, languages `en`/`zh`,
and only supported tags (`densek3`, `qwen`, `kda`, `mla`, `long-context`,
`custom-code`). The release remains an independent research model, not an
official Kimi/Moonshot release.

## 9. Restructure acceptance criteria

After the rewrite:

1. GitHub English/Chinese pages share the same emoji-led section order and open
   with the research question, not P11.6 or a delta checkpoint.
2. HF English/Chinese cards share a model-card section order and put Quick Start
   near the top.
3. P11.6 appears in front-page material only as explicit frozen provenance; it is
   not the public model name.
4. Training-data disclosure states verified sources/roles and marks anything not
   supported by frozen artifacts as `NOT VERIFIED FROM RELEASE ARTIFACTS`.
5. Link, YAML, command, naming, 512K, and metric audits pass without changing
   scientific values or model payloads.

This file is the pre-change audit snapshot. Completion and post-change checks are
recorded in [`DOCS_RESTRUCTURE_FINAL.md`](DOCS_RESTRUCTURE_FINAL.md).
