# Reproducibility guide

## 1. Environment

The authorized verification run used the server environment `zmm_2` on an RTX
5090. Its snapshot uses Python 3.13.11, PyTorch
2.8.0+cu128, Transformers 5.12.1, FLA 0.5.2, Triton 3.4.0, and NVIDIA driver
595.84. `requirements.txt` is the package snapshot; environment creation is
intentionally left to the user:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

CUDA wheels are platform-specific; use the official PyTorch CUDA 12.8 wheel
index when installing the pinned `torch==2.8.0+cu128` snapshot.

The historical P7-P11 Titan runs used a separate SM75/FP16 runtime with a
project-local Triton 3.3 overlay. Those scripts and exact contracts remain in
`titan/`, but their results must not be relabeled as native RTX 5090 BF16 runs.

## 2. Donor and datasets

No dataset is committed. Keep caches and generated data outside Git history:

```bash
export DENSEK3_ROOT="$PWD"
export HF_HOME="$DENSEK3_ROOT/.cache/huggingface"
export DENSEK3_EVAL_OUT="$DENSEK3_ROOT/results/reproduced"
python scripts/download_evaluation_assets.py --root "$DENSEK3_ROOT"
python evaluation/prepare_ruler_data.py
```

The downloader materializes:

- `Qwen/Qwen3.5-4B-Base` at revision
  `1001bb4d826a52d1f399e183466143f4da7b741b`;
- `cais/mmlu` (`all`), `Salesforce/wikitext`
  (`wikitext-103-raw-v1` test), and `THUDM/LongBench-v2` (train);
- LongBench at `2e00731f8d0bff23dc4325161044d0ed8af94c1e`;
- NVIDIA RULER at `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`.

P5/P10/P11 training data is also excluded. Its public materialization entry
points are `scripts/materialize_p5_probe_corpus.py` and
`titan/scripts/materialize_p11_general.py`; FineWeb-Edu must be downloaded by
the user under its upstream terms.

## 3. P0-P11 stage entry points

Run commands from the repository root. Every stage is fail-closed and expects
the preceding stage's approved artifacts.

```bash
# P2 KDA reference/parity
python scripts/run_p2.py --through P2.12

# P3 deterministic GDN-to-KDA transplant and P4 full hybrid conversion
python scripts/run_p3.py --model-dir models/Qwen3.5-4B-Base --device cuda
python scripts/run_p4.py --model-dir models/Qwen3.5-4B-Base

# P5 KDA recovery and P6 NoPE-MLA migration
python scripts/run_p5_rethink_v3_initialization.py
python scripts/run_p5_i1_closeout.py
python scripts/run_p6_initialization.py
python scripts/run_p6_function_calibration.py
python scripts/run_p6_qk_norm_bridge.py
python scripts/run_p6_qk_norm_scaleout.py

# P7 latent cache, P8 Block AttnRes, P9 SiTU-GLU, P10 joint recovery
python titan/scripts/run_p7.py
python titan/scripts/run_p8_reference.py
python titan/scripts/run_p9.py
python titan/scripts/run_p10_t.py

# P11 dual-teacher OPD; API key is read only from the environment
export MOONSHOT_API_KEY="<your-key>"
python titan/scripts/run_p11_cost_calibration.py
bash titan/scripts/run_p11_fast.sh --stage all
```

Consult the frozen YAML contracts before expensive runs. Do not relax gates or
replace missing inputs silently.

## 4. Canonical checkpoint placement and evaluation

The public DenseK3-4B package is a delta checkpoint over the reproduced P10-T
parent, not a standalone copy of all donor and intermediate weights. This
canonical state is internally provenance-tracked as P11.6. Place the delta
payload (overrides, AttnRes weights, candidate manifest, and runtime
configuration; the accompanying source note and SHA file are optional metadata)
at `models/densek3-p11.6/` and set the parent paths if they differ:

```bash
export DENSEK3_QWEN_PATH="$PWD/models/Qwen3.5-4B-Base"
export DENSEK3_P10_PATH="$PWD/outputs/densek3-core/checkpoints/p10-t/densek3-4b-core-k3-joint-reduced-titan"
export DENSEK3_P11_PATH="$PWD/models/densek3-p11.6"
export DENSEK3_TOKENIZER_PATH="$DENSEK3_QWEN_PATH"
python evaluation/run_capability_benchmarks.py prepare
python evaluation/run_capability_benchmarks.py orchestrate --poll-seconds 60
python evaluation/run_capability_benchmarks.py summarize
```

The orchestrator is inference-only, admits one job at a time according to free
GPU memory, resumes completed per-case files, and labels shared-GPU throughput
as diagnostic rather than a formal speed comparison.

## 5. Standalone `DenseK3-4B` export and parity

The export is inference-only. Freeze the authorized P11.6 reference first, then
materialize the effective state into a standalone Hugging Face directory. The
commands below keep all caches and outputs under the project volume:

```bash
export DENSEK3_ROOT="$PWD"
export HF_HOME="$DENSEK3_ROOT/.cache/huggingface"
export DENSEK3_QWEN_PATH="$DENSEK3_ROOT/models/Qwen3.5-4B-Base"
export DENSEK3_TOKENIZER_PATH="$DENSEK3_QWEN_PATH"
export DENSEK3_P10_PATH="$DENSEK3_ROOT/outputs/densek3-core/checkpoints/p10-t/densek3-4b-core-k3-joint-reduced-titan"
export DENSEK3_P11_PATH="$DENSEK3_ROOT/models/densek3-p11.6"
export DENSEK3_EXPORT="$DENSEK3_ROOT/release/hf/DenseK3-4B"

python release/hf_export/freeze_reference.py \
  --checkpoint "$DENSEK3_P11_PATH" \
  --output release/hf_export/canonical_reference.json
python release/scripts/export_hf_standalone.py \
  --checkpoint "$DENSEK3_P11_PATH" \
  --output "$DENSEK3_EXPORT" \
  --loader release.hf_export.reference_loader:load_canonical_densek3_reference
python release/scripts/validate_hf_standalone.py \
  --mode compare \
  --checkpoint "$DENSEK3_P11_PATH" \
  --model-dir "$DENSEK3_EXPORT" \
  --output-dir release/hf_export/validation \
  --loader release.hf_export.reference_loader:load_canonical_densek3_reference
```

The validator starts the standalone side with `python -I`, clears
`PYTHONPATH`, checks structure/logits/cache/greedy generation, and performs a
`save_pretrained` safetensors reload. It does not train, select a checkpoint, or
download a dataset. See `release/RELEASE_VALIDATION.md` for the recorded
authorized-host result and `release/hf_export/validation/PARITY_SUMMARY.json`
for machine-readable gates.

## 6. Optional publication commands

Publication is deliberately manual. After reviewing the release scope, model
card, notices, and security audit, push the GitHub source tree and upload the
standalone model directory separately:

```bash
# GitHub source tree
# Run from the local staging parent that contains `github/` and `release/`.
cd /path/to/open_source_release
export DENSEK3_STAGING_ROOT="$PWD"
cd "$DENSEK3_STAGING_ROOT/github"
git init
git add .
git commit -m "Release DenseK3-4B canonical checkpoint"
git branch -M main
git remote add origin https://github.com/yigu666/DenseK3.git
git push -u origin main

# Hugging Face model (requires a prior `hf auth login`)
hf upload yigu666/DenseK3-4B \
  "$DENSEK3_STAGING_ROOT/release/hf/DenseK3-4B" . --repo-type model
```

Do not commit datasets, donor weights, credentials, generated caches, or
unpromoted Q4/long-context artifacts.
