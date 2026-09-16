# Security audit for the public release

Audit date: 2026-09-16  
Scope: the staged GitHub tree and the standalone `DenseK3-4B` Hugging Face
directory.

This is a release-hygiene audit, not a security certification. The audit was
repeated after the final model export and parity reports were copied into the
staging area.

## Findings

- No supplied SSH address, password, private username, private host, local
  machine path, donor checkpoint path, dataset payload, or credential value is
  present in either public tree.
- No private key, Hugging Face token, or hard-coded API-key assignment was
  found. The only secret-like values are names of environment variables and
  the literal documentation placeholder `<your-key>`.
- `titan/scripts/activate_p11_kimi_api.sh` reads the Kimi key only from a
  project-local mode-600 file, exports it for the child shell, and never
  prints the value. The file is excluded by `.gitignore`; the script's status
  messages do not contain the secret.
- No Q4 implementation, Q4 checkpoint, or promoted long-context branch is in
  the runtime/model payload. The long-context exploration is disclosed as
  attempted and not promoted; related gate field names in the P11 contracts
  are retained to make the fail-closed decision auditable.
- The standalone bundle contains only Python source, configuration,
  tokenizer, notices, JSON manifests/reports, and three safetensors shards.
  Generated Python caches were removed before staging.
- SHA-256 entries in `DenseK3-4B/SHA256SUMS` were checked against every staged
  file; the remote copy had no mismatches (`sha_bad=[]`).

## Safe-to-publish boundary

The GitHub repository contains reproducible architecture, migration,
training-contract, evaluation, and release tooling. The Hugging Face payload is
the effective P11.6 state exported as a standalone `DenseK3-4B` model. Private
research logs, failed experiments, donor weights, datasets, API credentials,
and unpromoted Q4/P12 artifacts remain outside both publication trees.

Maintainers should still review `NOTICE` and `THIRD_PARTY_NOTICES.md` for exact
upstream provenance before pushing or uploading. This document does not waive
any upstream license or model-card obligations.
