# P6.1c — Activation-Aware Rank-512 MLA Initialization

Status: frozen, no-training 2x2 initialization experiment.

P6.1c reads the frozen P5 I1 checkpoint plus the completed P6.1/P6.1b
manifests. It changes only layer 3 in memory. Rank remains 512, position
encoding remains NoPE, Q/gate projections remain split, donor Q/K RMSNorm is
excluded, true latent cache is forbidden, and the other seven GQA layers are
untouched. It writes no model checkpoint and never reads heldout data.

## Cases

- C0: frozen weight-only SVD plus latent RMSNorm weight 1.
- C1: frozen weight-only SVD plus closed-form scale-matched latent RMSNorm.
- C2: activation-weighted SVD plus latent RMSNorm weight 1.
- C3: activation-weighted SVD plus closed-form scale-matched latent RMSNorm.

C0 must replay the frozen P6.1 factor and canonical hashes. C1 and C3 fit their
RMSNorm scales independently because their latent bases differ.

## W1 activation-weighted factorization

The calibration source is exactly train sequences 0 through 7 at length 2048.
Frozen P5 normalized layer-3 inputs define `G = X^T X / N`. The implementation
uses deterministic IEEE FP32 covariance accumulation, exact eigendecomposition,
and exact SVD with TF32 disabled. It factors the four unique donor GQA `[K;V]`
head blocks and expands the left factor to 16 heads; this is mathematically
equivalent to factoring the repeated full matrix.

W1 minimizes the calibration-weighted projection error. Weight-space energy and
Frobenius reconstruction remain diagnostics and are never used as acceptance
Gates.

## N1 scale matching

For each latent channel, N1 solves the closed-form least-squares problem
`gamma * RMSNorm(c) ~= c` on all frozen calibration latents. No optimizer,
gradient, seed-dependent approximation, or learned calibration is allowed.

## Selection and Gate

Candidate selection uses only the first 256 tokens of the same eight frozen
train sequences. Its lexicographic order is layer-3 mixer relative L2,
attention-output relative L2, post-residual relative L2, final-hidden cosine,
logits cosine, and calibration CE. Canonical results are diagnostic only.

Exactly one of C1/C2/C3 is selected and receives the only 97-sequence full-dev
evaluation. P6.1c passes only if all conditions hold:

- selected mixer relative L2 versus B is at most 0.45;
- selected full-dev CE minus B full-dev CE is at most 0.01;
- finite values and stable five-category generation;
- final-hidden and logits cosine both improve over C0 on the frozen calibration
  subset.

A pass authorizes P6.2 but never P7. A failure remains `REDESIGN`; it does not
authorize a rank sweep, training, a Gate change, or heldout access.
