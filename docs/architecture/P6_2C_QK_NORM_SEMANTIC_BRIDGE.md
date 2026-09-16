# P6.2c Q/K Norm Semantic Bridge

P6.2c is a two-phase local diagnostic and calibration stage for layers 3 and
23 only. It preserves the formal rank-512 NoPE MLA contract and restores the
frozen donor Q/K RMSNorm semantics as the sole architectural variable.

Phase A compares three no-training cases on the frozen P6 calibration
validation inputs:

- N0: C3 with donor Q/K RMSNorm removed.
- N1: the same C3 factorization and latent scale with donor Q/K RMSNorm kept.
- N2: the P6.2b-2 best local weights with donor Q/K RMSNorm restored post-hoc.

N2 is diagnostic-only. Phase B is allowed only when layer 23 N1 improves over
N0 by at least 10%, or N1 reaches mixer relative L2 at most 0.70. The 0.70
value makes the approved approximate signal threshold machine-replayable and
does not change the formal local Gate.

If Phase A is positive, Phase B starts from a fresh N1 C3 initialization. It
does not resume P6.2b weights. Q/K norms remain frozen while the same latent
down projection, interleaved K/V up projection, and latent RMSNorm scale are
optimized with the unchanged attention-core plus mixer-output loss. The hard
cap is 256 steps and 65,536 activation tokens per layer.

The formal P6.2c GO decision retains the P6.2b local Gate and therefore keeps
N0, the original C3 baseline, as its formal comparison baseline. N1 is only
the fresh Phase B training starting point. Layer 23 must reach mixer relative
L2 at most 0.40 and at least 45% improvement from N0; layer 3 must retain its
existing safety checks. No full-dev, heldout, canonical, full-model
checkpoint, rank change, global training, or eight-layer scaleout is allowed
in this stage.
