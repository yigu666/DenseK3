# P6.2d Eight-layer Q/K-norm-preserving scaleout

P6.2d applies the P6.2c protocol independently to all eight former GQA
layers: activation-weighted rank-512 C3, scale-matched latent RMSNorm, donor
Q/K RMSNorm present from initialization, and the unchanged local
attention-core plus mixer-output calibration objective.

Layers 3 and 23 reuse the validated P6.2c best trainable tensors only after
source, configuration, tensor, frozen-Q/K-norm, and validation-metric replay
checks pass. Layers 7, 11, 15, 19, 27, and 31 start from fresh N1 and receive
at most 256 optimizer steps or 65,536 activation tokens each. Every target
and teacher input is captured from the untouched frozen P5 donor; sequential
self-conditioning is forbidden.

The local scaleout safety contract does not impose the layer-23 pilot's 0.40
threshold on every layer. It requires finite values, a fresh-N1 mixer error
at most 0.70, and a selected calibrated mixer error no worse than that
layer's own N1 baseline and at most 0.70. Best validation checkpoint within
the fixed budget is used. Fresh N1 is the eligible step-zero checkpoint, so
an optimizer checkpoint that improves the combined selection objective but
worsens mixer error cannot displace the safe initialization.

Exactly one native 24-KDA/8-MLA candidate is assembled with formal rank 512,
NoPE, preserved donor Q/K RMSNorm, and expanded/reference KV cache. The
candidate is written once, then evaluated once on the frozen 97-sequence dev
split. The only capability Gate is full-dev CE no more than 0.15 above the
frozen P5 baseline, with finite execution and non-collapsed five-category
generation. Full-model drift is diagnostic; canonical and heldout data are
not accessed. P6.3 remains blocked unless this composite Gate passes.
