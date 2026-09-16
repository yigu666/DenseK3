# P6.2b-1 — Attention-Function-Aware Local Calibration

P6.2b-1 follows the P6.2a result that neither rank-512 capacity nor sequential
input-distribution amplification explains the eight-layer scale-out failure.
It is a two-layer local pilot, not full-model recovery training. Layers 3 and
23 represent the easiest and hardest frozen C3 conversions.

Both students start from the exact P6.2 C3 rank-512 factors and scale-matched
latent RMSNorm. The teacher is the matching full-rank NoPE case B on the same
frozen P5 hidden inputs. Formal NoPE, removed Q/K normalization, rank 512,
Sigmoid gate, and output-projection semantics remain unchanged.

Only `kv_a_proj.weight`, the interleaved K/V rows of `kv_b_proj.weight`, and
`kv_a_layernorm.weight` are optimized. Q, gate, and output projections are
hash-checked before and after fitting. Every remaining model parameter is
frozen. The loss is the sum of normalized MSE on attention-core and mixer
outputs; K/V reconstruction, LM CE, full-model hidden, canonical, dev, and
heldout losses are absent.

The frozen train corpus is split by document membership before local fitting.
The pilot uses 128 length-256 fit sequences (32,768 activation tokens per
layer) and 32 length-256 validation sequences (8,192 tokens per layer), with
zero document overlap. A single AdamW configuration runs one fixed-order pass
of 128 steps per layer. BF16 validation at steps 0, 32, 64, 96, and 128 selects
the checkpoint minimizing attention-core plus mixer relative-L2.

The pilot returns GO only when layer 23 reaches mixer relative-L2 at most 0.40
and improves by at least 45%, layer 3 remains at most 0.32 and within 20% of
its independent validation baseline, both attention-core errors improve by at
least 20%, and all values remain finite. These local thresholds do not modify
the frozen P6 full-dev Gate.

The best six trainable tensors may be saved as a local calibration artifact.
It is not a full-model checkpoint, formal candidate, optimizer resume, or P6
freeze. P6.3 and P7 remain blocked. Only a GO authorizes a separately approved
eight-layer calibration scale-out.
