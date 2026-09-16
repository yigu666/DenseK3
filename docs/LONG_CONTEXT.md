# Long-Context and Memory Evidence

DenseK3 separates runtime feasibility from effective-context task quality. The two
must not be conflated.

## Validated task quality: through 128K

The public quality record covers LongBench-v2 examples whose tokenized length is at
most 131,072 and a reduced fixed-sample RULER suite at 4K, 8K, 16K, 32K, 64K, and
128K. DenseK3-4B does not exceed the Qwen donor on these aggregate measurements. See
[RESULTS.md](RESULTS.md).

## Runtime feasibility: 512K

A separate exact-semantics P10-T runtime probe completed a 524,288-token full
prefill (**PASS**) followed by continued autoregressive decoding (**PASS**). It
used true latent cache, no sliding window, and no approximate attention. The
artifact recorded:

```text
peak allocated bytes:       15,762,103,296
peak reserved bytes:        15,994,978,304
persistent latent bytes:     4,294,967,296
persistent expanded K:                   0
persistent expanded V:                   0
```

The exact allocated value is approximately 14.68 GiB. Some historical summaries
rounded allocator observations differently; the byte count above is authoritative
for this published statement. This probe validates runtime execution and cache
semantics, not DenseK3-4B task quality at 512K.

## Derived cache scaling

For eight attention layers at two bytes per element:

```text
Qwen-equivalent GQA: 8 layers * 2 (K,V) * 4 KV heads * 256 dim * 2 bytes
                     = 32768 bytes/token = 32 KiB/token

DenseK3 latent:      8 layers * 512 latent dim * 2 bytes
                     = 8192 bytes/token = 8 KiB/token
```

| Context | Qwen-equivalent GQA | DenseK3 latent | Status |
|---:|---:|---:|---|
| 128K | ~4 GiB | ~1 GiB | derived |
| 256K | ~8 GiB | ~2 GiB | derived |
| 512K | ~16 GiB | ~4 GiB | derived; DenseK3 latent bytes also measured |

The 75% reduction concerns sequence-growing persistent attention state only. It
does not include weights, KDA recurrent/convolution state, temporary activations,
allocator fragmentation, tokenizer buffers, or framework overhead.

## Explicit limitations

- 512K runtime support is not 512K validated effective-context quality.
- Standard quality benchmarks currently stop at 128K.
- No speed advantage over Qwen is claimed.
- Long-context quality is not claimed to exceed Qwen.
- The later noncanonical research branch and its artifacts are outside the public
  implementation scope; see [RELEASE_SCOPE.md](RELEASE_SCOPE.md).
