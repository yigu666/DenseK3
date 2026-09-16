# P11.6 Runtime-Patch Audit

The canonical research loader assembles a P10-T model, applies the P11.6 delta,
then installs several process-local runtime behaviors. This audit is the bridge to
the native `DenseK3-4B` implementation. The final standalone directory must not
require a user to call any of these patch functions.

| Behavior | Source / entry | Current mutation | Serialization today | Native destination |
|---|---|---|---|---|
| SiTU-GLU | `titan/situ/runtime.py::enable_situ_glu_runtime` | Replaces each `model.model.layers[i].mlp.forward`; computes FP32 capped gate/up and restores projection dtype | No: Python bound method is not in `state_dict`; folded scales are in P11.6 weights | `modeling_densek3.py::_p11_situ_mlp_forward`, with beta fields in config |
| SiTU down-scale folding | `titan/situ/runtime.py::fold_down_projection_scales` | Deterministically multiplies selected `W_down` columns | Yes: modified `down_proj.weight` is in delta/effective weights | Exported effective `down_proj.weight`; no post-load fold |
| Block AttnRes | `titan/attnres/block_reference.py::enable_block_attnres_runtime` | Replaces `model.model.forward` with depth-state read/branch/write loop | No: bound forward method is not serialized; `attnres.safetensors` stores only site parameters | Registered `model.attnres` module plus native forward hook installed by the model constructor |
| AttnRes parameters | `titan/attnres/block_reference.py::BlockAttnResReference` | 65 query/RMSNorm site modules, zero-query initialization contract | Separate `attnres.safetensors`, not part of model state today | `DenseK3ForCausalLM.attnres`, included in standalone state dict |
| True latent cache | `titan/scripts/p7_latent_cache.py::enable_p7_latent_cache_runtime` | Replaces cache class and `DenseK3MLA.forward` in the current process | No: cache is runtime state, not weights | `DenseK3HybridCache` alias to a native latent-cache class and native MLA forward |
| Latent reconstruction | `p7_latent_cache.py::reconstruct_kv_from_cache_layer` | Reconstructs temporary normalized K/V from latent segments and preserves append boundaries | No persistent K/V serialization | `cache_densek3.py`/bundled runtime code inside `modeling_densek3.py` |
| KDA backend | `densek3_core/kda/fla_backend.py` and model `_operator` | Selects FLA chunk/recurrent operators; reference fallback is explicit | Weights/config serialize backend choice; operator code is dependency code | Bundled backend adapter with the same gate mode, V-first state, and FP32 state |
| MLA semantics | `src/densek3_core/modeling/modeling_densek3.py::DenseK3MLA` | Native P6 module is expanded-cache reference; P7 replaces forward for latent cache | Projection weights serialize; patch does not | Native model class keeps strict-NoPE projections and latent forward |
| Precision hooks | `titan` loaders and runtime scripts | Loads model FP16; recurrent state/reductions are FP32; disables gradients | Dtype is partly inferred from tensors, not bound methods | Export manifest/config declares precision-sensitive boundaries; loader honors saved dtype |
| Generation | `DenseK3ForCausalLM.greedy_generate` | Manual cached greedy loop and attention-mask extension | Method code is not a weight artifact | Native class provides deterministic `.generate(..., do_sample=False)` and the same loop |
| Cache hooks | `DenseK3HybridCache` API (`reorder_cache`, repeat/select, reset) | Custom cache object passed through model forward | Runtime-only | Native DenseK3 cache object returned by prefill and continuation |
| Loader overrides | `evaluation/model_loader.py::load_p11` and `run_p10.apply_dense_overrides` | Loads P10 parent, applies P11 delta and AttnRes artifact, freezes/evals | Delta files serialize only overrides | Exporter materializes all effective tensors and records source provenance |
| Forward hooks / observers | P8 diagnostics (`branch_observer`, capture flags) | Optional diagnostics lists/callbacks on AttnRes runtime | Not serialized; disabled for canonical inference | Omitted from product runtime; diagnostics remain research code |

## Required removal condition

The standalone model constructor must create all required modules and semantics from
`config.json`. No `patch_attnres`, `enable_p7_latent_cache_runtime`,
`enable_situ_glu_runtime`, donor checkpoint, P10 parent, or P11 delta may be needed
after `AutoModelForCausalLM.from_pretrained()` returns. The export and parity scripts
must fail if a required module is missing, a state key is dropped, or a cache reports
persistent expanded K/V.

