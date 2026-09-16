from __future__ import annotations

from pathlib import Path

import torch

from densek3_core.evaluation.p4_runtime import load_densek3_model
from densek3_core.modeling import DenseK3Config, DenseK3ForCausalLM, DenseK3HybridCache
from densek3_core.modeling.modeling_densek3 import DenseK3KDA
from densek3_core.transplant.full_model import DeterministicShardWriter
from densek3_core.transplant.gdn_to_kda import convert_gdn_to_kda
from densek3_core.transplant.single_layer import DenseK3QwenCompatMixer
from tests.p3.helpers import small_bridge_config, small_source_state


def small_config() -> DenseK3Config:
    return DenseK3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        mixer_types=["kda", "gqa", "kda", "gqa"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        partial_rotary_factor=0.5,
        kda_qk_num_heads=2,
        kda_value_num_heads=4,
        kda_qk_head_dim=4,
        kda_value_head_dim=4,
        kda_decay_projection_rank=8,
        kda_conv_kernel_size=3,
        kda_backend="reference",
    )


def test_full_forward_matches_tokenwise_hybrid_cache() -> None:
    torch.manual_seed(20260811)
    model = DenseK3ForCausalLM(small_config()).eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        full = model(tokens, use_cache=False).logits
        cache = None
        incremental = []
        for index in range(tokens.shape[1]):
            output = model(
                tokens[:, index : index + 1],
                attention_mask=torch.ones((1, index + 1)),
                past_key_values=cache,
                use_cache=True,
            )
            cache = output.past_key_values
            incremental.append(output.logits)
    assert torch.allclose(full, torch.cat(incremental, dim=1), rtol=2e-5, atol=2e-5)
    assert cache.seen_tokens == tokens.shape[1]
    assert all(cache.gqa_layer(index).sequence_length == tokens.shape[1] for index in (1, 3))
    assert all(cache.kda_layer(index).recurrent_state.dtype == torch.float32 for index in (0, 2))


def test_cache_reorder_and_batch_isolation() -> None:
    config = small_config()
    cache = DenseK3HybridCache(config)
    cache.kda_layer(0).recurrent_state = torch.arange(2 * 4 * 4 * 4).view(2, 4, 4, 4).float()
    cache.gqa_layer(1).update(torch.randn(2, 2, 3, 4), torch.randn(2, 2, 3, 4))
    cache.advance(3)
    expected_kda = cache.kda_layer(0).recurrent_state[1].clone()
    expected_gqa = cache.gqa_layer(1).key[1].clone()
    cache.reorder_cache(torch.tensor([1, 0]))
    assert torch.equal(cache.kda_layer(0).recurrent_state[0], expected_kda)
    assert torch.equal(cache.gqa_layer(1).key[0], expected_gqa)
    assert cache.seen_tokens == 3


def test_small_model_save_reload_is_exact(tmp_path: Path) -> None:
    torch.manual_seed(7)
    model = DenseK3ForCausalLM(small_config()).eval()
    tokens = torch.tensor([[3, 4, 5]])
    with torch.no_grad():
        expected = model(tokens, use_cache=False).logits
    model.save_pretrained(tmp_path)
    reloaded = DenseK3ForCausalLM.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        actual = reloaded(tokens, use_cache=False).logits
    assert torch.allclose(expected, actual, rtol=1e-7, atol=1e-7)
    assert reloaded.lm_head.weight is reloaded.model.embed_tokens.weight


def test_greedy_generation_uses_and_extends_hybrid_cache() -> None:
    torch.manual_seed(9)
    model = DenseK3ForCausalLM(small_config()).eval()
    prompt = torch.tensor([[1, 2, 3]])
    generated = model.greedy_generate(prompt, max_new_tokens=3, eos_token_id=[])
    assert generated.shape == (1, 6)
    assert torch.equal(generated[:, :3], prompt)


def test_integrated_kda_module_replays_frozen_p3_mixer() -> None:
    bridge = small_bridge_config()
    converted = convert_gdn_to_kda(small_source_state(), bridge)
    config = DenseK3Config(
        vocab_size=32,
        hidden_size=bridge.hidden_size,
        intermediate_size=16,
        num_hidden_layers=1,
        mixer_types=["kda"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        partial_rotary_factor=0.5,
        kda_qk_num_heads=bridge.qk_num_heads,
        kda_value_num_heads=bridge.value_num_heads,
        kda_qk_head_dim=bridge.qk_head_dim,
        kda_value_head_dim=bridge.value_head_dim,
        kda_decay_projection_rank=bridge.decay_projection_rank,
        kda_conv_kernel_size=bridge.conv_kernel_size,
        kda_backend="reference",
    )
    module = DenseK3KDA(config, layer_idx=0).eval()
    module.load_state_dict(converted.target_state, strict=True)
    oracle = DenseK3QwenCompatMixer(bridge, converted.target_state)
    torch.manual_seed(11)
    hidden = torch.randn(2, 7, bridge.hidden_size)
    with torch.no_grad():
        actual = module(hidden, torch.ones(2, 7), cache=None, use_cache=False)
        expected = oracle.forward(hidden).output
    assert torch.allclose(expected, actual, rtol=1e-6, atol=1e-6)


def test_indexed_runtime_loader_materializes_and_ties_small_model(tmp_path: Path) -> None:
    torch.manual_seed(13)
    source = DenseK3ForCausalLM(small_config()).eval()
    writer = DeterministicShardWriter(tmp_path, max_shard_size=2_000)
    for name, parameter in source.named_parameters():
        writer.add(name, parameter)
    writer.finish()
    source.config.save_pretrained(tmp_path)
    loaded = load_densek3_model(tmp_path, device="cpu", dtype=torch.float32)
    tokens = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        expected = source(tokens, use_cache=False).logits
        actual = loaded(tokens, use_cache=False).logits
    assert torch.allclose(expected, actual, rtol=1e-7, atol=1e-7)
    assert loaded.lm_head.weight is loaded.model.embed_tokens.weight
    assert not any(parameter.device.type == "meta" for parameter in loaded.parameters())
