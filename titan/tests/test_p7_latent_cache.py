from __future__ import annotations

# ruff: noqa: E402, I001 -- the Titan scripts directory is an intentional overlay.

import hashlib
import sys
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from densek3_core.modeling import DenseK3Config, DenseK3ForCausalLM
from densek3_core.modeling.hybrid_cache import DenseK3HybridCache
from densek3_core.modeling.modeling_densek3 import DenseK3MLA
from p7_latent_cache import (
    LatentCacheLayer,
    TitanP7HybridCache,
    disable_p7_latent_cache_runtime,
    enable_p7_latent_cache_runtime,
    reconstruct_kv_from_cache_layer,
    reconstruct_kv_from_latent,
)


def small_mla_config() -> DenseK3Config:
    return DenseK3Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=4,
        mixer_types=["kda", "kda", "kda", "mla"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        mla_kv_lora_rank=4,
        mla_qk_nope_head_dim=2,
        mla_value_head_dim=2,
        mla_qk_norm=True,
        partial_rotary_factor=1.0,
        kda_qk_num_heads=2,
        kda_value_num_heads=4,
        kda_qk_head_dim=2,
        kda_value_head_dim=2,
        kda_decay_projection_rank=4,
        kda_gate_mode="kimi_k3",
        kda_gate_lower_bound=-5.0,
        kda_output_gate_activation="silu",
        kda_backend="reference",
        mrope_section=(1, 0, 0),
    )


def parameter_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in module.named_parameters():
        digest.update(name.encode())
        digest.update(value.detach().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def test_latent_layer_append_clone_reorder_and_reset() -> None:
    layer = LatentCacheLayer(latent_dim=4)
    first = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    second = torch.full((2, 1, 4), 99.0)
    assert layer.update(first).shape == (2, 3, 4)
    assert layer.update(second).shape == (2, 4, 4)
    assert layer.append_lengths == [3, 1]
    assert [segment.shape[1] for segment in layer.segments()] == [3, 1]
    cloned = layer.clone()
    assert cloned.append_lengths == [3, 1]
    cloned.reorder(torch.tensor([1, 0]))
    assert torch.equal(cloned.latent[0], layer.latent[1])
    assert torch.equal(layer.latent[0, 0], first[0, 0])
    cloned.repeat_interleave(2)
    assert cloned.latent.shape == (4, 4, 4)
    cloned.reset()
    assert not cloned.initialized
    assert cloned.append_lengths == []


def test_segmented_reconstruction_preserves_projection_write_shapes() -> None:
    torch.manual_seed(11)
    config = small_mla_config()
    mla = DenseK3MLA(config, 3).eval()
    first = torch.randn(1, 7, config.mla_kv_lora_rank)
    second = torch.randn(1, 1, config.mla_kv_lora_rank)
    layer = LatentCacheLayer(latent_dim=config.mla_kv_lora_rank)
    layer.update(first)
    layer.update(second)

    expected_first = reconstruct_kv_from_latent(mla, first)
    expected_second = reconstruct_kv_from_latent(mla, second)
    actual_key, actual_value = reconstruct_kv_from_cache_layer(mla, layer)

    assert torch.equal(
        actual_key,
        torch.cat((expected_first[0], expected_second[0]), dim=2),
    )
    assert torch.equal(
        actual_value,
        torch.cat((expected_first[1], expected_second[1]), dim=2),
    )


def test_p7_cache_persists_only_latent_for_selected_mla() -> None:
    config = small_mla_config()
    TitanP7HybridCache.latent_layer_indices = (3,)
    cache = TitanP7HybridCache(config)
    cache.update_latent(torch.randn(2, 5, 4), 3)
    cache.advance(5)
    contract = cache.latent_contract()
    assert contract["latent_cache_layers"] == [3]
    assert contract["latent_shapes"] == {"3": [2, 5, 4]}
    assert contract["seen_tokens"] == 5
    assert contract["bytes"]["latent"] == 2 * 5 * 4 * 4
    assert contract["bytes"]["persistent_expanded_k_numel"] == 0
    assert contract["bytes"]["persistent_expanded_v_numel"] == 0


def test_cached_normalized_latent_reconstructs_original_kv() -> None:
    torch.manual_seed(7)
    config = small_mla_config()
    mla = DenseK3MLA(config, 3).eval()
    hidden = torch.randn(2, 5, config.hidden_size)
    _, reference_k, reference_v, _, raw_latent = mla.project_qkvg(hidden)
    cached = mla.kv_a_layernorm(raw_latent)
    actual_k, actual_v = reconstruct_kv_from_latent(mla, cached)
    assert torch.equal(actual_k, reference_k)
    assert torch.equal(actual_v, reference_v)


def test_incremental_latent_cache_matches_expanded_reference_and_preserves_weights() -> None:
    torch.manual_seed(11)
    config = small_mla_config()
    mla = DenseK3MLA(config, 3).eval()
    hidden = torch.randn(1, 7, config.hidden_size)
    positions = torch.arange(7).unsqueeze(0)
    full = mla(hidden, positions, torch.ones(1, 7), None, False)
    before = parameter_hash(mla)

    expanded = DenseK3HybridCache(config)
    mla(hidden[:, :3], positions[:, :3], torch.ones(1, 3), expanded, True)
    expanded.advance(3)
    expanded_pieces = []
    for index in range(3, 7):
        expanded_pieces.append(
            mla(
                hidden[:, index : index + 1],
                positions[:, index : index + 1],
                torch.ones(1, index + 1),
                expanded,
                True,
            )
        )
        expanded.advance(1)

    enable_p7_latent_cache_runtime((3,))
    try:
        latent = TitanP7HybridCache(config)
        mla(hidden[:, :3], positions[:, :3], torch.ones(1, 3), latent, True)
        latent.advance(3)
        branch = latent.clone()
        latent_pieces = []
        for index in range(3, 7):
            latent_pieces.append(
                mla(
                    hidden[:, index : index + 1],
                    positions[:, index : index + 1],
                    torch.ones(1, index + 1),
                    latent,
                    True,
                )
            )
            latent.advance(1)
        assert branch.seen_tokens == 3
        assert branch.latent_layer(3).sequence_length == 3
    finally:
        disable_p7_latent_cache_runtime()

    expanded_output = torch.cat(expanded_pieces, dim=1)
    latent_output = torch.cat(latent_pieces, dim=1)
    assert torch.allclose(expanded_output, full[:, 3:], rtol=1e-5, atol=1e-6)
    assert torch.allclose(latent_output, expanded_output, rtol=1e-5, atol=1e-6)
    assert parameter_hash(mla) == before
    assert latent.cache_bytes()["persistent_expanded_k_numel"] == 0
    assert latent.cache_bytes()["persistent_expanded_v_numel"] == 0


def test_model_automatically_materializes_p7_hybrid_cache() -> None:
    torch.manual_seed(13)
    config = small_mla_config()
    model = DenseK3ForCausalLM(config).eval()
    enable_p7_latent_cache_runtime((3,))
    try:
        prefix = model(torch.tensor([[1, 2, 3]]), use_cache=True, return_dict=True)
        assert isinstance(prefix.past_key_values, TitanP7HybridCache)
        assert prefix.past_key_values.seen_tokens == 3
        assert prefix.past_key_values.latent_layer(3).sequence_length == 3
        continuation = model(
            torch.tensor([[4]]),
            past_key_values=prefix.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        assert continuation.past_key_values.seen_tokens == 4
        assert continuation.past_key_values.latent_layer(3).sequence_length == 4
    finally:
        disable_p7_latent_cache_runtime()
