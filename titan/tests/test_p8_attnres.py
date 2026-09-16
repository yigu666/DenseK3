from __future__ import annotations

# ruff: noqa: E402, I001 -- Titan overlay paths are intentional.

import math
import sys
from pathlib import Path

import torch
import yaml

TITAN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TITAN))
sys.path.insert(0, str(TITAN / "scripts"))

from attnres.block_reference import (
    AttnResReadSite,
    BlockAttnResReference,
    disable_block_attnres_runtime,
    enable_block_attnres_runtime,
)
from attnres.contracts import FINAL_SITE, TOTAL_ROUTING_SITES
from attnres.metrics import tensor_metrics
from attnres.state import DepthBlockState
from densek3_core.modeling import DenseK3Config, DenseK3ForCausalLM


def tiny_config() -> DenseK3Config:
    return DenseK3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        mixer_types=["kda", "kda", "kda", "mla"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        mla_kv_lora_rank=8,
        mla_qk_nope_head_dim=4,
        mla_value_head_dim=4,
        mla_qk_norm=True,
        partial_rotary_factor=0.5,
        kda_qk_num_heads=2,
        kda_value_num_heads=4,
        kda_qk_head_dim=4,
        kda_value_head_dim=4,
        kda_decay_projection_rank=4,
        kda_gate_mode="kimi_k3",
        kda_gate_lower_bound=-5.0,
        kda_output_gate_activation="silu",
        kda_backend="reference",
        mrope_section=(1, 0, 0),
    )


def test_depth_state_partitions_standard_residual_stream() -> None:
    torch.manual_seed(1)
    embedding = torch.randn(2, 3, 8, dtype=torch.float64)
    state = DepthBlockState(embedding, block_size_residuals=2)
    stream = embedding
    for _ in range(6):
        branch = torch.randn_like(embedding)
        stream = stream + branch
        state.update(branch)
        assert torch.allclose(state.reconstruct(), stream, rtol=1e-15, atol=1e-15)
    assert state.completed_residual_blocks == 3
    assert state.partial is None


def test_zero_query_is_uniform_and_direction_preserving() -> None:
    torch.manual_seed(2)
    site = AttnResReadSite(hidden_size=8, eps=1e-6)
    sources = tuple(torch.randn(2, 5, 8) for _ in range(5))
    output, alpha, scores = site(sources)
    expected = sum(sources) / len(sources)
    assert torch.equal(scores, torch.zeros_like(scores))
    assert torch.equal(alpha.max(dim=0).values, alpha.min(dim=0).values)
    assert torch.allclose(output, expected, rtol=1e-6, atol=1e-6)
    entropy = -(alpha.double() * alpha.double().log()).sum(dim=0)
    assert torch.allclose(entropy, torch.full_like(entropy, math.log(5)), atol=1e-12)


def test_reference_has_64_reads_plus_final_and_zero_init() -> None:
    runtime = BlockAttnResReference(16, 1e-6)
    contract = runtime.parameter_contract()
    assert len(runtime.sites) == TOTAL_ROUTING_SITES
    assert FINAL_SITE == 64
    assert contract == {
        "site_count": 65,
        "query_count": 65,
        "norm_count": 65,
        "parameter_count": 65 * 16 * 2,
        "all_fp32": True,
        "all_queries_zero": True,
        "all_norms_one": True,
    }


def test_n1_zero_query_replays_standard_prenorm_model() -> None:
    torch.manual_seed(3)
    model = DenseK3ForCausalLM(tiny_config()).eval()
    tokens = torch.randint(1, model.config.vocab_size, (1, 7))
    standard = model(tokens, use_cache=False, return_dict=True).logits
    runtime = BlockAttnResReference(
        model.config.hidden_size,
        model.config.rms_norm_eps,
        block_size_residuals=8,
    )
    runtime.capture_diagnostics = True
    enable_block_attnres_runtime(model, runtime)
    try:
        actual = model(tokens, use_cache=False, return_dict=True).logits
    finally:
        disable_block_attnres_runtime(model)
    metrics = tensor_metrics(actual, standard)
    assert metrics["relative_l2"] < 0.01
    assert metrics["cosine"] > 0.9999
    assert len(runtime.routing_records) == 9
    assert runtime.routing_records[-1]["label"] == "final_output"
    assert all(item["scores_zero"] for item in runtime.routing_records)
    assert all(item["alpha_uniform_exact"] for item in runtime.routing_records)


def test_depth_state_is_not_added_to_sequence_cache() -> None:
    torch.manual_seed(4)
    model = DenseK3ForCausalLM(tiny_config()).eval()
    runtime = BlockAttnResReference(model.config.hidden_size, model.config.rms_norm_eps)
    enable_block_attnres_runtime(model, runtime)
    try:
        output = model(
            torch.randint(1, model.config.vocab_size, (1, 5)),
            use_cache=True,
            return_dict=True,
        )
    finally:
        disable_block_attnres_runtime(model)
    cache = output.past_key_values
    assert cache.seen_tokens == 5
    assert not hasattr(cache, "attnres")
    assert not hasattr(cache, "depth_blocks")


def test_p8_contract_freezes_titan_only_scope() -> None:
    config = yaml.safe_load(
        (TITAN / "configs/p8-titan-block-attnres.yaml").read_text(encoding="utf-8")
    )
    assert config["architecture"]["block_boundary_layers"] == [3, 7, 11, 15, 19, 23, 27, 31]
    assert config["architecture"]["routing_read_sites"] == 64
    assert config["architecture"]["final_aggregation_sites"] == 1
    assert config["frozen"]["p7_true_latent_cache"] is True
    assert config["frozen"]["original_weights_modified"] is False
    assert config["training_probe"]["max_effective_tokens"] == 131_072
    assert config["training_probe"]["heldout_accessed"] is False
    assert config["final"]["canonical_bf16_replay"] == "pending"


def test_routing_probe_keeps_heldout_out_of_evaluation() -> None:
    source = (TITAN / "scripts/run_p8_routing_probe.py").read_text(encoding="utf-8")
    assert 'PackedTokenDataset(DATA / "packed", "heldout")' not in source
    assert 'objective": "causal_lm_ce_only"' in source
    assert "for parameter in model.parameters():" in source
    assert "parameter.requires_grad_(False)" in source
    assert "P9_CANONICAL_ALLOWED=false" in source


def test_checkpointed_routing_backward_reaches_all_new_parameters() -> None:
    torch.manual_seed(5)
    model = DenseK3ForCausalLM(tiny_config())
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.train()
    model.model.gradient_checkpointing = True
    runtime = BlockAttnResReference(model.config.hidden_size, model.config.rms_norm_eps)
    runtime.train()
    enable_block_attnres_runtime(model, runtime)
    try:
        tokens = torch.randint(1, model.config.vocab_size, (1, 7))
        loss = model(tokens, labels=tokens, use_cache=False, return_dict=True).loss
        loss.backward()
    finally:
        disable_block_attnres_runtime(model)
    named = dict(runtime.named_parameters())
    used_names = [
        name
        for name in named
        if any(name.startswith(f"sites.{site}.") for site in (*range(8), 64))
    ]
    assert len(used_names) == 18
    assert all(named[name].grad is not None for name in used_names)
    assert all(torch.isfinite(named[name].grad).all() for name in used_names)
    assert all(
        parameter.grad is None
        for name, parameter in named.items()
        if name not in used_names
    )
