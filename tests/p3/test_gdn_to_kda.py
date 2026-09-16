from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from densek3_core.transplant.gdn_to_kda import (
    ALLOWED_OPERATIONS,
    FORBIDDEN_OPERATIONS,
    SOURCE_TENSOR_NAMES,
    GDNBridgeConfig,
    audit_source_state,
    convert_gdn_to_kda,
    hash_json,
    hash_state_dict,
    load_saved_conversion,
    save_conversion,
)
from tests.p3.helpers import small_bridge_config, small_source_state

ROOT = Path(__file__).resolve().parents[2]


def test_source_audit_requires_exact_names_and_shapes() -> None:
    config = small_bridge_config()
    source = small_source_state()
    audit = audit_source_state(source, config)
    assert audit["status"] == "PASS"
    assert audit["tensor_count"] == 9
    broken = dict(source)
    broken["dt_bias"] = torch.zeros(5)
    with pytest.raises(ValueError, match="source audit failed"):
        audit_source_state(broken, config)


@pytest.mark.skipif(
    not (ROOT / "models/Qwen3.5-4B-Base/model.safetensors.index.json").is_file(),
    reason="Qwen donor metadata is downloaded separately",
)
def test_real_qwen_metadata_locks_layer_zero_source_contract() -> None:
    model_dir = ROOT / "models/Qwen3.5-4B-Base"
    config = GDNBridgeConfig.from_model_directory(model_dir, layer_index=0)
    assert (config.hidden_size, config.qk_num_heads, config.value_num_heads) == (2560, 16, 32)
    assert (config.qk_head_dim, config.value_head_dim, config.conv_kernel_size) == (128, 128, 4)
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    actual = sorted(name for name in index["weight_map"] if name.startswith(config.source_prefix))
    expected = sorted(config.source_prefix + name for name in SOURCE_TENSOR_NAMES)
    assert actual == expected


def test_conversion_is_deterministic_and_lossless_where_required() -> None:
    config = small_bridge_config()
    source = small_source_state()
    first = convert_gdn_to_kda(source, config)
    second = convert_gdn_to_kda(source, config)
    assert first.parameter_hash == second.parameter_hash
    assert first.manifest_hash == second.manifest_hash

    qk = config.qk_dim
    qkv = source["in_proj_qkv.weight"]
    assert torch.equal(first.target_state["q_proj.weight"], qkv[:qk])
    assert torch.equal(first.target_state["k_proj.weight"], qkv[qk : 2 * qk])
    assert torch.equal(first.target_state["v_proj.weight"], qkv[2 * qk :])
    assert torch.equal(
        first.target_state["dt_bias"], source["dt_bias"].repeat_interleave(config.qk_head_dim)
    )


def test_scalar_decay_projection_embeds_each_head_in_every_channel() -> None:
    config = small_bridge_config()
    source = small_source_state()
    converted = convert_gdn_to_kda(source, config).target_state
    hidden = torch.randn(2, 7, config.hidden_size, generator=torch.Generator().manual_seed(42))
    source_scalar = F.linear(hidden, source["in_proj_a.weight"])
    target_channel = F.linear(
        F.linear(hidden, converted["f_a_proj.weight"]), converted["f_b_proj.weight"]
    ).view(2, 7, config.value_num_heads, config.qk_head_dim)
    expected = source_scalar.unsqueeze(-1).expand_as(target_channel)
    torch.testing.assert_close(target_channel, expected, rtol=0, atol=0)


def test_manifest_has_complete_coverage_and_no_implicit_initialization() -> None:
    result = convert_gdn_to_kda(small_source_state(), small_bridge_config())
    manifest = result.manifest
    operations = {entry["operation"] for entry in manifest["entries"]}
    assert operations <= ALLOWED_OPERATIONS
    assert not operations & FORBIDDEN_OPERATIONS
    assert manifest["source_tensor_count"] == 9
    assert manifest["target_tensor_count"] == 14
    assert manifest["source_total_parameters"] == sum(tensor.numel() for tensor in small_source_state().values())
    assert manifest["target_total_parameters"] == sum(tensor.numel() for tensor in result.target_state.values())


def test_save_reload_preserves_parameter_and_manifest_hashes(tmp_path) -> None:
    result = convert_gdn_to_kda(small_source_state(), small_bridge_config())
    report = save_conversion(result, tmp_path)
    state, manifest, reloaded_report = load_saved_conversion(tmp_path)
    assert report == reloaded_report
    assert hash_state_dict(state) == result.parameter_hash
    assert hash_json(manifest) == result.manifest_hash
    assert json.loads((tmp_path / "conversion-report.json").read_text())["parameter_hash"] == result.parameter_hash
