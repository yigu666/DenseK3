from pathlib import Path

import pytest
import yaml

from densek3_core.config import load_p1_config


def test_locked_p1_config_is_loaded_without_redefinition() -> None:
    config = load_p1_config()
    assert config.schema_version == 1
    assert config.status == "p1_architecture_locked"
    assert config.source_path.name == "densek3-4b-p1.yaml"
    assert config.kda.qk_num_heads == config.raw["kda"]["qk_num_heads"] == 16
    assert config.kda.value_num_heads == config.raw["kda"]["value_num_heads"] == 32
    assert config.kda.value_heads_per_qk_head == 2
    assert config.kda.qk_head_dim == config.kda.value_head_dim == 128
    assert config.kda.recurrent_state_dtype == "float32"
    assert config.kda.recurrent_state_layout == "v_first"
    assert config.kda.training_backend == "fla_chunk"
    assert config.kda.short_inference_backend == "fla_fused_recurrent"


def test_layer_topology_matches_p1() -> None:
    config = load_p1_config()
    layer_types = config.layers["layer_types"]
    assert len(layer_types) == config.model["num_hidden_layers"] == 32
    assert layer_types.count("kda") == 24
    assert [index for index, layer_type in enumerate(layer_types) if layer_type == "gqa"] == [3, 7, 11, 15, 19, 23, 27, 31]


def test_loader_rejects_unlocked_copy(tmp_path: Path) -> None:
    source = load_p1_config()
    raw = dict(source.raw)
    raw["status"] = "draft"
    path = tmp_path / "p1.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="not locked"):
        load_p1_config(path)

